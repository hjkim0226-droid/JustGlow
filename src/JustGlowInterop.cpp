/**
 * JustGlow DX12-CUDA Interop Implementation
 *
 * Implements shared resource management between DirectX 12 and CUDA.
 * Uses D3D12_HEAP_FLAG_SHARED + cudaImportExternalMemory for zero-copy sharing.
 *
 * Key implementation details:
 * - Shared Texture2D for DX12 SampleLevel() + CUDA surf2D access
 * - Fence synchronization via cudaExternalSemaphore
 * - Resource state transitions for CUDA access (COMMON state required)
 */

#ifdef _WIN32

#include "JustGlowInterop.h"
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <cstdarg>

// ============================================================================
// Debug Logging
// ============================================================================

#define JUSTGLOW_INTEROP_LOGGING 1

#if JUSTGLOW_INTEROP_LOGGING
static std::wstring GetInteropLogFilePath() {
    wchar_t tempPath[MAX_PATH];
    GetTempPathW(MAX_PATH, tempPath);
    return std::wstring(tempPath) + L"JustGlow_Interop_debug.log";
}

static void InteropLogMessage(const char* format, ...) {
    static std::ofstream logFile;
    static bool initialized = false;

    if (!initialized) {
        logFile.open(GetInteropLogFilePath(), std::ios::out | std::ios::trunc);
        initialized = true;
    }

    if (logFile.is_open()) {
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        logFile << std::put_time(&tm, "[%H:%M:%S] ");

        char buffer[1024];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);

        logFile << buffer << std::endl;
        logFile.flush();
    }
}

#define INTEROP_LOG(fmt, ...) InteropLogMessage(fmt, ##__VA_ARGS__)
#else
#define INTEROP_LOG(fmt, ...) ((void)0)
#endif

// ============================================================================
// Constructor / Destructor
// ============================================================================

JustGlowInterop::JustGlowInterop()
    : m_device(nullptr)
    , m_commandQueue(nullptr)
    , m_cudaContext(nullptr)
    , m_descriptorSize(0)
    , m_initialized(false)
{
}

JustGlowInterop::~JustGlowInterop() {
    Shutdown();
}

// ============================================================================
// Error Handling
// ============================================================================

bool JustGlowInterop::CheckCUDAError(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        const char* errName = cudaGetErrorName(err);
        const char* errStr = cudaGetErrorString(err);
        INTEROP_LOG("CUDA ERROR in %s: %s (%s)",
            context, errName ? errName : "Unknown", errStr ? errStr : "");
        return false;
    }
    return true;
}

bool JustGlowInterop::CheckDX12Error(HRESULT hr, const char* context) {
    if (FAILED(hr)) {
        INTEROP_LOG("DX12 ERROR in %s: HRESULT 0x%08X", context, hr);
        return false;
    }
    return true;
}

// ============================================================================
// Initialize / Shutdown
// ============================================================================

bool JustGlowInterop::Initialize(
    ID3D12Device* device,
    ID3D12CommandQueue* commandQueue,
    CUcontext cudaContext)
{
    INTEROP_LOG("=== JustGlow Interop Initialize ===");

    if (!device || !commandQueue || !cudaContext) {
        INTEROP_LOG("ERROR: Invalid parameters (device=%p, queue=%p, cuda=%p)",
            device, commandQueue, cudaContext);
        return false;
    }

    m_device = device;
    m_commandQueue = commandQueue;
    m_cudaContext = cudaContext;

    // Get descriptor size for SRV/UAV heap calculations
    m_descriptorSize = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    INTEROP_LOG("Interop initialized: device=%p, queue=%p, cudaContext=%p",
        m_device, m_commandQueue, m_cudaContext);
    INTEROP_LOG("Descriptor size: %u", m_descriptorSize);

    m_initialized = true;
    return true;
}

void JustGlowInterop::Shutdown() {
    INTEROP_LOG("=== JustGlow Interop Shutdown ===");
    m_device = nullptr;
    m_commandQueue = nullptr;
    m_cudaContext = nullptr;
    m_initialized = false;
}

// ============================================================================
// Shared Texture Creation
// ============================================================================

bool JustGlowInterop::CreateSharedTexture(
    int width, int height,
    InteropTexture& texture,
    ID3D12DescriptorHeap* descriptorHeap,
    UINT srvIndex, UINT uavIndex)
{
    INTEROP_LOG("CreateSharedTexture: %dx%d, srvIdx=%u, uavIdx=%u",
        width, height, srvIndex, uavIndex);

    if (!m_initialized) {
        INTEROP_LOG("ERROR: Interop not initialized");
        return false;
    }

    // Reset any existing texture
    texture.reset();
    texture.width = width;
    texture.height = height;
    texture.sizeBytes = static_cast<size_t>(width) * height * 4 * sizeof(float);  // RGBA32F

    // ========================================
    // Step 1: Create DX12 Shared Texture2D
    // ========================================

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Alignment = 0;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;  // RGBA32F
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // D3D12_HEAP_FLAG_SHARED is the key for interop!
    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_SHARED,  // Critical: enables sharing with CUDA
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&texture.d3d12Resource));

    if (!CheckDX12Error(hr, "CreateCommittedResource (shared texture)")) {
        return false;
    }

    INTEROP_LOG("DX12 Texture2D created: %dx%d, format=RGBA32F", width, height);

    // ========================================
    // Step 2: Create Shared Handle
    // ========================================

    hr = m_device->CreateSharedHandle(
        texture.d3d12Resource.Get(),
        nullptr,
        GENERIC_ALL,
        nullptr,  // Named handle not needed
        &texture.sharedHandle);

    if (!CheckDX12Error(hr, "CreateSharedHandle")) {
        texture.reset();
        return false;
    }

    INTEROP_LOG("Shared handle created: %p", texture.sharedHandle);

    // ========================================
    // Step 3: Create SRV and UAV descriptors
    // ========================================

    if (descriptorHeap) {
        texture.srvIndex = srvIndex;
        texture.uavIndex = uavIndex;

        D3D12_CPU_DESCRIPTOR_HANDLE heapStart = descriptorHeap->GetCPUDescriptorHandleForHeapStart();

        // SRV
        texture.srvCpuHandle.ptr = heapStart.ptr + srvIndex * m_descriptorSize;
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Texture2D.MostDetailedMip = 0;
        m_device->CreateShaderResourceView(texture.d3d12Resource.Get(), &srvDesc, texture.srvCpuHandle);

        // UAV
        texture.uavCpuHandle.ptr = heapStart.ptr + uavIndex * m_descriptorSize;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = 0;
        m_device->CreateUnorderedAccessView(texture.d3d12Resource.Get(), nullptr, &uavDesc, texture.uavCpuHandle);

        // GPU handles (for shader binding)
        D3D12_GPU_DESCRIPTOR_HANDLE gpuStart = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
        texture.srvGpuHandle.ptr = gpuStart.ptr + srvIndex * m_descriptorSize;
        texture.uavGpuHandle.ptr = gpuStart.ptr + uavIndex * m_descriptorSize;

        INTEROP_LOG("Descriptors created: SRV=%u, UAV=%u", srvIndex, uavIndex);
    }

    // ========================================
    // Step 4: Import to CUDA
    // ========================================

    if (!ImportTextureToGPU(texture)) {
        texture.reset();
        return false;
    }

    INTEROP_LOG("Shared texture created successfully: %dx%d", width, height);
    return true;
}

bool JustGlowInterop::ImportTextureToGPU(InteropTexture& texture) {
    INTEROP_LOG("Importing texture to CUDA: %dx%d", texture.width, texture.height);

    // ========================================
    // Step 1: Import External Memory
    // ========================================

    cudaExternalMemoryHandleDesc memDesc = {};
    memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    memDesc.handle.win32.handle = texture.sharedHandle;
    memDesc.size = texture.sizeBytes;
    memDesc.flags = cudaExternalMemoryDedicated;

    cudaError_t err = cudaImportExternalMemory(&texture.cudaExtMem, &memDesc);
    if (!CheckCUDAError(err, "cudaImportExternalMemory")) {
        return false;
    }

    INTEROP_LOG("External memory imported: size=%zu bytes", texture.sizeBytes);

    // ========================================
    // Step 2: Map as Mipmapped Array
    // ========================================

    // For Texture2D, we use cudaExternalMemoryGetMappedMipmappedArray
    cudaExternalMemoryMipmappedArrayDesc arrayDesc = {};
    arrayDesc.offset = 0;
    arrayDesc.formatDesc = cudaCreateChannelDesc<float4>();  // RGBA32F
    arrayDesc.extent = make_cudaExtent(texture.width, texture.height, 0);  // 2D
    arrayDesc.flags = cudaArraySurfaceLoadStore;  // Enable surf2D access
    arrayDesc.numLevels = 1;

    err = cudaExternalMemoryGetMappedMipmappedArray(
        &texture.cudaMipArray, texture.cudaExtMem, &arrayDesc);
    if (!CheckCUDAError(err, "cudaExternalMemoryGetMappedMipmappedArray")) {
        return false;
    }

    INTEROP_LOG("Mipmapped array created");

    // ========================================
    // Step 3: Get Level 0 Array
    // ========================================

    err = cudaGetMipmappedArrayLevel(&texture.cudaArray, texture.cudaMipArray, 0);
    if (!CheckCUDAError(err, "cudaGetMipmappedArrayLevel")) {
        return false;
    }

    // ========================================
    // Step 4: Create Surface Object
    // ========================================

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texture.cudaArray;

    err = cudaCreateSurfaceObject(&texture.cudaSurface, &resDesc);
    if (!CheckCUDAError(err, "cudaCreateSurfaceObject")) {
        return false;
    }

    INTEROP_LOG("Surface object created: %llu", (unsigned long long)texture.cudaSurface);

    // ========================================
    // Step 5: Create Texture Object (optional, for tex2D)
    // ========================================

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;  // Bilinear filtering
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;  // Use [0, 1] coordinates

    err = cudaCreateTextureObject(&texture.cudaTexture, &resDesc, &texDesc, nullptr);
    if (!CheckCUDAError(err, "cudaCreateTextureObject")) {
        // Not critical, surface object is sufficient
        texture.cudaTexture = 0;
        INTEROP_LOG("Warning: Texture object creation failed (optional)");
    } else {
        INTEROP_LOG("Texture object created: %llu", (unsigned long long)texture.cudaTexture);
    }

    return true;
}

void JustGlowInterop::DestroySharedTexture(InteropTexture& texture) {
    INTEROP_LOG("DestroySharedTexture: %dx%d", texture.width, texture.height);
    texture.reset();
}

// ============================================================================
// Fence Synchronization
// ============================================================================

bool JustGlowInterop::CreateFence(InteropFence& fence) {
    INTEROP_LOG("CreateFence");

    if (!m_initialized) {
        INTEROP_LOG("ERROR: Interop not initialized");
        return false;
    }

    fence.reset();

    // ========================================
    // Step 1: Create DX12 Shared Fence
    // ========================================

    HRESULT hr = m_device->CreateFence(
        0,
        D3D12_FENCE_FLAG_SHARED,  // Critical: enables sharing with CUDA
        IID_PPV_ARGS(&fence.d3d12Fence));

    if (!CheckDX12Error(hr, "CreateFence")) {
        return false;
    }

    fence.fenceValue = 0;

    // ========================================
    // Step 2: Create Shared Handle
    // ========================================

    hr = m_device->CreateSharedHandle(
        fence.d3d12Fence.Get(),
        nullptr,
        GENERIC_ALL,
        nullptr,
        &fence.sharedHandle);

    if (!CheckDX12Error(hr, "CreateSharedHandle (fence)")) {
        fence.reset();
        return false;
    }

    INTEROP_LOG("DX12 fence created with shared handle: %p", fence.sharedHandle);

    // ========================================
    // Step 3: Import to CUDA
    // ========================================

    cudaExternalSemaphoreHandleDesc semDesc = {};
    semDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    semDesc.handle.win32.handle = fence.sharedHandle;
    semDesc.flags = 0;

    cudaError_t err = cudaImportExternalSemaphore(&fence.cudaSemaphore, &semDesc);
    if (!CheckCUDAError(err, "cudaImportExternalSemaphore")) {
        fence.reset();
        return false;
    }

    INTEROP_LOG("Fence imported to CUDA successfully");
    return true;
}

void JustGlowInterop::DestroyFence(InteropFence& fence) {
    INTEROP_LOG("DestroyFence");
    fence.reset();
}

void JustGlowInterop::SignalFromDX12(InteropFence& fence) {
    fence.fenceValue++;
    HRESULT hr = m_commandQueue->Signal(fence.d3d12Fence.Get(), fence.fenceValue);
    if (FAILED(hr)) {
        INTEROP_LOG("ERROR: DX12 Signal failed: 0x%08X", hr);
    } else {
        INTEROP_LOG("DX12 Signal: value=%llu", fence.fenceValue);
    }
}

void JustGlowInterop::WaitOnCUDA(InteropFence& fence, CUstream stream) {
    cudaExternalSemaphoreWaitParams params = {};
    params.params.fence.value = fence.fenceValue;
    params.flags = 0;

    cudaError_t err = cudaWaitExternalSemaphoresAsync(
        &fence.cudaSemaphore, &params, 1,
        reinterpret_cast<cudaStream_t>(stream));

    if (err != cudaSuccess) {
        INTEROP_LOG("ERROR: CUDA Wait failed: %s", cudaGetErrorString(err));
    } else {
        INTEROP_LOG("CUDA Wait: value=%llu", fence.fenceValue);
    }
}

void JustGlowInterop::SignalFromCUDA(InteropFence& fence, CUstream stream) {
    fence.fenceValue++;

    cudaExternalSemaphoreSignalParams params = {};
    params.params.fence.value = fence.fenceValue;
    params.flags = 0;

    cudaError_t err = cudaSignalExternalSemaphoresAsync(
        &fence.cudaSemaphore, &params, 1,
        reinterpret_cast<cudaStream_t>(stream));

    if (err != cudaSuccess) {
        INTEROP_LOG("ERROR: CUDA Signal failed: %s", cudaGetErrorString(err));
    } else {
        INTEROP_LOG("CUDA Signal: value=%llu", fence.fenceValue);
    }
}

void JustGlowInterop::WaitOnDX12(InteropFence& fence) {
    HRESULT hr = m_commandQueue->Wait(fence.d3d12Fence.Get(), fence.fenceValue);
    if (FAILED(hr)) {
        INTEROP_LOG("ERROR: DX12 Wait failed: 0x%08X", hr);
    } else {
        INTEROP_LOG("DX12 Wait: value=%llu", fence.fenceValue);
    }
}

// ============================================================================
// Resource Transition Helpers
// ============================================================================

void JustGlowInterop::TransitionForCUDAAccess(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES currentState)
{
    if (currentState == D3D12_RESOURCE_STATE_COMMON) {
        return;  // Already in correct state
    }

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = currentState;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmdList->ResourceBarrier(1, &barrier);
    INTEROP_LOG("Transition to COMMON for CUDA access");
}

void JustGlowInterop::TransitionFromCUDAAccess(
    ID3D12GraphicsCommandList* cmdList,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES targetState)
{
    if (targetState == D3D12_RESOURCE_STATE_COMMON) {
        return;  // Already in correct state
    }

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = targetState;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmdList->ResourceBarrier(1, &barrier);
    INTEROP_LOG("Transition from COMMON to state %d", targetState);
}

#endif // _WIN32
