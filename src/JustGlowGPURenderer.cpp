/**
 * JustGlow GPU Renderer Implementation
 *
 * DirectX 12 Compute Shader based rendering pipeline.
 * Fully implements Dual Kawase blur with proper descriptor binding.
 */

#ifdef _WIN32

// Prevent Windows.h from defining min/max macros (conflicts with std::min/max)
#ifndef NOMINMAX
#define NOMINMAX
#endif

// IMPORTANT: CUDA headers MUST be included first to define vector types
// before JustGlowParams.h (which is included by JustGlowGPURenderer.h)
#if HAS_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "JustGlowInterop.h"
#include "JustGlowCUDARenderer.h"
#endif

#include "JustGlowGPURenderer.h"
#include <d3dcompiler.h>
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <cstdarg>
#include <algorithm>  // for std::min

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

// ============================================================================
// Debug Logging
// ============================================================================

#define JUSTGLOW_ENABLE_LOGGING 1

#if JUSTGLOW_ENABLE_LOGGING
static std::wstring GetLogFilePath() {
    wchar_t tempPath[MAX_PATH];
    GetTempPathW(MAX_PATH, tempPath);
    return std::wstring(tempPath) + L"JustGlow_debug.log";
}

static void LogMessage(const char* format, ...) {
    static std::ofstream logFile;
    static bool initialized = false;

    if (!initialized) {
        logFile.open(GetLogFilePath(), std::ios::out | std::ios::trunc);
        initialized = true;
    }

    if (logFile.is_open()) {
        // Timestamp
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        logFile << std::put_time(&tm, "[%H:%M:%S] ");

        // Message
        char buffer[1024];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);

        logFile << buffer << std::endl;
        logFile.flush();
    }
}

#define LOG(fmt, ...) LogMessage(fmt, ##__VA_ARGS__)
#else
#define LOG(fmt, ...) ((void)0)
#endif

// Thread group size for compute shaders (must match HLSL)
constexpr UINT THREAD_GROUP_SIZE = 16;

// Descriptor heap layout:
// [0-7]    : MIP chain SRVs (max 8 levels)
// [8-15]   : MIP chain UAVs (max 8 levels)
// [16-17]  : External input SRV pair (t0=input, t1=dummy for prefilter)
// [18-19]  : Composite SRV pair (t0=original, t1=glow/mip[0])
// [20]     : External output UAV
// [21-36]  : Downsample/Upsample SRV pairs (consecutive pairs for each pass)
constexpr UINT MAX_DESCRIPTORS = 64;
constexpr UINT MIP_SRV_START = 0;
constexpr UINT MIP_UAV_START = 8;
constexpr UINT PREFILTER_SRV_PAIR = 16;      // t0=input, t1=dummy
constexpr UINT COMPOSITE_SRV_PAIR = 18;      // t0=original, t1=glow
constexpr UINT EXTERNAL_OUTPUT_UAV = 20;
constexpr UINT PASS_SRV_PAIRS_START = 21;    // For downsample/upsample passes

// DispatchIndirect resource indices (for Refine shaders)
// SRV table must be consecutive: t0, t1, t2
constexpr UINT REFINE_SRV_START = 40;        // Start of SRV table for Refine shaders
constexpr UINT REFINE_INPUT_SRV = 40;        // t0: input texture
constexpr UINT REFINE_DUMMY_SRV = 41;        // t1: dummy (same as input)
constexpr UINT REFINE_BOUNDS_SRV = 42;       // t2: bounds input (StructuredBuffer)

// UAV table must be consecutive: u0, u1, u2, u3
constexpr UINT REFINE_UAV_START = 43;        // Start of UAV table for Refine shaders
constexpr UINT REFINE_OUTPUT_UAV = 43;       // u0: output texture
constexpr UINT ATOMIC_BOUNDS_UAV = 44;       // u1: atomic bounds
constexpr UINT INDIRECT_ARGS_UAV = 45;       // u2: indirect args
constexpr UINT BOUNDS_OUTPUT_UAV = 46;       // u3: bounds output

// Interop descriptor indices (for hybrid DX12-CUDA mode)
#if HAS_CUDA
constexpr UINT INTEROP_INPUT_SRV = 47;       // Interop input texture SRV
constexpr UINT INTEROP_INPUT_UAV = 48;       // Interop input texture UAV
constexpr UINT INTEROP_BLURRED_SRV_START = 49;  // Blurred levels SRV (6 levels)
constexpr UINT INTEROP_BLURRED_UAV_START = 55;  // Blurred levels UAV (6 levels)
constexpr UINT SCREEN_BLEND_OUTPUT_UAV = 61; // Screen blend output UAV
#endif

// ============================================================================
// Helper to get DLL module handle
// ============================================================================

static HMODULE GetCurrentModule() {
    HMODULE hModule = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&GetCurrentModule),
        &hModule);
    return hModule;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

JustGlowGPURenderer::JustGlowGPURenderer() = default;

JustGlowGPURenderer::~JustGlowGPURenderer() {
    Shutdown();
}

// ============================================================================
// Initialize
// ============================================================================

bool JustGlowGPURenderer::Initialize(
    ID3D12Device* device,
    ID3D12CommandQueue* commandQueue)
{
    LOG("=== JustGlow GPU Renderer Initialize ===");

    if (!device || !commandQueue) {
        LOG("ERROR: Invalid device or command queue");
        return false;
    }

    m_device = device;
    m_commandQueue = commandQueue;
    LOG("Device and command queue set");

    // Create command objects
    if (!CreateCommandObjects()) {
        LOG("ERROR: Failed to create command objects");
        return false;
    }
    LOG("Command objects created");

    // Create descriptor heaps
    if (!CreateDescriptorHeaps()) {
        LOG("ERROR: Failed to create descriptor heaps");
        return false;
    }
    LOG("Descriptor heaps created");

    // Create constant buffers
    if (!CreateConstantBuffers()) {
        LOG("ERROR: Failed to create constant buffers");
        return false;
    }
    LOG("Constant buffers created");

    // Create DispatchIndirect resources (for BoundingBox optimization)
    if (!CreateDispatchIndirectResources()) {
        LOG("ERROR: Failed to create DispatchIndirect resources");
        return false;
    }
    LOG("DispatchIndirect resources created");

    // Create Refine root signature (with additional UAVs)
    if (!CreateRefineRootSignature()) {
        LOG("ERROR: Failed to create Refine root signature");
        return false;
    }
    LOG("Refine root signature created");

    // Create command signature for ExecuteIndirect
    if (!CreateCommandSignature()) {
        LOG("ERROR: Failed to create command signature");
        return false;
    }
    LOG("Command signature created");

    // Load shaders
    if (!LoadShaders()) {
        LOG("ERROR: Failed to load shaders");
        return false;
    }
    LOG("Shaders loaded successfully");

    LOG("=== Initialize Complete ===");
    return true;
}

// ============================================================================
// Shutdown
// ============================================================================

void JustGlowGPURenderer::Shutdown() {
    WaitForGPU();

#if HAS_CUDA
    ShutdownInterop();
#endif

    ReleaseMipChain();

    if (m_constantBufferPtr) {
        m_constantBuffer->Unmap(0, nullptr);
        m_constantBufferPtr = nullptr;
    }

    if (m_blurPassBufferPtr) {
        m_blurPassBuffer->Unmap(0, nullptr);
        m_blurPassBufferPtr = nullptr;
    }

    // Release DispatchIndirect resources
    if (m_refineConstBufferPtr) {
        m_refineConstBuffer->Unmap(0, nullptr);
        m_refineConstBufferPtr = nullptr;
    }
    m_refineConstBuffer.Reset();
    m_atomicBoundsBuffer.Reset();
    m_indirectArgsBuffer.Reset();
    m_boundsOutputBuffer.Reset();
    m_dispatchIndirectSignature.Reset();
    m_refineRootSignature.Reset();

    if (m_fenceEvent) {
        CloseHandle(m_fenceEvent);
        m_fenceEvent = nullptr;
    }

    m_commandList.Reset();
    m_commandAllocator.Reset();
    m_fence.Reset();
    m_srvUavHeap.Reset();
    m_constantBuffer.Reset();
    m_blurPassBuffer.Reset();

    for (auto& shader : m_shaders) {
        shader.rootSignature.Reset();
        shader.pipelineState.Reset();
        shader.loaded = false;
    }

    m_device = nullptr;
    m_commandQueue = nullptr;
}

// ============================================================================
// Create Command Objects
// ============================================================================

bool JustGlowGPURenderer::CreateCommandObjects() {
    HRESULT hr;

    hr = m_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_PPV_ARGS(&m_commandAllocator));
    if (FAILED(hr)) return false;

    hr = m_device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        m_commandAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(&m_commandList));
    if (FAILED(hr)) return false;

    m_commandList->Close();

    hr = m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
    if (FAILED(hr)) return false;

    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent) return false;

    return true;
}

// ============================================================================
// Create Descriptor Heaps
// ============================================================================

bool JustGlowGPURenderer::CreateDescriptorHeaps() {
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = MAX_DESCRIPTORS;
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    HRESULT hr = m_device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_srvUavHeap));
    if (FAILED(hr)) return false;

    m_srvUavDescriptorSize = m_device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    return true;
}

// ============================================================================
// Create Constant Buffers
// ============================================================================

bool JustGlowGPURenderer::CreateConstantBuffers() {
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    // Main GlowParams constant buffer (b0)
    UINT cbSize0 = (sizeof(GlowParams) + 255) & ~255;
    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = cbSize0;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&m_constantBuffer));
    if (FAILED(hr)) return false;

    hr = m_constantBuffer->Map(0, nullptr, &m_constantBufferPtr);
    if (FAILED(hr)) return false;

    // BlurPassParams constant buffer (b1)
    UINT cbSize1 = (sizeof(BlurPassParams) + 255) & ~255;
    bufferDesc.Width = cbSize1;

    hr = m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&m_blurPassBuffer));
    if (FAILED(hr)) return false;

    hr = m_blurPassBuffer->Map(0, nullptr, &m_blurPassBufferPtr);
    if (FAILED(hr)) return false;

    return true;
}

// ============================================================================
// Create Root Signature with Samplers
// ============================================================================

bool JustGlowGPURenderer::CreateRootSignature(ComPtr<ID3D12RootSignature>& rootSig) {
    // Root parameters:
    // [0] CBV - GlowParams (b0)
    // [1] CBV - BlurPassParams (b1)
    // [2] Descriptor Table - SRVs (t0, t1)
    // [3] Descriptor Table - UAV (u0)

    D3D12_ROOT_PARAMETER rootParams[4] = {};

    // CBV for GlowParams (b0)
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].Descriptor.RegisterSpace = 0;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // CBV for BlurPassParams (b1)
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[1].Descriptor.ShaderRegister = 1;
    rootParams[1].Descriptor.RegisterSpace = 0;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // SRV descriptor table (t0, t1)
    D3D12_DESCRIPTOR_RANGE srvRange = {};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 2;
    srvRange.BaseShaderRegister = 0;
    srvRange.RegisterSpace = 0;
    srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[2].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[2].DescriptorTable.pDescriptorRanges = &srvRange;
    rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // UAV descriptor table (u0)
    D3D12_DESCRIPTOR_RANGE uavRange = {};
    uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange.NumDescriptors = 1;
    uavRange.BaseShaderRegister = 0;
    uavRange.RegisterSpace = 0;
    uavRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[3].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[3].DescriptorTable.pDescriptorRanges = &uavRange;
    rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Static samplers (s0 = linear, s1 = point)
    D3D12_STATIC_SAMPLER_DESC staticSamplers[2] = {};

    // Linear sampler (s0)
    staticSamplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    staticSamplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].MipLODBias = 0;
    staticSamplers[0].MaxAnisotropy = 1;
    staticSamplers[0].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    staticSamplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    staticSamplers[0].MinLOD = 0;
    staticSamplers[0].MaxLOD = D3D12_FLOAT32_MAX;
    staticSamplers[0].ShaderRegister = 0;
    staticSamplers[0].RegisterSpace = 0;
    staticSamplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Point sampler (s1)
    staticSamplers[1] = staticSamplers[0];
    staticSamplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    staticSamplers[1].ShaderRegister = 1;

    // Create root signature
    D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.NumParameters = 4;
    rootSigDesc.pParameters = rootParams;
    rootSigDesc.NumStaticSamplers = 2;
    rootSigDesc.pStaticSamplers = staticSamplers;
    rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serializedRootSig;
    ComPtr<ID3DBlob> errorBlob;

    HRESULT hr = D3D12SerializeRootSignature(
        &rootSigDesc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRootSig,
        &errorBlob);
    if (FAILED(hr)) return false;

    hr = m_device->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(&rootSig));
    if (FAILED(hr)) return false;

    return true;
}

// ============================================================================
// Create Refine Root Signature (with additional UAVs for atomics)
// ============================================================================

bool JustGlowGPURenderer::CreateRefineRootSignature() {
    // Root parameters for Refine shaders:
    // [0] CBV - GlowParams (b0)
    // [1] CBV - BlurPassParams (b1)
    // [2] CBV - RefineParams (b2)
    // [3] Descriptor Table - SRVs (t0, t1, t2)
    // [4] Descriptor Table - UAVs (u0, u1, u2, u3)

    D3D12_ROOT_PARAMETER rootParams[5] = {};

    // CBV for GlowParams (b0)
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // CBV for BlurPassParams (b1)
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[1].Descriptor.ShaderRegister = 1;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // CBV for RefineParams (b2)
    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[2].Descriptor.ShaderRegister = 2;
    rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // SRV descriptor table (t0, t1, t2)
    D3D12_DESCRIPTOR_RANGE srvRange = {};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 3;
    srvRange.BaseShaderRegister = 0;
    srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[3].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[3].DescriptorTable.pDescriptorRanges = &srvRange;
    rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // UAV descriptor table (u0, u1, u2, u3)
    D3D12_DESCRIPTOR_RANGE uavRange = {};
    uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange.NumDescriptors = 4;
    uavRange.BaseShaderRegister = 0;
    uavRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    rootParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[4].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[4].DescriptorTable.pDescriptorRanges = &uavRange;
    rootParams[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Static samplers
    D3D12_STATIC_SAMPLER_DESC staticSamplers[2] = {};
    staticSamplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    staticSamplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].ShaderRegister = 0;
    staticSamplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    staticSamplers[1] = staticSamplers[0];
    staticSamplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    staticSamplers[1].ShaderRegister = 1;

    D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.NumParameters = 5;
    rootSigDesc.pParameters = rootParams;
    rootSigDesc.NumStaticSamplers = 2;
    rootSigDesc.pStaticSamplers = staticSamplers;

    ComPtr<ID3DBlob> serializedRootSig;
    ComPtr<ID3DBlob> errorBlob;

    HRESULT hr = D3D12SerializeRootSignature(
        &rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRootSig, &errorBlob);
    if (FAILED(hr)) {
        LOG("ERROR: Failed to serialize Refine root signature");
        return false;
    }

    hr = m_device->CreateRootSignature(
        0, serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(&m_refineRootSignature));
    if (FAILED(hr)) {
        LOG("ERROR: Failed to create Refine root signature");
        return false;
    }

    LOG("Refine root signature created");
    return true;
}

// ============================================================================
// Create ScreenBlend Root Signature (for DX12-CUDA Interop hybrid mode)
// ============================================================================

bool JustGlowGPURenderer::CreateScreenBlendRootSignature() {
    // Root parameters for ScreenBlend shader:
    // [0] CBV - GlowParams (b0)
    // [1] CBV - BlurPassParams (b1)
    // [2] CBV - ScreenBlendParams (b2)
    // [3] Descriptor Table - SRVs t2-t7 (6 blurred level textures)
    // [4] Descriptor Table - UAV u0 (output)

    D3D12_ROOT_PARAMETER rootParams[5] = {};

    // CBV for GlowParams (b0)
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].Descriptor.RegisterSpace = 0;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // CBV for BlurPassParams (b1)
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[1].Descriptor.ShaderRegister = 1;
    rootParams[1].Descriptor.RegisterSpace = 0;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // CBV for ScreenBlendParams (b2)
    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[2].Descriptor.ShaderRegister = 2;
    rootParams[2].Descriptor.RegisterSpace = 0;
    rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // SRV descriptor table (t2-t7) - 6 blurred level textures
    D3D12_DESCRIPTOR_RANGE srvRange = {};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 6;  // t2, t3, t4, t5, t6, t7
    srvRange.BaseShaderRegister = 2;  // Start at t2
    srvRange.RegisterSpace = 0;
    srvRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[3].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[3].DescriptorTable.pDescriptorRanges = &srvRange;
    rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // UAV descriptor table (u0)
    D3D12_DESCRIPTOR_RANGE uavRange = {};
    uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange.NumDescriptors = 1;
    uavRange.BaseShaderRegister = 0;
    uavRange.RegisterSpace = 0;
    uavRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    rootParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[4].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[4].DescriptorTable.pDescriptorRanges = &uavRange;
    rootParams[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Static samplers
    D3D12_STATIC_SAMPLER_DESC staticSamplers[2] = {};

    // Linear sampler (s0)
    staticSamplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    staticSamplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamplers[0].MipLODBias = 0;
    staticSamplers[0].MaxAnisotropy = 1;
    staticSamplers[0].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    staticSamplers[0].BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    staticSamplers[0].MinLOD = 0;
    staticSamplers[0].MaxLOD = D3D12_FLOAT32_MAX;
    staticSamplers[0].ShaderRegister = 0;
    staticSamplers[0].RegisterSpace = 0;
    staticSamplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // Point sampler (s1)
    staticSamplers[1] = staticSamplers[0];
    staticSamplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    staticSamplers[1].ShaderRegister = 1;

    D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.NumParameters = 5;
    rootSigDesc.pParameters = rootParams;
    rootSigDesc.NumStaticSamplers = 2;
    rootSigDesc.pStaticSamplers = staticSamplers;
    rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serializedRootSig;
    ComPtr<ID3DBlob> errorBlob;

    HRESULT hr = D3D12SerializeRootSignature(
        &rootSigDesc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRootSig,
        &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) {
            LOG("ERROR: Root signature serialization failed: %s",
                (const char*)errorBlob->GetBufferPointer());
        }
        return false;
    }

    hr = m_device->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(&m_screenBlendRootSignature));
    if (FAILED(hr)) {
        LOG("ERROR: Failed to create ScreenBlend root signature, HR=0x%08X", hr);
        return false;
    }

    LOG("ScreenBlend root signature created");
    return true;
}

// ============================================================================
// Create DispatchIndirect Resources
// ============================================================================

bool JustGlowGPURenderer::CreateDispatchIndirectResources() {
    D3D12_HEAP_PROPERTIES defaultHeapProps = {};
    defaultHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    // 1. Atomic bounds buffer: 4 uints [minX, maxX, minY, maxY]
    {
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = 4 * sizeof(UINT);
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr = m_device->CreateCommittedResource(
            &defaultHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_atomicBoundsBuffer));
        if (FAILED(hr)) {
            LOG("ERROR: Failed to create atomic bounds buffer");
            return false;
        }
        LOG("Atomic bounds buffer created");
    }

    // 2. IndirectArgs buffer: 3 uints per MIP level (X, Y, Z thread group counts)
    {
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = MAX_MIP_LEVELS * 3 * sizeof(UINT);
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr = m_device->CreateCommittedResource(
            &defaultHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, nullptr,
            IID_PPV_ARGS(&m_indirectArgsBuffer));
        if (FAILED(hr)) {
            LOG("ERROR: Failed to create indirect args buffer");
            return false;
        }
        LOG("Indirect args buffer created (%d MIP levels)", MAX_MIP_LEVELS);
    }

    // 3. Bounds output buffer: 4 ints per MIP level [minX, maxX, minY, maxY]
    {
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = MAX_MIP_LEVELS * 4 * sizeof(INT);
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr = m_device->CreateCommittedResource(
            &defaultHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_boundsOutputBuffer));
        if (FAILED(hr)) {
            LOG("ERROR: Failed to create bounds output buffer");
            return false;
        }
        LOG("Bounds output buffer created");
    }

    // 4. Refine constant buffer (b2)
    {
        D3D12_HEAP_PROPERTIES uploadHeapProps = {};
        uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

        // RefineParams: 8 ints = 32 bytes, aligned to 256
        UINT cbSize = 256;
        D3D12_RESOURCE_DESC bufferDesc = {};
        bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufferDesc.Width = cbSize;
        bufferDesc.Height = 1;
        bufferDesc.DepthOrArraySize = 1;
        bufferDesc.MipLevels = 1;
        bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufferDesc.SampleDesc.Count = 1;
        bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        HRESULT hr = m_device->CreateCommittedResource(
            &uploadHeapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&m_refineConstBuffer));
        if (FAILED(hr)) {
            LOG("ERROR: Failed to create refine constant buffer");
            return false;
        }

        hr = m_refineConstBuffer->Map(0, nullptr, &m_refineConstBufferPtr);
        if (FAILED(hr)) {
            LOG("ERROR: Failed to map refine constant buffer");
            return false;
        }
        LOG("Refine constant buffer created");
    }

    return true;
}

// ============================================================================
// Create Command Signature for ExecuteIndirect
// ============================================================================

bool JustGlowGPURenderer::CreateCommandSignature() {
    // Command signature for Dispatch (3 uints: X, Y, Z)
    D3D12_INDIRECT_ARGUMENT_DESC argDesc = {};
    argDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

    D3D12_COMMAND_SIGNATURE_DESC sigDesc = {};
    sigDesc.ByteStride = 3 * sizeof(UINT);  // ThreadGroupCountX, Y, Z
    sigDesc.NumArgumentDescs = 1;
    sigDesc.pArgumentDescs = &argDesc;
    sigDesc.NodeMask = 0;

    HRESULT hr = m_device->CreateCommandSignature(
        &sigDesc, nullptr,  // No root signature needed for Dispatch
        IID_PPV_ARGS(&m_dispatchIndirectSignature));
    if (FAILED(hr)) {
        LOG("ERROR: Failed to create dispatch indirect command signature");
        return false;
    }

    LOG("Dispatch indirect command signature created");
    return true;
}

// ============================================================================
// Load Shaders
// ============================================================================

bool JustGlowGPURenderer::LoadShaders() {
    // Create shared root signature first
    ComPtr<ID3D12RootSignature> sharedRootSig;
    if (!CreateRootSignature(sharedRootSig)) {
        return false;
    }

    // Load each shader with the shared root signature
    bool success = true;
    success &= LoadShader(ShaderType::Prefilter, GetShaderPath(ShaderType::Prefilter), sharedRootSig);
    success &= LoadShader(ShaderType::Downsample, GetShaderPath(ShaderType::Downsample), sharedRootSig);
    success &= LoadShader(ShaderType::Upsample, GetShaderPath(ShaderType::Upsample), sharedRootSig);
    success &= LoadShader(ShaderType::Anamorphic, GetShaderPath(ShaderType::Anamorphic), sharedRootSig);
    success &= LoadShader(ShaderType::Composite, GetShaderPath(ShaderType::Composite), sharedRootSig);

    // Load PrefilterWithBounds with Refine root sig (needs t2 for bounds input)
    success &= LoadShader(ShaderType::PrefilterWithBounds, GetShaderPath(ShaderType::PrefilterWithBounds), m_refineRootSignature);

    // Load Refine shaders with Refine root signature (needs extra UAVs)
    success &= LoadShader(ShaderType::Refine, GetShaderPath(ShaderType::Refine), m_refineRootSignature);
    success &= LoadShader(ShaderType::CalcIndirectArgs, GetShaderPath(ShaderType::CalcIndirectArgs), m_refineRootSignature);
    success &= LoadShader(ShaderType::ResetBounds, GetShaderPath(ShaderType::ResetBounds), m_refineRootSignature);

    // Create ScreenBlend root signature (for hybrid Interop mode - needs t2-t7)
    if (!CreateScreenBlendRootSignature()) {
        LOG("WARNING: ScreenBlend root signature creation failed, Interop mode will be unavailable");
    }

    // Load ScreenBlend shaders with ScreenBlend root signature (t2-t7 for blurred levels)
    // These are optional - hybrid mode only works if CUDA is available
    if (m_screenBlendRootSignature) {
        LoadShader(ShaderType::ScreenBlend, GetShaderPath(ShaderType::ScreenBlend), m_screenBlendRootSignature);
        LoadShader(ShaderType::ScreenBlendDirect, GetShaderPath(ShaderType::ScreenBlendDirect), m_screenBlendRootSignature);
    }

    return success;
}

bool JustGlowGPURenderer::LoadShader(
    ShaderType type,
    const std::wstring& csoPath,
    const ComPtr<ID3D12RootSignature>& rootSig)
{
    auto& shader = m_shaders[static_cast<size_t>(type)];

    // Convert path to narrow string for logging
    char narrowPath[MAX_PATH];
    WideCharToMultiByte(CP_UTF8, 0, csoPath.c_str(), -1, narrowPath, MAX_PATH, nullptr, nullptr);
    LOG("Loading shader: %s", narrowPath);

    // Read CSO file
    std::ifstream file(csoPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG("ERROR: Failed to open shader file: %s", narrowPath);
        return false;
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    LOG("Shader file size: %zu bytes", fileSize);

    std::vector<char> shaderData(fileSize);
    file.read(shaderData.data(), fileSize);
    file.close();

    // Use shared root signature
    shader.rootSignature = rootSig;

    // Create pipeline state
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = shader.rootSignature.Get();
    psoDesc.CS.pShaderBytecode = shaderData.data();
    psoDesc.CS.BytecodeLength = shaderData.size();

    HRESULT hr = m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&shader.pipelineState));
    if (FAILED(hr)) {
        LOG("ERROR: CreateComputePipelineState failed, HRESULT=0x%08X", hr);
        return false;
    }

    shader.loaded = true;
    LOG("Shader loaded successfully: type=%d", static_cast<int>(type));
    return true;
}

std::wstring JustGlowGPURenderer::GetShaderPath(ShaderType type) {
    HMODULE hModule = GetCurrentModule();
    wchar_t modulePath[MAX_PATH];
    GetModuleFileNameW(hModule, modulePath, MAX_PATH);

    std::wstring path(modulePath);
    size_t lastSlash = path.find_last_of(L"\\/");
    if (lastSlash != std::wstring::npos) {
        path = path.substr(0, lastSlash + 1);
    }

    path += L"DirectX_Assets\\";

    switch (type) {
        case ShaderType::Prefilter:         return path + L"Prefilter.cso";
        case ShaderType::PrefilterWithBounds: return path + L"PrefilterWithBounds.cso";
        case ShaderType::Downsample:        return path + L"Downsample.cso";
        case ShaderType::Upsample:          return path + L"Upsample.cso";
        case ShaderType::Anamorphic:        return path + L"PostProcess.cso";
        case ShaderType::Composite:         return path + L"Composite.cso";
        case ShaderType::Refine:            return path + L"Refine.cso";
        case ShaderType::CalcIndirectArgs:  return path + L"CalcIndirectArgs.cso";
        case ShaderType::ResetBounds:       return path + L"ResetBounds.cso";
        case ShaderType::ScreenBlend:       return path + L"ScreenBlend.cso";
        case ShaderType::ScreenBlendDirect: return path + L"ScreenBlendDirect.cso";
        default:                            return L"";
    }
}

// ============================================================================
// Allocate MIP Chain
// ============================================================================

bool JustGlowGPURenderer::AllocateMipChain(int width, int height, int levels) {
    if (m_currentMipLevels == levels && !m_mipChain.empty()) {
        if (m_mipChain[0].width == width && m_mipChain[0].height == height) {
            return true;
        }
    }

    ReleaseMipChain();

    m_mipChain.resize(levels);
    m_currentMipLevels = levels;

    int w = width;
    int h = height;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    for (int i = 0; i < levels; ++i) {
        auto& mip = m_mipChain[i];
        mip.width = w;
        mip.height = h;

        D3D12_RESOURCE_DESC texDesc = {};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Width = w;
        texDesc.Height = h;
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        texDesc.SampleDesc.Count = 1;
        texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr = m_device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &texDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&mip.resource));
        if (FAILED(hr)) return false;

        // Create SRV at MIP_SRV_START + i
        UINT srvIndex = MIP_SRV_START + i;
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;

        m_device->CreateShaderResourceView(
            mip.resource.Get(),
            &srvDesc,
            GetCPUDescriptorHandle(srvIndex));
        mip.srvIndex = srvIndex;

        // Create UAV at MIP_UAV_START + i
        UINT uavIndex = MIP_UAV_START + i;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

        m_device->CreateUnorderedAccessView(
            mip.resource.Get(),
            nullptr,
            &uavDesc,
            GetCPUDescriptorHandle(uavIndex));
        mip.uavIndex = uavIndex;

        // Next level is half size
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
    }

    return true;
}

void JustGlowGPURenderer::ReleaseMipChain() {
    m_mipChain.clear();
    m_currentMipLevels = 0;
}

// ============================================================================
// Create Views for External Resources
// ============================================================================

void JustGlowGPURenderer::CreateExternalResourceViews(
    ID3D12Resource* input,
    ID3D12Resource* output)
{
    D3D12_RESOURCE_DESC inputDesc = input->GetDesc();
    D3D12_RESOURCE_DESC outputDesc = output->GetDesc();

    // Create SRV descriptor template
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = inputDesc.Format;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;

    // === Prefilter SRV Pair [16-17] ===
    // t0 = external input
    m_device->CreateShaderResourceView(
        input,
        &srvDesc,
        GetCPUDescriptorHandle(PREFILTER_SRV_PAIR));
    // t1 = same input (dummy, not used by prefilter)
    m_device->CreateShaderResourceView(
        input,
        &srvDesc,
        GetCPUDescriptorHandle(PREFILTER_SRV_PAIR + 1));

    // === Composite SRV Pair [18-19] ===
    // t0 = original input
    m_device->CreateShaderResourceView(
        input,
        &srvDesc,
        GetCPUDescriptorHandle(COMPOSITE_SRV_PAIR));
    // t1 = glow (mip[0]) - will be set up after mip chain is ready

    // Create UAV for output [20]
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = outputDesc.Format;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

    m_device->CreateUnorderedAccessView(
        output,
        nullptr,
        &uavDesc,
        GetCPUDescriptorHandle(EXTERNAL_OUTPUT_UAV));

    // Create glow SRV at COMPOSITE_SRV_PAIR + 1 if mip chain exists
    if (!m_mipChain.empty()) {
        D3D12_SHADER_RESOURCE_VIEW_DESC glowSrvDesc = {};
        glowSrvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        glowSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        glowSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        glowSrvDesc.Texture2D.MipLevels = 1;

        m_device->CreateShaderResourceView(
            m_mipChain[0].resource.Get(),
            &glowSrvDesc,
            GetCPUDescriptorHandle(COMPOSITE_SRV_PAIR + 1));
    }

    // === DispatchIndirect Resource Views ===
    if (m_useDispatchIndirect && m_atomicBoundsBuffer) {
        // SRV table for Refine shaders: t0, t1, t2 must be consecutive

        // t0: Input texture SRV
        m_device->CreateShaderResourceView(
            input,
            &srvDesc,
            GetCPUDescriptorHandle(REFINE_INPUT_SRV));

        // t1: Dummy SRV (same as input, not used by Prefilter)
        m_device->CreateShaderResourceView(
            input,
            &srvDesc,
            GetCPUDescriptorHandle(REFINE_DUMMY_SRV));

        // t2: Bounds SRV (StructuredBuffer<int>)
        D3D12_SHADER_RESOURCE_VIEW_DESC boundsSrvDesc = {};
        boundsSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
        boundsSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        boundsSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        boundsSrvDesc.Buffer.FirstElement = 0;
        boundsSrvDesc.Buffer.NumElements = MAX_MIP_LEVELS * 4;  // 4 ints per level
        boundsSrvDesc.Buffer.StructureByteStride = sizeof(INT);

        m_device->CreateShaderResourceView(
            m_boundsOutputBuffer.Get(),
            &boundsSrvDesc,
            GetCPUDescriptorHandle(REFINE_BOUNDS_SRV));

        // UAV table for Refine shaders: u0, u1, u2, u3 must be consecutive

        // u0: Output UAV (placeholder - will be overwritten in ExecutePrefilterIndirect)
        // Use atomicBoundsBuffer as placeholder since Refine doesn't write to u0
        D3D12_UNORDERED_ACCESS_VIEW_DESC placeholderUavDesc = {};
        placeholderUavDesc.Format = DXGI_FORMAT_UNKNOWN;
        placeholderUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        placeholderUavDesc.Buffer.FirstElement = 0;
        placeholderUavDesc.Buffer.NumElements = 4;
        placeholderUavDesc.Buffer.StructureByteStride = sizeof(UINT);

        m_device->CreateUnorderedAccessView(
            m_atomicBoundsBuffer.Get(),
            nullptr,
            &placeholderUavDesc,
            GetCPUDescriptorHandle(REFINE_OUTPUT_UAV));

        // u1: Atomic bounds UAV
        D3D12_UNORDERED_ACCESS_VIEW_DESC atomicUavDesc = {};
        atomicUavDesc.Format = DXGI_FORMAT_UNKNOWN;
        atomicUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        atomicUavDesc.Buffer.FirstElement = 0;
        atomicUavDesc.Buffer.NumElements = 4;
        atomicUavDesc.Buffer.StructureByteStride = sizeof(UINT);

        m_device->CreateUnorderedAccessView(
            m_atomicBoundsBuffer.Get(),
            nullptr,
            &atomicUavDesc,
            GetCPUDescriptorHandle(ATOMIC_BOUNDS_UAV));

        // Indirect args UAV (u2)
        D3D12_UNORDERED_ACCESS_VIEW_DESC indirectUavDesc = {};
        indirectUavDesc.Format = DXGI_FORMAT_UNKNOWN;
        indirectUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        indirectUavDesc.Buffer.FirstElement = 0;
        indirectUavDesc.Buffer.NumElements = MAX_MIP_LEVELS * 3;
        indirectUavDesc.Buffer.StructureByteStride = sizeof(UINT);

        m_device->CreateUnorderedAccessView(
            m_indirectArgsBuffer.Get(),
            nullptr,
            &indirectUavDesc,
            GetCPUDescriptorHandle(INDIRECT_ARGS_UAV));

        // Bounds output UAV (u3)
        D3D12_UNORDERED_ACCESS_VIEW_DESC boundsUavDesc = {};
        boundsUavDesc.Format = DXGI_FORMAT_UNKNOWN;
        boundsUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        boundsUavDesc.Buffer.FirstElement = 0;
        boundsUavDesc.Buffer.NumElements = MAX_MIP_LEVELS * 4;
        boundsUavDesc.Buffer.StructureByteStride = sizeof(INT);

        m_device->CreateUnorderedAccessView(
            m_boundsOutputBuffer.Get(),
            nullptr,
            &boundsUavDesc,
            GetCPUDescriptorHandle(BOUNDS_OUTPUT_UAV));
    }
}

// ============================================================================
// Render
// ============================================================================

bool JustGlowGPURenderer::Render(
    const RenderParams& params,
    ID3D12Resource* inputBuffer,
    ID3D12Resource* outputBuffer)
{
    LOG("=== Render Begin ===");
    LOG("Size: %dx%d, MipLevels: %d, Exposure: %.2f, Threshold: %.2f",
        params.width, params.height, params.mipLevels, params.exposure, params.threshold);

    if (!inputBuffer || !outputBuffer) {
        LOG("ERROR: Invalid input/output buffers");
        return false;
    }

#if HAS_CUDA
    // Use hybrid DX12-CUDA pipeline if Interop is enabled
    if (m_useInterop && m_interop && m_cudaRenderer) {
        LOG("=== Using Hybrid DX12-CUDA Pipeline ===");
        return RenderHybrid(params, inputBuffer, outputBuffer);
    }
#endif

    // Verify shaders are loaded
    bool shadersOk = m_shaders[static_cast<size_t>(ShaderType::Prefilter)].loaded &&
                     m_shaders[static_cast<size_t>(ShaderType::Downsample)].loaded &&
                     m_shaders[static_cast<size_t>(ShaderType::Upsample)].loaded &&
                     m_shaders[static_cast<size_t>(ShaderType::Composite)].loaded;
    if (!shadersOk) {
        LOG("ERROR: Shaders not loaded - Prefilter:%d, Downsample:%d, Upsample:%d, Composite:%d",
            m_shaders[static_cast<size_t>(ShaderType::Prefilter)].loaded,
            m_shaders[static_cast<size_t>(ShaderType::Downsample)].loaded,
            m_shaders[static_cast<size_t>(ShaderType::Upsample)].loaded,
            m_shaders[static_cast<size_t>(ShaderType::Composite)].loaded);
        return false;
    }

    // Allocate MIP chain
    if (!AllocateMipChain(params.width, params.height, params.mipLevels)) {
        LOG("ERROR: Failed to allocate MIP chain");
        return false;
    }
    LOG("MIP chain allocated: %d levels", params.mipLevels);

    // Create views for AE resources
    CreateExternalResourceViews(inputBuffer, outputBuffer);
    LOG("External resource views created");

    // Reset command list
    HRESULT hr = m_commandAllocator->Reset();
    if (FAILED(hr)) {
        LOG("ERROR: CommandAllocator Reset failed, HR=0x%08X", hr);
        return false;
    }
    hr = m_commandList->Reset(m_commandAllocator.Get(), nullptr);
    if (FAILED(hr)) {
        LOG("ERROR: CommandList Reset failed, HR=0x%08X", hr);
        return false;
    }

    // Set descriptor heap
    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
    m_commandList->SetDescriptorHeaps(1, heaps);

    // Update main constant buffer
    GlowParams cb;
    FillGlowParams(cb, params);
    memcpy(m_constantBufferPtr, &cb, sizeof(cb));
    LOG("Constant buffer updated");

    // === Pipeline Execution ===
    // Check if DispatchIndirect optimization is enabled and shaders are loaded
    bool useIndirect = m_useDispatchIndirect &&
        m_shaders[static_cast<size_t>(ShaderType::Refine)].loaded &&
        m_shaders[static_cast<size_t>(ShaderType::CalcIndirectArgs)].loaded &&
        m_shaders[static_cast<size_t>(ShaderType::ResetBounds)].loaded;

    if (useIndirect) {
        // DispatchIndirect optimized pipeline
        LOG("=== Using DispatchIndirect Pipeline ===");

        LOG("--- Refine (BoundingBox) ---");
        ExecuteRefine(params, inputBuffer, 0);

        LOG("--- PrefilterIndirect ---");
        ExecutePrefilterIndirect(params, inputBuffer);

        LOG("--- Downsample Chain ---");
        ExecuteDownsampleChainIndirect(params);
    } else {
        // Standard pipeline
        LOG("=== Using Standard Pipeline ===");

        LOG("--- Prefilter ---");
        ExecutePrefilter(params, inputBuffer);

        LOG("--- Downsample Chain ---");
        ExecuteDownsampleChain(params);
    }

    LOG("--- Upsample Chain ---");
    ExecuteUpsampleChain(params);

    if (params.anamorphic > 0.001f) {
        LOG("--- Anamorphic ---");
        ExecuteAnamorphic(params);
    }

    LOG("--- Composite ---");
    ExecuteComposite(params, inputBuffer, outputBuffer);

    // Close and execute
    hr = m_commandList->Close();
    if (FAILED(hr)) {
        LOG("ERROR: CommandList Close failed, HR=0x%08X", hr);
        return false;
    }

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(1, commandLists);
    LOG("Commands executed");

    WaitForGPU();
    LOG("=== Render Complete ===");

    return true;
}

// ============================================================================
// Pipeline Stages
// ============================================================================

void JustGlowGPURenderer::ExecutePrefilter(
    const RenderParams& params,
    ID3D12Resource* input)
{
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Prefilter)];
    if (!shader.loaded) {
        LOG("Prefilter shader not loaded!");
        return;
    }

    // Update BlurPassParams for this pass
    // For prefilter: src = original image, dst = mip[0]
    BlurPassParams passParams = {};
    passParams.srcWidth = params.width;
    passParams.srcHeight = params.height;
    passParams.dstWidth = m_mipChain[0].width;
    passParams.dstHeight = m_mipChain[0].height;
    passParams.srcTexelX = 1.0f / static_cast<float>(params.width);
    passParams.srcTexelY = 1.0f / static_cast<float>(params.height);
    passParams.dstTexelX = 1.0f / static_cast<float>(passParams.dstWidth);
    passParams.dstTexelY = 1.0f / static_cast<float>(passParams.dstHeight);
    passParams.blurOffset = 0.0f;
    passParams.passIndex = 0;
    passParams.totalPasses = params.mipLevels;
    memcpy(m_blurPassBufferPtr, &passParams, sizeof(passParams));

    LOG("Prefilter: %dx%d -> %dx%d", passParams.srcWidth, passParams.srcHeight,
        passParams.dstWidth, passParams.dstHeight);

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());

    // Set constant buffers
    m_commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(1, m_blurPassBuffer->GetGPUVirtualAddress());

    // Bind SRV pair at PREFILTER_SRV_PAIR (t0=input, t1=dummy)
    m_commandList->SetComputeRootDescriptorTable(2, GetGPUDescriptorHandle(PREFILTER_SRV_PAIR));

    // Bind UAV: mip[0]
    m_commandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(m_mipChain[0].uavIndex));

    // Dispatch
    UINT groupsX = (passParams.dstWidth + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
    UINT groupsY = (passParams.dstHeight + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
    LOG("Dispatch: %u x %u groups", groupsX, groupsY);
    m_commandList->Dispatch(groupsX, groupsY, 1);

    // UAV barrier for mip[0]
    InsertUAVBarrier(m_mipChain[0].resource.Get());
}

void JustGlowGPURenderer::ExecuteDownsampleChain(const RenderParams& params) {
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Downsample)];
    if (!shader.loaded) {
        LOG("Downsample shader not loaded!");
        return;
    }

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());
    m_commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    for (int i = 0; i < params.mipLevels - 1; ++i) {
        auto& srcMip = m_mipChain[i];
        auto& dstMip = m_mipChain[i + 1];

        LOG("Downsample pass %d: mip[%d](%dx%d) -> mip[%d](%dx%d)",
            i, i, srcMip.width, srcMip.height, i+1, dstMip.width, dstMip.height);

        // Update BlurPassParams (per-level blurOffset from v1.2.1)
        BlurPassParams passParams = {};
        passParams.srcWidth = srcMip.width;
        passParams.srcHeight = srcMip.height;
        passParams.dstWidth = dstMip.width;
        passParams.dstHeight = dstMip.height;
        passParams.srcPitch = srcMip.width;
        passParams.dstPitch = dstMip.width;
        passParams.srcTexelX = 1.0f / static_cast<float>(srcMip.width);
        passParams.srcTexelY = 1.0f / static_cast<float>(srcMip.height);
        passParams.dstTexelX = 1.0f / static_cast<float>(dstMip.width);
        passParams.dstTexelY = 1.0f / static_cast<float>(dstMip.height);
        passParams.blurOffset = params.blurOffsets[i];  // Per-level offset (decays to 1.5px)
        passParams.fractionalBlend = 0.0f;
        passParams.passIndex = i;
        passParams.totalPasses = params.mipLevels;
        memcpy(m_blurPassBufferPtr, &passParams, sizeof(passParams));
        m_commandList->SetComputeRootConstantBufferView(1, m_blurPassBuffer->GetGPUVirtualAddress());

        // Transition source to SRV
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Create SRV pair for this pass at PASS_SRV_PAIRS_START + i*2
        // t0 = source mip, t1 = same source (downsample doesn't use t1)
        UINT pairIndex = PASS_SRV_PAIRS_START + i * 2;
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;

        m_device->CreateShaderResourceView(srcMip.resource.Get(), &srvDesc,
            GetCPUDescriptorHandle(pairIndex));
        m_device->CreateShaderResourceView(srcMip.resource.Get(), &srvDesc,
            GetCPUDescriptorHandle(pairIndex + 1));

        // Bind SRV pair
        m_commandList->SetComputeRootDescriptorTable(2, GetGPUDescriptorHandle(pairIndex));

        // Bind destination UAV
        m_commandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(dstMip.uavIndex));

        // Dispatch
        UINT groupsX = (dstMip.width + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        UINT groupsY = (dstMip.height + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        m_commandList->Dispatch(groupsX, groupsY, 1);

        // Transition source back to UAV
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // UAV barrier for destination
        InsertUAVBarrier(dstMip.resource.Get());
    }
}

void JustGlowGPURenderer::ExecuteUpsampleChain(const RenderParams& params) {
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Upsample)];
    if (!shader.loaded) {
        LOG("Upsample shader not loaded!");
        return;
    }

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());
    m_commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    // Upsample from smallest to largest
    // Note: We use passIndex=0 to avoid the problematic blend with prevLevel
    // This simplifies the pipeline and avoids read-write conflicts
    for (int i = params.mipLevels - 2; i >= 0; --i) {
        auto& srcMip = m_mipChain[i + 1];  // Smaller (source)
        auto& dstMip = m_mipChain[i];      // Larger (destination)

        LOG("Upsample pass: mip[%d](%dx%d) -> mip[%d](%dx%d)",
            i+1, srcMip.width, srcMip.height, i, dstMip.width, dstMip.height);

        // Update BlurPassParams
        // Set passIndex to 0 to skip the blend logic in shader
        BlurPassParams passParams = {};
        passParams.srcWidth = srcMip.width;
        passParams.srcHeight = srcMip.height;
        passParams.dstWidth = dstMip.width;
        passParams.dstHeight = dstMip.height;
        passParams.srcTexelX = 1.0f / static_cast<float>(srcMip.width);
        passParams.srcTexelY = 1.0f / static_cast<float>(srcMip.height);
        passParams.dstTexelX = 1.0f / static_cast<float>(dstMip.width);
        passParams.dstTexelY = 1.0f / static_cast<float>(dstMip.height);
        passParams.blurOffset = params.blurOffsets[i];  // Per-level offset (decays to 1.5px)
        passParams.fractionalBlend = 0.0f;  // Disable to avoid reading from destination
        passParams.passIndex = 0;  // Always 0 to skip blend with prevLevel
        passParams.totalPasses = params.mipLevels;
        memcpy(m_blurPassBufferPtr, &passParams, sizeof(passParams));
        m_commandList->SetComputeRootConstantBufferView(1, m_blurPassBuffer->GetGPUVirtualAddress());

        // Transition source to SRV
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Create SRV pair: t0 = srcMip, t1 = srcMip (dummy, not used when passIndex=0)
        UINT pairIndex = PASS_SRV_PAIRS_START + (params.mipLevels + i) * 2;
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;

        m_device->CreateShaderResourceView(srcMip.resource.Get(), &srvDesc,
            GetCPUDescriptorHandle(pairIndex));
        m_device->CreateShaderResourceView(srcMip.resource.Get(), &srvDesc,
            GetCPUDescriptorHandle(pairIndex + 1));

        // Bind SRV pair
        m_commandList->SetComputeRootDescriptorTable(2, GetGPUDescriptorHandle(pairIndex));

        // Bind destination UAV
        m_commandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(dstMip.uavIndex));

        // Dispatch
        UINT groupsX = (dstMip.width + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        UINT groupsY = (dstMip.height + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        m_commandList->Dispatch(groupsX, groupsY, 1);

        // Transition source back to UAV
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // UAV barrier
        InsertUAVBarrier(dstMip.resource.Get());
    }
}

void JustGlowGPURenderer::ExecuteAnamorphic(const RenderParams& params) {
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Anamorphic)];
    if (!shader.loaded) return;

    // Anamorphic operates on mip[0]
    // This would stretch the glow in the specified direction
    // For now, we'll skip this as the effect is subtle
}

void JustGlowGPURenderer::ExecuteComposite(
    const RenderParams& params,
    ID3D12Resource* original,
    ID3D12Resource* output)
{
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Composite)];
    if (!shader.loaded) {
        LOG("Composite shader not loaded!");
        return;
    }

    LOG("Composite: original + mip[0] -> output (%dx%d)", params.width, params.height);

    // Update BlurPassParams
    BlurPassParams passParams = {};
    passParams.srcWidth = params.width;
    passParams.srcHeight = params.height;
    passParams.dstWidth = params.width;
    passParams.dstHeight = params.height;
    passParams.srcTexelX = 1.0f / static_cast<float>(params.width);
    passParams.srcTexelY = 1.0f / static_cast<float>(params.height);
    passParams.dstTexelX = passParams.srcTexelX;
    passParams.dstTexelY = passParams.srcTexelY;
    memcpy(m_blurPassBufferPtr, &passParams, sizeof(passParams));

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());

    m_commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(1, m_blurPassBuffer->GetGPUVirtualAddress());

    // Transition mip[0] to SRV (original is already in SRV state from AE)
    TransitionResource(m_mipChain[0].resource.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Create SRV pair at COMPOSITE_SRV_PAIR: t0 = original, t1 = glow (mip[0])
    D3D12_RESOURCE_DESC originalDesc = original->GetDesc();
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = originalDesc.Format;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;

    // t0 = original
    m_device->CreateShaderResourceView(original, &srvDesc,
        GetCPUDescriptorHandle(COMPOSITE_SRV_PAIR));

    // t1 = glow (mip[0])
    srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    m_device->CreateShaderResourceView(m_mipChain[0].resource.Get(), &srvDesc,
        GetCPUDescriptorHandle(COMPOSITE_SRV_PAIR + 1));

    // Bind SRV pair
    m_commandList->SetComputeRootDescriptorTable(2, GetGPUDescriptorHandle(COMPOSITE_SRV_PAIR));

    // Bind output UAV
    m_commandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(EXTERNAL_OUTPUT_UAV));

    // Dispatch
    UINT groupsX = (params.width + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
    UINT groupsY = (params.height + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
    LOG("Dispatch: %u x %u groups", groupsX, groupsY);
    m_commandList->Dispatch(groupsX, groupsY, 1);

    // Transition mip[0] back to UAV
    TransitionResource(m_mipChain[0].resource.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

// ============================================================================
// DispatchIndirect Optimized Stages
// ============================================================================

void JustGlowGPURenderer::ExecuteRefine(
    const RenderParams& params,
    ID3D12Resource* input,
    int mipLevel)
{
    if (!m_useDispatchIndirect) return;

    LOG("ExecuteRefine: mipLevel=%d, size=%dx%d",
        mipLevel, mipLevel == 0 ? params.width : m_mipChain[mipLevel-1].width,
        mipLevel == 0 ? params.height : m_mipChain[mipLevel-1].height);

    int refineWidth = mipLevel == 0 ? params.width : m_mipChain[mipLevel-1].width;
    int refineHeight = mipLevel == 0 ? params.height : m_mipChain[mipLevel-1].height;

    // Update RefineParams constant buffer (b2)
    struct RefineParams {
        int refineWidth;
        int refineHeight;
        float refineThreshold;
        int blurRadius;
        int mipLevel;
        int maxMipLevels;
        int pad0, pad1;
    };
    RefineParams refineParams = {};
    refineParams.refineWidth = refineWidth;
    refineParams.refineHeight = refineHeight;
    refineParams.refineThreshold = params.threshold;
    refineParams.blurRadius = static_cast<int>(params.offsetPrefilter);
    refineParams.mipLevel = mipLevel;
    refineParams.maxMipLevels = params.mipLevels;
    memcpy(m_refineConstBufferPtr, &refineParams, sizeof(refineParams));

    // === Step 1: Reset atomic bounds ===
    {
        auto& shader = m_shaders[static_cast<size_t>(ShaderType::ResetBounds)];
        if (!shader.loaded) {
            LOG("ResetBounds shader not loaded!");
            return;
        }

        m_commandList->SetComputeRootSignature(m_refineRootSignature.Get());
        m_commandList->SetPipelineState(shader.pipelineState.Get());

        m_commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
        m_commandList->SetComputeRootConstantBufferView(1, m_blurPassBuffer->GetGPUVirtualAddress());
        m_commandList->SetComputeRootConstantBufferView(2, m_refineConstBuffer->GetGPUVirtualAddress());

        // SRV table (t0, t1, t2) - we need 3 SRVs for Refine root sig
        m_commandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(REFINE_INPUT_SRV));

        // UAV table (u0, u1, u2, u3)
        m_commandList->SetComputeRootDescriptorTable(4, GetGPUDescriptorHandle(REFINE_OUTPUT_UAV));

        m_commandList->Dispatch(1, 1, 1);
        InsertUAVBarrier(m_atomicBoundsBuffer.Get());
    }

    // === Step 2: Execute RefineCS ===
    {
        auto& shader = m_shaders[static_cast<size_t>(ShaderType::Refine)];
        if (!shader.loaded) {
            LOG("Refine shader not loaded!");
            return;
        }

        m_commandList->SetPipelineState(shader.pipelineState.Get());

        UINT groupsX = (refineWidth + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        UINT groupsY = (refineHeight + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        LOG("RefineCS Dispatch: %u x %u groups", groupsX, groupsY);

        m_commandList->Dispatch(groupsX, groupsY, 1);
        InsertUAVBarrier(m_atomicBoundsBuffer.Get());
    }

    // === Step 3: Calculate IndirectArgs ===
    {
        auto& shader = m_shaders[static_cast<size_t>(ShaderType::CalcIndirectArgs)];
        if (!shader.loaded) {
            LOG("CalcIndirectArgs shader not loaded!");
            return;
        }

        // Transition IndirectArgs to UAV for writing
        TransitionResource(m_indirectArgsBuffer.Get(),
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        m_commandList->SetPipelineState(shader.pipelineState.Get());
        m_commandList->Dispatch(1, 1, 1);

        InsertUAVBarrier(m_indirectArgsBuffer.Get());
        InsertUAVBarrier(m_boundsOutputBuffer.Get());

        // Transition IndirectArgs back to INDIRECT_ARGUMENT for ExecuteIndirect
        TransitionResource(m_indirectArgsBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    }

    LOG("ExecuteRefine complete for mipLevel=%d", mipLevel);
}

void JustGlowGPURenderer::ExecutePrefilterIndirect(
    const RenderParams& params,
    ID3D12Resource* input)
{
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::PrefilterWithBounds)];
    if (!shader.loaded) {
        LOG("PrefilterWithBounds shader not loaded, falling back to regular Prefilter");
        ExecutePrefilter(params, input);
        return;
    }

    LOG("ExecutePrefilterIndirect: using DispatchIndirect");

    // Update BlurPassParams
    BlurPassParams passParams = {};
    passParams.srcWidth = params.width;
    passParams.srcHeight = params.height;
    passParams.dstWidth = m_mipChain[0].width;
    passParams.dstHeight = m_mipChain[0].height;
    passParams.srcTexelX = 1.0f / static_cast<float>(params.width);
    passParams.srcTexelY = 1.0f / static_cast<float>(params.height);
    passParams.dstTexelX = 1.0f / static_cast<float>(passParams.dstWidth);
    passParams.dstTexelY = 1.0f / static_cast<float>(passParams.dstHeight);
    passParams.blurOffset = 0.0f;
    passParams.passIndex = 0;
    passParams.totalPasses = params.mipLevels;
    memcpy(m_blurPassBufferPtr, &passParams, sizeof(passParams));

    // Refine root signature has 5 parameters:
    // [0] CBV - GlowParams (b0)
    // [1] CBV - BlurPassParams (b1)
    // [2] CBV - RefineParams (b2)
    // [3] Descriptor Table - SRVs (t0, t1, t2)
    // [4] Descriptor Table - UAVs (u0, u1, u2, u3)

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());

    m_commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(1, m_blurPassBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(2, m_refineConstBuffer->GetGPUVirtualAddress());

    // Set up SRV table (t0=input, t1=dummy, t2=bounds)
    // Use REFINE_INPUT_SRV which has consecutive descriptors for t0, t1, t2
    m_commandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(REFINE_INPUT_SRV));

    // Set up UAV table (u0=mip[0] output, u1-u3 unused for Prefilter)
    m_commandList->SetComputeRootDescriptorTable(4, GetGPUDescriptorHandle(REFINE_OUTPUT_UAV));

    // Create UAV for mip[0] at REFINE_OUTPUT_UAV position (u0)
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_device->CreateUnorderedAccessView(
        m_mipChain[0].resource.Get(),
        nullptr,
        &uavDesc,
        GetCPUDescriptorHandle(REFINE_OUTPUT_UAV));

    // ExecuteIndirect - GPU determines dispatch size from IndirectArgsBuffer
    // Offset = mipLevel * 3 * sizeof(UINT) = 0 for prefilter
    m_commandList->ExecuteIndirect(
        m_dispatchIndirectSignature.Get(),
        1,
        m_indirectArgsBuffer.Get(),
        0,  // Offset for mip level 0
        nullptr,
        0);

    InsertUAVBarrier(m_mipChain[0].resource.Get());
    LOG("ExecutePrefilterIndirect complete");
}

void JustGlowGPURenderer::ExecuteDownsampleChainIndirect(const RenderParams& params) {
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Downsample)];
    if (!shader.loaded) {
        LOG("Downsample shader not loaded!");
        return;
    }

    LOG("ExecuteDownsampleChainIndirect: %d levels", params.mipLevels);

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());
    m_commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());

    for (int i = 0; i < params.mipLevels - 1; ++i) {
        auto& srcMip = m_mipChain[i];
        auto& dstMip = m_mipChain[i + 1];

        LOG("Downsample pass %d (Indirect): mip[%d](%dx%d) -> mip[%d](%dx%d)",
            i, i, srcMip.width, srcMip.height, i+1, dstMip.width, dstMip.height);

        // Update BlurPassParams
        BlurPassParams passParams = {};
        passParams.srcWidth = srcMip.width;
        passParams.srcHeight = srcMip.height;
        passParams.dstWidth = dstMip.width;
        passParams.dstHeight = dstMip.height;
        passParams.srcPitch = srcMip.width;
        passParams.dstPitch = dstMip.width;
        passParams.srcTexelX = 1.0f / static_cast<float>(srcMip.width);
        passParams.srcTexelY = 1.0f / static_cast<float>(srcMip.height);
        passParams.dstTexelX = 1.0f / static_cast<float>(dstMip.width);
        passParams.dstTexelY = 1.0f / static_cast<float>(dstMip.height);
        passParams.blurOffset = params.blurOffsets[i];
        passParams.fractionalBlend = 0.0f;
        passParams.passIndex = i;
        passParams.totalPasses = params.mipLevels;
        memcpy(m_blurPassBufferPtr, &passParams, sizeof(passParams));
        m_commandList->SetComputeRootConstantBufferView(1, m_blurPassBuffer->GetGPUVirtualAddress());

        // Transition source to SRV
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Create SRV pair for this pass
        UINT pairIndex = PASS_SRV_PAIRS_START + i * 2;
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;

        m_device->CreateShaderResourceView(srcMip.resource.Get(), &srvDesc,
            GetCPUDescriptorHandle(pairIndex));
        m_device->CreateShaderResourceView(srcMip.resource.Get(), &srvDesc,
            GetCPUDescriptorHandle(pairIndex + 1));

        m_commandList->SetComputeRootDescriptorTable(2, GetGPUDescriptorHandle(pairIndex));
        m_commandList->SetComputeRootDescriptorTable(3, GetGPUDescriptorHandle(dstMip.uavIndex));

        // For now, use regular Dispatch for downsample
        // (Full DispatchIndirect for downsample would require per-level Refine)
        UINT groupsX = (dstMip.width + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        UINT groupsY = (dstMip.height + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        m_commandList->Dispatch(groupsX, groupsY, 1);

        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        InsertUAVBarrier(dstMip.resource.Get());
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void JustGlowGPURenderer::TransitionResource(
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after)
{
    if (before == after) return;

    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    m_commandList->ResourceBarrier(1, &barrier);
}

void JustGlowGPURenderer::InsertUAVBarrier(ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    m_commandList->ResourceBarrier(1, &barrier);
}

void JustGlowGPURenderer::WaitForGPU() {
    m_fenceValue++;
    m_commandQueue->Signal(m_fence.Get(), m_fenceValue);

    if (m_fence->GetCompletedValue() < m_fenceValue) {
        m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent);
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }
}

D3D12_GPU_DESCRIPTOR_HANDLE JustGlowGPURenderer::GetGPUDescriptorHandle(UINT index) {
    D3D12_GPU_DESCRIPTOR_HANDLE handle = m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * m_srvUavDescriptorSize;
    return handle;
}

D3D12_CPU_DESCRIPTOR_HANDLE JustGlowGPURenderer::GetCPUDescriptorHandle(UINT index) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * m_srvUavDescriptorSize;
    return handle;
}

// ============================================================================
// DX12-CUDA Interop Methods (Hybrid Mode)
// ============================================================================

#if HAS_CUDA

bool JustGlowGPURenderer::EnableInterop(CUcontext cudaContext, CUstream cudaStream) {
    LOG("=== EnableInterop ===");

    if (!m_device || !m_commandQueue) {
        LOG("ERROR: DX12 device not initialized");
        return false;
    }

    if (!cudaContext || !cudaStream) {
        LOG("ERROR: Invalid CUDA context or stream");
        return false;
    }

    // Store CUDA handles
    m_cudaContext = cudaContext;
    m_cudaStream = cudaStream;

    // Create Interop manager
    m_interop = new JustGlowInterop();
    if (!m_interop->Initialize(m_device, m_commandQueue, cudaContext)) {
        LOG("ERROR: Failed to initialize Interop");
        delete m_interop;
        m_interop = nullptr;
        return false;
    }

    // Create fence for DX12-CUDA synchronization
    m_interopFence = new InteropFence();
    if (!m_interop->CreateFence(*m_interopFence)) {
        LOG("ERROR: Failed to create Interop fence");
        ShutdownInterop();
        return false;
    }

    // Create CUDA renderer (for blur operations)
    m_cudaRenderer = new JustGlowCUDARenderer();
    if (!m_cudaRenderer->Initialize(cudaContext, cudaStream)) {
        LOG("ERROR: Failed to initialize CUDA renderer");
        ShutdownInterop();
        return false;
    }

    // Create ScreenBlend constant buffer
    if (!CreateScreenBlendResources()) {
        LOG("ERROR: Failed to create ScreenBlend resources");
        ShutdownInterop();
        return false;
    }

    // Verify ScreenBlend shaders are loaded
    if (!m_shaders[static_cast<size_t>(ShaderType::ScreenBlend)].loaded) {
        LOG("WARNING: ScreenBlend shader not loaded, hybrid mode may not work optimally");
    }

    m_useInterop = true;
    LOG("Interop enabled successfully");
    return true;
}

void JustGlowGPURenderer::ShutdownInterop() {
    LOG("=== ShutdownInterop ===");

    m_useInterop = false;

    // Release Interop textures
    ReleaseInteropTextures();

    // Release ScreenBlend buffer
    if (m_screenBlendBufferPtr) {
        m_screenBlendBuffer->Unmap(0, nullptr);
        m_screenBlendBufferPtr = nullptr;
    }
    m_screenBlendBuffer.Reset();

    // Release fence
    if (m_interopFence) {
        if (m_interop) {
            m_interop->DestroyFence(*m_interopFence);
        }
        delete m_interopFence;
        m_interopFence = nullptr;
    }

    // Release CUDA renderer
    if (m_cudaRenderer) {
        m_cudaRenderer->Shutdown();
        delete m_cudaRenderer;
        m_cudaRenderer = nullptr;
    }

    // Release Interop manager
    if (m_interop) {
        m_interop->Shutdown();
        delete m_interop;
        m_interop = nullptr;
    }

    m_cudaContext = nullptr;
    m_cudaStream = nullptr;

    LOG("Interop shutdown complete");
}

bool JustGlowGPURenderer::CreateInteropTextures(int width, int height) {
    LOG("CreateInteropTextures: %dx%d", width, height);

    // Release existing textures
    ReleaseInteropTextures();

    // Create input texture
    m_interopInput = new InteropTexture();
    if (!m_interop->CreateSharedTexture(width, height, *m_interopInput,
            m_srvUavHeap.Get(), INTEROP_INPUT_SRV, INTEROP_INPUT_UAV)) {
        LOG("ERROR: Failed to create Interop input texture");
        ReleaseInteropTextures();
        return false;
    }

    // Create blurred textures for each MIP level (6 levels max)
    for (int i = 0; i < MAX_INTEROP_LEVELS; i++) {
        int levelWidth = width >> (i + 1);
        int levelHeight = height >> (i + 1);

        if (levelWidth < 16 || levelHeight < 16) {
            LOG("Interop level %d: skipped (too small)", i);
            break;
        }

        m_interopBlurred[i] = new InteropTexture();
        if (!m_interop->CreateSharedTexture(levelWidth, levelHeight, *m_interopBlurred[i],
                m_srvUavHeap.Get(),
                INTEROP_BLURRED_SRV_START + i,
                INTEROP_BLURRED_UAV_START + i)) {
            LOG("ERROR: Failed to create Interop blurred texture %d", i);
            ReleaseInteropTextures();
            return false;
        }

        LOG("Interop blurred[%d]: %dx%d", i, levelWidth, levelHeight);
    }

    LOG("Interop textures created successfully");
    return true;
}

void JustGlowGPURenderer::ReleaseInteropTextures() {
    LOG("ReleaseInteropTextures");

    if (m_interopInput) {
        if (m_interop) {
            m_interop->DestroySharedTexture(*m_interopInput);
        }
        delete m_interopInput;
        m_interopInput = nullptr;
    }

    for (int i = 0; i < MAX_INTEROP_LEVELS; i++) {
        if (m_interopBlurred[i]) {
            if (m_interop) {
                m_interop->DestroySharedTexture(*m_interopBlurred[i]);
            }
            delete m_interopBlurred[i];
            m_interopBlurred[i] = nullptr;
        }
    }
}

bool JustGlowGPURenderer::CreateScreenBlendResources() {
    LOG("CreateScreenBlendResources");

    // ScreenBlend constant buffer (b2)
    // Contains: numLevels, baseWeight, falloff, levelWeights[6]
    UINT cbSize = 256;  // Aligned to 256 bytes

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = cbSize;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&m_screenBlendBuffer));

    if (FAILED(hr)) {
        LOG("ERROR: Failed to create ScreenBlend buffer, HR=0x%08X", hr);
        return false;
    }

    hr = m_screenBlendBuffer->Map(0, nullptr, &m_screenBlendBufferPtr);
    if (FAILED(hr)) {
        LOG("ERROR: Failed to map ScreenBlend buffer, HR=0x%08X", hr);
        return false;
    }

    LOG("ScreenBlend resources created");
    return true;
}

bool JustGlowGPURenderer::RenderHybrid(
    const RenderParams& params,
    ID3D12Resource* inputBuffer,
    ID3D12Resource* outputBuffer)
{
    LOG("=== RenderHybrid Begin ===");

    // Ensure Interop textures are allocated
    if (!m_interopInput || m_interopInput->width != params.width || m_interopInput->height != params.height) {
        if (!CreateInteropTextures(params.width, params.height)) {
            LOG("ERROR: Failed to create Interop textures, falling back to DX12");
            m_useInterop = false;
            return Render(params, inputBuffer, outputBuffer);
        }
    }

    // Reset command list
    HRESULT hr = m_commandAllocator->Reset();
    if (FAILED(hr)) {
        LOG("ERROR: CommandAllocator Reset failed");
        return false;
    }
    hr = m_commandList->Reset(m_commandAllocator.Get(), nullptr);
    if (FAILED(hr)) {
        LOG("ERROR: CommandList Reset failed");
        return false;
    }

    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
    m_commandList->SetDescriptorHeaps(1, heaps);

    // Update main constant buffer
    GlowParams cb;
    FillGlowParams(cb, params);
    memcpy(m_constantBufferPtr, &cb, sizeof(cb));

    // ========================================
    // Step 1: DX12 - Copy input to Interop texture
    // ========================================
    LOG("--- Step 1: Copy Input to Interop ---");
    CopyInputToInterop(inputBuffer);

    // Execute and wait (DX12 side complete)
    hr = m_commandList->Close();
    if (FAILED(hr)) {
        LOG("ERROR: CommandList Close failed");
        return false;
    }

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(1, commandLists);

    // Signal DX12 -> CUDA
    m_interop->SignalFromDX12(*m_interopFence);
    LOG("DX12 signaled, waiting on CUDA...");

    // ========================================
    // Step 2: CUDA - Blur operations
    // ========================================
    LOG("--- Step 2: CUDA Blur Operations ---");

    // Wait for DX12 signal
    m_interop->WaitOnCUDA(*m_interopFence, m_cudaStream);

    // CUDA processes the shared texture:
    // Unmult → Prefilter → Downsample → Log-Transmittance Pre-blur

    // Calculate number of blur levels (max 6 for Interop)
    int numBlurLevels = std::min(params.mipLevels, MAX_INTEROP_LEVELS);
    LOG("CUDA: Processing %d blur levels", numBlurLevels);

    // Call CUDA renderer with Interop textures
    if (!m_cudaRenderer->RenderWithInterop(params, m_interopInput, m_interopBlurred, numBlurLevels)) {
        LOG("ERROR: CUDA RenderWithInterop failed, falling back to DX12");
        // On failure, disable Interop and retry with standard pipeline
        m_useInterop = false;
        return Render(params, inputBuffer, outputBuffer);
    }

    // Signal CUDA -> DX12
    m_interop->SignalFromCUDA(*m_interopFence, m_cudaStream);
    LOG("CUDA signaled, waiting on DX12...");

    // ========================================
    // Step 3: DX12 - Screen blend + Composite
    // ========================================
    LOG("--- Step 3: DX12 Screen Blend + Composite ---");

    // Wait for CUDA signal
    m_interop->WaitOnDX12(*m_interopFence);

    // Reset command list for second DX12 pass
    hr = m_commandAllocator->Reset();
    if (FAILED(hr)) return false;
    hr = m_commandList->Reset(m_commandAllocator.Get(), nullptr);
    if (FAILED(hr)) return false;

    m_commandList->SetDescriptorHeaps(1, heaps);

    // Execute Screen blend (combines blurred levels)
    ExecuteScreenBlend(params);

    // Execute Composite (blend glow with original)
    // For now, fall back to standard Composite
    // which reads from m_mipChain[0] (we need to copy ScreenBlend result there first)

    // Allocate MIP chain for Composite
    if (!AllocateMipChain(params.width, params.height, params.mipLevels)) {
        LOG("ERROR: Failed to allocate MIP chain");
        return false;
    }

    CreateExternalResourceViews(inputBuffer, outputBuffer);
    ExecuteComposite(params, inputBuffer, outputBuffer);

    // Close and execute
    hr = m_commandList->Close();
    if (FAILED(hr)) return false;

    m_commandQueue->ExecuteCommandLists(1, commandLists);
    WaitForGPU();

    LOG("=== RenderHybrid Complete ===");
    return true;
}

void JustGlowGPURenderer::CopyInputToInterop(ID3D12Resource* input) {
    if (!m_interopInput || !m_interopInput->d3d12Resource) {
        LOG("ERROR: Interop input texture not allocated");
        return;
    }

    // Transition Interop texture to COPY_DEST
    TransitionResource(m_interopInput->d3d12Resource.Get(),
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);

    // Copy from AE input to Interop texture
    m_commandList->CopyResource(m_interopInput->d3d12Resource.Get(), input);

    // Transition back to COMMON for CUDA access
    TransitionResource(m_interopInput->d3d12Resource.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);

    LOG("Input copied to Interop texture");
}

void JustGlowGPURenderer::ExecuteScreenBlend(const RenderParams& params) {
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::ScreenBlend)];
    if (!shader.loaded) {
        LOG("ScreenBlend shader not loaded, skipping");
        return;
    }

    // Count available blurred levels
    int numLevels = 0;
    for (int i = 0; i < MAX_INTEROP_LEVELS; i++) {
        if (m_interopBlurred[i] && m_interopBlurred[i]->isValid()) {
            numLevels++;
        }
    }

    if (numLevels == 0) {
        LOG("No blurred levels available for ScreenBlend");
        return;
    }

    // Update ScreenBlend constant buffer
    struct ScreenBlendParams {
        int numLevels;
        float baseWeight;
        float falloff;
        float _pad0;
        float levelWeights[4];
        float levelWeights56[2];
        float _pad1[2];
    };

    ScreenBlendParams sbParams = {};
    sbParams.numLevels = numLevels;
    sbParams.baseWeight = params.level1Weight;
    sbParams.falloff = params.decayK / 100.0f;  // decayK is 0-100, normalize to 0-1

    // Calculate level weights: weight = baseWeight * pow(falloff, level)
    float decayRate = 1.0f - (sbParams.falloff - 0.5f);
    for (int i = 0; i < 4 && i < numLevels; i++) {
        sbParams.levelWeights[i] = params.level1Weight * powf(decayRate, static_cast<float>(i + 1));
    }
    for (int i = 0; i < 2 && (i + 4) < numLevels; i++) {
        sbParams.levelWeights56[i] = params.level1Weight * powf(decayRate, static_cast<float>(i + 5));
    }

    memcpy(m_screenBlendBufferPtr, &sbParams, sizeof(sbParams));

    LOG("ScreenBlend: %d levels, intensity=%.2f, falloff=%.2f",
        numLevels, params.level1Weight, sbParams.falloff);

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());

    // ScreenBlend Root Signature layout:
    // [0] CBV - GlowParams (b0)
    // [1] CBV - BlurPassParams (b1)
    // [2] CBV - ScreenBlendParams (b2)
    // [3] Descriptor Table - SRVs t2-t7 (6 blurred level textures)
    // [4] Descriptor Table - UAV u0 (output)

    // Set constant buffers
    m_commandList->SetComputeRootConstantBufferView(0, m_constantBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(1, m_blurPassBuffer->GetGPUVirtualAddress());
    m_commandList->SetComputeRootConstantBufferView(2, m_screenBlendBuffer->GetGPUVirtualAddress());

    // Transition blurred textures to PIXEL_SHADER_RESOURCE for reading
    for (int i = 0; i < numLevels && i < MAX_INTEROP_LEVELS; i++) {
        if (m_interopBlurred[i] && m_interopBlurred[i]->d3d12Resource) {
            TransitionResource(m_interopBlurred[i]->d3d12Resource.Get(),
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
    }

    // Bind blurred textures as SRVs (t2-t7)
    // SRVs are at descriptor indices INTEROP_BLURRED_SRV_START + i
    // The descriptor table expects 6 consecutive SRVs starting at t2
    D3D12_GPU_DESCRIPTOR_HANDLE srvTableStart = GetGPUDescriptorHandle(INTEROP_BLURRED_SRV_START);
    m_commandList->SetComputeRootDescriptorTable(3, srvTableStart);

    // Bind output UAV (u0) - use mipChain[0] as output target
    if (m_mipChain.size() > 0) {
        D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = GetGPUDescriptorHandle(m_mipChain[0].uavIndex);
        m_commandList->SetComputeRootDescriptorTable(4, uavHandle);
    }

    // Dispatch
    UINT groupsX = (params.width + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
    UINT groupsY = (params.height + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

    LOG("ScreenBlend Dispatch: %u x %u groups, SRV table at index %u",
        groupsX, groupsY, INTEROP_BLURRED_SRV_START);
    m_commandList->Dispatch(groupsX, groupsY, 1);

    // Transition blurred textures back to COMMON for CUDA access
    for (int i = 0; i < numLevels && i < MAX_INTEROP_LEVELS; i++) {
        if (m_interopBlurred[i] && m_interopBlurred[i]->d3d12Resource) {
            TransitionResource(m_interopBlurred[i]->d3d12Resource.Get(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
        }
    }

    // UAV barrier
    if (m_mipChain.size() > 0) {
        InsertUAVBarrier(m_mipChain[0].resource.Get());
    }
}

#endif // HAS_CUDA

#endif // _WIN32
