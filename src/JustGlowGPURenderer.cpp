/**
 * JustGlow GPU Renderer Implementation
 *
 * DirectX 12 Compute Shader based rendering pipeline.
 * Fully implements Dual Kawase blur with proper descriptor binding.
 */

#ifdef _WIN32

#include "JustGlowGPURenderer.h"
#include <d3dcompiler.h>
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <cstdarg>

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

    ReleaseMipChain();

    if (m_constantBufferPtr) {
        m_constantBuffer->Unmap(0, nullptr);
        m_constantBufferPtr = nullptr;
    }

    if (m_blurPassBufferPtr) {
        m_blurPassBuffer->Unmap(0, nullptr);
        m_blurPassBufferPtr = nullptr;
    }

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
        case ShaderType::Prefilter:     return path + L"Prefilter.cso";
        case ShaderType::Downsample:    return path + L"Downsample.cso";
        case ShaderType::Upsample:      return path + L"Upsample.cso";
        case ShaderType::Anamorphic:    return path + L"PostProcess.cso";
        case ShaderType::Composite:     return path + L"Composite.cso";
        default:                        return L"";
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
    LOG("--- Prefilter ---");
    ExecutePrefilter(params, inputBuffer);

    LOG("--- Downsample Chain ---");
    ExecuteDownsampleChain(params);

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

        // Update BlurPassParams manually (params.mipChain removed in v1.2.0)
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
        passParams.blurOffset = params.blurOffset;  // Fixed offset from Spread parameter
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
        passParams.blurOffset = params.blurOffset;  // Fixed offset from Spread parameter
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

#endif // _WIN32
