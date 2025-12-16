/**
 * JustGlow GPU Renderer Implementation
 *
 * DirectX 12 Compute Shader based rendering pipeline.
 */

#ifdef _WIN32

#include "JustGlowGPURenderer.h"
#include <d3dcompiler.h>
#include <fstream>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

// Thread group size for compute shaders
constexpr UINT THREAD_GROUP_SIZE_X = 16;
constexpr UINT THREAD_GROUP_SIZE_Y = 16;

// Maximum descriptors in heap
constexpr UINT MAX_DESCRIPTORS = 256;

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
    if (!device || !commandQueue) {
        return false;
    }

    m_device = device;
    m_commandQueue = commandQueue;

    // Create command objects
    if (!CreateCommandObjects()) {
        return false;
    }

    // Create descriptor heaps
    if (!CreateDescriptorHeaps()) {
        return false;
    }

    // Create constant buffer
    if (!CreateConstantBuffer()) {
        return false;
    }

    // Load shaders
    if (!LoadShaders()) {
        return false;
    }

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

    if (m_fenceEvent) {
        CloseHandle(m_fenceEvent);
        m_fenceEvent = nullptr;
    }

    m_commandList.Reset();
    m_commandAllocator.Reset();
    m_fence.Reset();
    m_srvUavHeap.Reset();
    m_constantBuffer.Reset();
    m_glowBuffer.Reset();
    m_tempBuffer.Reset();

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

    // Create command allocator
    hr = m_device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        IID_PPV_ARGS(&m_commandAllocator));
    if (FAILED(hr)) return false;

    // Create command list
    hr = m_device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_COMPUTE,
        m_commandAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(&m_commandList));
    if (FAILED(hr)) return false;

    // Close command list (will be reset before use)
    m_commandList->Close();

    // Create fence
    hr = m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
    if (FAILED(hr)) return false;

    // Create fence event
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
// Create Constant Buffer
// ============================================================================

bool JustGlowGPURenderer::CreateConstantBuffer() {
    // Constant buffer size (aligned to 256 bytes)
    UINT cbSize = (sizeof(GlowParams) + 255) & ~255;

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
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_constantBuffer));
    if (FAILED(hr)) return false;

    // Map constant buffer
    hr = m_constantBuffer->Map(0, nullptr, &m_constantBufferPtr);
    if (FAILED(hr)) return false;

    return true;
}

// ============================================================================
// Load Shaders
// ============================================================================

bool JustGlowGPURenderer::LoadShaders() {
    // Load all shaders
    bool success = true;

    success &= LoadShader(ShaderType::Prefilter, GetShaderPath(ShaderType::Prefilter));
    success &= LoadShader(ShaderType::Downsample, GetShaderPath(ShaderType::Downsample));
    success &= LoadShader(ShaderType::Upsample, GetShaderPath(ShaderType::Upsample));
    success &= LoadShader(ShaderType::Anamorphic, GetShaderPath(ShaderType::Anamorphic));
    success &= LoadShader(ShaderType::Composite, GetShaderPath(ShaderType::Composite));

    return success;
}

bool JustGlowGPURenderer::LoadShader(ShaderType type, const std::wstring& csoPath) {
    auto& shader = m_shaders[static_cast<size_t>(type)];

    // Read CSO file
    std::ifstream file(csoPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        // Shader file not found - required for plugin to work
        return false;
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> shaderData(fileSize);
    file.read(shaderData.data(), fileSize);
    file.close();

    // Create root signature
    // Simple root signature: CBV at b0, SRV at t0, UAV at u0
    D3D12_ROOT_PARAMETER rootParams[3] = {};

    // CBV
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].Descriptor.RegisterSpace = 0;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // SRV descriptor table
    D3D12_DESCRIPTOR_RANGE srvRange = {};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 2;  // Input + previous level
    srvRange.BaseShaderRegister = 0;

    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[1].DescriptorTable.pDescriptorRanges = &srvRange;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    // UAV descriptor table
    D3D12_DESCRIPTOR_RANGE uavRange = {};
    uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange.NumDescriptors = 1;
    uavRange.BaseShaderRegister = 0;

    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[2].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[2].DescriptorTable.pDescriptorRanges = &uavRange;
    rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.NumParameters = 3;
    rootSigDesc.pParameters = rootParams;

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
        IID_PPV_ARGS(&shader.rootSignature));
    if (FAILED(hr)) return false;

    // Create pipeline state
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = shader.rootSignature.Get();
    psoDesc.CS.pShaderBytecode = shaderData.data();
    psoDesc.CS.BytecodeLength = shaderData.size();

    hr = m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&shader.pipelineState));
    if (FAILED(hr)) return false;

    shader.loaded = true;
    return true;
}

std::wstring JustGlowGPURenderer::GetShaderPath(ShaderType type) {
    // Get the DLL module path (not the exe path!)
    HMODULE hModule = nullptr;

    // Get handle to this DLL by using address of a function in this module
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&JustGlowGPURenderer::GetShaderPath),
        &hModule);

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
        case ShaderType::Anamorphic:    return path + L"PostProcess.cso";  // Anamorphic uses PostProcess.hlsl
        case ShaderType::Composite:     return path + L"Composite.cso";
        default:                        return L"";
    }
}

// ============================================================================
// Allocate MIP Chain
// ============================================================================

bool JustGlowGPURenderer::AllocateMipChain(int width, int height, int levels) {
    // Check if we need to reallocate
    if (m_currentMipLevels == levels && !m_mipChain.empty()) {
        if (m_mipChain[0].width == width && m_mipChain[0].height == height) {
            return true;  // Already allocated with correct size
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

        // Create SRV
        UINT srvIndex = AllocateDescriptor();
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;

        m_device->CreateShaderResourceView(
            mip.resource.Get(),
            &srvDesc,
            GetCPUDescriptorHandle(srvIndex));
        mip.srvHandle = GetGPUDescriptorHandle(srvIndex);

        // Create UAV
        UINT uavIndex = AllocateDescriptor();
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;

        m_device->CreateUnorderedAccessView(
            mip.resource.Get(),
            nullptr,
            &uavDesc,
            GetCPUDescriptorHandle(uavIndex));
        mip.uavHandle = GetGPUDescriptorHandle(uavIndex);

        // Halve dimensions for next level
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
    m_currentDescriptorIndex = 0;  // Reset descriptor allocation
}

// ============================================================================
// Render
// ============================================================================

bool JustGlowGPURenderer::Render(
    const RenderParams& params,
    ID3D12Resource* inputBuffer,
    ID3D12Resource* outputBuffer)
{
    if (!inputBuffer || !outputBuffer) {
        return false;
    }

    // Allocate MIP chain if needed
    if (!AllocateMipChain(params.width, params.height, params.mipLevels)) {
        return false;
    }

    // Reset command list
    m_commandAllocator->Reset();
    m_commandList->Reset(m_commandAllocator.Get(), nullptr);

    // Set descriptor heap
    ID3D12DescriptorHeap* heaps[] = { m_srvUavHeap.Get() };
    m_commandList->SetDescriptorHeaps(1, heaps);

    // Update constant buffer
    GlowParams cb;
    FillGlowParams(cb, params);
    memcpy(m_constantBufferPtr, &cb, sizeof(cb));

    // Execute pipeline stages
    ExecutePrefilter(params, inputBuffer);
    ExecuteDownsampleChain(params);
    ExecuteUpsampleChain(params);

    if (params.anamorphic > 0.0f) {
        ExecuteAnamorphic(params);
    }

    ExecuteComposite(params, inputBuffer, outputBuffer);

    // Close and execute command list
    m_commandList->Close();

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(1, commandLists);

    WaitForGPU();

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
    if (!shader.loaded) return;

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());

    // Set CBV
    m_commandList->SetComputeRootConstantBufferView(0,
        m_constantBuffer->GetGPUVirtualAddress());

    // Transition input to SRV
    TransitionResource(input,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Set SRV (input) and UAV (mip 0)
    // Note: Actual descriptor binding would need proper setup
    // This is simplified for illustration

    // Dispatch
    UINT groupsX = (m_mipChain[0].width + THREAD_GROUP_SIZE_X - 1) / THREAD_GROUP_SIZE_X;
    UINT groupsY = (m_mipChain[0].height + THREAD_GROUP_SIZE_Y - 1) / THREAD_GROUP_SIZE_Y;
    m_commandList->Dispatch(groupsX, groupsY, 1);

    // Transition input back
    TransitionResource(input,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void JustGlowGPURenderer::ExecuteDownsampleChain(const RenderParams& params) {
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Downsample)];
    if (!shader.loaded) return;

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());

    // Downsample from level 0 to level N-1
    for (int i = 0; i < params.mipLevels - 1; ++i) {
        auto& srcMip = m_mipChain[i];
        auto& dstMip = m_mipChain[i + 1];

        // Update constant buffer for this pass
        BlurPassParams passParams;
        FillBlurPassParams(passParams, params.mipChain, i, true);

        // Transition source to SRV
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Dispatch
        UINT groupsX = (dstMip.width + THREAD_GROUP_SIZE_X - 1) / THREAD_GROUP_SIZE_X;
        UINT groupsY = (dstMip.height + THREAD_GROUP_SIZE_Y - 1) / THREAD_GROUP_SIZE_Y;
        m_commandList->Dispatch(groupsX, groupsY, 1);

        // Transition source back
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // UAV barrier for destination
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = dstMip.resource.Get();
        m_commandList->ResourceBarrier(1, &barrier);
    }
}

void JustGlowGPURenderer::ExecuteUpsampleChain(const RenderParams& params) {
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Upsample)];
    if (!shader.loaded) return;

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());

    // Upsample from level N-1 to level 0
    for (int i = params.mipLevels - 2; i >= 0; --i) {
        auto& srcMip = m_mipChain[i + 1];
        auto& dstMip = m_mipChain[i];

        // Update constant buffer for this pass
        BlurPassParams passParams;
        float fractional = (i == 0) ? params.fractionalAmount : 0.0f;
        FillBlurPassParams(passParams, params.mipChain, i, false, fractional);

        // Transition source to SRV
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // Dispatch
        UINT groupsX = (dstMip.width + THREAD_GROUP_SIZE_X - 1) / THREAD_GROUP_SIZE_X;
        UINT groupsY = (dstMip.height + THREAD_GROUP_SIZE_Y - 1) / THREAD_GROUP_SIZE_Y;
        m_commandList->Dispatch(groupsX, groupsY, 1);

        // Transition source back
        TransitionResource(srcMip.resource.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        // UAV barrier
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = dstMip.resource.Get();
        m_commandList->ResourceBarrier(1, &barrier);
    }
}

void JustGlowGPURenderer::ExecuteAnamorphic(const RenderParams& params) {
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Anamorphic)];
    if (!shader.loaded) return;

    // Apply anamorphic stretch to mip level 0
    // Implementation would stretch in the specified direction
}

void JustGlowGPURenderer::ExecuteComposite(
    const RenderParams& params,
    ID3D12Resource* original,
    ID3D12Resource* output)
{
    auto& shader = m_shaders[static_cast<size_t>(ShaderType::Composite)];
    if (!shader.loaded) return;

    m_commandList->SetComputeRootSignature(shader.rootSignature.Get());
    m_commandList->SetPipelineState(shader.pipelineState.Get());

    // Set CBV
    m_commandList->SetComputeRootConstantBufferView(0,
        m_constantBuffer->GetGPUVirtualAddress());

    // Transition resources
    TransitionResource(original,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    TransitionResource(m_mipChain[0].resource.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // Dispatch
    UINT groupsX = (params.width + THREAD_GROUP_SIZE_X - 1) / THREAD_GROUP_SIZE_X;
    UINT groupsY = (params.height + THREAD_GROUP_SIZE_Y - 1) / THREAD_GROUP_SIZE_Y;
    m_commandList->Dispatch(groupsX, groupsY, 1);

    // Transition back
    TransitionResource(original,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

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

UINT JustGlowGPURenderer::AllocateDescriptor() {
    UINT index = m_currentDescriptorIndex++;
    if (m_currentDescriptorIndex >= MAX_DESCRIPTORS) {
        m_currentDescriptorIndex = 0;  // Wrap around (should handle better in production)
    }
    return index;
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
