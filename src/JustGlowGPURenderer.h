/**
 * JustGlow GPU Renderer
 *
 * DirectX 12 Compute Shader based rendering pipeline.
 * Implements Dual Kawase blur with all enhancements.
 */

#pragma once
#ifndef JUSTGLOW_GPU_RENDERER_H
#define JUSTGLOW_GPU_RENDERER_H

#ifdef _WIN32

#include "JustGlowParams.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <array>
#include <memory>

using Microsoft::WRL::ComPtr;

// ============================================================================
// Shader Types
// ============================================================================

enum class ShaderType {
    Prefilter,          // Soft Threshold + Karis Average + 13-Tap Downsample
    Downsample,         // Dual Kawase 4-tap downsample
    Upsample,           // 9-Tap Tent upsample with progressive blend
    Anamorphic,         // Directional stretch
    Composite,          // Final blend with original
    COUNT
};

// ============================================================================
// Shader Resources
// ============================================================================

struct ShaderResource {
    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> pipelineState;
    bool loaded = false;
};

// ============================================================================
// MIP Chain Buffer
// ============================================================================

struct MipBuffer {
    ComPtr<ID3D12Resource> resource;
    D3D12_GPU_DESCRIPTOR_HANDLE srvHandle;
    D3D12_GPU_DESCRIPTOR_HANDLE uavHandle;
    int width;
    int height;
};

// ============================================================================
// JustGlowGPURenderer
// ============================================================================

class JustGlowGPURenderer {
public:
    JustGlowGPURenderer();
    ~JustGlowGPURenderer();

    // Initialization
    bool Initialize(ID3D12Device* device, ID3D12CommandQueue* commandQueue);
    void Shutdown();

    // Rendering
    bool Render(
        const RenderParams& params,
        ID3D12Resource* inputBuffer,
        ID3D12Resource* outputBuffer);

private:
    // DirectX objects
    ID3D12Device* m_device = nullptr;
    ID3D12CommandQueue* m_commandQueue = nullptr;

    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent = nullptr;
    UINT64 m_fenceValue = 0;

    // Descriptor heaps
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    UINT m_srvUavDescriptorSize = 0;
    UINT m_currentDescriptorIndex = 0;

    // Constant buffer
    ComPtr<ID3D12Resource> m_constantBuffer;
    void* m_constantBufferPtr = nullptr;

    // Shaders
    std::array<ShaderResource, static_cast<size_t>(ShaderType::COUNT)> m_shaders;

    // MIP chain buffers
    std::vector<MipBuffer> m_mipChain;
    int m_currentMipLevels = 0;

    // Temporary buffers
    ComPtr<ID3D12Resource> m_glowBuffer;        // Accumulated glow
    ComPtr<ID3D12Resource> m_tempBuffer;        // Temporary for passes

    // Initialization helpers
    bool CreateCommandObjects();
    bool CreateDescriptorHeaps();
    bool CreateConstantBuffer();
    bool LoadShaders();
    bool LoadShader(ShaderType type, const std::wstring& csoPath);

    // Resource management
    bool AllocateMipChain(int width, int height, int levels);
    void ReleaseMipChain();
    bool AllocateTemporaryBuffers(int width, int height);

    // Rendering stages
    void ExecutePrefilter(const RenderParams& params, ID3D12Resource* input);
    void ExecuteDownsampleChain(const RenderParams& params);
    void ExecuteUpsampleChain(const RenderParams& params);
    void ExecuteAnamorphic(const RenderParams& params);
    void ExecuteComposite(const RenderParams& params,
        ID3D12Resource* original, ID3D12Resource* output);

    // Utility
    void TransitionResource(ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
    void WaitForGPU();
    UINT AllocateDescriptor();
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUDescriptorHandle(UINT index);
    D3D12_CPU_DESCRIPTOR_HANDLE GetCPUDescriptorHandle(UINT index);

    // Shader path helper
    std::wstring GetShaderPath(ShaderType type);
};

#endif // _WIN32

#endif // JUSTGLOW_GPU_RENDERER_H
