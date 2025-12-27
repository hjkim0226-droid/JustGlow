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

// Forward declarations for Interop (CUDA support)
#if HAS_CUDA
class JustGlowInterop;
class JustGlowCUDARenderer;
struct InteropTexture;
struct InteropFence;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
#endif

// ============================================================================
// Shader Types
// ============================================================================

enum class ShaderType {
    Prefilter,              // Soft Threshold + Karis Average + 13-Tap Downsample
    PrefilterWithBounds,    // Same as Prefilter but reads BoundingBox for offset
    Downsample,             // Dual Kawase 4-tap downsample
    Upsample,               // 9-Tap Tent upsample with progressive blend
    Anamorphic,             // Directional stretch
    Composite,              // Final blend with original
    Refine,                 // Calculate BoundingBox using atomics
    CalcIndirectArgs,       // Convert bounds to thread group counts
    ResetBounds,            // Reset atomic bounds for next frame
    ScreenBlend,            // Log-Transmittance Screen blend (Interop mode)
    ScreenBlendDirect,      // Direct Screen blend (fallback)
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
    UINT srvIndex;      // Index in descriptor heap for SRV
    UINT uavIndex;      // Index in descriptor heap for UAV
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

#if HAS_CUDA
    // Interop initialization (call after Initialize)
    // Enables hybrid DX12-CUDA rendering pipeline
    bool EnableInterop(CUcontext cudaContext, CUstream cudaStream);
    bool IsInteropEnabled() const { return m_useInterop; }
#endif

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

    // Constant buffers
    ComPtr<ID3D12Resource> m_constantBuffer;      // GlowParams (b0)
    void* m_constantBufferPtr = nullptr;

    ComPtr<ID3D12Resource> m_blurPassBuffer;      // BlurPassParams (b1)
    void* m_blurPassBufferPtr = nullptr;

    // Shaders
    std::array<ShaderResource, static_cast<size_t>(ShaderType::COUNT)> m_shaders;

    // MIP chain buffers
    std::vector<MipBuffer> m_mipChain;
    int m_currentMipLevels = 0;

    // =========================================
    // DispatchIndirect Optimization Resources
    // =========================================

    // Atomic bounds buffer: [minX, maxX, minY, maxY] - written by RefineCS
    ComPtr<ID3D12Resource> m_atomicBoundsBuffer;
    UINT m_atomicBoundsUavIndex = 0;

    // IndirectArgs buffer: { ThreadGroupCountX, Y, Z } per MIP level
    // Used by ExecuteIndirect to determine dispatch size
    ComPtr<ID3D12Resource> m_indirectArgsBuffer;

    // Bounds output buffer: [minX, maxX, minY, maxY] per MIP level
    // Read by Prefilter/Downsample to offset coordinates
    ComPtr<ID3D12Resource> m_boundsOutputBuffer;
    UINT m_boundsOutputSrvIndex = 0;
    UINT m_boundsOutputUavIndex = 0;

    // Refine constant buffer (b2)
    ComPtr<ID3D12Resource> m_refineConstBuffer;
    void* m_refineConstBufferPtr = nullptr;

    // Command signature for ExecuteIndirect
    ComPtr<ID3D12CommandSignature> m_dispatchIndirectSignature;

    // Root signature for Refine shaders (needs additional UAVs)
    ComPtr<ID3D12RootSignature> m_refineRootSignature;

    // Root signature for ScreenBlend shader (needs t2-t7 for blurred levels)
    ComPtr<ID3D12RootSignature> m_screenBlendRootSignature;

    // Flag to enable/disable DispatchIndirect optimization
    bool m_useDispatchIndirect = true;

    static constexpr int MAX_MIP_LEVELS = 12;

    // =========================================
    // DX12-CUDA Interop Resources (Hybrid Mode)
    // =========================================
#if HAS_CUDA
    // Interop manager
    JustGlowInterop* m_interop = nullptr;

    // CUDA renderer for blur operations
    JustGlowCUDARenderer* m_cudaRenderer = nullptr;

    // CUDA context and stream (from AE)
    CUcontext m_cudaContext = nullptr;
    CUstream m_cudaStream = nullptr;

    // Shared textures for DX12-CUDA data exchange
    // These are created with D3D12_HEAP_FLAG_SHARED and imported to CUDA
    static constexpr int MAX_INTEROP_LEVELS = 6;

    // Input texture (AE input -> CUDA processing)
    InteropTexture* m_interopInput = nullptr;

    // Blurred output textures (CUDA blur results -> DX12 Screen blend)
    InteropTexture* m_interopBlurred[MAX_INTEROP_LEVELS] = {};

    // Fence for DX12-CUDA synchronization
    InteropFence* m_interopFence = nullptr;

    // Screen blend constant buffer (b2 for ScreenBlend shader)
    ComPtr<ID3D12Resource> m_screenBlendBuffer;
    void* m_screenBlendBufferPtr = nullptr;

    // Flag to enable/disable Interop (set during initialization)
    bool m_useInterop = false;

    // Interop initialization
    bool InitializeInterop(CUcontext cudaContext, CUstream cudaStream);
    void ShutdownInterop();

    // Interop resource management
    bool CreateInteropTextures(int width, int height);
    void ReleaseInteropTextures();
    bool CreateScreenBlendResources();

    // Hybrid rendering pipeline
    bool RenderHybrid(
        const RenderParams& params,
        ID3D12Resource* inputBuffer,
        ID3D12Resource* outputBuffer);

    // DX12 -> CUDA -> DX12 stages
    void CopyInputToInterop(ID3D12Resource* input);
    void ExecuteScreenBlend(const RenderParams& params);
#endif

    // Initialization helpers
    bool CreateCommandObjects();
    bool CreateDescriptorHeaps();
    bool CreateConstantBuffers();
    bool CreateRootSignature(ComPtr<ID3D12RootSignature>& rootSig);
    bool CreateRefineRootSignature();      // Root sig for Refine shaders (extra UAVs)
    bool CreateScreenBlendRootSignature(); // Root sig for ScreenBlend (t2-t7 blurred levels)
    bool CreateDispatchIndirectResources(); // IndirectArgs, bounds buffers
    bool CreateCommandSignature();          // For ExecuteIndirect
    bool LoadShaders();
    bool LoadShader(ShaderType type, const std::wstring& csoPath,
                   const ComPtr<ID3D12RootSignature>& rootSig);

    // Resource management
    bool AllocateMipChain(int width, int height, int levels);
    void ReleaseMipChain();
    void CreateExternalResourceViews(ID3D12Resource* input, ID3D12Resource* output);

    // Rendering stages
    void ExecutePrefilter(const RenderParams& params, ID3D12Resource* input);
    void ExecuteDownsampleChain(const RenderParams& params);
    void ExecuteUpsampleChain(const RenderParams& params);
    void ExecuteAnamorphic(const RenderParams& params);
    void ExecuteComposite(const RenderParams& params,
        ID3D12Resource* original, ID3D12Resource* output);

    // DispatchIndirect optimized stages
    void ExecuteRefine(const RenderParams& params, ID3D12Resource* input, int mipLevel);
    void ExecutePrefilterIndirect(const RenderParams& params, ID3D12Resource* input);
    void ExecuteDownsampleChainIndirect(const RenderParams& params);

    // Utility
    void TransitionResource(ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
    void InsertUAVBarrier(ID3D12Resource* resource);
    void WaitForGPU();
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUDescriptorHandle(UINT index);
    D3D12_CPU_DESCRIPTOR_HANDLE GetCPUDescriptorHandle(UINT index);

    // Shader path helper
    std::wstring GetShaderPath(ShaderType type);
};

#endif // _WIN32

#endif // JUSTGLOW_GPU_RENDERER_H
