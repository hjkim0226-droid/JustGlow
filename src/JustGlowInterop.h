/**
 * JustGlow DX12-CUDA Interop
 *
 * Shared resource management between DirectX 12 and CUDA.
 * Enables zero-copy buffer sharing for hybrid rendering pipeline.
 *
 * Key concepts:
 * - Shared Texture2D: D3D12_HEAP_FLAG_SHARED + cudaImportExternalMemory
 * - Fence sync: ID3D12Fence + cudaExternalSemaphore
 * - Surface objects: CUDA surf2Dread/write for texture access
 */

#pragma once
#ifndef JUSTGLOW_INTEROP_H
#define JUSTGLOW_INTEROP_H

#ifdef _WIN32

#include <d3d12.h>
#include <wrl/client.h>

// Forward declare CUDA types to avoid header conflicts with DirectX
// Actual CUDA headers are included only in .cpp files
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef void* cudaExternalMemory_t;
typedef void* cudaExternalSemaphore_t;
typedef struct cudaMipmappedArray* cudaMipmappedArray_t;
typedef struct cudaArray* cudaArray_t;
typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaTextureObject_t;
typedef int cudaError_t;  // Actually enum, but int works for forward declaration

using Microsoft::WRL::ComPtr;

// ============================================================================
// Interop Texture (Shared between DX12 and CUDA)
// ============================================================================

/**
 * InteropTexture - Shared Texture2D accessible by both DX12 and CUDA
 *
 * Why Texture2D instead of Buffer:
 * - DX12: SampleLevel() for hardware filtering (needed for 6 Draw blend)
 * - CUDA: surf2Dread/write for 2D access patterns
 * - Buffer only supports linear 1D access
 */
struct InteropTexture {
    // DX12 side
    ComPtr<ID3D12Resource> d3d12Resource;
    HANDLE sharedHandle = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpuHandle = {};  // For descriptor copy
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpuHandle = {};  // For shader binding
    D3D12_CPU_DESCRIPTOR_HANDLE uavCpuHandle = {};
    D3D12_GPU_DESCRIPTOR_HANDLE uavGpuHandle = {};
    UINT srvIndex = 0;
    UINT uavIndex = 0;

    // CUDA side
    cudaExternalMemory_t cudaExtMem = nullptr;
    cudaMipmappedArray_t cudaMipArray = nullptr;
    cudaArray_t cudaArray = nullptr;
    cudaSurfaceObject_t cudaSurface = 0;  // For surf2Dread/write
    cudaTextureObject_t cudaTexture = 0;  // For tex2D (optional)

    // Common properties
    int width = 0;
    int height = 0;
    size_t sizeBytes = 0;

    bool isValid() const {
        return d3d12Resource != nullptr && cudaSurface != 0;
    }

    // Note: reset() is implemented in JustGlowInterop.cpp to avoid CUDA header dependency
    void reset();
};

// ============================================================================
// Interop Fence (Synchronization between DX12 and CUDA)
// ============================================================================

/**
 * InteropFence - Shared fence for GPU-GPU synchronization
 *
 * Sync flow:
 * 1. DX12: Signal(N) after work complete
 * 2. CUDA: Wait(N) before starting work
 * 3. CUDA: Signal(N+1) after work complete
 * 4. DX12: Wait(N+1) before continuing
 */
struct InteropFence {
    // DX12 side
    ComPtr<ID3D12Fence> d3d12Fence;
    HANDLE sharedHandle = nullptr;
    UINT64 fenceValue = 0;

    // CUDA side
    cudaExternalSemaphore_t cudaSemaphore = nullptr;

    bool isValid() const {
        return d3d12Fence != nullptr && cudaSemaphore != nullptr;
    }

    // Note: reset() is implemented in JustGlowInterop.cpp to avoid CUDA header dependency
    void reset();
};

// ============================================================================
// JustGlowInterop Class
// ============================================================================

/**
 * JustGlowInterop - Manages DX12-CUDA interop resources
 *
 * Usage:
 * 1. Initialize() with DX12 device and CUDA context
 * 2. CreateSharedTexture() for input/output buffers
 * 3. CreateFence() for synchronization
 * 4. Signal/Wait methods for GPU-GPU sync
 * 5. Shutdown() to release resources
 */
class JustGlowInterop {
public:
    JustGlowInterop();
    ~JustGlowInterop();

    // ========================================
    // Initialization
    // ========================================

    /**
     * Initialize interop with DX12 device and CUDA context
     * @param device DX12 device from AE
     * @param commandQueue DX12 command queue for signaling
     * @param cudaContext CUDA context from AE
     * @return true if successful
     */
    bool Initialize(
        ID3D12Device* device,
        ID3D12CommandQueue* commandQueue,
        CUcontext cudaContext);

    void Shutdown();

    bool IsInitialized() const { return m_initialized; }

    // ========================================
    // Shared Texture Management
    // ========================================

    /**
     * Create a shared RGBA32F texture accessible by both DX12 and CUDA
     *
     * DX12 side:
     * - D3D12_RESOURCE_DIMENSION_TEXTURE2D
     * - DXGI_FORMAT_R32G32B32A32_FLOAT
     * - D3D12_HEAP_FLAG_SHARED
     *
     * CUDA side:
     * - cudaExternalMemory (imported from DX12)
     * - cudaMipmappedArray → cudaArray
     * - cudaSurfaceObject for surf2Dread/write
     *
     * @param width Texture width in pixels
     * @param height Texture height in pixels
     * @param texture Output InteropTexture struct
     * @param descriptorHeap DX12 descriptor heap for SRV/UAV
     * @param srvIndex Index in heap for SRV
     * @param uavIndex Index in heap for UAV
     * @return true if successful
     */
    bool CreateSharedTexture(
        int width, int height,
        InteropTexture& texture,
        ID3D12DescriptorHeap* descriptorHeap,
        UINT srvIndex, UINT uavIndex);

    void DestroySharedTexture(InteropTexture& texture);

    // ========================================
    // Fence Synchronization
    // ========================================

    /**
     * Create a shared fence for DX12-CUDA synchronization
     * Uses D3D12_FENCE_FLAG_SHARED + cudaImportExternalSemaphore
     */
    bool CreateFence(InteropFence& fence);
    void DestroyFence(InteropFence& fence);

    /**
     * Signal from DX12 - increments fence value and signals
     * Call after DX12 work is complete, before CUDA needs the data
     */
    void SignalFromDX12(InteropFence& fence);

    /**
     * Wait on CUDA - waits for DX12's signal
     * Call before CUDA starts using data written by DX12
     */
    void WaitOnCUDA(InteropFence& fence, CUstream stream);

    /**
     * Signal from CUDA - increments fence value and signals
     * Call after CUDA work is complete, before DX12 needs the data
     */
    void SignalFromCUDA(InteropFence& fence, CUstream stream);

    /**
     * Wait on DX12 - waits for CUDA's signal
     * Call before DX12 starts using data written by CUDA
     */
    void WaitOnDX12(InteropFence& fence);

    // ========================================
    // Resource Transition Helpers
    // ========================================

    /**
     * Transition DX12 resource state before/after CUDA access
     * CUDA requires resources in D3D12_RESOURCE_STATE_COMMON
     */
    void TransitionForCUDAAccess(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES currentState);

    void TransitionFromCUDAAccess(
        ID3D12GraphicsCommandList* cmdList,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES targetState);

private:
    // Core objects
    ID3D12Device* m_device = nullptr;
    ID3D12CommandQueue* m_commandQueue = nullptr;
    CUcontext m_cudaContext = nullptr;
    UINT m_descriptorSize = 0;

    // State
    bool m_initialized = false;

    // Internal helpers
    bool ImportTextureToGPU(InteropTexture& texture);
    bool CheckCUDAError(cudaError_t err, const char* context);
    bool CheckDX12Error(HRESULT hr, const char* context);
};

// ============================================================================
// Constants
// ============================================================================

static constexpr int MAX_INTEROP_MIP_LEVELS = 12;
static constexpr float BASE_BLUR_SIGMA = 16.0f;  // Base sigma for Pre-blur

#endif // _WIN32

#endif // JUSTGLOW_INTEROP_H
