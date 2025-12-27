/**
 * JustGlow CUDA Renderer
 *
 * CUDA-based rendering pipeline for the glow effect.
 * Uses CUDA Driver API for best compatibility with future drivers.
 */

#pragma once
#ifndef JUSTGLOW_CUDA_RENDERER_H
#define JUSTGLOW_CUDA_RENDERER_H

#include "JustGlowParams.h"
#include <cuda.h>  // Driver API
#include <cuda_runtime.h>  // For cudaSurfaceObject_t
#include <vector>

// Forward declarations for Interop
struct InteropTexture;
struct InteropFence;

// ============================================================================
// MIP Buffer Structure for CUDA
// ============================================================================

struct CUDAMipBuffer {
    CUdeviceptr devicePtr;
    int width;          // Pixels
    int height;         // Pixels
    size_t pitchBytes;  // Row stride in BYTES (width * 4 * sizeof(float) for RGBA F32)
    size_t sizeBytes;   // Total allocation size
};

// ============================================================================
// BoundingBox Structure for Adaptive Resolution
// ============================================================================

struct BoundingBox {
    int minX, minY, maxX, maxY;

    int width() const { return (maxX >= minX) ? (maxX - minX + 1) : 0; }
    int height() const { return (maxY >= minY) ? (maxY - minY + 1) : 0; }
    bool valid() const { return maxX >= minX && maxY >= minY; }

    // Initialize to invalid state (will be updated by atomic operations)
    void reset(int imgWidth, int imgHeight) {
        minX = imgWidth;   // Start at max, atomicMin will reduce
        minY = imgHeight;
        maxX = -1;         // Start at min, atomicMax will increase
        maxY = -1;
    }

    // Initialize to full image (fallback)
    void setFull(int imgWidth, int imgHeight) {
        minX = 0;
        minY = 0;
        maxX = imgWidth - 1;
        maxY = imgHeight - 1;
    }
};

static const int MAX_MIP_LEVELS = 12;

// ============================================================================
// JustGlowCUDARenderer Class
// ============================================================================

class JustGlowCUDARenderer {
public:
    JustGlowCUDARenderer();
    ~JustGlowCUDARenderer();

    // Initialization with AE-provided CUDA context and stream
    bool Initialize(CUcontext context, CUstream stream);
    void Shutdown();

    // Rendering - input/output are CUdeviceptr from AE
    bool Render(
        const RenderParams& params,
        CUdeviceptr inputBuffer,
        CUdeviceptr outputBuffer);

    // ========================================
    // Interop Rendering (Hybrid DX12-CUDA)
    // ========================================

    /**
     * Render with DX12-CUDA Interop textures
     *
     * Pipeline:
     * 1. Read from input InteropTexture (shared with DX12)
     * 2. Unmult → Prefilter → Downsample → Log-Transmittance Pre-blur
     * 3. Write results to blurred InteropTextures (shared with DX12)
     *
     * @param params Render parameters
     * @param input Input texture (AE buffer copied by DX12)
     * @param blurredOutputs Array of 6 output textures for blurred MIP levels
     * @param numLevels Number of MIP levels to process (1-6)
     * @return true if successful
     */
    bool RenderWithInterop(
        const RenderParams& params,
        InteropTexture* input,
        InteropTexture** blurredOutputs,
        int numLevels);

private:
    // CUDA objects from AE
    CUcontext m_context;
    CUstream m_stream;

    // Synchronization event for inter-kernel dependencies
    CUevent m_syncEvent;

    // CUDA modules and kernels
    CUmodule m_module;
    CUfunction m_prefilterKernel;
    CUfunction m_prefilter25TapKernel;
    CUfunction m_prefilterSep5HKernel;
    CUfunction m_prefilterSep5VKernel;
    CUfunction m_prefilterSep9HKernel;
    CUfunction m_prefilterSep9VKernel;
    CUfunction m_downsampleKernel;
    CUfunction m_upsampleKernel;
    CUfunction m_compositeKernel;
    CUfunction m_horizontalBlurKernel;
    // Gaussian downsample kernels
    CUfunction m_gaussian2DDownsampleKernel;  // 9-tap 2D + ZeroPad (primary)
    CUfunction m_gaussianDownsampleHKernel;   // [deprecated] kept for fallback
    CUfunction m_gaussianDownsampleVKernel;   // [deprecated] kept for fallback
    // Debug output kernel
    CUfunction m_debugOutputKernel;
    // Desaturation kernel (runs after Prefilter, before Downsample)
    CUfunction m_desaturationKernel;
    // Refine kernel (BoundingBox calculation)
    CUfunction m_refineKernel;

    // ========================================
    // Interop kernels (Surface-based)
    // ========================================
    CUfunction m_unmultSurfaceKernel;          // Unmult with √max formula (surface I/O)
    CUfunction m_prefilterSurfaceKernel;       // Prefilter (surface I/O)
    CUfunction m_downsampleSurfaceKernel;      // Downsample (surface I/O)
    CUfunction m_logTransPreblurHKernel;       // Log-Transmittance H-blur (separable)
    CUfunction m_logTransPreblurVKernel;       // Log-Transmittance V-blur (separable)
    CUfunction m_clearSurfaceKernel;           // Clear surface to zero

    // ========================================
    // CUDA Streams for parallel Pre-blur
    // ========================================
    static const int MAX_PARALLEL_STREAMS = 6;
    CUstream m_preblurStreams[MAX_PARALLEL_STREAMS];  // Parallel streams for Pre-blur
    bool m_streamsInitialized;

    // MIP chain buffers (stores downsampled textures - read during upsample)
    std::vector<CUDAMipBuffer> m_mipChain;
    // Upsample result buffers (separate from mipChain to avoid race condition)
    std::vector<CUDAMipBuffer> m_upsampleChain;
    // Temp buffer for horizontal blur (separable Gaussian first pass during upsample)
    CUDAMipBuffer m_horizontalTemp;
    // Temp buffer for Gaussian downsample H-blur (source resolution)
    CUDAMipBuffer m_gaussianDownsampleTemp;
    // Temp buffer for separable prefilter H-pass
    CUDAMipBuffer m_prefilterSepTemp;
    int m_currentMipLevels;
    int m_currentWidth;
    int m_currentHeight;

    // BoundingBox for adaptive resolution
    BoundingBox m_mipBounds[MAX_MIP_LEVELS];  // BoundingBox for each MIP level
    CUdeviceptr m_refineBoundsGPU;            // GPU buffer for 4 ints (minX, maxX, minY, maxY)

    // State
    bool m_initialized;

    // Resource management
    bool AllocateMipChain(int width, int height, int levels);
    void ReleaseMipChain();
    bool LoadKernels();

    // Rendering stages
    bool ExecuteRefine(CUdeviceptr input, int width, int height, int pitchPixels,
                       float threshold, int blurRadius, int mipLevel);
    bool ExecutePrefilter(const RenderParams& params, CUdeviceptr input);
    bool ExecuteDownsampleChain(const RenderParams& params);
    bool ExecuteUpsampleChain(const RenderParams& params);
    bool ExecuteComposite(const RenderParams& params, CUdeviceptr original, CUdeviceptr output);

    // Error handling
    bool CheckCUDAError(CUresult err, const char* context);

    // ========================================
    // Interop rendering helpers (Surface + BoundingBox)
    // ========================================

    // Execute Unmult on input texture (√max formula)
    bool ExecuteUnmultInterop(
        cudaSurfaceObject_t input,
        cudaSurfaceObject_t output,
        int width, int height,
        const BoundingBox* bounds = nullptr);  // Optional BoundingBox

    // Execute Prefilter on unmulted input (threshold + 13-tap blur)
    bool ExecutePrefilterInterop(
        const RenderParams& params,
        cudaSurfaceObject_t input,
        cudaSurfaceObject_t output,
        int width, int height,
        const BoundingBox* bounds = nullptr);

    // Execute Downsample chain with surface objects
    bool ExecuteDownsampleChainInterop(
        const RenderParams& params,
        cudaSurfaceObject_t* mipSurfaces,
        int* mipWidths, int* mipHeights,
        int numLevels,
        BoundingBox* mipBounds = nullptr);  // Output: BoundingBox per level

    // Execute Log-Transmittance Pre-blur (separable Gaussian) - parallel with streams
    bool ExecuteLogTransPreblurInterop(
        cudaSurfaceObject_t input,
        cudaSurfaceObject_t tempBuffer,
        cudaSurfaceObject_t output,
        int width, int height,
        int level,
        float baseSigma,
        const BoundingBox* bounds = nullptr,
        CUstream stream = nullptr);  // Optional stream for parallel execution

    // Clear surface to zero (for initialization)
    bool ExecuteClearSurface(
        cudaSurfaceObject_t surface,
        int width, int height);

    // Initialize/Destroy parallel streams
    bool InitializeStreams();
    void DestroyStreams();
};

#endif // JUSTGLOW_CUDA_RENDERER_H
