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
#include <vector>

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

    // State
    bool m_initialized;

    // Resource management
    bool AllocateMipChain(int width, int height, int levels);
    void ReleaseMipChain();
    bool LoadKernels();

    // Rendering stages
    bool ExecutePrefilter(const RenderParams& params, CUdeviceptr input);
    bool ExecuteDownsampleChain(const RenderParams& params);
    bool ExecuteUpsampleChain(const RenderParams& params);
    bool ExecuteComposite(const RenderParams& params, CUdeviceptr original, CUdeviceptr output);

    // Error handling
    bool CheckCUDAError(CUresult err, const char* context);
};

#endif // JUSTGLOW_CUDA_RENDERER_H
