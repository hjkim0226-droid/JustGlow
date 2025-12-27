/**
 * JustGlow Surface Kernels Reference
 *
 * ARCHIVED: 2025-12-28
 *
 * Surface-based CUDA kernels for DX12-CUDA Interop.
 * These kernels use cudaSurfaceObject_t for zero-copy
 * memory access shared between DX12 and CUDA.
 *
 * KEY DIFFERENCE from CUdeviceptr kernels:
 * - Surface I/O: surf2Dread() / surf2Dwrite()
 * - Direct access to DX12 shared memory
 * - No explicit pitch calculation needed
 */

#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define EPSILON 1e-6f
#define THREAD_BLOCK_SIZE 16

__device__ inline float smoothstepf(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ inline float gaussianWeight(float dist, float sigma) {
    return expf(-0.5f * (dist * dist) / (sigma * sigma));
}

// ============================================================================
// Surface Unmult Kernel (sqrt(max) formula)
// ============================================================================

extern "C" __global__ void UnmultSurfaceKernel(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    int width, int height,
    int boundMinX, int boundMinY,
    int boundWidth, int boundHeight)
{
    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;

    if (localX >= boundWidth || localY >= boundHeight)
        return;

    int x = localX + boundMinX;
    int y = localY + boundMinY;

    if (x >= width || y >= height)
        return;

    // Read premultiplied RGBA from surface
    float4 premult;
    surf2Dread(&premult, input, x * sizeof(float4), y);

    // sqrt(max) formula for Unmult
    float maxP = fmaxf(fmaxf(premult.x, premult.y), premult.z);

    float4 result;
    if (maxP < EPSILON) {
        result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        float estA = sqrtf(maxP);
        float invEstA = 1.0f / estA;
        result = make_float4(
            premult.x * invEstA,
            premult.y * invEstA,
            premult.z * invEstA,
            estA
        );
    }

    surf2Dwrite(result, output, x * sizeof(float4), y);
}

// ============================================================================
// Surface Prefilter Kernel (Soft Threshold + 13-tap Gaussian + Unmult)
// ============================================================================

extern "C" __global__ void PrefilterSurfaceKernel(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    int width, int height,
    float threshold, float softKnee,
    float glowR, float glowG, float glowB,
    float preserveColor,
    int boundMinX, int boundMinY,
    int boundWidth, int boundHeight)
{
    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;

    if (localX >= boundWidth || localY >= boundHeight)
        return;

    int x = localX + boundMinX;
    int y = localY + boundMinY;

    if (x >= width || y >= height)
        return;

    // 13-tap star pattern
    const int2 offsets[13] = {
        {0, 0},
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-2, 0}, {2, 0}, {0, -2}, {0, 2},
        {-1, -1}, {1, -1}, {-1, 1}, {1, 1}
    };
    const float weights[13] = {
        0.25f,
        0.125f, 0.125f, 0.125f, 0.125f,
        0.0625f, 0.0625f, 0.0625f, 0.0625f,
        0.03125f, 0.03125f, 0.03125f, 0.03125f
    };

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;

    for (int i = 0; i < 13; i++) {
        int sx = x + offsets[i].x;
        int sy = y + offsets[i].y;

        if (sx < 0 || sx >= width || sy < 0 || sy >= height)
            continue;

        float4 sample;
        surf2Dread(&sample, input, sx * sizeof(float4), sy);

        // Unmult (sqrt(max))
        float maxP = fmaxf(fmaxf(sample.x, sample.y), sample.z);
        if (maxP > EPSILON) {
            float estA = sqrtf(maxP);
            float invEstA = 1.0f / estA;
            sample.x *= invEstA;
            sample.y *= invEstA;
            sample.z *= invEstA;
            sample.w = estA;
        } else {
            sample = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // Soft threshold
        float luma = sample.x * 0.2126f + sample.y * 0.7152f + sample.z * 0.0722f;
        float knee = threshold * softKnee;
        float t = smoothstepf(threshold - knee, threshold + knee, luma);

        if (t > 0.0f) {
            // Color blending with glow color
            float3 blended = make_float3(
                sample.x * glowR + (sample.x - sample.x * glowR) * preserveColor,
                sample.y * glowG + (sample.y - sample.y * glowG) * preserveColor,
                sample.z * glowB + (sample.z - sample.z * glowB) * preserveColor
            );

            float w = weights[i];
            sum.x += blended.x * t * w;
            sum.y += blended.y * t * w;
            sum.z += blended.z * t * w;
            sum.w += sample.w * t * w;
            weightSum += w;
        }
    }

    float4 result;
    if (weightSum > 0.0f) {
        float inv = 1.0f / weightSum;
        result = make_float4(sum.x * inv, sum.y * inv, sum.z * inv, sum.w * inv);
    } else {
        result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    surf2Dwrite(result, output, x * sizeof(float4), y);
}

// ============================================================================
// Surface 9-tap 2D Gaussian Downsample (ZeroPad)
// ============================================================================

extern "C" __global__ void DownsampleSurfaceKernel(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight,
    float offset,
    int boundMinX, int boundMinY,
    int boundWidth, int boundHeight)
{
    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;

    if (localX >= boundWidth || localY >= boundHeight)
        return;

    int dstX = localX + boundMinX;
    int dstY = localY + boundMinY;

    if (dstX >= dstWidth || dstY >= dstHeight)
        return;

    float srcCenterX = (dstX + 0.5f) * 2.0f - 0.5f;
    float srcCenterY = (dstY + 0.5f) * 2.0f - 0.5f;

    const float2 offs[9] = {
        {0, 0},
        {-offset, 0}, {offset, 0}, {0, -offset}, {0, offset},
        {-offset, -offset}, {offset, -offset}, {-offset, offset}, {offset, offset}
    };
    const float wts[9] = {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.0625f, 0.0625f, 0.0625f, 0.0625f};

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float wSum = 0.0f;

    for (int i = 0; i < 9; i++) {
        int sx = (int)(srcCenterX + offs[i].x + 0.5f);
        int sy = (int)(srcCenterY + offs[i].y + 0.5f);

        // ZeroPad: skip out-of-bounds samples
        if (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight)
            continue;

        float4 s;
        surf2Dread(&s, input, sx * sizeof(float4), sy);

        sum.x += s.x * wts[i];
        sum.y += s.y * wts[i];
        sum.z += s.z * wts[i];
        sum.w += s.w * wts[i];
        wSum += wts[i];
    }

    float4 result = (wSum > 0.0f)
        ? make_float4(sum.x/wSum, sum.y/wSum, sum.z/wSum, sum.w/wSum)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    surf2Dwrite(result, output, dstX * sizeof(float4), dstY);
}

// ============================================================================
// Surface Log-Transmittance Pre-blur H (Horizontal, separable)
//
// Algorithm: T = 1 - color, blur(log(T)), then exp back
// This produces physically correct light transmission blur.
// ============================================================================

extern "C" __global__ void LogTransPreblurHSurfaceKernel(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    int width, int height,
    int level,
    float baseSigma,
    int boundMinX, int boundMinY,
    int boundWidth, int boundHeight)
{
    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;

    if (localX >= boundWidth || localY >= boundHeight)
        return;

    int x = localX + boundMinX;
    int y = localY + boundMinY;

    if (x >= width || y >= height)
        return;

    // sigma = baseSigma * sqrt(level) for progressive blur
    float sigma = baseSigma * sqrtf((float)level);
    int radius = (int)ceilf(sigma * 3.0f);

    float4 sumLogT = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float wSum = 0.0f;

    for (int kx = -radius; kx <= radius; kx++) {
        int sx = x + kx;
        if (sx < 0 || sx >= width) continue;

        float4 sample;
        surf2Dread(&sample, input, sx * sizeof(float4), y);

        // Transmittance: T = 1 - color
        float4 T = make_float4(
            fmaxf(1.0f - sample.x, 1e-6f),
            fmaxf(1.0f - sample.y, 1e-6f),
            fmaxf(1.0f - sample.z, 1e-6f),
            fmaxf(1.0f - sample.w, 1e-6f)
        );
        // Log space
        float4 logT = make_float4(logf(T.x), logf(T.y), logf(T.z), logf(T.w));

        float w = gaussianWeight((float)kx, sigma);
        sumLogT.x += logT.x * w;
        sumLogT.y += logT.y * w;
        sumLogT.z += logT.z * w;
        sumLogT.w += logT.w * w;
        wSum += w;
    }

    if (wSum > 0.0f) {
        float inv = 1.0f / wSum;
        sumLogT.x *= inv; sumLogT.y *= inv; sumLogT.z *= inv; sumLogT.w *= inv;
    }

    // Output is still in log space (intermediate result)
    surf2Dwrite(sumLogT, output, x * sizeof(float4), y);
}

// ============================================================================
// Surface Log-Transmittance Pre-blur V (Vertical, final output)
// ============================================================================

extern "C" __global__ void LogTransPreblurVSurfaceKernel(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    int width, int height,
    int level,
    float baseSigma,
    int boundMinX, int boundMinY,
    int boundWidth, int boundHeight)
{
    int localX = blockIdx.x * blockDim.x + threadIdx.x;
    int localY = blockIdx.y * blockDim.y + threadIdx.y;

    if (localX >= boundWidth || localY >= boundHeight)
        return;

    int x = localX + boundMinX;
    int y = localY + boundMinY;

    if (x >= width || y >= height)
        return;

    float sigma = baseSigma * sqrtf((float)level);
    int radius = (int)ceilf(sigma * 3.0f);

    float4 sumLogT = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float wSum = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        int sy = y + ky;
        if (sy < 0 || sy >= height) continue;

        // Read log-space intermediate from H-pass
        float4 logT;
        surf2Dread(&logT, input, x * sizeof(float4), sy);

        float w = gaussianWeight((float)ky, sigma);
        sumLogT.x += logT.x * w;
        sumLogT.y += logT.y * w;
        sumLogT.z += logT.z * w;
        sumLogT.w += logT.w * w;
        wSum += w;
    }

    if (wSum > 0.0f) {
        float inv = 1.0f / wSum;
        sumLogT.x *= inv; sumLogT.y *= inv; sumLogT.z *= inv; sumLogT.w *= inv;
    }

    // Convert back from log space: color = 1 - exp(logT)
    float4 result = make_float4(
        1.0f - expf(sumLogT.x),
        1.0f - expf(sumLogT.y),
        1.0f - expf(sumLogT.z),
        1.0f - expf(sumLogT.w)
    );

    surf2Dwrite(result, output, x * sizeof(float4), y);
}

// ============================================================================
// Surface Clear Kernel
// ============================================================================

extern "C" __global__ void ClearSurfaceKernel(
    cudaSurfaceObject_t surface,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    surf2Dwrite(zero, surface, x * sizeof(float4), y);
}
