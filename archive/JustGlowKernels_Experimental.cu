/**
 * JustGlow Experimental Kernels Reference
 *
 * ARCHIVED: 2025-12-28
 * STATUS: INACTIVE (Artifacts observed in Karis Average v1.4.0)
 *
 * Log-Transmittance Pre-blur Algorithm:
 * - Converts color to transmittance: T = 1 - color
 * - Blurs in log space: blur(log(T))
 * - Converts back: color = 1 - exp(blurred_log)
 *
 * This produces physically correct light transmission blur,
 * but was disabled due to visual artifacts in certain edge cases.
 *
 * The algorithm is mathematically sound for simulating how light
 * transmits through multiple layers of semi-transparent material.
 *
 * Theory:
 * - N sequential blurs with sigma = sigma * sqrt(N) single blur
 * - Level i: sigma_effective = baseSigma * sqrt(i)
 */

#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define EPSILON 1e-6f
#define THREAD_BLOCK_SIZE 16

// Gaussian weight function
__device__ __forceinline__ float gaussianWeight(float x, float sigma) {
    return expf(-x * x / (2.0f * sigma * sigma));
}

// ============================================================================
// LogTransmittancePreblurKernel (Full 2D version - slower but complete)
//
// Single-pass 2D Gaussian in log-transmittance space.
// Complexity: O(kernelRadius^2) per pixel
// ============================================================================

extern "C" __global__ void LogTransmittancePreblurKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int srcPitch, int dstPitch,
    int level,          // MIP level (1-6)
    float baseSigma)    // Base sigma (typically 16.0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Gaussian variance composition: sigma = baseSigma * sqrt(level)
    float sigma = baseSigma * sqrtf((float)level);
    int kernelRadius = (int)ceilf(sigma * 3.0f);  // 3 sigma covers 99.7%

    // Accumulate in log-transmittance space
    float4 sumLogT = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;

    for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
        for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
            int sx = x + kx;
            int sy = y + ky;

            // ZeroPad: out-of-bounds contributes T=1 (fully transparent)
            // log(1) = 0, so we can skip these samples
            if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
                continue;
            }

            // Read sample
            int srcIdx = (sy * srcPitch + sx) * 4;
            float4 sample = make_float4(
                input[srcIdx + 0],
                input[srcIdx + 1],
                input[srcIdx + 2],
                input[srcIdx + 3]
            );

            // Convert to transmittance: T = 1 - color
            float4 T = make_float4(
                1.0f - sample.x,
                1.0f - sample.y,
                1.0f - sample.z,
                1.0f - sample.w
            );

            // Clamp to avoid log(0)
            T.x = fmaxf(T.x, EPSILON);
            T.y = fmaxf(T.y, EPSILON);
            T.z = fmaxf(T.z, EPSILON);
            T.w = fmaxf(T.w, EPSILON);

            // Convert to log space
            float4 logT = make_float4(
                logf(T.x),
                logf(T.y),
                logf(T.z),
                logf(T.w)
            );

            // Gaussian weight
            float dist2 = (float)(kx * kx + ky * ky);
            float weight = gaussianWeight(sqrtf(dist2), sigma);

            // Accumulate weighted log-transmittance
            sumLogT.x += logT.x * weight;
            sumLogT.y += logT.y * weight;
            sumLogT.z += logT.z * weight;
            sumLogT.w += logT.w * weight;
            weightSum += weight;
        }
    }

    // Normalize
    if (weightSum > 0.0f) {
        float invWeight = 1.0f / weightSum;
        sumLogT.x *= invWeight;
        sumLogT.y *= invWeight;
        sumLogT.z *= invWeight;
        sumLogT.w *= invWeight;
    }

    // Convert back: T = exp(logT), color = 1 - T
    float4 result = make_float4(
        1.0f - expf(sumLogT.x),
        1.0f - expf(sumLogT.y),
        1.0f - expf(sumLogT.z),
        1.0f - expf(sumLogT.w)
    );

    // Write output
    int dstIdx = (y * dstPitch + x) * 4;
    output[dstIdx + 0] = result.x;
    output[dstIdx + 1] = result.y;
    output[dstIdx + 2] = result.z;
    output[dstIdx + 3] = result.w;
}

// ============================================================================
// Separable Log-Transmittance Pre-blur (Optimized)
//
// Two-pass separable Gaussian for better performance:
// 1. Horizontal pass: LogTransmittancePreblurHKernel
// 2. Vertical pass: LogTransmittancePreblurVKernel
//
// Total complexity: O(2 * kernelRadius) vs O(kernelRadius^2)
// ============================================================================

// Horizontal pass - outputs to intermediate buffer (in log space)
extern "C" __global__ void LogTransmittancePreblurHKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int srcPitch, int dstPitch,
    int level,
    float baseSigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sigma = baseSigma * sqrtf((float)level);
    int kernelRadius = (int)ceilf(sigma * 3.0f);

    // Accumulate in log-transmittance space (horizontal pass)
    float4 sumLogT = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;

    for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
        int sx = x + kx;

        if (sx < 0 || sx >= width) continue;

        int srcIdx = (y * srcPitch + sx) * 4;
        float4 sample = make_float4(
            input[srcIdx + 0],
            input[srcIdx + 1],
            input[srcIdx + 2],
            input[srcIdx + 3]
        );

        // T = 1 - color, clamped
        float4 T = make_float4(
            fmaxf(1.0f - sample.x, EPSILON),
            fmaxf(1.0f - sample.y, EPSILON),
            fmaxf(1.0f - sample.z, EPSILON),
            fmaxf(1.0f - sample.w, EPSILON)
        );

        // Convert to log space
        float4 logT = make_float4(logf(T.x), logf(T.y), logf(T.z), logf(T.w));

        float weight = gaussianWeight((float)kx, sigma);
        sumLogT.x += logT.x * weight;
        sumLogT.y += logT.y * weight;
        sumLogT.z += logT.z * weight;
        sumLogT.w += logT.w * weight;
        weightSum += weight;
    }

    if (weightSum > 0.0f) {
        float invWeight = 1.0f / weightSum;
        sumLogT.x *= invWeight;
        sumLogT.y *= invWeight;
        sumLogT.z *= invWeight;
        sumLogT.w *= invWeight;
    }

    // Keep in log space for vertical pass
    int dstIdx = (y * dstPitch + x) * 4;
    output[dstIdx + 0] = sumLogT.x;
    output[dstIdx + 1] = sumLogT.y;
    output[dstIdx + 2] = sumLogT.z;
    output[dstIdx + 3] = sumLogT.w;
}

// Vertical pass - reads intermediate log-space buffer, outputs final color
extern "C" __global__ void LogTransmittancePreblurVKernel(
    const float* __restrict__ input,  // Log-transmittance from H pass
    float* __restrict__ output,
    int width, int height, int srcPitch, int dstPitch,
    int level,
    float baseSigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sigma = baseSigma * sqrtf((float)level);
    int kernelRadius = (int)ceilf(sigma * 3.0f);

    // Accumulate log-transmittance (vertical pass)
    // Input is already in log space from H pass
    float4 sumLogT = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;

    for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
        int sy = y + ky;

        if (sy < 0 || sy >= height) continue;

        int srcIdx = (sy * srcPitch + x) * 4;
        float4 logT = make_float4(
            input[srcIdx + 0],
            input[srcIdx + 1],
            input[srcIdx + 2],
            input[srcIdx + 3]
        );

        float weight = gaussianWeight((float)ky, sigma);
        sumLogT.x += logT.x * weight;
        sumLogT.y += logT.y * weight;
        sumLogT.z += logT.z * weight;
        sumLogT.w += logT.w * weight;
        weightSum += weight;
    }

    if (weightSum > 0.0f) {
        float invWeight = 1.0f / weightSum;
        sumLogT.x *= invWeight;
        sumLogT.y *= invWeight;
        sumLogT.z *= invWeight;
        sumLogT.w *= invWeight;
    }

    // Convert back: T = exp(logT), color = 1 - T
    float4 result = make_float4(
        1.0f - expf(sumLogT.x),
        1.0f - expf(sumLogT.y),
        1.0f - expf(sumLogT.z),
        1.0f - expf(sumLogT.w)
    );

    int dstIdx = (y * dstPitch + x) * 4;
    output[dstIdx + 0] = result.x;
    output[dstIdx + 1] = result.y;
    output[dstIdx + 2] = result.z;
    output[dstIdx + 3] = result.w;
}

// ============================================================================
// Notes on why this was disabled (Karis Average v1.4.0)
// ============================================================================

/*
ARTIFACTS OBSERVED:
1. Edge ringing at high-contrast boundaries
2. Color bleeding in semi-transparent areas
3. Halo inversion in certain HDR scenarios

POSSIBLE CAUSES:
1. Log domain amplifies small differences near T=1
2. Weight normalization issues at image boundaries
3. Precision loss in exp/log conversions

POTENTIAL FIXES (if revisiting):
1. Use double precision for accumulation
2. Add special handling for T near 0 or 1
3. Blend with regular Gaussian for low-alpha areas
4. Use iterative log-space accumulation instead of single pass

MATHEMATICAL NOTE:
The algorithm is correct for physical light transmission:
- T_total = T_1 * T_2 * ... * T_n
- log(T_total) = log(T_1) + log(T_2) + ... + log(T_n)
- blur(log(T)) = weighted average of log transmittances
- T_blurred = exp(blur(log(T)))
- color = 1 - T_blurred

The issue is numerical stability, not mathematical correctness.
*/
