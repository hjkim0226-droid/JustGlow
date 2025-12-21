/**
 * JustGlow CUDA Kernels
 *
 * GPU compute kernels for the glow effect pipeline.
 * Uses Driver API compatible signatures for PTX loading.
 *
 * Kernels:
 * - PrefilterKernel: 13-tap downsample + soft threshold + alpha-weighted average
 * - GaussianDownsampleH/VKernel: Separable 5-tap Gaussian (Level 0-4)
 * - DownsampleKernel: Dual Kawase 5-tap with X/+ rotation (Level 5+)
 * - UpsampleKernel: 9-tap tent filter with progressive blend
 * - DebugOutputKernel: Final composite + debug view modes
 *
 * Note: Karis Average removed (v1.4.0) - caused artifacts on transparent backgrounds
 */

#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define THREAD_BLOCK_SIZE 16
#define EPSILON 0.0001f

// ============================================================================
// Helper Device Functions
// ============================================================================

__device__ __forceinline__ float clampf(float x, float minVal, float maxVal) {
    return fminf(fmaxf(x, minVal), maxVal);
}

__device__ __forceinline__ float luminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Smooth step function for fade transitions
__device__ __forceinline__ float smoothstepf(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ============================================================================
// Color Space Conversion (sRGB <-> Linear)
// Glow must be calculated in Linear space for physically correct light addition
// ============================================================================

// sRGB to Linear (remove gamma, called at input)
__device__ __forceinline__ float srgbToLinear(float c) {
    // Clamp negative values (can happen with some footage)
    if (c <= 0.0f) return 0.0f;
    // Standard sRGB transfer function
    return (c <= 0.04045f)
        ? c / 12.92f
        : powf((c + 0.055f) / 1.055f, 2.4f);
}

// Linear to sRGB (apply gamma, called at output)
__device__ __forceinline__ float linearToSrgb(float c) {
    // Clamp negative values
    if (c <= 0.0f) return 0.0f;
    // Standard sRGB transfer function
    return (c <= 0.0031308f)
        ? c * 12.92f
        : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

// ============================================================================
// Additional Color Profile Conversions
// ============================================================================

// Rec.709 to Linear (BT.709 gamma ~2.4)
__device__ __forceinline__ float rec709ToLinear(float c) {
    if (c <= 0.0f) return 0.0f;
    return (c < 0.081f)
        ? c / 4.5f
        : powf((c + 0.099f) / 1.099f, 1.0f / 0.45f);
}

// Linear to Rec.709
__device__ __forceinline__ float linearToRec709(float c) {
    if (c <= 0.0f) return 0.0f;
    return (c < 0.018f)
        ? c * 4.5f
        : 1.099f * powf(c, 0.45f) - 0.099f;
}

// Gamma 2.2 to Linear (pure power function)
__device__ __forceinline__ float gamma22ToLinear(float c) {
    if (c <= 0.0f) return 0.0f;
    return powf(c, 2.2f);
}

// Linear to Gamma 2.2
__device__ __forceinline__ float linearToGamma22(float c) {
    if (c <= 0.0f) return 0.0f;
    return powf(c, 1.0f / 2.2f);
}

// Generic toLinear based on profile (1=sRGB, 2=Rec709, 3=Gamma2.2)
__device__ __forceinline__ float toLinear(float c, int profile) {
    switch (profile) {
        case 2:  return rec709ToLinear(c);
        case 3:  return gamma22ToLinear(c);
        default: return srgbToLinear(c);  // 1 or default = sRGB
    }
}

// Generic fromLinear based on profile
__device__ __forceinline__ float fromLinear(float c, int profile) {
    switch (profile) {
        case 2:  return linearToRec709(c);
        case 3:  return linearToGamma22(c);
        default: return linearToSrgb(c);  // 1 or default = sRGB
    }
}

// Unpremultiply alpha (AE uses premultiplied alpha)
// Converts from premultiplied to straight alpha for correct threshold calculation
__device__ __forceinline__ void unpremultiply(float& r, float& g, float& b, float a) {
    // Higher threshold (0.01) to avoid extreme values at anti-aliased edges
    // Also clamp result to prevent fireflies from edge artifacts
    if (a > 0.01f) {
        float invA = 1.0f / a;
        r = fminf(r * invA, 10.0f);  // Clamp to reasonable max
        g = fminf(g * invA, 10.0f);
        b = fminf(b * invA, 10.0f);
    } else {
        // Very low alpha - treat as transparent (zero contribution)
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
    }
}

// Calculate weight based on level, falloff, and intensity
// Level 0: always 100%
// Level 1: level1Weight (controlled by Intensity parameter, 50%-100%)
// Level 2+: level1Weight * pow(decayRate, level-1)
// Falloff: 0%=boost outer, 50%=neutral, 100%=decay to core
__device__ __forceinline__ float calculatePhysicalWeight(float level, float falloff, int falloffType, float level1Weight) {
    // Level 0 is always 100%
    if (level < 0.5f) return 1.0f;

    // Calculate decayRate from Falloff (0-100, 50=neutral)
    // Symmetric around neutral: decayRate = 1.0 - normalizedFalloff * 0.5
    // Falloff 0%   -> decayRate 1.5  (boost outer levels)
    // Falloff 50%  -> decayRate 1.0  (neutral, natural decay only)
    // Falloff 100% -> decayRate 0.5  (strong decay to core)
    float normalizedFalloff = (falloff - 50.0f) / 50.0f;  // -1 to 1
    float decayRate = 1.0f - normalizedFalloff * 0.5f;    // 0.5 to 1.5

    // Level 1 starts at level1Weight, then multiplies by decayRate per level
    // weight = level1Weight * pow(decayRate, level - 1)
    float weight = level1Weight * powf(decayRate, level - 1.0f);

    return fmaxf(0.0f, weight);
}

// ============================================================================
// Bilinear Sampling
// ============================================================================

__device__ void sampleBilinear(
    const float* src, float u, float v,
    int width, int height, int pitch,
    float& outR, float& outG, float& outB, float& outA)
{
    u = clampf(u, 0.0f, 1.0f);
    v = clampf(v, 0.0f, 1.0f);

    float px = u * (float)width - 0.5f;
    float py = v * (float)height - 0.5f;

    int x0 = (int)floorf(px);
    int y0 = (int)floorf(py);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = px - (float)x0;
    float fy = py - (float)y0;

    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    // pitch is in pixels (not floats)
    int idx00 = (y0 * pitch + x0) * 4;
    int idx10 = (y0 * pitch + x1) * 4;
    int idx01 = (y1 * pitch + x0) * 4;
    int idx11 = (y1 * pitch + x1) * 4;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    outR = w00 * src[idx00 + 0] + w10 * src[idx10 + 0] + w01 * src[idx01 + 0] + w11 * src[idx11 + 0];
    outG = w00 * src[idx00 + 1] + w10 * src[idx10 + 1] + w01 * src[idx01 + 1] + w11 * src[idx11 + 1];
    outB = w00 * src[idx00 + 2] + w10 * src[idx10 + 2] + w01 * src[idx01 + 2] + w11 * src[idx11 + 2];
    outA = w00 * src[idx00 + 3] + w10 * src[idx10 + 3] + w01 * src[idx01 + 3] + w11 * src[idx11 + 3];
}

// ============================================================================
// Bilinear Sampling with Zero Padding
// Returns black (0,0,0,0) for UV coordinates outside [0,1] range
// Used in Prefilter to prevent edge pixel repetition causing light clumping
// ============================================================================

__device__ void sampleBilinearZeroPad(
    const float* src, float u, float v,
    int width, int height, int pitch,
    float& outR, float& outG, float& outB, float& outA)
{
    // Return black for out-of-bounds UV
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        outR = 0.0f;
        outG = 0.0f;
        outB = 0.0f;
        outA = 0.0f;
        return;
    }

    float px = u * (float)width - 0.5f;
    float py = v * (float)height - 0.5f;

    int x0 = (int)floorf(px);
    int y0 = (int)floorf(py);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = px - (float)x0;
    float fy = py - (float)y0;

    // For pixels near edge, check each sample point individually
    // If a sample point is out of bounds, treat it as black (0,0,0,0)
    float r00 = 0.0f, g00 = 0.0f, b00 = 0.0f, a00 = 0.0f;
    float r10 = 0.0f, g10 = 0.0f, b10 = 0.0f, a10 = 0.0f;
    float r01 = 0.0f, g01 = 0.0f, b01 = 0.0f, a01 = 0.0f;
    float r11 = 0.0f, g11 = 0.0f, b11 = 0.0f, a11 = 0.0f;

    // Sample only if within bounds
    if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
        int idx = (y0 * pitch + x0) * 4;
        r00 = src[idx + 0]; g00 = src[idx + 1]; b00 = src[idx + 2]; a00 = src[idx + 3];
    }
    if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
        int idx = (y0 * pitch + x1) * 4;
        r10 = src[idx + 0]; g10 = src[idx + 1]; b10 = src[idx + 2]; a10 = src[idx + 3];
    }
    if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
        int idx = (y1 * pitch + x0) * 4;
        r01 = src[idx + 0]; g01 = src[idx + 1]; b01 = src[idx + 2]; a01 = src[idx + 3];
    }
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
        int idx = (y1 * pitch + x1) * 4;
        r11 = src[idx + 0]; g11 = src[idx + 1]; b11 = src[idx + 2]; a11 = src[idx + 3];
    }

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    outR = w00 * r00 + w10 * r10 + w01 * r01 + w11 * r11;
    outG = w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11;
    outB = w00 * b00 + w10 * b10 + w01 * b01 + w11 * b11;
    outA = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11;
}

// ============================================================================
// Soft Threshold (Symmetric around T)
// Dynamic K: maxK = min(T, 1-T), actualK = maxK * softness
// Effective range: [T-K, T+K] (symmetric)
// - Below T-K: contribution = 0
// - T-K ~ T+K: smooth S-curve
// - Above T+K: contribution = 1.0 (pass through)
// ============================================================================

__device__ void softThreshold(
    float& r, float& g, float& b,
    float threshold, float softness)
{
    // Threshold 0 = bypass (모든 픽셀 통과)
    if (threshold <= 0.001f) {
        return;
    }

    float brightness = fmaxf(fmaxf(r, g), b);

    // Dynamic K calculation: maxK = min(T, 1-T), actualK = maxK * softness
    float maxK = fminf(threshold, 1.0f - threshold);
    float K = maxK * softness;

    // Threshold range: [T-K, T+K] (symmetric around T)
    float lowerBound = threshold - K;
    float upperBound = threshold + K;

    // Below T-K: contribution = 0
    if (brightness <= lowerBound) {
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
        return;
    }

    // Above T+K: contribution = 1.0 (pass through)
    if (brightness >= upperBound) {
        return;
    }

    // Hard threshold (no softness)
    if (K <= 0.001f) {
        if (brightness < threshold) {
            r = 0.0f;
            g = 0.0f;
            b = 0.0f;
        }
        return;
    }

    // Soft curve in [T-K, T+K] range
    // Normalize position: 0 at T-K, 1 at T+K
    float t = (brightness - lowerBound) / (2.0f * K);

    // Smooth step curve (S-curve)
    float contribution = t * t * (3.0f - 2.0f * t);

    r *= contribution;
    g *= contribution;
    b *= contribution;
}

// ============================================================================
// Prefilter Kernel
// Driver API entry point
// ============================================================================

extern "C" __global__ void PrefilterKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    int inputWidth, int inputHeight,
    float threshold, float softKnee, float intensity,
    float colorR, float colorG, float colorB,
    float colorTempR, float colorTempG, float colorTempB,
    float preserveColor, int useHDR, int useLinear, int inputProfile)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    // Calculate offset for centering input within expanded output
    int offsetX = (dstWidth - inputWidth) / 2;
    int offsetY = (dstHeight - inputHeight) / 2;

    // Map output coordinates to input UV coordinates
    // No early return - all pixels calculated with ZeroPad sampling
    // This ensures natural fade-out at edges without stretching
    float srcX = (float)x - (float)offsetX;
    float srcY = (float)y - (float)offsetY;

    // UV coordinates (can be outside [0,1] for padding area)
    float u = (srcX + 0.5f) / (float)inputWidth;
    float v = (srcY + 0.5f) / (float)inputHeight;

    float texelX = 1.0f / (float)inputWidth;
    float texelY = 1.0f / (float)inputHeight;

    // 13-tap sampling
    float Ar, Ag, Ab, Aa;
    float Br, Bg, Bb, Ba;
    float Cr, Cg, Cb, Ca;
    float Dr, Dg, Db, Da;
    float Er, Eg, Eb, Ea;
    float Fr, Fg, Fb, Fa;
    float Gr, Gg, Gb, Ga;
    float Hr, Hg, Hb, Ha;
    float Ir, Ig, Ib, Ia;
    float Jr, Jg, Jb, Ja;
    float Kr, Kg, Kb, Ka;
    float Lr, Lg, Lb, La;
    float Mr, Mg, Mb, Ma;

    // =========================================
    // 13-tap sampling with ZERO PADDING
    // Out-of-bounds samples return 0 for natural edge fade
    // No edge stretching - pixels blend with black at boundaries
    // =========================================

    // Outer corners
    sampleBilinearZeroPad(input, u - 2.0f * texelX, v - 2.0f * texelY, inputWidth, inputHeight, srcPitch, Ar, Ag, Ab, Aa);
    sampleBilinearZeroPad(input, u + 2.0f * texelX, v - 2.0f * texelY, inputWidth, inputHeight, srcPitch, Cr, Cg, Cb, Ca);
    sampleBilinearZeroPad(input, u - 2.0f * texelX, v + 2.0f * texelY, inputWidth, inputHeight, srcPitch, Kr, Kg, Kb, Ka);
    sampleBilinearZeroPad(input, u + 2.0f * texelX, v + 2.0f * texelY, inputWidth, inputHeight, srcPitch, Mr, Mg, Mb, Ma);

    // Outer cross
    sampleBilinearZeroPad(input, u, v - 2.0f * texelY, inputWidth, inputHeight, srcPitch, Br, Bg, Bb, Ba);
    sampleBilinearZeroPad(input, u - 2.0f * texelX, v, inputWidth, inputHeight, srcPitch, Fr, Fg, Fb, Fa);
    sampleBilinearZeroPad(input, u + 2.0f * texelX, v, inputWidth, inputHeight, srcPitch, Hr, Hg, Hb, Ha);
    sampleBilinearZeroPad(input, u, v + 2.0f * texelY, inputWidth, inputHeight, srcPitch, Lr, Lg, Lb, La);

    // Inner corners
    sampleBilinearZeroPad(input, u - texelX, v - texelY, inputWidth, inputHeight, srcPitch, Dr, Dg, Db, Da);
    sampleBilinearZeroPad(input, u + texelX, v - texelY, inputWidth, inputHeight, srcPitch, Er, Eg, Eb, Ea);
    sampleBilinearZeroPad(input, u - texelX, v + texelY, inputWidth, inputHeight, srcPitch, Ir, Ig, Ib, Ia);
    sampleBilinearZeroPad(input, u + texelX, v + texelY, inputWidth, inputHeight, srcPitch, Jr, Jg, Jb, Ja);

    // Center
    sampleBilinearZeroPad(input, u, v, inputWidth, inputHeight, srcPitch, Gr, Gg, Gb, Ga);

    // =========================================
    // Weighted Average (kernel weights only)
    // Input is premultiplied, so RGB already contains alpha info
    // =========================================
    (void)useHDR;  // Suppress unused parameter warning

    // Kernel weights for 13-tap sampling (normalized to sum = 1.0):
    // Center G: 4/11, Inner DEIJ: 1/11 each (4/11 total), Outer cross BFHL: 1/22 each (2/11 total), Corners ACKM: 1/44 each (1/11 total)
    // Previous weights summed to 0.34375, causing ~66% brightness loss
    const float wCenter = 4.0f / 11.0f;      // 0.3636
    const float wInner = 1.0f / 11.0f;       // 0.0909
    const float wCross = 1.0f / 22.0f;       // 0.0455
    const float wCorner = 1.0f / 44.0f;      // 0.0227

    // Simple weighted average of premultiplied RGB
    // Zero-padded samples contribute 0, which is correct for blur (light spreads into empty space)
    float sumR = Gr * wCenter + (Dr + Er + Ir + Jr) * wInner + (Br + Fr + Hr + Lr) * wCross + (Ar + Cr + Kr + Mr) * wCorner;
    float sumG = Gg * wCenter + (Dg + Eg + Ig + Jg) * wInner + (Bg + Fg + Hg + Lg) * wCross + (Ag + Cg + Kg + Mg) * wCorner;
    float sumB = Gb * wCenter + (Db + Eb + Ib + Jb) * wInner + (Bb + Fb + Hb + Lb) * wCross + (Ab + Cb + Kb + Mb) * wCorner;
    float sumA = Ga * wCenter + (Da + Ea + Ia + Ja) * wInner + (Ba + Fa + Ha + La) * wCross + (Aa + Ca + Ka + Ma) * wCorner;

    // =========================================
    // Color Space Conversion
    // OFF: Premultiplied sRGB/Rec709/Gamma2.2 유지
    // ON:  Unmult → Input Profile→Linear → Premult
    // Threshold는 항상 Premultiplied 상태에서 적용 (통일)
    // inputProfile: 1=sRGB, 2=Rec709, 3=Gamma2.2
    // =========================================
    float resR, resG, resB;

    if (useLinear) {
        // Step 1: Unpremultiply
        float straightR, straightG, straightB;
        if (sumA > 0.001f) {
            straightR = sumR / sumA;
            straightG = sumG / sumA;
            straightB = sumB / sumA;
        } else {
            straightR = 0.0f;
            straightG = 0.0f;
            straightB = 0.0f;
        }

        // Step 2: Input Profile → Linear
        straightR = toLinear(straightR, inputProfile);
        straightG = toLinear(straightG, inputProfile);
        straightB = toLinear(straightB, inputProfile);

        // Step 3: Premultiply
        resR = straightR * sumA;
        resG = straightG * sumA;
        resB = straightB * sumA;
    } else {
        // No conversion: stay in Premultiplied (sRGB/Rec709/Gamma2.2)
        resR = sumR;
        resG = sumG;
        resB = sumB;
    }

    // Soft threshold on Premultiplied (통일된 위치)
    softThreshold(resR, resG, resB, threshold, softKnee);

    // Write output
    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = sumA;
}

// ============================================================================
// 2D Gaussian Downsample (9-tap, single pass with ZeroPad)
// Replaces separable H+V passes for consistency across different buffer sizes
// Uses ZeroPad sampling to prevent edge energy concentration
// 3×3 Gaussian weights: [1,2,1; 2,4,2; 1,2,1] / 16
// ============================================================================

extern "C" __global__ void Gaussian2DDownsampleKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    // Map destination pixel to source UV (center of the 2x2 block we're downsampling)
    float u = ((float)x + 0.5f) / (float)dstWidth;
    float v = ((float)y + 0.5f) / (float)dstHeight;

    float texelX = 1.0f / (float)srcWidth;
    float texelY = 1.0f / (float)srcHeight;

    // 3×3 Gaussian weights: [1,2,1; 2,4,2; 1,2,1] / 16
    const float wCenter = 4.0f / 16.0f;    // 0.25
    const float wCross = 2.0f / 16.0f;     // 0.125
    const float wDiagonal = 1.0f / 16.0f;  // 0.0625

    // Sample 9 points with ZeroPad (energy escapes at edges instead of being trapped)
    float TLr, TLg, TLb, TLa;  // Top-Left
    float Tr, Tg, Tb, Ta;       // Top
    float TRr, TRg, TRb, TRa;  // Top-Right
    float Lr, Lg, Lb, La;       // Left
    float Cr, Cg, Cb, Ca;       // Center
    float Rr, Rg, Rb, Ra;       // Right
    float BLr, BLg, BLb, BLa;  // Bottom-Left
    float Br, Bg, Bb, Ba;       // Bottom
    float BRr, BRg, BRb, BRa;  // Bottom-Right

    // Sample all 9 points with ZeroPad (out-of-bounds = black)
    sampleBilinearZeroPad(input, u - texelX, v - texelY, srcWidth, srcHeight, srcPitch, TLr, TLg, TLb, TLa);
    sampleBilinearZeroPad(input, u,          v - texelY, srcWidth, srcHeight, srcPitch, Tr, Tg, Tb, Ta);
    sampleBilinearZeroPad(input, u + texelX, v - texelY, srcWidth, srcHeight, srcPitch, TRr, TRg, TRb, TRa);

    sampleBilinearZeroPad(input, u - texelX, v,          srcWidth, srcHeight, srcPitch, Lr, Lg, Lb, La);
    sampleBilinearZeroPad(input, u,          v,          srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);
    sampleBilinearZeroPad(input, u + texelX, v,          srcWidth, srcHeight, srcPitch, Rr, Rg, Rb, Ra);

    sampleBilinearZeroPad(input, u - texelX, v + texelY, srcWidth, srcHeight, srcPitch, BLr, BLg, BLb, BLa);
    sampleBilinearZeroPad(input, u,          v + texelY, srcWidth, srcHeight, srcPitch, Br, Bg, Bb, Ba);
    sampleBilinearZeroPad(input, u + texelX, v + texelY, srcWidth, srcHeight, srcPitch, BRr, BRg, BRb, BRa);

    // Weighted sum
    float outR = Cr * wCenter +
                 (Tr + Lr + Rr + Br) * wCross +
                 (TLr + TRr + BLr + BRr) * wDiagonal;
    float outG = Cg * wCenter +
                 (Tg + Lg + Rg + Bg) * wCross +
                 (TLg + TRg + BLg + BRg) * wDiagonal;
    float outB = Cb * wCenter +
                 (Tb + Lb + Rb + Bb) * wCross +
                 (TLb + TRb + BLb + BRb) * wDiagonal;
    float outA = Ca * wCenter +
                 (Ta + La + Ra + Ba) * wCross +
                 (TLa + TRa + BLa + BRa) * wDiagonal;

    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = outR;
    output[outIdx + 1] = outG;
    output[outIdx + 2] = outB;
    output[outIdx + 3] = outA;
}

// ============================================================================
// [DEPRECATED] Gaussian Downsample - Horizontal Pass
// Kept for reference, replaced by Gaussian2DDownsampleKernel
// ============================================================================

extern "C" __global__ void GaussianDownsampleHKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int pitch,
    float blurOffset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    float texelX = 1.0f / (float)width;

    const float offset1 = 1.2f;
    const float weight0 = 0.375f;
    const float weight1 = 0.3125f;

    float cR, cG, cB, cA;
    float lR, lG, lB, lA;
    float rR, rG, rB, rA;

    sampleBilinear(input, u, v, width, height, pitch, cR, cG, cB, cA);
    sampleBilinear(input, u - offset1 * texelX, v, width, height, pitch, lR, lG, lB, lA);
    sampleBilinear(input, u + offset1 * texelX, v, width, height, pitch, rR, rG, rB, rA);

    float outR = cR * weight0 + (lR + rR) * weight1;
    float outG = cG * weight0 + (lG + rG) * weight1;
    float outB = cB * weight0 + (lB + rB) * weight1;
    float outA = cA * weight0 + (lA + rA) * weight1;

    int outIdx = (y * pitch + x) * 4;
    output[outIdx + 0] = outR;
    output[outIdx + 1] = outG;
    output[outIdx + 2] = outB;
    output[outIdx + 3] = outA;
}

// ============================================================================
// [DEPRECATED] Gaussian Downsample - Vertical Pass with 2x Downsample
// Kept for reference, replaced by Gaussian2DDownsampleKernel
// ============================================================================

extern "C" __global__ void GaussianDownsampleVKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float u = ((float)x + 0.5f) / (float)dstWidth;
    float v = ((float)y + 0.5f) / (float)dstHeight;

    float texelY = 1.0f / (float)srcHeight;

    const float offset1 = 1.2f;
    const float weight0 = 0.375f;
    const float weight1 = 0.3125f;

    float cR, cG, cB, cA;
    float tR, tG, tB, tA;
    float bR, bG, bB, bA;

    sampleBilinear(input, u, v, srcWidth, srcHeight, srcPitch, cR, cG, cB, cA);
    sampleBilinear(input, u, v - offset1 * texelY, srcWidth, srcHeight, srcPitch, tR, tG, tB, tA);
    sampleBilinear(input, u, v + offset1 * texelY, srcWidth, srcHeight, srcPitch, bR, bG, bB, bA);

    float outR = cR * weight0 + (tR + bR) * weight1;
    float outG = cG * weight0 + (tG + bG) * weight1;
    float outB = cB * weight0 + (tB + bB) * weight1;
    float outA = cA * weight0 + (tA + bA) * weight1;

    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = outR;
    output[outIdx + 1] = outG;
    output[outIdx + 2] = outB;
    output[outIdx + 3] = outA;
}

// ============================================================================
// Downsample Kernel (Dual Kawase 5-Tap with X/+ Rotation)
// rotationMode: 0 = X (diagonal), 1 = + (cross)
// Alternating patterns breaks up boxy artifacts -> rounder glow
// Used for Level 5+ where speed matters more than detail
// Uses ZeroPad sampling for consistent edge behavior across buffer sizes
// ============================================================================

extern "C" __global__ void DownsampleKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset, int rotationMode)  // blurOffset ignored, using fixed 0.5
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float u = ((float)x + 0.5f) / (float)dstWidth;
    float v = ((float)y + 0.5f) / (float)dstHeight;

    float texelX = 1.0f / (float)srcWidth;
    float texelY = 1.0f / (float)srcHeight;

    // Standard Dual Kawase downsample: fixed 0.5px offset
    // Samples at pixel boundaries for optimal bilinear blend
    const float offset = 0.5f;

    float Ar, Ag, Ab, Aa;
    float Br, Bg, Bb, Ba;
    float Cr, Cg, Cb, Ca;
    float Dr, Dg, Db, Da;
    float centerR, centerG, centerB, centerA;

    // Use ZeroPad sampling for consistent edge behavior
    // Energy escapes at edges instead of being trapped (clamped)
    if (rotationMode == 0) {
        // X pattern (diagonal) - default
        sampleBilinearZeroPad(input, u - offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, Ar, Ag, Ab, Aa);
        sampleBilinearZeroPad(input, u + offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, Br, Bg, Bb, Ba);
        sampleBilinearZeroPad(input, u - offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);
        sampleBilinearZeroPad(input, u + offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, Dr, Dg, Db, Da);
    } else {
        // + pattern (cross) - 45 degree rotation
        // Slightly larger offset (1.414x) to maintain similar coverage area
        float crossOffset = offset * 1.414f;
        sampleBilinearZeroPad(input, u, v - crossOffset * texelY, srcWidth, srcHeight, srcPitch, Ar, Ag, Ab, Aa);  // Top
        sampleBilinearZeroPad(input, u + crossOffset * texelX, v, srcWidth, srcHeight, srcPitch, Br, Bg, Bb, Ba);  // Right
        sampleBilinearZeroPad(input, u, v + crossOffset * texelY, srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);  // Bottom
        sampleBilinearZeroPad(input, u - crossOffset * texelX, v, srcWidth, srcHeight, srcPitch, Dr, Dg, Db, Da);  // Left
    }

    sampleBilinearZeroPad(input, u, v, srcWidth, srcHeight, srcPitch, centerR, centerG, centerB, centerA);

    float resR = centerR * 0.5f + (Ar + Br + Cr + Dr) * 0.125f;
    float resG = centerG * 0.5f + (Ag + Bg + Cg + Dg) * 0.125f;
    float resB = centerB * 0.5f + (Ab + Bb + Cb + Db) * 0.125f;

    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = 1.0f;
}

// ============================================================================
// Horizontal Blur Kernel (9-Tap Gaussian, 5-fetch linear optimized)
// First pass of separable Gaussian blur for upsample
// 9-tap Discrete [1,8,28,56,70,56,28,8,1]/256 -> 5-fetch Linear
// Offsets: [0, 1.33, 3.11] with weights [0.27343750, 0.32812500, 0.03515625]
// ============================================================================

extern "C" __global__ void HorizontalBlurKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int pitch,
    float blurOffset)  // ignored, using fixed offsets
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    float texelX = 1.0f / (float)width;

    // Linear sampling optimized offsets and weights
    // 9-tap [1,8,28,56,70,56,28,8,1]/256 -> 5-fetch
    const float offset1 = 1.33f;   // Between pixel 1 and 2
    const float offset2 = 3.11f;   // Between pixel 3 and 4
    const float weight0 = 0.27343750f;  // Center (70/256)
    const float weight1 = 0.32812500f;  // Inner pair (84/256 each side)
    const float weight2 = 0.03515625f;  // Outer pair (9/256 each side)

    // Sample 5 points (center + 2 inner + 2 outer with linear interpolation)
    float cR, cG, cB, cA;   // Center
    float l1R, l1G, l1B, l1A;  // Left inner
    float r1R, r1G, r1B, r1A;  // Right inner
    float l2R, l2G, l2B, l2A;  // Left outer
    float r2R, r2G, r2B, r2A;  // Right outer

    sampleBilinear(input, u, v, width, height, pitch, cR, cG, cB, cA);
    sampleBilinear(input, u - offset1 * texelX, v, width, height, pitch, l1R, l1G, l1B, l1A);
    sampleBilinear(input, u + offset1 * texelX, v, width, height, pitch, r1R, r1G, r1B, r1A);
    sampleBilinear(input, u - offset2 * texelX, v, width, height, pitch, l2R, l2G, l2B, l2A);
    sampleBilinear(input, u + offset2 * texelX, v, width, height, pitch, r2R, r2G, r2B, r2A);

    // Weighted sum
    float outR = cR * weight0 + (l1R + r1R) * weight1 + (l2R + r2R) * weight2;
    float outG = cG * weight0 + (l1G + r1G) * weight1 + (l2G + r2G) * weight2;
    float outB = cB * weight0 + (l1B + r1B) * weight1 + (l2B + r2B) * weight2;
    float outA = cA * weight0 + (l1A + r1A) * weight1 + (l2A + r2A) * weight2;

    int outIdx = (y * pitch + x) * 4;
    output[outIdx + 0] = outR;
    output[outIdx + 1] = outG;
    output[outIdx + 2] = outB;
    output[outIdx + 3] = outA;
}

// ============================================================================
// Upsample Kernel (9-Tap Discrete Gaussian with Advanced Falloff)
// Standard 3x3 Gaussian Upsample - NO Linear Optimization (prevents shift)
// Parameters:
// - levelIndex: current MIP level (0 = core, higher = atmosphere)
// - activeLimit: Radius-controlled threshold base (0-1, from radius param)
// - decayK: Falloff decay constant (0-100, 50=neutral)
// - falloffType: 0=Exponential, 1=InverseSquare, 2=Linear
// - maxLevels: total MIP levels for threshold scaling
// NEW: Radius now applies soft threshold per level instead of hard cutoff
// ============================================================================

extern "C" __global__ void UpsampleKernel(
    const float* __restrict__ input,
    const float* __restrict__ prevLevel,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int prevWidth, int prevHeight, int prevPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset,  // ignored, using fixed 1.0
    int levelIndex,
    float activeLimit,  // now 0-1 (radius / 100)
    float decayK,
    float level1Weight,
    int falloffType,
    int maxLevels,      // total MIP levels
    int blurMode)       // ignored, always use 9-tap Discrete Gaussian
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float u = ((float)x + 0.5f) / (float)dstWidth;
    float v = ((float)y + 0.5f) / (float)dstHeight;

    float resR = 0.0f, resG = 0.0f, resB = 0.0f;

    // =========================================================
    // STEP 1: Upsample from Previous Level (smaller texture)
    // 9-Tap Discrete Gaussian (3x3 pattern)
    // Dynamic offset (1.5 + 0.3*level) - prevents center clumping at higher levels
    // Weights: Center=4/16, Cross=2/16, Diagonal=1/16 (total=16)
    // =========================================================
    if (prevLevel != nullptr) {
        float texelX = 1.0f / (float)prevWidth;
        float texelY = 1.0f / (float)prevHeight;

        // Fixed offset for consistent bilinear sampling
        // 1.0 = sample exactly at neighboring pixel centers
        // Prevents ghosting at high MIP levels where dynamic offset exceeded texture size
        const float offset = 1.0f;

        // =========================================
        // 9-Tap Discrete Gaussian (3x3 pattern)
        // For upsampling, we sample 9 points honestly
        // No linear optimization - prevents shift artifacts
        // =========================================
        float TLr, TLg, TLb, TLa;  // Top-Left (diagonal)
        float Tr, Tg, Tb, Ta;       // Top (cross)
        float TRr, TRg, TRb, TRa;  // Top-Right (diagonal)
        float Lr, Lg, Lb, La;       // Left (cross)
        float Cr, Cg, Cb, Ca;       // Center
        float Rr, Rg, Rb, Ra;       // Right (cross)
        float BLr, BLg, BLb, BLa;  // Bottom-Left (diagonal)
        float Bor, Bog, Bob, Boa;  // Bottom (cross)
        float BRr, BRg, BRb, BRa;  // Bottom-Right (diagonal)

        // Sample all 9 points with ZeroPad (consistent with downsample)
        sampleBilinearZeroPad(prevLevel, u - offset * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, TLr, TLg, TLb, TLa);
        sampleBilinearZeroPad(prevLevel, u, v - offset * texelY, prevWidth, prevHeight, prevPitch, Tr, Tg, Tb, Ta);
        sampleBilinearZeroPad(prevLevel, u + offset * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, TRr, TRg, TRb, TRa);

        sampleBilinearZeroPad(prevLevel, u - offset * texelX, v, prevWidth, prevHeight, prevPitch, Lr, Lg, Lb, La);
        sampleBilinearZeroPad(prevLevel, u, v, prevWidth, prevHeight, prevPitch, Cr, Cg, Cb, Ca);
        sampleBilinearZeroPad(prevLevel, u + offset * texelX, v, prevWidth, prevHeight, prevPitch, Rr, Rg, Rb, Ra);

        sampleBilinearZeroPad(prevLevel, u - offset * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, BLr, BLg, BLb, BLa);
        sampleBilinearZeroPad(prevLevel, u, v + offset * texelY, prevWidth, prevHeight, prevPitch, Bor, Bog, Bob, Boa);
        sampleBilinearZeroPad(prevLevel, u + offset * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, BRr, BRg, BRb, BRa);

        // Gaussian weights: Center=4/16, Cross=2/16, Diagonal=1/16
        // This is the standard "국룰" for smooth upsampling
        const float wCenter = 0.25f;     // 4/16
        const float wCross = 0.125f;     // 2/16
        const float wDiagonal = 0.0625f; // 1/16

        resR = Cr * wCenter +
               (Tr + Lr + Rr + Bor) * wCross +
               (TLr + TRr + BLr + BRr) * wDiagonal;
        resG = Cg * wCenter +
               (Tg + Lg + Rg + Bog) * wCross +
               (TLg + TRg + BLg + BRg) * wDiagonal;
        resB = Cb * wCenter +
               (Tb + Lb + Rb + Bob) * wCross +
               (TLb + TRb + BLb + BRb) * wDiagonal;
    }

    // =========================================================
    // STEP 2: Add Current Level's Contribution with Weight
    // "현재 층의 디테일을 가중치 적용해서 더한다"
    // =========================================================
    float currR, currG, currB, currA;
    sampleBilinearZeroPad(input, u, v, srcWidth, srcHeight, srcPitch, currR, currG, currB, currA);

    // Weight calculation for current level
    // A. Physical decay weight (The Shape) - Level 0=100%, Level 1=level1Weight, then decay
    float physicalWeight = calculatePhysicalWeight((float)levelIndex, decayK, falloffType, level1Weight);

    // Apply physical weight first
    float contribR = currR * physicalWeight;
    float contribG = currG * physicalWeight;
    float contribB = currB * physicalWeight;

    // B. Radius-based Soft Threshold (replaces hard level cutoff)
    // - Higher level = more threshold applied
    // - Lower radius = more threshold applied
    // - Removes "faint" outer glow first, keeps bright core
    float levelRatio = (float)levelIndex / fmaxf((float)maxLevels, 1.0f);  // 0 to 1
    float radiusFactor = 1.0f - activeLimit;  // activeLimit is radius/100, so inverted

    // Threshold increases with level and decreases with radius
    // At radius=100% (activeLimit=1): threshold = 0 for all levels
    // At radius=0% (activeLimit=0): threshold = levelRatio (higher for outer levels)
    float levelThreshold = radiusFactor * levelRatio * 0.5f;  // max threshold = 0.5 at level=max, radius=0

    // Soft knee = 50% of threshold (automatic)
    float knee = levelThreshold * 0.5f;

    // Apply soft threshold to contribution (per channel)
    if (levelThreshold > 0.001f) {
        float brightness = fmaxf(fmaxf(contribR, contribG), contribB);

        float lowerBound = levelThreshold - knee;
        float upperBound = levelThreshold + knee;

        float contribution = 1.0f;
        if (brightness <= lowerBound) {
            contribution = 0.0f;
        } else if (brightness < upperBound) {
            // Soft curve in transition zone
            float t = (brightness - lowerBound) / fmaxf(2.0f * knee, 0.001f);
            contribution = t * t * (3.0f - 2.0f * t);  // smoothstep
        }
        // else contribution = 1.0 (full pass)

        contribR *= contribution;
        contribG *= contribution;
        contribB *= contribution;
    }

    // Add to upsampled base
    // Result = TentUpsample(Previous) + Current × Weight (after threshold)
    resR = resR + contribR;
    resG = resG + contribG;
    resB = resB + contribB;

    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = 1.0f;
}

// ============================================================================
// Composite Kernel
// ============================================================================

extern "C" __global__ void CompositeKernel(
    const float* __restrict__ original,
    const float* __restrict__ glow,
    float* __restrict__ output,
    int width, int height,
    int inputWidth, int inputHeight,
    int originalPitch, int glowPitch, int outputPitch,
    int compositeMode,
    float exposure)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Calculate offset for centering original within expanded output
    // The original layer is centered within the glow-expanded output
    int offsetX = (width - inputWidth) / 2;
    int offsetY = (height - inputHeight) / 2;

    // Read original pixel (with bounds check accounting for expansion offset)
    float origR = 0.0f, origG = 0.0f, origB = 0.0f, origA = 0.0f;
    int srcX = x - offsetX;
    int srcY = y - offsetY;

    if (srcX >= 0 && srcX < inputWidth && srcY >= 0 && srcY < inputHeight) {
        int origIdx = (srcY * originalPitch + srcX) * 4;
        origR = original[origIdx + 0];
        origG = original[origIdx + 1];
        origB = original[origIdx + 2];
        origA = original[origIdx + 3];
    }

    // Sample glow with bilinear
    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    float glowR, glowG, glowB, glowA;
    sampleBilinear(glow, u, v, width, height, glowPitch, glowR, glowG, glowB, glowA);

    // Apply exposure here (once) instead of accumulating in UpsampleKernel
    // This prevents color blowout from exposure being multiplied at each level
    glowR *= exposure;
    glowG *= exposure;
    glowB *= exposure;

    float resR, resG, resB;

    // Composite modes: 0=Add, 1=Screen, 2=Overlay
    switch (compositeMode) {
        case 0: // Add
            resR = origR + glowR;
            resG = origG + glowG;
            resB = origB + glowB;
            break;
        case 1: // Screen
            resR = 1.0f - (1.0f - origR) * (1.0f - glowR);
            resG = 1.0f - (1.0f - origG) * (1.0f - glowG);
            resB = 1.0f - (1.0f - origB) * (1.0f - glowB);
            break;
        case 2: // Overlay
            resR = origR < 0.5f ? 2.0f * origR * glowR : 1.0f - 2.0f * (1.0f - origR) * (1.0f - glowR);
            resG = origG < 0.5f ? 2.0f * origG * glowG : 1.0f - 2.0f * (1.0f - origG) * (1.0f - glowG);
            resB = origB < 0.5f ? 2.0f * origB * glowB : 1.0f - 2.0f * (1.0f - origB) * (1.0f - glowB);
            break;
        default:
            resR = origR + glowR;
            resG = origG + glowG;
            resB = origB + glowB;
            break;
    }

    int outIdx = (y * outputPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;

    // Expand alpha to include glow contribution (for transparent backgrounds)
    float glowLum = fmaxf(fmaxf(glowR, glowG), glowB);
    float expandedAlpha = fmaxf(origA, clampf(glowLum, 0.0f, 1.0f));
    output[outIdx + 3] = expandedAlpha;
}

// ============================================================================
// Debug Output Kernel
// Outputs a specific MIP buffer upsampled to full resolution
// Used for visualizing individual pipeline stages
// ============================================================================

extern "C" __global__ void DebugOutputKernel(
    const float* __restrict__ original,
    const float* __restrict__ debugBuffer,
    const float* __restrict__ glow,
    float* __restrict__ output,
    int width, int height,
    int inputWidth, int inputHeight,
    int originalPitch, int debugWidth, int debugHeight, int debugPitch,
    int glowWidth, int glowHeight, int glowPitch, int outputPitch,
    int debugMode,          // DebugViewMode enum value
    float exposure,
    float sourceOpacity,    // 0-1
    float glowOpacity,      // 0-2
    int compositeMode,
    int useLinear,          // Enable sRGB to Linear conversion
    int inputProfile,       // 1=sRGB, 2=Rec709, 3=Gamma2.2
    float glowTintR,        // Glow tint color R
    float glowTintG,        // Glow tint color G
    float glowTintB,        // Glow tint color B
    float dither)           // Dithering amount (0-1)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Calculate offset for centering original within expanded output
    int offsetX = (width - inputWidth) / 2;
    int offsetY = (height - inputHeight) / 2;

    // Read original pixel (with bounds check accounting for expansion offset)
    float origR = 0.0f, origG = 0.0f, origB = 0.0f, origA = 0.0f;
    int srcX = x - offsetX;
    int srcY = y - offsetY;

    if (srcX >= 0 && srcX < inputWidth && srcY >= 0 && srcY < inputHeight) {
        int origIdx = (srcY * originalPitch + srcX) * 4;
        origR = original[origIdx + 0];
        origG = original[origIdx + 1];
        origB = original[origIdx + 2];
        origA = original[origIdx + 3];

        if (useLinear) {
            // Convert original from Premultiplied Input Profile to Premultiplied Linear
            // VFX rule: Unpremult → Input Profile→Linear → Premult
            if (origA > 0.001f) {
                float invA = 1.0f / origA;
                origR = toLinear(origR * invA, inputProfile) * origA;
                origG = toLinear(origG * invA, inputProfile) * origA;
                origB = toLinear(origB * invA, inputProfile) * origA;
            }
        }
        // When useLinear=false, original stays in Premultiplied (sRGB/Rec709/Gamma2.2)
    }

    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    float resR, resG, resB, resA;

    // debugMode: 1=Final, 2=Prefilter, 3-8=Down1-6, 9-15=Up0-6, 16=GlowOnly
    if (debugMode == 1) {
        // Final: normal composite with opacity controls
        // Color space: Premultiplied Linear (if useLinear) or Premultiplied sRGB (if not)
        float glowR, glowG, glowB, glowA;
        sampleBilinear(glow, u, v, glowWidth, glowHeight, glowPitch, glowR, glowG, glowB, glowA);

        // Note: Glow color/tint is already applied in Prefilter stage
        // (via glowColor with preserveColor blending)
        // Do NOT apply again here - causes blue channel to zero out completely

        // Apply exposure and glow opacity
        glowR *= exposure * glowOpacity;
        glowG *= exposure * glowOpacity;
        glowB *= exposure * glowOpacity;

        // Highlight desaturation: brightness > 1.0 gradually desaturates toward white
        // Prevents over-saturated "burning" colors in bright areas
        float glowBrightness = fmaxf(fmaxf(glowR, glowG), glowB);
        if (glowBrightness > 1.0f) {
            float glowLum = 0.2126f * glowR + 0.7152f * glowG + 0.0722f * glowB;
            // Soft ramp: 0 at brightness=1, 1 at brightness=3
            float desatT = fminf((glowBrightness - 1.0f) / 2.0f, 1.0f);
            // Apply 30% desaturation (conservative to avoid too white)
            float desatAmount = desatT * 0.3f;
            glowR = glowR + (glowLum - glowR) * desatAmount;
            glowG = glowG + (glowLum - glowG) * desatAmount;
            glowB = glowB + (glowLum - glowB) * desatAmount;
        }

        // Apply source opacity (premultiplied)
        float srcR = origR * sourceOpacity;
        float srcG = origG * sourceOpacity;
        float srcB = origB * sourceOpacity;
        float srcA = origA * sourceOpacity;

        // Composite based on mode
        // Note: Case values match CompositeMode enum (1=Add, 2=Screen, 3=Overlay)
        switch (compositeMode) {
            case 1: // Add - additive blending (standard glow)
                // Add works directly on premultiplied values
                resR = srcR + glowR;
                resG = srcG + glowG;
                resB = srcB + glowB;
                break;

            case 2: // Screen - premultiplied formula: A + B - AB
                // Both srcR and glowR are light contributions
                // Screen combines them: result = A + B - A*B
                resR = srcR + glowR - srcR * glowR;
                resG = srcG + glowG - srcG * glowG;
                resB = srcB + glowB - srcB * glowB;
                break;

            case 3: { // Overlay - premultiplied: conditional multiply/screen
                // Decision based on straight source luminance
                float straightSrcR = (srcA > 0.001f) ? srcR / srcA : 0.0f;
                float straightSrcG = (srcA > 0.001f) ? srcG / srcA : 0.0f;
                float straightSrcB = (srcA > 0.001f) ? srcB / srcA : 0.0f;

                // Overlay on premultiplied values:
                // < 0.5: Multiply-like: 2 * src * glow
                // >= 0.5: Screen-like: src + 2*glow*(1-src)
                resR = (straightSrcR < 0.5f)
                    ? 2.0f * srcR * glowR
                    : srcR + 2.0f * glowR * (1.0f - srcR);
                resG = (straightSrcG < 0.5f)
                    ? 2.0f * srcG * glowG
                    : srcG + 2.0f * glowG * (1.0f - srcG);
                resB = (straightSrcB < 0.5f)
                    ? 2.0f * srcB * glowB
                    : srcB + 2.0f * glowB * (1.0f - srcB);
                break;
            }

            default: // Fallback to Add
                resR = srcR + glowR;
                resG = srcG + glowG;
                resB = srcB + glowB;
                break;
        }

        // Calculate alpha from blended result
        // Coverage = max RGB of blended result, clamped to [0,1]
        float blendedCoverage = clampf(fmaxf(fmaxf(resR, resG), resB), 0.0f, 1.0f);
        resA = fmaxf(srcA, blendedCoverage);
    }
    else if (debugMode == 16) {
        // GlowOnly: just glow with exposure and opacity
        // Color space: Premultiplied Linear (if useLinear) or Premultiplied sRGB (if not)
        float glowR, glowG, glowB, glowA;
        sampleBilinear(glow, u, v, glowWidth, glowHeight, glowPitch, glowR, glowG, glowB, glowA);

        // Note: Glow color already applied in Prefilter

        // Apply exposure and opacity
        resR = glowR * exposure * glowOpacity;
        resG = glowG * exposure * glowOpacity;
        resB = glowB * exposure * glowOpacity;

        // Highlight desaturation (same as Final mode)
        float glowBrightness = fmaxf(fmaxf(resR, resG), resB);
        if (glowBrightness > 1.0f) {
            float glowLum = 0.2126f * resR + 0.7152f * resG + 0.0722f * resB;
            float desatT = fminf((glowBrightness - 1.0f) / 2.0f, 1.0f);
            float desatAmount = desatT * 0.3f;
            resR = resR + (glowLum - resR) * desatAmount;
            resG = resG + (glowLum - resG) * desatAmount;
            resB = resB + (glowLum - resB) * desatAmount;
        }

        // Alpha from glow coverage
        float glowCoverage = fmaxf(fmaxf(resR, resG), resB);
        resA = clampf(glowCoverage, 0.0f, 1.0f);
    }
    else {
        // Debug view: show specific buffer (Prefilter, Down1-6, Up0-6)
        // Check for null buffer or zero dimensions (level doesn't exist)
        if (debugBuffer == nullptr || debugWidth <= 0 || debugHeight <= 0) {
            // Show magenta placeholder with checkerboard pattern for missing levels
            int checkerX = (x / 16) % 2;
            int checkerY = (y / 16) % 2;
            bool checker = (checkerX ^ checkerY) == 0;
            resR = checker ? 1.0f : 0.5f;
            resG = 0.0f;
            resB = checker ? 1.0f : 0.5f;
            resA = 1.0f;
        }
        else {
            // debugBuffer is in Linear space
            float dbgR, dbgG, dbgB, dbgA;
            sampleBilinear(debugBuffer, u, v, debugWidth, debugHeight, debugPitch, dbgR, dbgG, dbgB, dbgA);

            // For debug views, apply exposure so we can see threshold results
            resR = dbgR * exposure;
            resG = dbgG * exposure;
            resB = dbgB * exposure;
            resA = 1.0f;
        }

        // =========================================
        // Algorithm Indicator for debug views
        // Top-left 24x24 square with color indicator:
        // - Yellow = Prefilter (debugMode 2)
        // - Green = Gaussian Downsample (debugMode 3-7 = Down1-5)
        // - Blue = Kawase Downsample (debugMode 8 = Down6+)
        // - Cyan = Upsample (debugMode 9-15 = Up0-6)
        // - Magenta = Level not available
        // =========================================
        const int indicatorSize = 24;
        const int borderSize = 2;

        if (x < indicatorSize && y < indicatorSize) {
            // Inside indicator box
            bool isBorder = (x < borderSize || x >= indicatorSize - borderSize ||
                             y < borderSize || y >= indicatorSize - borderSize);

            if (isBorder) {
                // White border for visibility
                resR = 1.0f;
                resG = 1.0f;
                resB = 1.0f;
            } else {
                // Check if level is available (buffer is valid)
                bool levelAvailable = (debugBuffer != nullptr && debugWidth > 0 && debugHeight > 0);

                if (!levelAvailable) {
                    // Magenta = Level not available
                    resR = 1.0f;
                    resG = 0.0f;
                    resB = 1.0f;
                } else if (debugMode == 2) {
                    // Prefilter - Yellow
                    resR = 1.0f;
                    resG = 0.9f;
                    resB = 0.2f;
                } else if (debugMode >= 3 && debugMode <= 7) {
                    // Gaussian Downsample (Down1-5) - Green
                    resR = 0.2f;
                    resG = 0.8f;
                    resB = 0.2f;
                } else if (debugMode == 8) {
                    // Kawase Downsample (Down6+) - Blue
                    resR = 0.2f;
                    resG = 0.4f;
                    resB = 0.9f;
                } else if (debugMode >= 9 && debugMode <= 15) {
                    // Upsample (Up0-6) - Cyan
                    resR = 0.2f;
                    resG = 0.8f;
                    resB = 0.9f;
                }
            }
            resA = 1.0f;
        }
    }

    // =========================================
    // Output Color Space Conversion
    // =========================================
    if (useLinear) {
        // Convert from Premultiplied Linear to Premultiplied Input Profile
        // VFX rule: Unpremult → Linear→Input Profile → Premult
        if (resA > 0.001f) {
            float invA = 1.0f / resA;
            resR = fromLinear(resR * invA, inputProfile) * resA;
            resG = fromLinear(resG * invA, inputProfile) * resA;
            resB = fromLinear(resB * invA, inputProfile) * resA;
        }
    }
    // When useLinear=false, output stays in Premultiplied (sRGB/Rec709/Gamma2.2)

    // =========================================
    // Dithering (banding prevention)
    // =========================================
    if (dither > 0.0f) {
        // Pseudo-random noise based on pixel position (fast hash)
        float noise = fmodf(sinf((float)x * 12.9898f + (float)y * 78.233f) * 43758.5453f, 1.0f) - 0.5f;
        // Scale to ~1-2 bits range (dither=1.0 -> ~2/255 amplitude)
        float ditherAmount = dither * (2.0f / 255.0f) * noise;
        resR += ditherAmount;
        resG += ditherAmount;
        resB += ditherAmount;
    }

    int outIdx = (y * outputPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = resA;
}
