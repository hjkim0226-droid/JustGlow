/**
 * JustGlow CUDA Kernels
 *
 * GPU compute kernels for the glow effect pipeline.
 * Uses Driver API compatible signatures for PTX loading.
 *
 * Kernels:
 * - PrefilterKernel: 13-tap downsample + soft threshold + alpha-weighted average
 * - Gaussian2DDownsampleKernel: 9-tap 2D Gaussian for ALL levels (temporal stability)
 * - UpsampleKernel: 9-tap Gaussian with progressive blend
 * - DebugOutputKernel: Final composite + debug view modes
 *
 * Notes:
 * - Karis Average removed (v1.4.0) - caused artifacts on transparent backgrounds
 * - Kawase downsample removed (v1.5.0) - caused temporal flickering on movement
 * - All sampling uses ZeroPad for consistent edge behavior
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
// Level 0: level1Weight (controlled by Intensity parameter)
// Level 1: level1Weight * decayRate
// Level N: level1Weight * pow(decayRate, N)
// Falloff: 0%=boost outer, 50%=neutral, 100%=decay to core
__device__ __forceinline__ float calculatePhysicalWeight(float level, float falloff, int falloffType, float level1Weight) {
    // Calculate decayRate from Falloff (0-100, 50=neutral)
    // Symmetric around neutral: decayRate = 1.0 - normalizedFalloff * 0.5
    // Falloff 0%   -> decayRate 1.5  (boost outer levels)
    // Falloff 50%  -> decayRate 1.0  (neutral, natural decay only)
    // Falloff 100% -> decayRate 0.5  (strong decay to core)
    float normalizedFalloff = (falloff - 50.0f) / 50.0f;  // -1 to 1
    float decayRate = 1.0f - normalizedFalloff * 0.5f;    // 0.5 to 1.5

    // Level 0 starts at level1Weight, then multiplies by decayRate per level
    // weight = level1Weight * pow(decayRate, level)
    float weight = level1Weight * powf(decayRate, level);

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
// Desaturation Kernel (Max-based, in-place)
// Runs before Prefilter to desaturate input toward max channel
// This simulates natural highlight blowout to white
// ============================================================================

extern "C" __global__ void DesaturationKernel(
    float* __restrict__ data,
    int width, int height, int pitch,
    float desaturation)  // 0-1: amount to blend toward max
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * pitch + x) * 4;
    float r = data[idx + 0];
    float g = data[idx + 1];
    float b = data[idx + 2];
    // Alpha unchanged

    // Max-based desaturation: blend toward max channel (adds only)
    float maxVal = fmaxf(fmaxf(r, g), b);
    r = r + (maxVal - r) * desaturation;
    g = g + (maxVal - g) * desaturation;
    b = b + (maxVal - b) * desaturation;

    data[idx + 0] = r;
    data[idx + 1] = g;
    data[idx + 2] = b;
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
    float preserveColor, int useHDR, int useLinear, int inputProfile,
    float offsetPrefilter)  // 0-10: sampling offset multiplier
    // Note: Desaturation now applied via separate DesaturationKernel before Prefilter
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
    // offsetPrefilter controls the sampling distance (default 1.0)
    // =========================================
    float outerOffset = 2.0f * offsetPrefilter;
    float innerOffset = 1.0f * offsetPrefilter;

    // Outer corners
    sampleBilinearZeroPad(input, u - outerOffset * texelX, v - outerOffset * texelY, inputWidth, inputHeight, srcPitch, Ar, Ag, Ab, Aa);
    sampleBilinearZeroPad(input, u + outerOffset * texelX, v - outerOffset * texelY, inputWidth, inputHeight, srcPitch, Cr, Cg, Cb, Ca);
    sampleBilinearZeroPad(input, u - outerOffset * texelX, v + outerOffset * texelY, inputWidth, inputHeight, srcPitch, Kr, Kg, Kb, Ka);
    sampleBilinearZeroPad(input, u + outerOffset * texelX, v + outerOffset * texelY, inputWidth, inputHeight, srcPitch, Mr, Mg, Mb, Ma);

    // Outer cross
    sampleBilinearZeroPad(input, u, v - outerOffset * texelY, inputWidth, inputHeight, srcPitch, Br, Bg, Bb, Ba);
    sampleBilinearZeroPad(input, u - outerOffset * texelX, v, inputWidth, inputHeight, srcPitch, Fr, Fg, Fb, Fa);
    sampleBilinearZeroPad(input, u + outerOffset * texelX, v, inputWidth, inputHeight, srcPitch, Hr, Hg, Hb, Ha);
    sampleBilinearZeroPad(input, u, v + outerOffset * texelY, inputWidth, inputHeight, srcPitch, Lr, Lg, Lb, La);

    // Inner corners
    sampleBilinearZeroPad(input, u - innerOffset * texelX, v - innerOffset * texelY, inputWidth, inputHeight, srcPitch, Dr, Dg, Db, Da);
    sampleBilinearZeroPad(input, u + innerOffset * texelX, v - innerOffset * texelY, inputWidth, inputHeight, srcPitch, Er, Eg, Eb, Ea);
    sampleBilinearZeroPad(input, u - innerOffset * texelX, v + innerOffset * texelY, inputWidth, inputHeight, srcPitch, Ir, Ig, Ib, Ia);
    sampleBilinearZeroPad(input, u + innerOffset * texelX, v + innerOffset * texelY, inputWidth, inputHeight, srcPitch, Jr, Jg, Jb, Ja);

    // Center
    sampleBilinearZeroPad(input, u, v, inputWidth, inputHeight, srcPitch, Gr, Gg, Gb, Ga);

    // =========================================
    // Weighted Average using Gaussian kernel
    // Input is premultiplied, so RGB already contains alpha info
    // =========================================
    (void)useHDR;  // Suppress unused parameter warning

    // Gaussian kernel weights: weight = exp(-d² / (2σ²))
    // σ = outerOffset * 0.85 provides good coverage for 13-tap pattern
    // Weights calculated based on actual sample distances
    float sigma = outerOffset * 0.85f;
    float sigma2x2 = 2.0f * sigma * sigma;

    // Distance² for each sample group:
    // Center (0,0): d² = 0
    // Inner corners (±inner, ±inner): d² = 2 * inner²
    // Outer cross (±outer, 0): d² = outer²
    // Outer corners (±outer, ±outer): d² = 2 * outer²
    float innerDist2 = 2.0f * innerOffset * innerOffset;
    float crossDist2 = outerOffset * outerOffset;
    float cornerDist2 = 2.0f * outerOffset * outerOffset;

    // Gaussian weights (unnormalized)
    float wCenter_raw = 1.0f;  // exp(0) = 1
    float wInner_raw = expf(-innerDist2 / sigma2x2);
    float wCross_raw = expf(-crossDist2 / sigma2x2);
    float wCorner_raw = expf(-cornerDist2 / sigma2x2);

    // Normalize so sum = 1.0
    float wSum = wCenter_raw + 4.0f * wInner_raw + 4.0f * wCross_raw + 4.0f * wCorner_raw;
    float wCenter = wCenter_raw / wSum;
    float wInner = wInner_raw / wSum;
    float wCross = wCross_raw / wSum;
    float wCorner = wCorner_raw / wSum;

    // Simple weighted average of premultiplied RGB
    // Zero-padded samples contribute 0, which is correct for blur (light spreads into empty space)
    float sumR = Gr * wCenter + (Dr + Er + Ir + Jr) * wInner + (Br + Fr + Hr + Lr) * wCross + (Ar + Cr + Kr + Mr) * wCorner;
    float sumG = Gg * wCenter + (Dg + Eg + Ig + Jg) * wInner + (Bg + Fg + Hg + Lg) * wCross + (Ag + Cg + Kg + Mg) * wCorner;
    float sumB = Gb * wCenter + (Db + Eb + Ib + Jb) * wInner + (Bb + Fb + Hb + Lb) * wCross + (Ab + Cb + Kb + Mb) * wCorner;
    float sumA = Ga * wCenter + (Da + Ea + Ia + Ja) * wInner + (Ba + Fa + Ha + La) * wCross + (Aa + Ca + Ka + Ma) * wCorner;

    // Clamp to 1.0 for consistency (text layer vs adjustment layer)
    sumR = fminf(sumR, 1.0f);
    sumG = fminf(sumG, 1.0f);
    sumB = fminf(sumB, 1.0f);

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

    // Note: Desaturation applied via separate kernel after Prefilter

    // Write output - alpha=1 for glow buffer (padding area needs full opacity for blending)
    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = 1.0f;
}

// ============================================================================
// Prefilter 25-Tap (5x5 Discrete Gaussian)
// True Gaussian kernel on regular grid for highest quality
// ============================================================================

extern "C" __global__ void Prefilter25TapKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    int inputWidth, int inputHeight,
    float threshold, float softKnee, float intensity,
    float colorR, float colorG, float colorB,
    float colorTempR, float colorTempG, float colorTempB,
    float preserveColor, int useHDR, int useLinear, int inputProfile,
    float offsetPrefilter)
    // Note: Desaturation now applied via separate DesaturationKernel after Prefilter
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    int offsetX = (dstWidth - inputWidth) / 2;
    int offsetY = (dstHeight - inputHeight) / 2;

    float srcX = (float)x - (float)offsetX;
    float srcY = (float)y - (float)offsetY;

    float u = (srcX + 0.5f) / (float)inputWidth;
    float v = (srcY + 0.5f) / (float)inputHeight;

    float texelX = 1.0f / (float)inputWidth;
    float texelY = 1.0f / (float)inputHeight;

    // 5x5 Gaussian kernel with adjustable offset
    // Standard weights: σ ≈ 1.0
    // [1  4  6  4 1]
    // [4 16 24 16 4]
    // [6 24 36 24 6] / 256
    // [4 16 24 16 4]
    // [1  4  6  4 1]
    const float w0 = 1.0f / 256.0f;   // corners
    const float w1 = 4.0f / 256.0f;   // edge
    const float w2 = 6.0f / 256.0f;   // edge center
    const float w3 = 16.0f / 256.0f;  // inner corner
    const float w4 = 24.0f / 256.0f;  // inner edge
    const float w5 = 36.0f / 256.0f;  // center

    float offset1 = 1.0f * offsetPrefilter;
    float offset2 = 2.0f * offsetPrefilter;

    float sumR = 0, sumG = 0, sumB = 0, sumA = 0;
    float r, g, b, a;

    // Row -2
    sampleBilinearZeroPad(input, u - offset2*texelX, v - offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w0; sumG += g*w0; sumB += b*w0; sumA += a*w0;
    sampleBilinearZeroPad(input, u - offset1*texelX, v - offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinearZeroPad(input, u, v - offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w2; sumG += g*w2; sumB += b*w2; sumA += a*w2;
    sampleBilinearZeroPad(input, u + offset1*texelX, v - offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinearZeroPad(input, u + offset2*texelX, v - offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w0; sumG += g*w0; sumB += b*w0; sumA += a*w0;

    // Row -1
    sampleBilinearZeroPad(input, u - offset2*texelX, v - offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinearZeroPad(input, u - offset1*texelX, v - offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w3; sumG += g*w3; sumB += b*w3; sumA += a*w3;
    sampleBilinearZeroPad(input, u, v - offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w4; sumG += g*w4; sumB += b*w4; sumA += a*w4;
    sampleBilinearZeroPad(input, u + offset1*texelX, v - offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w3; sumG += g*w3; sumB += b*w3; sumA += a*w3;
    sampleBilinearZeroPad(input, u + offset2*texelX, v - offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;

    // Row 0 (center)
    sampleBilinearZeroPad(input, u - offset2*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w2; sumG += g*w2; sumB += b*w2; sumA += a*w2;
    sampleBilinearZeroPad(input, u - offset1*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w4; sumG += g*w4; sumB += b*w4; sumA += a*w4;
    sampleBilinearZeroPad(input, u, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w5; sumG += g*w5; sumB += b*w5; sumA += a*w5;
    sampleBilinearZeroPad(input, u + offset1*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w4; sumG += g*w4; sumB += b*w4; sumA += a*w4;
    sampleBilinearZeroPad(input, u + offset2*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w2; sumG += g*w2; sumB += b*w2; sumA += a*w2;

    // Row +1
    sampleBilinearZeroPad(input, u - offset2*texelX, v + offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinearZeroPad(input, u - offset1*texelX, v + offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w3; sumG += g*w3; sumB += b*w3; sumA += a*w3;
    sampleBilinearZeroPad(input, u, v + offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w4; sumG += g*w4; sumB += b*w4; sumA += a*w4;
    sampleBilinearZeroPad(input, u + offset1*texelX, v + offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w3; sumG += g*w3; sumB += b*w3; sumA += a*w3;
    sampleBilinearZeroPad(input, u + offset2*texelX, v + offset1*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;

    // Row +2
    sampleBilinearZeroPad(input, u - offset2*texelX, v + offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w0; sumG += g*w0; sumB += b*w0; sumA += a*w0;
    sampleBilinearZeroPad(input, u - offset1*texelX, v + offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinearZeroPad(input, u, v + offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w2; sumG += g*w2; sumB += b*w2; sumA += a*w2;
    sampleBilinearZeroPad(input, u + offset1*texelX, v + offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinearZeroPad(input, u + offset2*texelX, v + offset2*texelY, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w0; sumG += g*w0; sumB += b*w0; sumA += a*w0;

    // Clamp
    sumR = fminf(sumR, 1.0f);
    sumG = fminf(sumG, 1.0f);
    sumB = fminf(sumB, 1.0f);

    // Color space conversion (same as 13-tap)
    float resR, resG, resB;
    (void)useHDR;

    if (useLinear) {
        float straightR, straightG, straightB;
        if (sumA > 0.001f) {
            straightR = sumR / sumA;
            straightG = sumG / sumA;
            straightB = sumB / sumA;
        } else {
            straightR = straightG = straightB = 0.0f;
        }
        straightR = toLinear(straightR, inputProfile);
        straightG = toLinear(straightG, inputProfile);
        straightB = toLinear(straightB, inputProfile);
        resR = straightR * sumA;
        resG = straightG * sumA;
        resB = straightB * sumA;
    } else {
        resR = sumR;
        resG = sumG;
        resB = sumB;
    }

    softThreshold(resR, resG, resB, threshold, softKnee);

    // Write output - alpha=1 for glow buffer (padding area needs full opacity for blending)
    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = 1.0f;
}

// ============================================================================
// Prefilter Separable 5-tap Horizontal
// First pass of separable Gaussian blur
// ============================================================================

extern "C" __global__ void PrefilterSep5HKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    int inputWidth, int inputHeight,
    float offsetPrefilter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    int offsetX = (dstWidth - inputWidth) / 2;
    int offsetY = (dstHeight - inputHeight) / 2;

    float srcX = (float)x - (float)offsetX;
    float srcY = (float)y - (float)offsetY;

    float u = (srcX + 0.5f) / (float)inputWidth;
    float v = (srcY + 0.5f) / (float)inputHeight;

    float texelX = 1.0f / (float)inputWidth;

    // 5-tap Gaussian: [1, 4, 6, 4, 1] / 16
    const float w0 = 1.0f / 16.0f;
    const float w1 = 4.0f / 16.0f;
    const float w2 = 6.0f / 16.0f;

    float offset1 = 1.0f * offsetPrefilter;
    float offset2 = 2.0f * offsetPrefilter;

    float r0, g0, b0, a0, r1, g1, b1, a1, r2, g2, b2, a2, r3, g3, b3, a3, r4, g4, b4, a4;

    sampleBilinearZeroPad(input, u - offset2*texelX, v, inputWidth, inputHeight, srcPitch, r0, g0, b0, a0);
    sampleBilinearZeroPad(input, u - offset1*texelX, v, inputWidth, inputHeight, srcPitch, r1, g1, b1, a1);
    sampleBilinearZeroPad(input, u, v, inputWidth, inputHeight, srcPitch, r2, g2, b2, a2);
    sampleBilinearZeroPad(input, u + offset1*texelX, v, inputWidth, inputHeight, srcPitch, r3, g3, b3, a3);
    sampleBilinearZeroPad(input, u + offset2*texelX, v, inputWidth, inputHeight, srcPitch, r4, g4, b4, a4);

    float sumR = r0*w0 + r1*w1 + r2*w2 + r3*w1 + r4*w0;
    float sumG = g0*w0 + g1*w1 + g2*w2 + g3*w1 + g4*w0;
    float sumB = b0*w0 + b1*w1 + b2*w2 + b3*w1 + b4*w0;
    float sumA = a0*w0 + a1*w1 + a2*w2 + a3*w1 + a4*w0;

    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = sumR;
    output[outIdx + 1] = sumG;
    output[outIdx + 2] = sumB;
    output[outIdx + 3] = sumA;
}

// ============================================================================
// Prefilter Separable 5-tap Vertical + Threshold
// Second pass: vertical blur + threshold + color processing
// ============================================================================

extern "C" __global__ void PrefilterSep5VKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float threshold, float softKnee,
    float colorR, float colorG, float colorB,
    float colorTempR, float colorTempG, float colorTempB,
    float preserveColor, int useHDR, int useLinear, int inputProfile,
    float offsetPrefilter)
    // Note: Desaturation now applied via separate DesaturationKernel after Prefilter
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float u = ((float)x + 0.5f) / (float)srcWidth;
    float v = ((float)y + 0.5f) / (float)srcHeight;

    float texelY = 1.0f / (float)srcHeight;

    // 5-tap Gaussian: [1, 4, 6, 4, 1] / 16
    const float w0 = 1.0f / 16.0f;
    const float w1 = 4.0f / 16.0f;
    const float w2 = 6.0f / 16.0f;

    float offset1 = 1.0f * offsetPrefilter;
    float offset2 = 2.0f * offsetPrefilter;

    float r0, g0, b0, a0, r1, g1, b1, a1, r2, g2, b2, a2, r3, g3, b3, a3, r4, g4, b4, a4;

    sampleBilinear(input, u, v - offset2*texelY, srcWidth, srcHeight, srcPitch, r0, g0, b0, a0);
    sampleBilinear(input, u, v - offset1*texelY, srcWidth, srcHeight, srcPitch, r1, g1, b1, a1);
    sampleBilinear(input, u, v, srcWidth, srcHeight, srcPitch, r2, g2, b2, a2);
    sampleBilinear(input, u, v + offset1*texelY, srcWidth, srcHeight, srcPitch, r3, g3, b3, a3);
    sampleBilinear(input, u, v + offset2*texelY, srcWidth, srcHeight, srcPitch, r4, g4, b4, a4);

    float sumR = r0*w0 + r1*w1 + r2*w2 + r3*w1 + r4*w0;
    float sumG = g0*w0 + g1*w1 + g2*w2 + g3*w1 + g4*w0;
    float sumB = b0*w0 + b1*w1 + b2*w2 + b3*w1 + b4*w0;
    float sumA = a0*w0 + a1*w1 + a2*w2 + a3*w1 + a4*w0;

    sumR = fminf(sumR, 1.0f);
    sumG = fminf(sumG, 1.0f);
    sumB = fminf(sumB, 1.0f);

    float resR, resG, resB;
    (void)useHDR;

    if (useLinear) {
        float straightR, straightG, straightB;
        if (sumA > 0.001f) {
            straightR = sumR / sumA;
            straightG = sumG / sumA;
            straightB = sumB / sumA;
        } else {
            straightR = straightG = straightB = 0.0f;
        }
        straightR = toLinear(straightR, inputProfile);
        straightG = toLinear(straightG, inputProfile);
        straightB = toLinear(straightB, inputProfile);
        resR = straightR * sumA;
        resG = straightG * sumA;
        resB = straightB * sumA;
    } else {
        resR = sumR;
        resG = sumG;
        resB = sumB;
    }

    softThreshold(resR, resG, resB, threshold, softKnee);

    // Write output - alpha=1 for glow buffer (padding area needs full opacity for blending)
    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = 1.0f;
}

// ============================================================================
// Prefilter Separable 9-tap Horizontal
// Higher quality separable blur
// ============================================================================

extern "C" __global__ void PrefilterSep9HKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    int inputWidth, int inputHeight,
    float offsetPrefilter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    int offsetX = (dstWidth - inputWidth) / 2;
    int offsetY = (dstHeight - inputHeight) / 2;

    float srcX = (float)x - (float)offsetX;
    float srcY = (float)y - (float)offsetY;

    float u = (srcX + 0.5f) / (float)inputWidth;
    float v = (srcY + 0.5f) / (float)inputHeight;

    float texelX = 1.0f / (float)inputWidth;

    // 9-tap Gaussian: [1, 8, 28, 56, 70, 56, 28, 8, 1] / 256
    const float w0 = 1.0f / 256.0f;
    const float w1 = 8.0f / 256.0f;
    const float w2 = 28.0f / 256.0f;
    const float w3 = 56.0f / 256.0f;
    const float w4 = 70.0f / 256.0f;

    float off1 = 1.0f * offsetPrefilter;
    float off2 = 2.0f * offsetPrefilter;
    float off3 = 3.0f * offsetPrefilter;
    float off4 = 4.0f * offsetPrefilter;

    float sumR = 0, sumG = 0, sumB = 0, sumA = 0;
    float r, g, b, a;

    sampleBilinearZeroPad(input, u - off4*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w0; sumG += g*w0; sumB += b*w0; sumA += a*w0;
    sampleBilinearZeroPad(input, u - off3*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinearZeroPad(input, u - off2*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w2; sumG += g*w2; sumB += b*w2; sumA += a*w2;
    sampleBilinearZeroPad(input, u - off1*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w3; sumG += g*w3; sumB += b*w3; sumA += a*w3;
    sampleBilinearZeroPad(input, u, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w4; sumG += g*w4; sumB += b*w4; sumA += a*w4;
    sampleBilinearZeroPad(input, u + off1*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w3; sumG += g*w3; sumB += b*w3; sumA += a*w3;
    sampleBilinearZeroPad(input, u + off2*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w2; sumG += g*w2; sumB += b*w2; sumA += a*w2;
    sampleBilinearZeroPad(input, u + off3*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinearZeroPad(input, u + off4*texelX, v, inputWidth, inputHeight, srcPitch, r, g, b, a);
    sumR += r*w0; sumG += g*w0; sumB += b*w0; sumA += a*w0;

    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = sumR;
    output[outIdx + 1] = sumG;
    output[outIdx + 2] = sumB;
    output[outIdx + 3] = sumA;
}

// ============================================================================
// Prefilter Separable 9-tap Vertical + Threshold
// ============================================================================

extern "C" __global__ void PrefilterSep9VKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float threshold, float softKnee,
    float colorR, float colorG, float colorB,
    float colorTempR, float colorTempG, float colorTempB,
    float preserveColor, int useHDR, int useLinear, int inputProfile,
    float offsetPrefilter)
    // Note: Desaturation now applied via separate DesaturationKernel after Prefilter
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float u = ((float)x + 0.5f) / (float)srcWidth;
    float v = ((float)y + 0.5f) / (float)srcHeight;

    float texelY = 1.0f / (float)srcHeight;

    // 9-tap Gaussian: [1, 8, 28, 56, 70, 56, 28, 8, 1] / 256
    const float w0 = 1.0f / 256.0f;
    const float w1 = 8.0f / 256.0f;
    const float w2 = 28.0f / 256.0f;
    const float w3 = 56.0f / 256.0f;
    const float w4 = 70.0f / 256.0f;

    float off1 = 1.0f * offsetPrefilter;
    float off2 = 2.0f * offsetPrefilter;
    float off3 = 3.0f * offsetPrefilter;
    float off4 = 4.0f * offsetPrefilter;

    float sumR = 0, sumG = 0, sumB = 0, sumA = 0;
    float r, g, b, a;

    sampleBilinear(input, u, v - off4*texelY, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w0; sumG += g*w0; sumB += b*w0; sumA += a*w0;
    sampleBilinear(input, u, v - off3*texelY, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinear(input, u, v - off2*texelY, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w2; sumG += g*w2; sumB += b*w2; sumA += a*w2;
    sampleBilinear(input, u, v - off1*texelY, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w3; sumG += g*w3; sumB += b*w3; sumA += a*w3;
    sampleBilinear(input, u, v, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w4; sumG += g*w4; sumB += b*w4; sumA += a*w4;
    sampleBilinear(input, u, v + off1*texelY, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w3; sumG += g*w3; sumB += b*w3; sumA += a*w3;
    sampleBilinear(input, u, v + off2*texelY, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w2; sumG += g*w2; sumB += b*w2; sumA += a*w2;
    sampleBilinear(input, u, v + off3*texelY, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w1; sumG += g*w1; sumB += b*w1; sumA += a*w1;
    sampleBilinear(input, u, v + off4*texelY, srcWidth, srcHeight, srcPitch, r, g, b, a);
    sumR += r*w0; sumG += g*w0; sumB += b*w0; sumA += a*w0;

    sumR = fminf(sumR, 1.0f);
    sumG = fminf(sumG, 1.0f);
    sumB = fminf(sumB, 1.0f);

    float resR, resG, resB;
    (void)useHDR;

    if (useLinear) {
        float straightR, straightG, straightB;
        if (sumA > 0.001f) {
            straightR = sumR / sumA;
            straightG = sumG / sumA;
            straightB = sumB / sumA;
        } else {
            straightR = straightG = straightB = 0.0f;
        }
        straightR = toLinear(straightR, inputProfile);
        straightG = toLinear(straightG, inputProfile);
        straightB = toLinear(straightB, inputProfile);
        resR = straightR * sumA;
        resG = straightG * sumA;
        resB = straightB * sumA;
    } else {
        resR = sumR;
        resG = sumG;
        resB = sumB;
    }

    softThreshold(resR, resG, resB, threshold, softKnee);

    // Write output - alpha=1 for glow buffer (padding area needs full opacity for blending)
    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = 1.0f;
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
    int dstWidth, int dstHeight, int dstPitch,
    float offsetDown, float spreadDown, int level, int maxLevels,
    float paddingThreshold)  // Clip dark values for padding optimization
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

    // Dynamic offset: offsetDown at level 0, offsetDown + spreadDown at max level
    // offsetDown: base offset (default 1.0), spreadDown: 0-10
    float levelRatio = (float)level / fmaxf((float)(maxLevels - 1), 1.0f);
    float offset = offsetDown + spreadDown * levelRatio;

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

    // Sample all 9 points with ZeroPad, using dynamic offset
    sampleBilinearZeroPad(input, u - offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, TLr, TLg, TLb, TLa);
    sampleBilinearZeroPad(input, u,                   v - offset * texelY, srcWidth, srcHeight, srcPitch, Tr, Tg, Tb, Ta);
    sampleBilinearZeroPad(input, u + offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, TRr, TRg, TRb, TRa);

    sampleBilinearZeroPad(input, u - offset * texelX, v,                   srcWidth, srcHeight, srcPitch, Lr, Lg, Lb, La);
    sampleBilinearZeroPad(input, u,                   v,                   srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);
    sampleBilinearZeroPad(input, u + offset * texelX, v,                   srcWidth, srcHeight, srcPitch, Rr, Rg, Rb, Ra);

    sampleBilinearZeroPad(input, u - offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, BLr, BLg, BLb, BLa);
    sampleBilinearZeroPad(input, u,                   v + offset * texelY, srcWidth, srcHeight, srcPitch, Br, Bg, Bb, Ba);
    sampleBilinearZeroPad(input, u + offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, BRr, BRg, BRb, BRa);

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

    // Padding threshold clipping: clip very dark values to 0 for padding optimization
    // This allows smaller padding sizes by making dark edges fade to true black
    if (paddingThreshold > 0.0f && (outR + outG + outB) < paddingThreshold) {
        outR = outG = outB = outA = 0.0f;
    }

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
    float resA = centerA * 0.5f + (Aa + Ba + Ca + Da) * 0.125f;

    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = resA;
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
    float offsetUp,    // base offset (default 1.0)
    float spreadUp,    // 0-10: added to base offset at max level
    int levelIndex,
    float activeLimit,  // now 0-1 (radius / 100)
    float decayK,
    float level1Weight,
    int falloffType,
    int maxLevels,      // total MIP levels
    int blurMode,       // ignored, always use 9-tap Discrete Gaussian
    int compositeMode)  // 1=Add, 2=Screen, 3=Overlay
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float u = ((float)x + 0.5f) / (float)dstWidth;
    float v = ((float)y + 0.5f) / (float)dstHeight;

    float resR = 0.0f, resG = 0.0f, resB = 0.0f, resA = 0.0f;

    // =========================================================
    // STEP 1: Upsample from Previous Level (smaller texture)
    // blurMode=1: 9-Tap 3x3 Gaussian (fast)
    // blurMode=2: 25-Tap 5x5 Gaussian (better diagonal coverage)
    // =========================================================
    if (prevLevel != nullptr) {
        float texelX = 1.0f / (float)prevWidth;
        float texelY = 1.0f / (float)prevHeight;

        // Dynamic offset: offsetUp at level 0, offsetUp + spreadUp at max level
        float levelRatio = (float)levelIndex / fmaxf((float)(maxLevels - 1), 1.0f);
        float offset = offsetUp + spreadUp * levelRatio;

        if (blurMode == 2) {
            // =========================================
            // 25-Tap 5x5 Gaussian (blurMode == 2)
            // Weights from Pascal's triangle: 1 4 6 4 1
            // Total = 256, better diagonal coverage
            // =========================================
            const float w0 = 36.0f / 256.0f;   // center (0,0)
            const float w1 = 24.0f / 256.0f;   // cross ±1 (0,±1), (±1,0)
            const float w2 = 16.0f / 256.0f;   // diagonal ±1 (±1,±1)
            const float w3 = 6.0f / 256.0f;    // cross ±2 (0,±2), (±2,0)
            const float w4 = 4.0f / 256.0f;    // edge corners (±1,±2), (±2,±1)
            const float w5 = 1.0f / 256.0f;    // corners (±2,±2)

            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f, sumA = 0.0f;
            float r, g, b, a;

            // Center (0,0) - weight 36
            sampleBilinearZeroPad(prevLevel, u, v, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w0; sumG += g * w0; sumB += b * w0; sumA += a * w0;

            // Cross ±1 (4 samples) - weight 24 each
            sampleBilinearZeroPad(prevLevel, u, v - offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w1; sumG += g * w1; sumB += b * w1; sumA += a * w1;
            sampleBilinearZeroPad(prevLevel, u, v + offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w1; sumG += g * w1; sumB += b * w1; sumA += a * w1;
            sampleBilinearZeroPad(prevLevel, u - offset * texelX, v, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w1; sumG += g * w1; sumB += b * w1; sumA += a * w1;
            sampleBilinearZeroPad(prevLevel, u + offset * texelX, v, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w1; sumG += g * w1; sumB += b * w1; sumA += a * w1;

            // Diagonal ±1 (4 samples) - weight 16 each
            sampleBilinearZeroPad(prevLevel, u - offset * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w2; sumG += g * w2; sumB += b * w2; sumA += a * w2;
            sampleBilinearZeroPad(prevLevel, u + offset * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w2; sumG += g * w2; sumB += b * w2; sumA += a * w2;
            sampleBilinearZeroPad(prevLevel, u - offset * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w2; sumG += g * w2; sumB += b * w2; sumA += a * w2;
            sampleBilinearZeroPad(prevLevel, u + offset * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w2; sumG += g * w2; sumB += b * w2; sumA += a * w2;

            // Cross ±2 (4 samples) - weight 6 each
            float offset2 = offset * 2.0f;
            sampleBilinearZeroPad(prevLevel, u, v - offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w3; sumG += g * w3; sumB += b * w3; sumA += a * w3;
            sampleBilinearZeroPad(prevLevel, u, v + offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w3; sumG += g * w3; sumB += b * w3; sumA += a * w3;
            sampleBilinearZeroPad(prevLevel, u - offset2 * texelX, v, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w3; sumG += g * w3; sumB += b * w3; sumA += a * w3;
            sampleBilinearZeroPad(prevLevel, u + offset2 * texelX, v, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w3; sumG += g * w3; sumB += b * w3; sumA += a * w3;

            // Edge corners (8 samples) - weight 4 each: (±1,±2) and (±2,±1)
            sampleBilinearZeroPad(prevLevel, u - offset * texelX, v - offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w4; sumG += g * w4; sumB += b * w4; sumA += a * w4;
            sampleBilinearZeroPad(prevLevel, u + offset * texelX, v - offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w4; sumG += g * w4; sumB += b * w4; sumA += a * w4;
            sampleBilinearZeroPad(prevLevel, u - offset * texelX, v + offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w4; sumG += g * w4; sumB += b * w4; sumA += a * w4;
            sampleBilinearZeroPad(prevLevel, u + offset * texelX, v + offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w4; sumG += g * w4; sumB += b * w4; sumA += a * w4;
            sampleBilinearZeroPad(prevLevel, u - offset2 * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w4; sumG += g * w4; sumB += b * w4; sumA += a * w4;
            sampleBilinearZeroPad(prevLevel, u + offset2 * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w4; sumG += g * w4; sumB += b * w4; sumA += a * w4;
            sampleBilinearZeroPad(prevLevel, u - offset2 * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w4; sumG += g * w4; sumB += b * w4; sumA += a * w4;
            sampleBilinearZeroPad(prevLevel, u + offset2 * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w4; sumG += g * w4; sumB += b * w4; sumA += a * w4;

            // Corners (4 samples) - weight 1 each: (±2,±2)
            sampleBilinearZeroPad(prevLevel, u - offset2 * texelX, v - offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w5; sumG += g * w5; sumB += b * w5; sumA += a * w5;
            sampleBilinearZeroPad(prevLevel, u + offset2 * texelX, v - offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w5; sumG += g * w5; sumB += b * w5; sumA += a * w5;
            sampleBilinearZeroPad(prevLevel, u - offset2 * texelX, v + offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w5; sumG += g * w5; sumB += b * w5; sumA += a * w5;
            sampleBilinearZeroPad(prevLevel, u + offset2 * texelX, v + offset2 * texelY, prevWidth, prevHeight, prevPitch, r, g, b, a);
            sumR += r * w5; sumG += g * w5; sumB += b * w5; sumA += a * w5;

            resR = sumR;
            resG = sumG;
            resB = sumB;
            resA = sumA;
        } else {
            // =========================================
            // 9-Tap 3x3 Gaussian (blurMode == 1, default)
            // Weights: Center=4/16, Cross=2/16, Diagonal=1/16
            // =========================================
            float TLr, TLg, TLb, TLa;
            float Tr, Tg, Tb, Ta;
            float TRr, TRg, TRb, TRa;
            float Lr, Lg, Lb, La;
            float Cr, Cg, Cb, Ca;
            float Rr, Rg, Rb, Ra;
            float BLr, BLg, BLb, BLa;
            float Bor, Bog, Bob, Boa;
            float BRr, BRg, BRb, BRa;

            sampleBilinearZeroPad(prevLevel, u - offset * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, TLr, TLg, TLb, TLa);
            sampleBilinearZeroPad(prevLevel, u, v - offset * texelY, prevWidth, prevHeight, prevPitch, Tr, Tg, Tb, Ta);
            sampleBilinearZeroPad(prevLevel, u + offset * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, TRr, TRg, TRb, TRa);
            sampleBilinearZeroPad(prevLevel, u - offset * texelX, v, prevWidth, prevHeight, prevPitch, Lr, Lg, Lb, La);
            sampleBilinearZeroPad(prevLevel, u, v, prevWidth, prevHeight, prevPitch, Cr, Cg, Cb, Ca);
            sampleBilinearZeroPad(prevLevel, u + offset * texelX, v, prevWidth, prevHeight, prevPitch, Rr, Rg, Rb, Ra);
            sampleBilinearZeroPad(prevLevel, u - offset * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, BLr, BLg, BLb, BLa);
            sampleBilinearZeroPad(prevLevel, u, v + offset * texelY, prevWidth, prevHeight, prevPitch, Bor, Bog, Bob, Boa);
            sampleBilinearZeroPad(prevLevel, u + offset * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, BRr, BRg, BRb, BRa);

            const float wCenter = 0.25f;     // 4/16
            const float wCross = 0.125f;     // 2/16
            const float wDiagonal = 0.0625f; // 1/16

            resR = Cr * wCenter + (Tr + Lr + Rr + Bor) * wCross + (TLr + TRr + BLr + BRr) * wDiagonal;
            resG = Cg * wCenter + (Tg + Lg + Rg + Bog) * wCross + (TLg + TRg + BLg + BRg) * wDiagonal;
            resB = Cb * wCenter + (Tb + Lb + Rb + Bob) * wCross + (TLb + TRb + BLb + BRb) * wDiagonal;
            resA = Ca * wCenter + (Ta + La + Ra + Boa) * wCross + (TLa + TRa + BLa + BRa) * wDiagonal;
        }
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
    float contribA = currA * physicalWeight;

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
        contribA *= contribution;
    }

    // Blend current level contribution with upsampled base
    // RGB = based on compositeMode (1=Add, 2=Screen, 3=Overlay)
    // Alpha = OVER formula (standard alpha compositing)
    switch (compositeMode) {
        case 2: { // Screen: A + B - A*B
            resR = resR + contribR - resR * contribR;
            resG = resG + contribG - resG * contribG;
            resB = resB + contribB - resB * contribB;
            break;
        }
        case 3: { // Overlay: 2*A*B if A<0.5, else 1-2*(1-A)*(1-B)
            resR = (resR < 0.5f) ? 2.0f * resR * contribR : resR + 2.0f * contribR * (1.0f - resR);
            resG = (resG < 0.5f) ? 2.0f * resG * contribG : resG + 2.0f * contribG * (1.0f - resG);
            resB = (resB < 0.5f) ? 2.0f * resB * contribB : resB + 2.0f * contribB * (1.0f - resB);
            break;
        }
        default: { // Add (case 1 and fallback)
            resR = resR + contribR;
            resG = resG + contribG;
            resB = resB + contribB;
            break;
        }
    }
    resA = resA + contribA * (1.0f - resA);  // OVER: prevents alpha > 1.0

    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = resA;
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
    float dither,           // Dithering amount (0-1)
    float chromaticAberration,  // CA amount (0-100)
    float caTintRr, float caTintRg, float caTintRb,  // CA Red channel tint
    float caTintBr, float caTintBg, float caTintBb)  // CA Blue channel tint
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

    // DEBUG: Show debugMode value indicator in top-left corner (32x32 pixels)
    // Color encodes mode: R=mode/16, G=(mode!=1)?1:0, B=(mode>8)?1:0
    if (x < 32 && y < 32) {
        int idx = (y * outputPitch + x) * 4;
        output[idx + 0] = (float)debugMode / 16.0f;  // R = mode/16
        output[idx + 1] = (debugMode != 1) ? 1.0f : 0.0f;  // G = not Final
        output[idx + 2] = (debugMode > 8) ? 1.0f : 0.0f;   // B = Up modes
        output[idx + 3] = 1.0f;
        return;
    }

    // debugMode: 1=Final, 2=Prefilter, 3-8=Down1-6, 9-15=Up0-6, 16=GlowOnly
    if (debugMode == 1) {
        // Final: normal composite with opacity controls
        // Color space: Premultiplied Linear (if useLinear) or Premultiplied sRGB (if not)
        float glowR, glowG, glowB, glowA;

        if (chromaticAberration > 0.001f) {
            // Chromatic aberration: sample R/G/B at different UV offsets
            // R shifts outward (toward edge), B shifts inward (toward center)
            float caAmount = chromaticAberration * 0.002f;  // Scale factor
            float dirX = u - 0.5f;
            float dirY = v - 0.5f;

            // R channel: sample from outer position
            float uR = u + dirX * caAmount;
            float vR = v + dirY * caAmount;
            // B channel: sample from inner position
            float uB = u - dirX * caAmount;
            float vB = v - dirY * caAmount;

            // Sample each channel at its offset position
            float rSampleR, rSampleG, rSampleB, rSampleA;
            float gSampleR, gSampleG, gSampleB, gSampleA;
            float bSampleR, bSampleG, bSampleB, bSampleA;

            sampleBilinear(glow, uR, vR, glowWidth, glowHeight, glowPitch, rSampleR, rSampleG, rSampleB, rSampleA);
            sampleBilinear(glow, u,  v,  glowWidth, glowHeight, glowPitch, gSampleR, gSampleG, gSampleB, gSampleA);
            sampleBilinear(glow, uB, vB, glowWidth, glowHeight, glowPitch, bSampleR, bSampleG, bSampleB, bSampleA);

            // Take R from outer sample, G from center, B from inner sample
            // Then apply CA tint colors
            float rChannel = rSampleR;
            float gChannel = gSampleG;
            float bChannel = bSampleB;

            // Apply tint: blend R channel with caTintR color, B channel with caTintB color
            glowR = rChannel * caTintRr + bChannel * caTintBr;
            glowG = rChannel * caTintRg + gChannel + bChannel * caTintBg;
            glowB = rChannel * caTintRb + bChannel * caTintBb;
            glowA = gSampleA;  // Use center alpha
        } else {
            // No chromatic aberration - standard sampling
            sampleBilinear(glow, u, v, glowWidth, glowHeight, glowPitch, glowR, glowG, glowB, glowA);
        }

        // Note: Glow color/tint is already applied in Prefilter stage
        // (via glowColor with preserveColor blending)
        // Do NOT apply again here - causes blue channel to zero out completely

        // Glow stays in premultiplied space (AE native format)
        // Do NOT unpremultiply - causes artifacts at transparent edges

        // Apply exposure and glow opacity
        glowR *= exposure * glowOpacity;
        glowG *= exposure * glowOpacity;
        glowB *= exposure * glowOpacity;

        // Note: Max-based desaturation is applied in Prefilter stage

        // Apply source opacity (premultiplied)
        float srcR = origR * sourceOpacity;
        float srcG = origG * sourceOpacity;
        float srcB = origB * sourceOpacity;
        float srcA = origA * sourceOpacity;

        // 1. Estimate glow alpha from matted-with-black RGB
        float glowAlpha = fminf(fmaxf(fmaxf(glowR, glowG), glowB), 1.0f);

        // 2. Calculate result alpha: max of src and glow
        resA = fmaxf(srcA, glowAlpha);

        // 3. Blend assuming alpha=1 (both are matted with black)
        float tempR, tempG, tempB;
        switch (compositeMode) {
            case 1: { // Add
                tempR = srcR + glowR;
                tempG = srcG + glowG;
                tempB = srcB + glowB;
                break;
            }

            case 2: { // Screen: A + B - A*B
                tempR = srcR + glowR - srcR * glowR;
                tempG = srcG + glowG - srcG * glowG;
                tempB = srcB + glowB - srcB * glowB;
                break;
            }

            case 3: { // Overlay
                float straightSrcR = (srcA > 0.001f) ? srcR / srcA : 0.0f;
                float straightSrcG = (srcA > 0.001f) ? srcG / srcA : 0.0f;
                float straightSrcB = (srcA > 0.001f) ? srcB / srcA : 0.0f;

                tempR = (straightSrcR < 0.5f)
                    ? 2.0f * srcR * glowR
                    : srcR + 2.0f * glowR * (1.0f - srcR);
                tempG = (straightSrcG < 0.5f)
                    ? 2.0f * srcG * glowG
                    : srcG + 2.0f * glowG * (1.0f - srcG);
                tempB = (straightSrcB < 0.5f)
                    ? 2.0f * srcB * glowB
                    : srcB + 2.0f * glowB * (1.0f - srcB);
                break;
            }

            default: { // Fallback to Add
                tempR = srcR + glowR;
                tempG = srcG + glowG;
                tempB = srcB + glowB;
                break;
            }
        }

        // 4. Unmult: divide by resA to get normalized premultiplied RGB
        resR = (resA > 0.001f) ? tempR / resA : 0.0f;
        resG = (resA > 0.001f) ? tempG / resA : 0.0f;
        resB = (resA > 0.001f) ? tempB / resA : 0.0f;
    }
    else if (debugMode == 16) {
        // GlowOnly: just glow with exposure and opacity
        // Color space: Premultiplied Linear (if useLinear) or Premultiplied sRGB (if not)
        float glowR, glowG, glowB, glowA;

        if (chromaticAberration > 0.001f) {
            // Chromatic aberration (same as Final mode)
            float caAmount = chromaticAberration * 0.002f;
            float dirX = u - 0.5f;
            float dirY = v - 0.5f;

            float uR = u + dirX * caAmount;
            float vR = v + dirY * caAmount;
            float uB = u - dirX * caAmount;
            float vB = v - dirY * caAmount;

            float rSampleR, rSampleG, rSampleB, rSampleA;
            float gSampleR, gSampleG, gSampleB, gSampleA;
            float bSampleR, bSampleG, bSampleB, bSampleA;

            sampleBilinear(glow, uR, vR, glowWidth, glowHeight, glowPitch, rSampleR, rSampleG, rSampleB, rSampleA);
            sampleBilinear(glow, u,  v,  glowWidth, glowHeight, glowPitch, gSampleR, gSampleG, gSampleB, gSampleA);
            sampleBilinear(glow, uB, vB, glowWidth, glowHeight, glowPitch, bSampleR, bSampleG, bSampleB, bSampleA);

            float rChannel = rSampleR;
            float gChannel = gSampleG;
            float bChannel = bSampleB;

            glowR = rChannel * caTintRr + bChannel * caTintBr;
            glowG = rChannel * caTintRg + gChannel + bChannel * caTintBg;
            glowB = rChannel * caTintRb + bChannel * caTintBb;
            glowA = gSampleA;
        } else {
            sampleBilinear(glow, u, v, glowWidth, glowHeight, glowPitch, glowR, glowG, glowB, glowA);
        }

        // Note: Glow color applied in Prefilter, desaturation removed

        // Apply exposure and opacity
        resR = glowR * exposure * glowOpacity;
        resG = glowG * exposure * glowOpacity;
        resB = glowB * exposure * glowOpacity;

        // Alpha for GlowOnly: based on glow brightness
        float glowBrightness = fmaxf(fmaxf(resR, resG), resB);
        resA = fminf(glowBrightness, 1.0f);
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
            // Note: Internal buffers may have alpha=0 (paddingThreshold optimization)
            // For debug visualization, always use alpha=1 to see RGB values
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
