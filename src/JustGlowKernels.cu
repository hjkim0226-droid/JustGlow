/**
 * JustGlow CUDA Kernels
 *
 * GPU compute kernels for the glow effect pipeline.
 * Uses Driver API compatible signatures for PTX loading.
 *
 * Kernels:
 * - PrefilterKernel: 13-tap downsample + soft threshold + Karis average
 * - DownsampleKernel: Dual Kawase 4-tap
 * - UpsampleKernel: 9-tap tent filter with progressive blend
 * - CompositeKernel: Final blend with original
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

__device__ __forceinline__ float karisWeight(float r, float g, float b) {
    return 1.0f / (1.0f + luminance(r, g, b));
}

// Smooth step function for fade transitions
__device__ __forceinline__ float smoothstepf(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Calculate weight based on level and intensity
// Level 0: always 100%
// Level 1: level1Weight (controlled by Intensity parameter, 50%-100%)
// Level 2+: level1Weight * pow(decayRate, level-1)
// decayRate is derived from Falloff parameter
__device__ __forceinline__ float calculatePhysicalWeight(float level, float decayK, int falloffType, float level1Weight) {
    // Level 0 is always 100%
    if (level < 0.5f) return 1.0f;

    // Decay rate per level (from Falloff)
    // decayK range 0.2-3.0 -> decayRate 0.95-0.5
    float decayRate = 1.0f - (decayK - 0.2f) / 2.8f * 0.5f;  // 0.5 ~ 1.0

    // Level 1 starts at level1Weight, then decays
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
// Soft Threshold (Fixed: threshold 아래 픽셀은 무조건 0)
// Effective range: [threshold, threshold + 2*knee]
// - Below threshold: 0
// - threshold ~ threshold+2*knee: quadratic curve
// - Above threshold+2*knee: linear
// ============================================================================

__device__ void softThreshold(
    float& r, float& g, float& b,
    float threshold, float knee)
{
    float brightness = fmaxf(fmaxf(r, g), b);
    float contribution;

    // 1. Knee가 0일 때: hard threshold
    if (knee <= 0.001f) {
        contribution = fmaxf(0.0f, brightness - threshold);
        contribution /= fmaxf(brightness, EPSILON);
        contribution = fmaxf(contribution, 0.0f);
        r *= contribution;
        g *= contribution;
        b *= contribution;
        return;
    }

    // 2. Threshold 오프셋 (선형 구간 시작점)
    float curveThreshold = threshold + knee;

    // 3. Soft curve 계산
    // brightness < threshold: soft = 0 (clamp)
    // brightness in [threshold, threshold+2*knee]: quadratic
    float soft = brightness - threshold;
    soft = clampf(soft, 0.0f, 2.0f * knee);
    soft = (soft * soft) / (4.0f * knee);

    // 4. 곡선 vs 선형 중 큰 값 선택
    // brightness > threshold+2*knee 이면 선형이 더 큼
    contribution = fmaxf(soft, brightness - curveThreshold);

    // 5. 정규화
    contribution /= fmaxf(brightness, EPSILON);
    contribution = fmaxf(contribution, 0.0f);

    r *= contribution;
    g *= contribution;
    b *= contribution;
}

// ============================================================================
// Karis Average for 4 samples
// ============================================================================

__device__ void karisAverage4(
    float r0, float g0, float b0,
    float r1, float g1, float b1,
    float r2, float g2, float b2,
    float r3, float g3, float b3,
    float& outR, float& outG, float& outB)
{
    float w0 = karisWeight(r0, g0, b0);
    float w1 = karisWeight(r1, g1, b1);
    float w2 = karisWeight(r2, g2, b2);
    float w3 = karisWeight(r3, g3, b3);

    float totalWeight = w0 + w1 + w2 + w3;

    outR = (r0 * w0 + r1 * w1 + r2 * w2 + r3 * w3) / totalWeight;
    outG = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3) / totalWeight;
    outB = (b0 * w0 + b1 * w1 + b2 * w2 + b3 * w3) / totalWeight;
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
    float preserveColor, int useHDR)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    // Calculate offset for centering input within expanded output
    int offsetX = (dstWidth - inputWidth) / 2;
    int offsetY = (dstHeight - inputHeight) / 2;

    // Map output coordinates to input coordinates
    int srcX = x - offsetX;
    int srcY = y - offsetY;

    // Check if we're outside the input bounds - output black
    if (srcX < 0 || srcX >= inputWidth || srcY < 0 || srcY >= inputHeight) {
        int outIdx = (y * dstPitch + x) * 4;
        output[outIdx + 0] = 0.0f;
        output[outIdx + 1] = 0.0f;
        output[outIdx + 2] = 0.0f;
        output[outIdx + 3] = 0.0f;
        return;
    }

    // UV coordinates within the input region
    float u = ((float)srcX + 0.5f) / (float)inputWidth;
    float v = ((float)srcY + 0.5f) / (float)inputHeight;

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

    // Outer corners
    sampleBilinear(input, u - 2.0f * texelX, v - 2.0f * texelY, inputWidth, inputHeight, srcPitch, Ar, Ag, Ab, Aa);
    sampleBilinear(input, u + 2.0f * texelX, v - 2.0f * texelY, inputWidth, inputHeight, srcPitch, Cr, Cg, Cb, Ca);
    sampleBilinear(input, u - 2.0f * texelX, v + 2.0f * texelY, inputWidth, inputHeight, srcPitch, Kr, Kg, Kb, Ka);
    sampleBilinear(input, u + 2.0f * texelX, v + 2.0f * texelY, inputWidth, inputHeight, srcPitch, Mr, Mg, Mb, Ma);

    // Outer cross
    sampleBilinear(input, u, v - 2.0f * texelY, inputWidth, inputHeight, srcPitch, Br, Bg, Bb, Ba);
    sampleBilinear(input, u - 2.0f * texelX, v, inputWidth, inputHeight, srcPitch, Fr, Fg, Fb, Fa);
    sampleBilinear(input, u + 2.0f * texelX, v, inputWidth, inputHeight, srcPitch, Hr, Hg, Hb, Ha);
    sampleBilinear(input, u, v + 2.0f * texelY, inputWidth, inputHeight, srcPitch, Lr, Lg, Lb, La);

    // Inner corners
    sampleBilinear(input, u - texelX, v - texelY, inputWidth, inputHeight, srcPitch, Dr, Dg, Db, Da);
    sampleBilinear(input, u + texelX, v - texelY, inputWidth, inputHeight, srcPitch, Er, Eg, Eb, Ea);
    sampleBilinear(input, u - texelX, v + texelY, inputWidth, inputHeight, srcPitch, Ir, Ig, Ib, Ia);
    sampleBilinear(input, u + texelX, v + texelY, inputWidth, inputHeight, srcPitch, Jr, Jg, Jb, Ja);

    // Center
    sampleBilinear(input, u, v, inputWidth, inputHeight, srcPitch, Gr, Gg, Gb, Ga);

    float resR, resG, resB;

    if (useHDR) {
        // Karis average groups
        float g1r, g1g, g1b;
        float g2r, g2g, g2b;
        float g3r, g3g, g3b;
        float g4r, g4g, g4b;
        float g5r, g5g, g5b;

        karisAverage4(Dr, Dg, Db, Er, Eg, Eb, Ir, Ig, Ib, Jr, Jg, Jb, g1r, g1g, g1b);
        karisAverage4(Ar, Ag, Ab, Br, Bg, Bb, Fr, Fg, Fb, Gr, Gg, Gb, g2r, g2g, g2b);
        karisAverage4(Br, Bg, Bb, Cr, Cg, Cb, Gr, Gg, Gb, Hr, Hg, Hb, g3r, g3g, g3b);
        karisAverage4(Fr, Fg, Fb, Gr, Gg, Gb, Kr, Kg, Kb, Lr, Lg, Lb, g4r, g4g, g4b);
        karisAverage4(Gr, Gg, Gb, Hr, Hg, Hb, Lr, Lg, Lb, Mr, Mg, Mb, g5r, g5g, g5b);

        resR = g1r * 0.5f + (g2r + g3r + g4r + g5r) * 0.125f;
        resG = g1g * 0.5f + (g2g + g3g + g4g + g5g) * 0.125f;
        resB = g1b * 0.5f + (g2b + g3b + g4b + g5b) * 0.125f;
    } else {
        resR = Gr * 0.125f + (Dr + Er + Ir + Jr) * 0.125f + (Br + Fr + Hr + Lr) * 0.0625f + (Ar + Cr + Kr + Mr) * 0.03125f;
        resG = Gg * 0.125f + (Dg + Eg + Ig + Jg) * 0.125f + (Bg + Fg + Hg + Lg) * 0.0625f + (Ag + Cg + Kg + Mg) * 0.03125f;
        resB = Gb * 0.125f + (Db + Eb + Ib + Jb) * 0.125f + (Bb + Fb + Hb + Lb) * 0.0625f + (Ab + Cb + Kb + Mb) * 0.03125f;
    }

    // Apply soft threshold
    softThreshold(resR, resG, resB, threshold, softKnee);

    // Write output
    int outIdx = (y * dstPitch + x) * 4;
    output[outIdx + 0] = resR;
    output[outIdx + 1] = resG;
    output[outIdx + 2] = resB;
    output[outIdx + 3] = 1.0f;
}

// ============================================================================
// Downsample Kernel (Dual Kawase 5-Tap with X/+ Rotation)
// rotationMode: 0 = X (diagonal), 1 = + (cross)
// Alternating patterns breaks up boxy artifacts -> rounder glow
// ============================================================================

extern "C" __global__ void DownsampleKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset, int rotationMode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float u = ((float)x + 0.5f) / (float)dstWidth;
    float v = ((float)y + 0.5f) / (float)dstHeight;

    float texelX = 1.0f / (float)srcWidth;
    float texelY = 1.0f / (float)srcHeight;
    float offset = blurOffset + 0.5f;

    float Ar, Ag, Ab, Aa;
    float Br, Bg, Bb, Ba;
    float Cr, Cg, Cb, Ca;
    float Dr, Dg, Db, Da;
    float centerR, centerG, centerB, centerA;

    if (rotationMode == 0) {
        // X pattern (diagonal) - default
        sampleBilinear(input, u - offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, Ar, Ag, Ab, Aa);
        sampleBilinear(input, u + offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, Br, Bg, Bb, Ba);
        sampleBilinear(input, u - offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);
        sampleBilinear(input, u + offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, Dr, Dg, Db, Da);
    } else {
        // + pattern (cross) - 45 degree rotation
        // Slightly larger offset (1.414x) to maintain similar coverage area
        float crossOffset = offset * 1.414f;
        sampleBilinear(input, u, v - crossOffset * texelY, srcWidth, srcHeight, srcPitch, Ar, Ag, Ab, Aa);  // Top
        sampleBilinear(input, u + crossOffset * texelX, v, srcWidth, srcHeight, srcPitch, Br, Bg, Bb, Ba);  // Right
        sampleBilinear(input, u, v + crossOffset * texelY, srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);  // Bottom
        sampleBilinear(input, u - crossOffset * texelX, v, srcWidth, srcHeight, srcPitch, Dr, Dg, Db, Da);  // Left
    }

    sampleBilinear(input, u, v, srcWidth, srcHeight, srcPitch, centerR, centerG, centerB, centerA);

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
// Horizontal Blur Kernel (5-Tap Gaussian)
// First pass of separable Gaussian blur
// Weights: [1, 4, 6, 4, 1] / 16
// ============================================================================

extern "C" __global__ void HorizontalBlurKernel(
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
    float offset = blurOffset + 0.5f;
    float offset2 = offset * 2.0f;

    // Sample 5 horizontal points
    float L2r, L2g, L2b, L2a;
    float L1r, L1g, L1b, L1a;
    float Cr, Cg, Cb, Ca;
    float R1r, R1g, R1b, R1a;
    float R2r, R2g, R2b, R2a;

    sampleBilinear(input, u - offset2 * texelX, v, width, height, pitch, L2r, L2g, L2b, L2a);
    sampleBilinear(input, u - offset * texelX, v, width, height, pitch, L1r, L1g, L1b, L1a);
    sampleBilinear(input, u, v, width, height, pitch, Cr, Cg, Cb, Ca);
    sampleBilinear(input, u + offset * texelX, v, width, height, pitch, R1r, R1g, R1b, R1a);
    sampleBilinear(input, u + offset2 * texelX, v, width, height, pitch, R2r, R2g, R2b, R2a);

    // Gaussian weights: [1, 4, 6, 4, 1] / 16
    float outR = (L2r * 1.0f + L1r * 4.0f + Cr * 6.0f + R1r * 4.0f + R2r * 1.0f) / 16.0f;
    float outG = (L2g * 1.0f + L1g * 4.0f + Cg * 6.0f + R1g * 4.0f + R2g * 1.0f) / 16.0f;
    float outB = (L2b * 1.0f + L1b * 4.0f + Cb * 6.0f + R1b * 4.0f + R2b * 1.0f) / 16.0f;
    float outA = (L2a * 1.0f + L1a * 4.0f + Ca * 6.0f + R1a * 4.0f + R2a * 1.0f) / 16.0f;

    int outIdx = (y * pitch + x) * 4;
    output[outIdx + 0] = outR;
    output[outIdx + 1] = outG;
    output[outIdx + 2] = outB;
    output[outIdx + 3] = outA;
}

// ============================================================================
// Upsample Kernel (Separable Gaussian / Tent Filter with Advanced Falloff)
// Parameters:
// - levelIndex: current MIP level (0 = core, higher = atmosphere)
// - activeLimit: Radius-controlled level limit (fade out beyond this)
// - decayK: Falloff decay constant (0.2-3.0, higher = steeper)
// - exposure: HDR exposure multiplier pow(2, intensity)
// - falloffType: 0=Exponential, 1=InverseSquare, 2=Linear
// - blurMode: 0=Tent filter (3x3), 1=Vertical Gaussian (5-tap, assumes H-blur done)
// ============================================================================

extern "C" __global__ void UpsampleKernel(
    const float* __restrict__ input,
    const float* __restrict__ prevLevel,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int prevWidth, int prevHeight, int prevPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset,
    int levelIndex,
    float activeLimit,
    float decayK,
    float level1Weight,
    int falloffType,
    int blurMode)  // 0=Tent (3x3), 1=Vertical Gaussian (5-tap)
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
    // blurMode 0: Tent filter (3x3, 9 samples)
    // blurMode 1: Vertical Gaussian (5-tap, assumes H-blur already done)
    // =========================================================
    if (prevLevel != nullptr) {
        float texelX = 1.0f / (float)prevWidth;
        float texelY = 1.0f / (float)prevHeight;
        float offset = blurOffset + 0.5f;

        if (blurMode == 1) {
            // =========================================
            // Vertical Gaussian (5-tap) - Second pass of separable blur
            // Input (prevLevel) is already horizontally blurred
            // Weights: [1, 4, 6, 4, 1] / 16
            // =========================================
            float T2r, T2g, T2b, T2a;  // Top 2
            float T1r, T1g, T1b, T1a;  // Top 1
            float Cr, Cg, Cb, Ca;       // Center
            float B1r, B1g, B1b, B1a;  // Bottom 1
            float B2r, B2g, B2b, B2a;  // Bottom 2

            float offset2 = offset * 2.0f;

            // Vertical samples only (horizontal blur already done in HorizontalBlurKernel)
            sampleBilinear(prevLevel, u, v - offset2 * texelY, prevWidth, prevHeight, prevPitch, T2r, T2g, T2b, T2a);
            sampleBilinear(prevLevel, u, v - offset * texelY, prevWidth, prevHeight, prevPitch, T1r, T1g, T1b, T1a);
            sampleBilinear(prevLevel, u, v, prevWidth, prevHeight, prevPitch, Cr, Cg, Cb, Ca);
            sampleBilinear(prevLevel, u, v + offset * texelY, prevWidth, prevHeight, prevPitch, B1r, B1g, B1b, B1a);
            sampleBilinear(prevLevel, u, v + offset2 * texelY, prevWidth, prevHeight, prevPitch, B2r, B2g, B2b, B2a);

            // Gaussian weights: [1, 4, 6, 4, 1] / 16
            resR = (T2r * 1.0f + T1r * 4.0f + Cr * 6.0f + B1r * 4.0f + B2r * 1.0f) / 16.0f;
            resG = (T2g * 1.0f + T1g * 4.0f + Cg * 6.0f + B1g * 4.0f + B2g * 1.0f) / 16.0f;
            resB = (T2b * 1.0f + T1b * 4.0f + Cb * 6.0f + B1b * 4.0f + B2b * 1.0f) / 16.0f;
        } else {
            // =========================================
            // Tent Filter (9-tap, 3x3 pattern)
            // For deeper levels where speed matters more
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

            sampleBilinear(prevLevel, u - offset * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, TLr, TLg, TLb, TLa);
            sampleBilinear(prevLevel, u, v - offset * texelY, prevWidth, prevHeight, prevPitch, Tr, Tg, Tb, Ta);
            sampleBilinear(prevLevel, u + offset * texelX, v - offset * texelY, prevWidth, prevHeight, prevPitch, TRr, TRg, TRb, TRa);

            sampleBilinear(prevLevel, u - offset * texelX, v, prevWidth, prevHeight, prevPitch, Lr, Lg, Lb, La);
            sampleBilinear(prevLevel, u, v, prevWidth, prevHeight, prevPitch, Cr, Cg, Cb, Ca);
            sampleBilinear(prevLevel, u + offset * texelX, v, prevWidth, prevHeight, prevPitch, Rr, Rg, Rb, Ra);

            sampleBilinear(prevLevel, u - offset * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, BLr, BLg, BLb, BLa);
            sampleBilinear(prevLevel, u, v + offset * texelY, prevWidth, prevHeight, prevPitch, Bor, Bog, Bob, Boa);
            sampleBilinear(prevLevel, u + offset * texelX, v + offset * texelY, prevWidth, prevHeight, prevPitch, BRr, BRg, BRb, BRa);

            // Tent filter: corners=1, edges=2, center=4 (total=16)
            resR = (TLr + TRr + BLr + BRr) * 1.0f + (Tr + Lr + Rr + Bor) * 2.0f + Cr * 4.0f;
            resG = (TLg + TRg + BLg + BRg) * 1.0f + (Tg + Lg + Rg + Bog) * 2.0f + Cg * 4.0f;
            resB = (TLb + TRb + BLb + BRb) * 1.0f + (Tb + Lb + Rb + Bob) * 2.0f + Cb * 4.0f;

            resR /= 16.0f;
            resG /= 16.0f;
            resB /= 16.0f;
        }
    }

    // =========================================================
    // STEP 2: Add Current Level's Contribution with Weight
    // "현재 층의 디테일을 가중치 적용해서 더한다"
    // =========================================================
    float currR, currG, currB, currA;
    sampleBilinear(input, u, v, srcWidth, srcHeight, srcPitch, currR, currG, currB, currA);

    // Weight calculation for current level
    // A. Physical decay weight (The Shape) - Level 0=100%, Level 1=level1Weight, then decay
    float physicalWeight = calculatePhysicalWeight((float)levelIndex, decayK, falloffType, level1Weight);

    // B. Distance fade weight (The Cutoff)
    // Smooth fade out for levels beyond activeLimit (controlled by Radius)
    float fadeWeight = 1.0f - smoothstepf(activeLimit, activeLimit + 1.0f, (float)levelIndex);

    // C. Final weight combines:
    //    - Physical decay (falloff shape)
    //    - Distance cutoff (radius control)
    //    - Exposure is NOW applied in CompositeKernel to prevent accumulation explosion
    float finalWeight = physicalWeight * fadeWeight;

    // Add weighted current level contribution to upsampled base
    // Result = TentUpsample(Previous) + Current × Weight
    // Note: Exposure will be applied once in Composite, not accumulated per level
    resR = resR + currR * finalWeight;
    resG = resG + currG * finalWeight;
    resB = resB + currB * finalWeight;

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

    // Brightness overflow → white desaturation
    // When brightness exceeds 1.0, colors shift toward white (like real light bloom)
    float maxVal = fmaxf(fmaxf(resR, resG), resB);
    if (maxVal > 1.0f) {
        // Normalize to preserve hue
        float invMax = 1.0f / maxVal;
        float normR = resR * invMax;
        float normG = resG * invMax;
        float normB = resB * invMax;

        // Blend toward white based on overbright amount
        // Uses soft curve: overbright/(overbright+1000) → barely noticeable desaturation
        float overbright = maxVal - 1.0f;
        float blendFactor = overbright / (overbright + 1000.0f);

        resR = normR + blendFactor * (1.0f - normR);
        resG = normG + blendFactor * (1.0f - normG);
        resB = normB + blendFactor * (1.0f - normB);
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
