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
// Soft Threshold
// ============================================================================

__device__ void softThreshold(
    float& r, float& g, float& b,
    float threshold, float knee)
{
    float brightness = fmaxf(fmaxf(r, g), b);

    float soft = brightness - threshold + knee;
    soft = clampf(soft, 0.0f, 2.0f * knee);
    soft = soft * soft / (4.0f * knee + EPSILON);

    float contribution = fmaxf(soft, brightness - threshold);
    contribution = contribution / fmaxf(brightness, EPSILON);
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
    float threshold, float softKnee, float intensity,
    float colorR, float colorG, float colorB,
    float colorTempR, float colorTempG, float colorTempB,
    float preserveColor, int useHDR)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight)
        return;

    float u = ((float)x + 0.5f) / (float)dstWidth;
    float v = ((float)y + 0.5f) / (float)dstHeight;

    float texelX = 1.0f / (float)srcWidth;
    float texelY = 1.0f / (float)srcHeight;

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
    sampleBilinear(input, u - 2.0f * texelX, v - 2.0f * texelY, srcWidth, srcHeight, srcPitch, Ar, Ag, Ab, Aa);
    sampleBilinear(input, u + 2.0f * texelX, v - 2.0f * texelY, srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);
    sampleBilinear(input, u - 2.0f * texelX, v + 2.0f * texelY, srcWidth, srcHeight, srcPitch, Kr, Kg, Kb, Ka);
    sampleBilinear(input, u + 2.0f * texelX, v + 2.0f * texelY, srcWidth, srcHeight, srcPitch, Mr, Mg, Mb, Ma);

    // Outer cross
    sampleBilinear(input, u, v - 2.0f * texelY, srcWidth, srcHeight, srcPitch, Br, Bg, Bb, Ba);
    sampleBilinear(input, u - 2.0f * texelX, v, srcWidth, srcHeight, srcPitch, Fr, Fg, Fb, Fa);
    sampleBilinear(input, u + 2.0f * texelX, v, srcWidth, srcHeight, srcPitch, Hr, Hg, Hb, Ha);
    sampleBilinear(input, u, v + 2.0f * texelY, srcWidth, srcHeight, srcPitch, Lr, Lg, Lb, La);

    // Inner corners
    sampleBilinear(input, u - texelX, v - texelY, srcWidth, srcHeight, srcPitch, Dr, Dg, Db, Da);
    sampleBilinear(input, u + texelX, v - texelY, srcWidth, srcHeight, srcPitch, Er, Eg, Eb, Ea);
    sampleBilinear(input, u - texelX, v + texelY, srcWidth, srcHeight, srcPitch, Ir, Ig, Ib, Ia);
    sampleBilinear(input, u + texelX, v + texelY, srcWidth, srcHeight, srcPitch, Jr, Jg, Jb, Ja);

    // Center
    sampleBilinear(input, u, v, srcWidth, srcHeight, srcPitch, Gr, Gg, Gb, Ga);

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
// Downsample Kernel (Dual Kawase 4-Tap)
// ============================================================================

extern "C" __global__ void DownsampleKernel(
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

    float texelX = 1.0f / (float)srcWidth;
    float texelY = 1.0f / (float)srcHeight;
    float offset = blurOffset + 0.5f;

    float Ar, Ag, Ab, Aa;
    float Br, Bg, Bb, Ba;
    float Cr, Cg, Cb, Ca;
    float Dr, Dg, Db, Da;
    float centerR, centerG, centerB, centerA;

    sampleBilinear(input, u - offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, Ar, Ag, Ab, Aa);
    sampleBilinear(input, u + offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, Br, Bg, Bb, Ba);
    sampleBilinear(input, u - offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);
    sampleBilinear(input, u + offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, Dr, Dg, Db, Da);
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
// Upsample Kernel (9-Tap Tent Filter)
// ============================================================================

extern "C" __global__ void UpsampleKernel(
    const float* __restrict__ input,
    const float* __restrict__ prevLevel,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset, float blendFactor)
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

    float TLr, TLg, TLb, TLa;
    float Tr, Tg, Tb, Ta;
    float TRr, TRg, TRb, TRa;
    float Lr, Lg, Lb, La;
    float Cr, Cg, Cb, Ca;
    float Rr, Rg, Rb, Ra;
    float BLr, BLg, BLb, BLa;
    float Bor, Bog, Bob, Boa;
    float BRr, BRg, BRb, BRa;

    sampleBilinear(input, u - offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, TLr, TLg, TLb, TLa);
    sampleBilinear(input, u, v - offset * texelY, srcWidth, srcHeight, srcPitch, Tr, Tg, Tb, Ta);
    sampleBilinear(input, u + offset * texelX, v - offset * texelY, srcWidth, srcHeight, srcPitch, TRr, TRg, TRb, TRa);

    sampleBilinear(input, u - offset * texelX, v, srcWidth, srcHeight, srcPitch, Lr, Lg, Lb, La);
    sampleBilinear(input, u, v, srcWidth, srcHeight, srcPitch, Cr, Cg, Cb, Ca);
    sampleBilinear(input, u + offset * texelX, v, srcWidth, srcHeight, srcPitch, Rr, Rg, Rb, Ra);

    sampleBilinear(input, u - offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, BLr, BLg, BLb, BLa);
    sampleBilinear(input, u, v + offset * texelY, srcWidth, srcHeight, srcPitch, Bor, Bog, Bob, Boa);
    sampleBilinear(input, u + offset * texelX, v + offset * texelY, srcWidth, srcHeight, srcPitch, BRr, BRg, BRb, BRa);

    // Tent filter: corners=1, edges=2, center=4 (total=16)
    float resR = (TLr + TRr + BLr + BRr) * 1.0f + (Tr + Lr + Rr + Bor) * 2.0f + Cr * 4.0f;
    float resG = (TLg + TRg + BLg + BRg) * 1.0f + (Tg + Lg + Rg + Bog) * 2.0f + Cg * 4.0f;
    float resB = (TLb + TRb + BLb + BRb) * 1.0f + (Tb + Lb + Rb + Bob) * 2.0f + Cb * 4.0f;

    resR /= 16.0f;
    resG /= 16.0f;
    resB /= 16.0f;

    // Blend with previous level if needed
    if (blendFactor > 0.0f && prevLevel != nullptr) {
        float prevR, prevG, prevB, prevA;
        sampleBilinear(prevLevel, u, v, dstWidth, dstHeight, dstPitch, prevR, prevG, prevB, prevA);
        resR = resR + prevR * blendFactor;
        resG = resG + prevG * blendFactor;
        resB = resB + prevB * blendFactor;
    }

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
    int originalPitch, int glowPitch, int outputPitch,
    float intensity, int compositeMode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int origIdx = (y * originalPitch + x) * 4;
    float origR = original[origIdx + 0];
    float origG = original[origIdx + 1];
    float origB = original[origIdx + 2];
    float origA = original[origIdx + 3];

    // Sample glow with bilinear
    float u = ((float)x + 0.5f) / (float)width;
    float v = ((float)y + 0.5f) / (float)height;

    float glowR, glowG, glowB, glowA;
    sampleBilinear(glow, u, v, width, height, glowPitch, glowR, glowG, glowB, glowA);

    // Apply intensity
    glowR *= intensity;
    glowG *= intensity;
    glowB *= intensity;

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
    output[outIdx + 3] = origA;
}
