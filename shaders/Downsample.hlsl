/**
 * JustGlow Downsample Shader
 *
 * Dual Kawase blur downsample pass.
 * Uses 4-tap pattern with bilinear filtering for efficient blur.
 *
 * Based on ARM's SIGGRAPH 2015 presentation:
 * "Bandwidth-Efficient Rendering" by Marius Bjorge
 */

#include "Common.hlsli"

#define THREAD_GROUP_SIZE 16

// ============================================================================
// Dual Kawase Downsample Pattern
// ============================================================================
//
//     +---+---+
//     | A | B |
//     +---X---+   X = output pixel (center of 4 samples)
//     | C | D |
//     +---+---+
//
// Samples at corners of a 2x2 pixel region, exploiting bilinear filtering.
// The offset controls the blur spread.

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    // Bounds check
    if (dispatchID.x >= (uint)g_dstWidth || dispatchID.y >= (uint)g_dstHeight)
        return;

    // Calculate UV coordinates for output pixel
    float2 uv = PixelToUV(dispatchID.xy, g_dstWidth, g_dstHeight);
    float2 texelSize = float2(g_srcTexelX, g_srcTexelY);

    // Kawase blur offset (increases with each pass)
    float offset = g_blurOffset + 0.5f;

    // 4-tap Kawase downsample pattern
    float3 A = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-offset, -offset) * texelSize, 0).rgb;
    float3 B = g_inputTex.SampleLevel(g_linearSampler, uv + float2( offset, -offset) * texelSize, 0).rgb;
    float3 C = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-offset,  offset) * texelSize, 0).rgb;
    float3 D = g_inputTex.SampleLevel(g_linearSampler, uv + float2( offset,  offset) * texelSize, 0).rgb;

    // Center sample (for better quality)
    float3 center = g_inputTex.SampleLevel(g_linearSampler, uv, 0).rgb;

    float3 result;

    if (g_useHDR)
    {
        // Karis weighted average for HDR
        result = KarisAverage4(A, B, C, D);

        // Blend with center sample
        float centerWeight = KarisWeight(center);
        float totalWeight = 4.0f + centerWeight;
        result = (result * 4.0f + center * centerWeight) / totalWeight;
    }
    else
    {
        // Standard average
        // Weight: center = 0.5, corners = 0.125 each
        result = center * 0.5f + (A + B + C + D) * 0.125f;
    }

    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}

// ============================================================================
// Alternative: Extended 9-Tap Downsample
// ============================================================================
// For even better quality at the cost of more samples

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_9tap(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_dstWidth || dispatchID.y >= (uint)g_dstHeight)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_dstWidth, g_dstHeight);
    float2 texelSize = float2(g_srcTexelX, g_srcTexelY);

    // 9-tap pattern: center + 8 surrounding samples
    //   A B C
    //   D E F
    //   G H I

    float3 A = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-1.0f, -1.0f) * texelSize, 0).rgb;
    float3 B = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 0.0f, -1.0f) * texelSize, 0).rgb;
    float3 C = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 1.0f, -1.0f) * texelSize, 0).rgb;
    float3 D = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-1.0f,  0.0f) * texelSize, 0).rgb;
    float3 E = g_inputTex.SampleLevel(g_linearSampler, uv, 0).rgb;
    float3 F = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 1.0f,  0.0f) * texelSize, 0).rgb;
    float3 G = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-1.0f,  1.0f) * texelSize, 0).rgb;
    float3 H = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 0.0f,  1.0f) * texelSize, 0).rgb;
    float3 I = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 1.0f,  1.0f) * texelSize, 0).rgb;

    // Gaussian-like weights
    // Center: 4, Cross: 2, Corners: 1 (total = 16)
    float3 result = (E * 4.0f +
                    (B + D + F + H) * 2.0f +
                    (A + C + G + I)) / 16.0f;

    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}
