/**
 * JustGlow Prefilter Shader
 *
 * First stage of the glow pipeline:
 * 1. 13-Tap Downsample (Call of Duty method)
 * 2. Soft Threshold with Knee
 * 3. Karis Average for HDR anti-firefly
 */

#include "Common.hlsli"

// Thread group size
#define THREAD_GROUP_SIZE 16

// ============================================================================
// 13-Tap Downsample Pattern (Call of Duty: Advanced Warfare)
// ============================================================================
//
//     A   B   C
//       D   E
//     F   G   H
//       I   J
//     K   L   M
//
// Sampling pattern with bilinear filtering exploits hardware interpolation
// for better quality and cache efficiency.

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    // Bounds check
    if (dispatchID.x >= (uint)g_dstWidth || dispatchID.y >= (uint)g_dstHeight)
        return;

    // Calculate UV coordinates
    float2 uv = PixelToUV(dispatchID.xy, g_dstWidth, g_dstHeight);
    float2 texelSize = float2(g_srcTexelX, g_srcTexelY);

    // 13-Tap sample positions
    // Outer corners (weight: 0.03125 each = 0.125 total)
    float3 A = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-2.0f, -2.0f) * texelSize, 0).rgb;
    float3 C = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 2.0f, -2.0f) * texelSize, 0).rgb;
    float3 K = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-2.0f,  2.0f) * texelSize, 0).rgb;
    float3 M = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 2.0f,  2.0f) * texelSize, 0).rgb;

    // Outer cross (weight: 0.0625 each = 0.25 total)
    float3 B = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 0.0f, -2.0f) * texelSize, 0).rgb;
    float3 F = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-2.0f,  0.0f) * texelSize, 0).rgb;
    float3 H = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 2.0f,  0.0f) * texelSize, 0).rgb;
    float3 L = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 0.0f,  2.0f) * texelSize, 0).rgb;

    // Inner corners (weight: 0.125 each = 0.5 total)
    float3 D = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-1.0f, -1.0f) * texelSize, 0).rgb;
    float3 E = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 1.0f, -1.0f) * texelSize, 0).rgb;
    float3 I = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-1.0f,  1.0f) * texelSize, 0).rgb;
    float3 J = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 1.0f,  1.0f) * texelSize, 0).rgb;

    // Center
    float3 G = g_inputTex.SampleLevel(g_linearSampler, uv, 0).rgb;

    float3 result;

    if (g_useHDR)
    {
        // Karis Average for HDR content
        // Prevents bright pixels from dominating the blur

        // Group samples into 5 regions for weighted averaging
        float3 group1 = KarisAverage4(D, E, I, J);     // Inner 4 (highest weight)
        float3 group2 = KarisAverage4(A, B, F, G);     // Top-left quad
        float3 group3 = KarisAverage4(B, C, G, H);     // Top-right quad
        float3 group4 = KarisAverage4(F, G, K, L);     // Bottom-left quad
        float3 group5 = KarisAverage4(G, H, L, M);     // Bottom-right quad

        // Weighted combination (inner region weighted higher)
        result = group1 * 0.5f + (group2 + group3 + group4 + group5) * 0.125f;
    }
    else
    {
        // Standard weighted average for non-HDR content
        float3 innerSum = D + E + I + J;
        float3 crossSum = B + F + H + L;
        float3 cornerSum = A + C + K + M;

        result = G * 0.125f +
                 innerSum * 0.125f +
                 crossSum * 0.0625f +
                 cornerSum * 0.03125f;
    }

    // Apply soft threshold
    result = SoftThreshold(result, g_threshold, g_softKnee);

    // Write output
    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}
