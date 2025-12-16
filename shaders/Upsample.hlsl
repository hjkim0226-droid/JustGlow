/**
 * JustGlow Upsample Shader
 *
 * 9-Tap Tent filter upsample with progressive blending.
 * Each upsample pass blends with the previous level for accumulated glow.
 *
 * Features:
 * - 9-Tap tent filter for smooth upsampling
 * - Progressive blend with previous MIP level
 * - Fractional interpolation for smooth radius transitions
 */

#include "Common.hlsli"

#define THREAD_GROUP_SIZE 16

// ============================================================================
// 9-Tap Tent Upsample Pattern
// ============================================================================
//
//     1   2   1
//     2   4   2   / 16
//     1   2   1
//
// Tent filter provides smooth interpolation during upsampling.
// Combined with bilinear filtering for high quality.

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    // Bounds check
    if (dispatchID.x >= (uint)g_dstWidth || dispatchID.y >= (uint)g_dstHeight)
        return;

    // Calculate UV coordinates
    float2 uv = PixelToUV(dispatchID.xy, g_dstWidth, g_dstHeight);
    float2 texelSize = float2(g_srcTexelX, g_srcTexelY);

    // Kawase offset for this pass
    float offset = g_blurOffset + 0.5f;

    // 9-Tap tent filter
    // Weights: corners=1, edges=2, center=4 (total=16)

    float3 TL = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-offset, -offset) * texelSize, 0).rgb;
    float3 T  = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 0.0f,   -offset) * texelSize, 0).rgb;
    float3 TR = g_inputTex.SampleLevel(g_linearSampler, uv + float2( offset, -offset) * texelSize, 0).rgb;

    float3 L  = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-offset,  0.0f)   * texelSize, 0).rgb;
    float3 C  = g_inputTex.SampleLevel(g_linearSampler, uv, 0).rgb;
    float3 R  = g_inputTex.SampleLevel(g_linearSampler, uv + float2( offset,  0.0f)   * texelSize, 0).rgb;

    float3 BL = g_inputTex.SampleLevel(g_linearSampler, uv + float2(-offset,  offset) * texelSize, 0).rgb;
    float3 B  = g_inputTex.SampleLevel(g_linearSampler, uv + float2( 0.0f,    offset) * texelSize, 0).rgb;
    float3 BR = g_inputTex.SampleLevel(g_linearSampler, uv + float2( offset,  offset) * texelSize, 0).rgb;

    // Apply tent filter weights
    float3 upsampled = (TL + TR + BL + BR) * 1.0f +    // Corners: 1
                       (T + L + R + B) * 2.0f +         // Edges: 2
                       C * 4.0f;                        // Center: 4
    upsampled /= 16.0f;

    // Progressive blend with previous (higher resolution) level
    // This accumulates the glow from all MIP levels
    if (g_passIndex > 0)
    {
        float3 prevLevel = g_prevLevelTex.SampleLevel(g_linearSampler, uv, 0).rgb;

        // Blend factor: can be uniform or weighted by pass index
        // Using equal blending gives good results for bloom
        float blendFactor = 0.5f;

        upsampled = lerp(upsampled, prevLevel + upsampled, blendFactor);
    }

    // Fractional interpolation for smooth radius control
    // Only applies at the final (highest resolution) pass
    if (g_fractionalBlend > 0.0f && g_passIndex == 0)
    {
        float3 prevLevel = g_prevLevelTex.SampleLevel(g_linearSampler, uv, 0).rgb;
        upsampled = lerp(prevLevel, upsampled, g_fractionalBlend);
    }

    g_outputTex[dispatchID.xy] = float4(upsampled, 1.0f);
}

// ============================================================================
// Alternative: Bilinear Upsample with Additive Blend
// ============================================================================
// Simpler but effective for certain styles

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_bilinear(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_dstWidth || dispatchID.y >= (uint)g_dstHeight)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_dstWidth, g_dstHeight);

    // Simple bilinear upsample (hardware filtering)
    float3 upsampled = g_inputTex.SampleLevel(g_linearSampler, uv, 0).rgb;

    // Additive blend with previous level
    float3 prevLevel = g_prevLevelTex.SampleLevel(g_linearSampler, uv, 0).rgb;

    // Progressive weight based on MIP level
    // Higher MIP levels (smaller textures) contribute less
    float weight = 1.0f / (float)(g_passIndex + 1);

    float3 result = prevLevel + upsampled * weight;

    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}

// ============================================================================
// High Quality: Catmull-Rom Upsample
// ============================================================================
// For maximum quality at higher cost

float3 CatmullRom(float3 p0, float3 p1, float3 p2, float3 p3, float t)
{
    float t2 = t * t;
    float t3 = t2 * t;

    return 0.5f * (
        (2.0f * p1) +
        (-p0 + p2) * t +
        (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
        (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
    );
}

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_catmull(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_dstWidth || dispatchID.y >= (uint)g_dstHeight)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_dstWidth, g_dstHeight);
    float2 texelSize = float2(g_srcTexelX, g_srcTexelY);

    // Calculate sample position in source texture
    float2 srcPos = uv * float2(g_srcWidth, g_srcHeight) - 0.5f;
    float2 texPos = floor(srcPos);
    float2 f = srcPos - texPos;

    // Sample 4x4 neighborhood
    float3 samples[4][4];
    for (int y = -1; y <= 2; y++)
    {
        for (int x = -1; x <= 2; x++)
        {
            float2 sampleUV = (texPos + float2(x, y) + 0.5f) * texelSize;
            samples[y + 1][x + 1] = g_inputTex.SampleLevel(g_pointSampler, sampleUV, 0).rgb;
        }
    }

    // Interpolate horizontally
    float3 rows[4];
    for (int y = 0; y < 4; y++)
    {
        rows[y] = CatmullRom(samples[y][0], samples[y][1], samples[y][2], samples[y][3], f.x);
    }

    // Interpolate vertically
    float3 result = CatmullRom(rows[0], rows[1], rows[2], rows[3], f.y);

    // Blend with previous level
    float3 prevLevel = g_prevLevelTex.SampleLevel(g_linearSampler, uv, 0).rgb;
    result = prevLevel + result;

    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}
