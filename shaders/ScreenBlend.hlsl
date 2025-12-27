/**
 * JustGlow Screen Blend Shader (Log-Transmittance)
 *
 * Combines multiple MIP levels using Screen blending in log-transmittance space.
 *
 * Mathematical basis:
 * - Screen: result = 1 - (1-A)(1-B) = 1 - T_A × T_B
 * - Log space: log(T_A × T_B) = log(T_A) + log(T_B)
 * - This makes Screen blending ADDITIVE in log space!
 *
 * Why Log-Transmittance:
 * - Screen blending is based on light transmittance (Beer-Lambert law)
 * - T = 1 - color = how much light passes through
 * - Multiple Screen = multiply transmittances
 * - Log converts multiply to add = linear = commutative with blur!
 *
 * Performance:
 * - Single dispatch instead of 6 draw calls
 * - All MIP levels sampled in one shader
 * - Log/exp operations are fast on modern GPUs
 */

#include "Common.hlsli"

#define THREAD_GROUP_SIZE 16
#define MAX_LEVELS 6

// ============================================================================
// Screen Blend Constant Buffer
// ============================================================================

cbuffer ScreenBlendParams : register(b2)
{
    int     g_numLevels;        // Number of MIP levels to blend (1-6)
    float   g_baseWeight;       // Base intensity weight
    float   g_falloff;          // Falloff rate per level (0-1)
    float   _sbPad0;

    float4  g_levelWeights;     // Pre-computed weights for levels 1-4
    float2  g_levelWeights56;   // Pre-computed weights for levels 5-6
    float2  _sbPad1;
};

// ============================================================================
// Additional Texture Resources for MIP levels
// ============================================================================

// Blurred MIP level textures (from CUDA Pre-blur)
Texture2D<float4> g_blurredLevel1 : register(t2);
Texture2D<float4> g_blurredLevel2 : register(t3);
Texture2D<float4> g_blurredLevel3 : register(t4);
Texture2D<float4> g_blurredLevel4 : register(t5);
Texture2D<float4> g_blurredLevel5 : register(t6);
Texture2D<float4> g_blurredLevel6 : register(t7);

// ============================================================================
// Helper Functions
// ============================================================================

// Safe log for transmittance (clamp to avoid log(0))
float4 SafeLogTransmittance(float4 color)
{
    float4 T = 1.0f - saturate(color);  // Transmittance = 1 - color
    T = max(T, 1e-6f);                   // Avoid log(0)
    return log(T);
}

// Restore color from log-transmittance
float4 ExpToColor(float4 logT)
{
    float4 T = exp(logT);               // Transmittance
    return 1.0f - T;                    // Color = 1 - T
}

// ============================================================================
// Log-Transmittance Screen Blend Kernel
// ============================================================================

/**
 * Accumulates all MIP levels using Screen blending in log space.
 *
 * Process:
 * 1. For each level, compute weighted color
 * 2. Convert to log-transmittance: L = log(1 - color × weight)
 * 3. Sum all log-transmittances (equivalent to multiply in linear space)
 * 4. Convert back: result = 1 - exp(sum_L)
 */
[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_dstWidth || dispatchID.y >= (uint)g_dstHeight)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_dstWidth, g_dstHeight);

    // Accumulate log-transmittance
    float4 logT_sum = float4(0, 0, 0, 0);

    // Sample and accumulate each level
    // Weight = baseWeight * pow(falloff, level)

    // Level 1
    if (g_numLevels >= 1)
    {
        float4 color = g_blurredLevel1.SampleLevel(g_linearSampler, uv, 0);
        color *= g_levelWeights.x;
        logT_sum += SafeLogTransmittance(color);
    }

    // Level 2
    if (g_numLevels >= 2)
    {
        float4 color = g_blurredLevel2.SampleLevel(g_linearSampler, uv, 0);
        color *= g_levelWeights.y;
        logT_sum += SafeLogTransmittance(color);
    }

    // Level 3
    if (g_numLevels >= 3)
    {
        float4 color = g_blurredLevel3.SampleLevel(g_linearSampler, uv, 0);
        color *= g_levelWeights.z;
        logT_sum += SafeLogTransmittance(color);
    }

    // Level 4
    if (g_numLevels >= 4)
    {
        float4 color = g_blurredLevel4.SampleLevel(g_linearSampler, uv, 0);
        color *= g_levelWeights.w;
        logT_sum += SafeLogTransmittance(color);
    }

    // Level 5
    if (g_numLevels >= 5)
    {
        float4 color = g_blurredLevel5.SampleLevel(g_linearSampler, uv, 0);
        color *= g_levelWeights56.x;
        logT_sum += SafeLogTransmittance(color);
    }

    // Level 6
    if (g_numLevels >= 6)
    {
        float4 color = g_blurredLevel6.SampleLevel(g_linearSampler, uv, 0);
        color *= g_levelWeights56.y;
        logT_sum += SafeLogTransmittance(color);
    }

    // Convert accumulated log-transmittance back to color
    float4 result = ExpToColor(logT_sum);

    // Write result
    g_outputTex[dispatchID.xy] = result;
}

// ============================================================================
// Alternative: Direct Screen Blend (without Log space)
// ============================================================================

/**
 * Traditional Screen blending without log-transmittance.
 * Mathematically equivalent but may have different numerical properties.
 *
 * Screen(A, B) = 1 - (1-A)(1-B) = A + B - AB
 */
[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_direct(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_dstWidth || dispatchID.y >= (uint)g_dstHeight)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_dstWidth, g_dstHeight);

    // Start with black (identity for Screen blend)
    float4 result = float4(0, 0, 0, 0);

    // Accumulate using Screen blend formula
    // Screen(A, B) = 1 - (1-A)(1-B)

    if (g_numLevels >= 1)
    {
        float4 color = g_blurredLevel1.SampleLevel(g_linearSampler, uv, 0) * g_levelWeights.x;
        result = BlendScreen(result, color);
    }

    if (g_numLevels >= 2)
    {
        float4 color = g_blurredLevel2.SampleLevel(g_linearSampler, uv, 0) * g_levelWeights.y;
        result = BlendScreen(result, color);
    }

    if (g_numLevels >= 3)
    {
        float4 color = g_blurredLevel3.SampleLevel(g_linearSampler, uv, 0) * g_levelWeights.z;
        result = BlendScreen(result, color);
    }

    if (g_numLevels >= 4)
    {
        float4 color = g_blurredLevel4.SampleLevel(g_linearSampler, uv, 0) * g_levelWeights.w;
        result = BlendScreen(result, color);
    }

    if (g_numLevels >= 5)
    {
        float4 color = g_blurredLevel5.SampleLevel(g_linearSampler, uv, 0) * g_levelWeights56.x;
        result = BlendScreen(result, color);
    }

    if (g_numLevels >= 6)
    {
        float4 color = g_blurredLevel6.SampleLevel(g_linearSampler, uv, 0) * g_levelWeights56.y;
        result = BlendScreen(result, color);
    }

    g_outputTex[dispatchID.xy] = result;
}

// ============================================================================
// Utility: Blend with float4 (for alpha channel)
// ============================================================================

float4 BlendScreen(float4 base, float4 blend)
{
    return float4(
        1.0f - (1.0f - base.rgb) * (1.0f - blend.rgb),
        1.0f - (1.0f - base.a) * (1.0f - blend.a)
    );
}
