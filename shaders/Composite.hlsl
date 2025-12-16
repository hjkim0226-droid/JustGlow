/**
 * JustGlow Composite Shader
 *
 * Final compositing stage:
 * - Apply glow color and intensity
 * - Apply color temperature
 * - Blend with original image using selected composite mode
 */

#include "Common.hlsli"

#define THREAD_GROUP_SIZE 16

// ============================================================================
// Composite Kernel
// ============================================================================

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    // Bounds check
    if (dispatchID.x >= (uint)g_width || dispatchID.y >= (uint)g_height)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_width, g_height);

    // Sample original image
    float4 original = g_inputTex.SampleLevel(g_linearSampler, uv, 0);

    // Sample glow (accumulated blur result)
    float3 glow = g_prevLevelTex.SampleLevel(g_linearSampler, uv, 0).rgb;

    // Apply color temperature
    glow.r *= g_colorTempR;
    glow.g *= g_colorTempG;
    glow.b *= g_colorTempB;

    // Apply glow color tint
    float3 glowColor = float3(g_glowColorR, g_glowColorG, g_glowColorB);

    // Blend between colored glow and original color preservation
    // preserveColor = 1.0 means keep original glow colors
    // preserveColor = 0.0 means fully tint with glow color
    float3 coloredGlow = lerp(glow * glowColor, glow, g_preserveColor);

    // Apply intensity
    coloredGlow *= g_intensity;

    // Composite with original
    float3 result = ApplyCompositeMode(original.rgb, coloredGlow, g_compositeMode);

    // Preserve original alpha
    g_outputTex[dispatchID.xy] = float4(result, original.a);
}

// ============================================================================
// Alternative: Composite with Bloom Dirt/Lens Texture
// ============================================================================
// For adding lens imperfection effects

Texture2D<float4> g_lensDirtTex : register(t2);

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_with_dirt(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_width || dispatchID.y >= (uint)g_height)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_width, g_height);

    float4 original = g_inputTex.SampleLevel(g_linearSampler, uv, 0);
    float3 glow = g_prevLevelTex.SampleLevel(g_linearSampler, uv, 0).rgb;

    // Sample lens dirt texture
    float3 lensDirt = g_lensDirtTex.SampleLevel(g_linearSampler, uv, 0).rgb;

    // Modulate glow by lens dirt
    glow *= (1.0f + lensDirt * 2.0f);

    // Apply color temperature
    glow.r *= g_colorTempR;
    glow.g *= g_colorTempG;
    glow.b *= g_colorTempB;

    // Apply glow color
    float3 glowColor = float3(g_glowColorR, g_glowColorG, g_glowColorB);
    float3 coloredGlow = lerp(glow * glowColor, glow, g_preserveColor);

    coloredGlow *= g_intensity;

    float3 result = ApplyCompositeMode(original.rgb, coloredGlow, g_compositeMode);

    g_outputTex[dispatchID.xy] = float4(result, original.a);
}

// ============================================================================
// Composite with Chromatic Aberration
// ============================================================================
// Adds RGB channel offset for a more stylized look

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_chromatic(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_width || dispatchID.y >= (uint)g_height)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_width, g_height);
    float2 center = float2(0.5f, 0.5f);

    // Calculate distance from center for aberration
    float2 dir = uv - center;
    float dist = length(dir);
    float2 aberrationOffset = normalize(dir) * dist * 0.01f; // Adjust strength

    float4 original = g_inputTex.SampleLevel(g_linearSampler, uv, 0);

    // Sample glow with chromatic offset
    float glowR = g_prevLevelTex.SampleLevel(g_linearSampler, uv + aberrationOffset, 0).r;
    float glowG = g_prevLevelTex.SampleLevel(g_linearSampler, uv, 0).g;
    float glowB = g_prevLevelTex.SampleLevel(g_linearSampler, uv - aberrationOffset, 0).b;

    float3 glow = float3(glowR, glowG, glowB);

    // Apply color temperature
    glow.r *= g_colorTempR;
    glow.g *= g_colorTempG;
    glow.b *= g_colorTempB;

    // Apply glow color
    float3 glowColor = float3(g_glowColorR, g_glowColorG, g_glowColorB);
    float3 coloredGlow = lerp(glow * glowColor, glow, g_preserveColor);

    coloredGlow *= g_intensity;

    float3 result = ApplyCompositeMode(original.rgb, coloredGlow, g_compositeMode);

    g_outputTex[dispatchID.xy] = float4(result, original.a);
}
