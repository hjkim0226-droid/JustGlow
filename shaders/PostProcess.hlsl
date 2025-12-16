/**
 * JustGlow Post Process Shader
 *
 * Optional post-processing effects:
 * - Anamorphic stretch (directional blur/streak)
 * - Gaussian polish pass
 */

#include "Common.hlsli"

#define THREAD_GROUP_SIZE 16
#define ANAMORPHIC_SAMPLES 15

// ============================================================================
// Anamorphic Stretch
// ============================================================================
// Creates directional streak effect common in cinematic looks.
// Can be horizontal (classic anamorphic) or at any angle.

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_anamorphic(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_width || dispatchID.y >= (uint)g_height)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_width, g_height);
    float2 texelSize = float2(g_texelSizeX, g_texelSizeY);

    // Skip if anamorphic is disabled
    if (g_anamorphicAmount < 0.001f)
    {
        g_outputTex[dispatchID.xy] = g_inputTex.SampleLevel(g_linearSampler, uv, 0);
        return;
    }

    // Calculate stretch direction from angle
    float cosAngle = cos(g_anamorphicAngle);
    float sinAngle = sin(g_anamorphicAngle);
    float2 stretchDir = float2(cosAngle, sinAngle);

    // Scale factor for stretch
    float2 scale = float2(g_anamorphicScaleX, g_anamorphicScaleY);

    // Accumulate samples along stretch direction
    float3 result = float3(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;

    // Sample spread based on anamorphic amount
    float spread = g_anamorphicAmount * 50.0f; // Max 50 pixel spread

    for (int i = -ANAMORPHIC_SAMPLES; i <= ANAMORPHIC_SAMPLES; i++)
    {
        float t = (float)i / (float)ANAMORPHIC_SAMPLES;

        // Gaussian-like weight falloff
        float weight = exp(-t * t * 3.0f);

        // Calculate sample offset
        float2 offset = stretchDir * t * spread * texelSize * scale;

        float3 sample = g_inputTex.SampleLevel(g_linearSampler, uv + offset, 0).rgb;

        result += sample * weight;
        totalWeight += weight;
    }

    result /= totalWeight;

    // Blend with original based on anamorphic amount
    float3 original = g_inputTex.SampleLevel(g_linearSampler, uv, 0).rgb;
    result = lerp(original, result, g_anamorphicAmount);

    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}

// ============================================================================
// Gaussian Polish Pass
// ============================================================================
// Optional final smoothing pass using separable Gaussian blur.
// Helps blend the MIP level transitions for smoother results.

// 7-tap Gaussian weights (sigma ~= 1.5)
static const float GaussianWeights[7] = {
    0.0702f, 0.1311f, 0.1907f, 0.2160f, 0.1907f, 0.1311f, 0.0702f
};

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_gaussian_h(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_width || dispatchID.y >= (uint)g_height)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_width, g_height);
    float2 texelSize = float2(g_texelSizeX, g_texelSizeY);

    float3 result = float3(0.0f, 0.0f, 0.0f);

    // Horizontal pass
    for (int i = -3; i <= 3; i++)
    {
        float2 offset = float2((float)i * texelSize.x, 0.0f);
        result += g_inputTex.SampleLevel(g_linearSampler, uv + offset, 0).rgb * GaussianWeights[i + 3];
    }

    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_gaussian_v(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_width || dispatchID.y >= (uint)g_height)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_width, g_height);
    float2 texelSize = float2(g_texelSizeX, g_texelSizeY);

    float3 result = float3(0.0f, 0.0f, 0.0f);

    // Vertical pass
    for (int i = -3; i <= 3; i++)
    {
        float2 offset = float2(0.0f, (float)i * texelSize.y);
        result += g_inputTex.SampleLevel(g_linearSampler, uv + offset, 0).rgb * GaussianWeights[i + 3];
    }

    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}

// ============================================================================
// Star/Diffraction Pattern
// ============================================================================
// Creates star-shaped diffraction patterns around bright points

#define STAR_POINTS 6
#define STAR_SAMPLES 16

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main_star(uint3 dispatchID : SV_DispatchThreadID)
{
    if (dispatchID.x >= (uint)g_width || dispatchID.y >= (uint)g_height)
        return;

    float2 uv = PixelToUV(dispatchID.xy, g_width, g_height);
    float2 texelSize = float2(g_texelSizeX, g_texelSizeY);

    float3 center = g_inputTex.SampleLevel(g_linearSampler, uv, 0).rgb;
    float3 result = center;

    // Calculate brightness
    float brightness = Luminance(center);

    // Only apply star effect to bright areas
    if (brightness > g_threshold)
    {
        float starIntensity = (brightness - g_threshold) * g_intensity * 0.1f;

        // Sample along star point directions
        for (int p = 0; p < STAR_POINTS; p++)
        {
            float angle = (float)p * PI * 2.0f / (float)STAR_POINTS + g_anamorphicAngle;
            float2 dir = float2(cos(angle), sin(angle));

            for (int s = 1; s <= STAR_SAMPLES; s++)
            {
                float t = (float)s / (float)STAR_SAMPLES;
                float weight = (1.0f - t) * starIntensity;

                float2 offset = dir * t * 100.0f * texelSize; // 100 pixel max spread

                result += g_inputTex.SampleLevel(g_linearSampler, uv + offset, 0).rgb * weight;
            }
        }
    }

    g_outputTex[dispatchID.xy] = float4(result, 1.0f);
}
