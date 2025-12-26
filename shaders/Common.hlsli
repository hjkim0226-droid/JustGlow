/**
 * JustGlow Common Shader Utilities
 *
 * Shared constants, structures, and utility functions for all shaders.
 */

#ifndef JUSTGLOW_COMMON_HLSLI
#define JUSTGLOW_COMMON_HLSLI

// ============================================================================
// Constants
// ============================================================================

static const float PI = 3.14159265359f;
static const float EPSILON = 0.0001f;

// Luminance weights (Rec. 709)
static const float3 LUMA_WEIGHTS = float3(0.2126f, 0.7152f, 0.0722f);

// ============================================================================
// Constant Buffer - Must match C++ GlowParams struct
// ============================================================================

cbuffer GlowParams : register(b0)
{
    // Image dimensions (16 bytes)
    int     g_width;
    int     g_height;
    int     g_srcPitch;
    int     g_dstPitch;

    // Threshold parameters (16 bytes)
    float   g_threshold;
    float   g_softKnee;
    float   g_intensity;
    float   _pad0;

    // Glow color (16 bytes)
    float   g_glowColorR;
    float   g_glowColorG;
    float   g_glowColorB;
    float   g_preserveColor;

    // Color temperature (16 bytes)
    float   g_colorTempR;
    float   g_colorTempG;
    float   g_colorTempB;
    float   _pad1;

    // Anamorphic (16 bytes)
    float   g_anamorphicScaleX;
    float   g_anamorphicScaleY;
    float   g_anamorphicAngle;
    float   g_anamorphicAmount;

    // Composite settings (16 bytes)
    int     g_compositeMode;
    int     g_useHDR;
    float   g_texelSizeX;
    float   g_texelSizeY;

    // Debug options (16 bytes)
    int     g_unpremultiply;
    int     _debugPad0;
    int     _debugPad1;
    int     _debugPad2;
};

// ============================================================================
// Blur Pass Constant Buffer
// ============================================================================

cbuffer BlurPassParams : register(b1)
{
    int     g_srcWidth;
    int     g_srcHeight;
    int     g_dstWidth;
    int     g_dstHeight;

    int     g_passSrcPitch;
    int     g_passDstPitch;
    int     _passPad0;
    int     _passPad1;

    float   g_srcTexelX;
    float   g_srcTexelY;
    float   g_dstTexelX;
    float   g_dstTexelY;

    float   g_blurOffset;
    float   g_fractionalBlend;
    int     g_passIndex;
    int     g_totalPasses;
};

// ============================================================================
// Texture Resources
// ============================================================================

Texture2D<float4> g_inputTex : register(t0);
Texture2D<float4> g_prevLevelTex : register(t1);

RWTexture2D<float4> g_outputTex : register(u0);

SamplerState g_linearSampler : register(s0);
SamplerState g_pointSampler : register(s1);

// ============================================================================
// Utility Functions
// ============================================================================

// Calculate luminance
float Luminance(float3 color)
{
    return dot(color, LUMA_WEIGHTS);
}

// Karis Average weight - prevents fireflies in HDR
// Based on "Call of Duty: Advanced Warfare" presentation
float KarisWeight(float3 color)
{
    float luma = Luminance(color);
    return 1.0f / (1.0f + luma);
}

// Weighted Karis average of 4 samples
float3 KarisAverage4(float3 a, float3 b, float3 c, float3 d)
{
    float wa = KarisWeight(a);
    float wb = KarisWeight(b);
    float wc = KarisWeight(c);
    float wd = KarisWeight(d);

    float totalWeight = wa + wb + wc + wd;
    return (a * wa + b * wb + c * wc + d * wd) / totalWeight;
}

// Soft threshold curve
// Creates a smooth transition around the threshold value
float3 SoftThreshold(float3 color, float threshold, float knee)
{
    float brightness = max(max(color.r, color.g), color.b);

    // Quadratic curve for soft knee
    float soft = brightness - threshold + knee;
    soft = clamp(soft, 0.0f, 2.0f * knee);
    soft = soft * soft / (4.0f * knee + EPSILON);

    // Contribution factor
    float contribution = max(soft, brightness - threshold);
    contribution = contribution / max(brightness, EPSILON);

    return color * max(contribution, 0.0f);
}

// Composite blend modes
float3 BlendAdd(float3 base, float3 blend)
{
    return base + blend;
}

float3 BlendScreen(float3 base, float3 blend)
{
    return 1.0f - (1.0f - base) * (1.0f - blend);
}

float3 BlendOverlay(float3 base, float3 blend)
{
    float3 result;
    result.r = base.r < 0.5f ? 2.0f * base.r * blend.r : 1.0f - 2.0f * (1.0f - base.r) * (1.0f - blend.r);
    result.g = base.g < 0.5f ? 2.0f * base.g * blend.g : 1.0f - 2.0f * (1.0f - base.g) * (1.0f - blend.g);
    result.b = base.b < 0.5f ? 2.0f * base.b * blend.b : 1.0f - 2.0f * (1.0f - base.b) * (1.0f - blend.b);
    return result;
}

float3 ApplyCompositeMode(float3 base, float3 glow, int mode)
{
    switch (mode)
    {
        case 0: return BlendAdd(base, glow);
        case 1: return BlendScreen(base, glow);
        case 2: return BlendOverlay(base, glow);
        default: return BlendAdd(base, glow);
    }
}

// Convert UV to pixel coordinates
int2 UVToPixel(float2 uv, int width, int height)
{
    return int2(uv * float2(width, height));
}

// Convert pixel coordinates to UV
float2 PixelToUV(int2 pixel, int width, int height)
{
    return (float2(pixel) + 0.5f) / float2(width, height);
}

// Safe texture sample with bounds checking
float4 SampleTextureSafe(Texture2D<float4> tex, SamplerState samp, float2 uv)
{
    uv = clamp(uv, 0.0f, 1.0f);
    return tex.SampleLevel(samp, uv, 0);
}

#endif // JUSTGLOW_COMMON_HLSLI
