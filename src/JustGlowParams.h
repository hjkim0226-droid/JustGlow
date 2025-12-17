/**
 * JustGlow GPU Parameters
 *
 * Constant buffer structures shared between CPU (C++) and GPU (HLSL).
 * These structures must maintain identical memory layout on both sides.
 */

#pragma once
#ifndef JUSTGLOW_PARAMS_H
#define JUSTGLOW_PARAMS_H

#ifdef __cplusplus
    #include <cstdint>
    #include <cmath>

    // C++ type aliases matching HLSL
    struct float2 { float x, y; };
    struct float3 { float x, y, z; };
    struct float4 { float x, y, z, w; };
    struct int2 { int x, y; };
    struct int4 { int x, y, z, w; };

    #define CONSTANT_BUFFER_BEGIN(name) struct alignas(16) name {
    #define CONSTANT_BUFFER_END };
#else
    // HLSL
    #define CONSTANT_BUFFER_BEGIN(name) cbuffer name : register(b0) {
    #define CONSTANT_BUFFER_END };
#endif

// ============================================================================
// Main Glow Parameters (Constant Buffer 0)
// ============================================================================

CONSTANT_BUFFER_BEGIN(GlowParams)
    // Image dimensions (16 bytes)
    int     width;              // Image width
    int     height;             // Image height
    int     srcPitch;           // Source buffer pitch (bytes per row / sizeof(float4))
    int     dstPitch;           // Destination buffer pitch

    // Threshold parameters (16 bytes)
    float   threshold;          // Brightness threshold (0-1)
    float   softKnee;           // Soft knee width (0-1)
    float   intensity;          // Glow intensity multiplier
    float   _pad0;

    // Glow color (16 bytes)
    float   glowColorR;         // Glow tint R
    float   glowColorG;         // Glow tint G
    float   glowColorB;         // Glow tint B
    float   preserveColor;      // Original color preservation (0-1)

    // Color temperature (16 bytes)
    float   colorTempR;         // Temperature R multiplier
    float   colorTempG;         // Temperature G multiplier
    float   colorTempB;         // Temperature B multiplier
    float   _pad1;

    // Anamorphic (16 bytes)
    float   anamorphicScaleX;   // Horizontal scale
    float   anamorphicScaleY;   // Vertical scale
    float   anamorphicAngle;    // Rotation angle (radians)
    float   anamorphicAmount;   // Blend amount (0-1)

    // Composite settings (16 bytes)
    int     compositeMode;      // 0=Add, 1=Screen, 2=Overlay
    int     useHDR;             // Enable Karis Average
    float   texelSizeX;         // 1.0 / width
    float   texelSizeY;         // 1.0 / height
CONSTANT_BUFFER_END

// ============================================================================
// Blur Pass Parameters (Constant Buffer 1)
// ============================================================================

CONSTANT_BUFFER_BEGIN(BlurPassParams)
    // Dimensions for current pass (16 bytes)
    int     srcWidth;           // Source texture width
    int     srcHeight;          // Source texture height
    int     dstWidth;           // Destination texture width
    int     dstHeight;          // Destination texture height

    // Pitch values (16 bytes)
    int     srcPitch;           // Source pitch
    int     dstPitch;           // Destination pitch
    int     _pad0;
    int     _pad1;

    // Texel size (16 bytes)
    float   srcTexelX;          // 1.0 / srcWidth
    float   srcTexelY;          // 1.0 / srcHeight
    float   dstTexelX;          // 1.0 / dstWidth
    float   dstTexelY;          // 1.0 / dstHeight

    // Kawase blur offset (16 bytes)
    float   blurOffset;         // Sample offset for current pass
    float   fractionalBlend;    // Fractional interpolation amount
    int     passIndex;          // Current pass index
    int     totalPasses;        // Total number of passes
CONSTANT_BUFFER_END

// ============================================================================
// MIP Level Info
// ============================================================================

#ifdef __cplusplus

// Maximum supported MIP levels (12 supports up to 8K resolution)
constexpr int MAX_MIP_LEVELS = 12;

// MIP chain configuration
struct MipChainConfig {
    int     levelCount;                     // Number of MIP levels
    int     widths[MAX_MIP_LEVELS];         // Width at each level
    int     heights[MAX_MIP_LEVELS];        // Height at each level
    float   blurOffsets[MAX_MIP_LEVELS];    // Kawase offset per level
};

// Calculate dynamic MIP levels based on resolution
// Goes until minimum dimension reaches minSize (default 16px for Deep Glow-like depth)
// maxLevels can limit the depth for performance (0 = no limit)
inline int CalculateDynamicMipLevels(int width, int height, int minSize = 16, int maxLevels = 0) {
    int levels = 1;  // Start with level 0 (original size)
    int w = width;
    int h = height;

    while (w > minSize && h > minSize && (maxLevels == 0 || levels < maxLevels)) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        levels++;

        if (levels >= MAX_MIP_LEVELS) break;
    }

    return levels;
}

// Calculate MIP chain configuration
// radius: blur radius parameter (default 50.0 as baseline)
// Dynamic levels: calculates until min dimension < 16px
inline MipChainConfig CalculateMipChain(int baseWidth, int baseHeight, int levels, float radius = 50.0f) {
    MipChainConfig config = {};
    config.levelCount = levels;

    // Optimal Kawase offsets (fitted to approximate Gaussian distribution)
    // These values are based on research from ARM's SIGGRAPH 2015 presentation
    const float kawaseOffsets[] = {
        0.0f,   // Level 0 (prefilter, not used for blur)
        1.0f,   // Level 1
        1.0f,   // Level 2
        2.0f,   // Level 3
        2.0f,   // Level 4
        3.0f,   // Level 5
        3.0f,   // Level 6
        3.0f,   // Level 7
        4.0f,   // Level 8
        4.0f,   // Level 9
        4.0f,   // Level 10
        4.0f    // Level 11
    };

    // Scale blur offsets by radius (50.0 as baseline)
    float radiusScale = radius / 50.0f;
    if (radiusScale < 0.1f) radiusScale = 0.1f;  // Minimum scale

    int w = baseWidth;
    int h = baseHeight;

    for (int i = 0; i < levels && i < MAX_MIP_LEVELS; ++i) {
        config.widths[i] = w;
        config.heights[i] = h;
        float baseOffset = (i < 12) ? kawaseOffsets[i] : 4.0f;
        config.blurOffsets[i] = baseOffset * radiusScale;

        // Halve dimensions for next level
        w = (w + 1) / 2;  // Round up
        h = (h + 1) / 2;

        // Minimum size
        if (w < 1) w = 1;
        if (h < 1) h = 1;
    }

    return config;
}

// Render parameters collected from AE params
struct RenderParams {
    // From UI
    float   intensity;
    float   radius;
    float   threshold;
    float   softKnee;

    int     quality;            // BlurQuality enum value
    bool    fractionalBlend;

    float   glowColor[3];
    float   colorTemp;
    float   preserveColor;

    float   anamorphic;
    float   anamorphicAngle;
    int     compositeMode;      // CompositeMode enum value
    bool    hdrMode;
    float   falloff;            // Light falloff (0-1, controls distance decay)

    // Image info
    int     width;
    int     height;
    int     srcPitch;
    int     dstPitch;

    // Computed
    int     mipLevels;
    float   fractionalAmount;
    MipChainConfig mipChain;
};

// Fill GlowParams constant buffer from RenderParams
inline void FillGlowParams(GlowParams& cb, const RenderParams& rp) {
    cb.width = rp.width;
    cb.height = rp.height;
    cb.srcPitch = rp.srcPitch;
    cb.dstPitch = rp.dstPitch;

    cb.threshold = rp.threshold / 100.0f;       // Convert from percentage
    cb.softKnee = rp.softKnee / 100.0f;
    cb.intensity = rp.intensity / 100.0f;

    cb.glowColorR = rp.glowColor[0];
    cb.glowColorG = rp.glowColor[1];
    cb.glowColorB = rp.glowColor[2];
    cb.preserveColor = rp.preserveColor / 100.0f;

    // Color temperature to RGB (approximate Planckian locus)
    float t = rp.colorTemp / 100.0f;  // Normalize to -1..1
    if (t >= 0) {
        // Warm: boost red, reduce blue
        cb.colorTempR = 1.0f + t * 0.3f;
        cb.colorTempG = 1.0f;
        cb.colorTempB = 1.0f - t * 0.3f;
    } else {
        // Cool: reduce red, boost blue
        cb.colorTempR = 1.0f + t * 0.3f;
        cb.colorTempG = 1.0f;
        cb.colorTempB = 1.0f - t * 0.3f;
    }

    // Anamorphic stretch
    float anamount = rp.anamorphic / 100.0f;
    float angleRad = rp.anamorphicAngle * 3.14159265f / 180.0f;
    cb.anamorphicScaleX = 1.0f + anamount * std::cos(angleRad);
    cb.anamorphicScaleY = 1.0f + anamount * std::sin(angleRad);
    cb.anamorphicAngle = angleRad;
    cb.anamorphicAmount = anamount;

    cb.compositeMode = rp.compositeMode;
    cb.useHDR = rp.hdrMode ? 1 : 0;

    cb.texelSizeX = 1.0f / static_cast<float>(rp.width);
    cb.texelSizeY = 1.0f / static_cast<float>(rp.height);
}

// Fill BlurPassParams for a specific pass
inline void FillBlurPassParams(
    BlurPassParams& cb,
    const MipChainConfig& mipChain,
    int passIndex,
    bool isDownsample,
    float fractionalBlend = 0.0f)
{
    int srcLevel = isDownsample ? passIndex : passIndex + 1;
    int dstLevel = isDownsample ? passIndex + 1 : passIndex;

    // Clamp to valid range
    srcLevel = (srcLevel < mipChain.levelCount) ? srcLevel : mipChain.levelCount - 1;
    dstLevel = (dstLevel < mipChain.levelCount) ? dstLevel : mipChain.levelCount - 1;

    cb.srcWidth = mipChain.widths[srcLevel];
    cb.srcHeight = mipChain.heights[srcLevel];
    cb.dstWidth = mipChain.widths[dstLevel];
    cb.dstHeight = mipChain.heights[dstLevel];

    // Pitch calculation (assuming RGBA float)
    cb.srcPitch = cb.srcWidth;
    cb.dstPitch = cb.dstWidth;

    cb.srcTexelX = 1.0f / static_cast<float>(cb.srcWidth);
    cb.srcTexelY = 1.0f / static_cast<float>(cb.srcHeight);
    cb.dstTexelX = 1.0f / static_cast<float>(cb.dstWidth);
    cb.dstTexelY = 1.0f / static_cast<float>(cb.dstHeight);

    cb.blurOffset = mipChain.blurOffsets[srcLevel];
    cb.fractionalBlend = fractionalBlend;
    cb.passIndex = passIndex;
    cb.totalPasses = mipChain.levelCount;
}

#endif // __cplusplus

#endif // JUSTGLOW_PARAMS_H
