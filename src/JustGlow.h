/**
 * JustGlow - High Quality GPU Glow Effect for After Effects
 *
 * Based on Dual Kawase Blur with modern enhancements:
 * - 13-Tap Downsample (Call of Duty method)
 * - Karis Average (HDR Firefly prevention)
 * - 9-Tap Tent Upsample
 * - Soft Knee Threshold
 * - Fractional Pass Interpolation
 * - Anamorphic Stretch
 * - Color Temperature Control
 */

#pragma once
#ifndef JUSTGLOW_H
#define JUSTGLOW_H

// AE SDK Headers
#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHelper.h"
#include "String_Utils.h"
#include "Param_Utils.h"

#ifdef _WIN32
    #define HAS_DIRECTX 1
    #define HAS_CUDA 1
    #define HAS_METAL 0
#else
    #define HAS_DIRECTX 0
    #define HAS_CUDA 0
    #define HAS_METAL 1
#endif

// ============================================================================
// Plugin Metadata
// ============================================================================

#define PLUGIN_NAME             "JustGlow"
#define PLUGIN_MATCH_NAME       "com.justglow.effect"
#define PLUGIN_CATEGORY         "Stylize"
#define PLUGIN_DESCRIPTION      "High-quality GPU Glow Effect with Dual Kawase Blur"

#define MAJOR_VERSION           1
#define MINOR_VERSION           0
#define BUG_VERSION             0
#define STAGE_VERSION           PF_Stage_DEVELOP
#define BUILD_VERSION           1

// ============================================================================
// Parameter Definitions
// ============================================================================

enum ParamID {
    PARAM_INPUT = 0,

    // === Basic Parameters ===
    PARAM_INTENSITY,            // Glow intensity (0-200%)
    PARAM_RADIUS,               // Blur radius in pixels (0-500)
    PARAM_THRESHOLD,            // Brightness threshold (0-100%)
    PARAM_SOFT_KNEE,            // Soft knee width (0-100%)

    // === Blur Options ===
    PARAM_QUALITY,              // Quality level (Low/Medium/High/Ultra)
    PARAM_FRACTIONAL_BLEND,     // Enable fractional pass interpolation

    // === Color Options ===
    PARAM_GLOW_COLOR,           // Glow tint color
    PARAM_COLOR_TEMP,           // Color temperature shift (-100 to +100)
    PARAM_PRESERVE_COLOR,       // Original color preservation (0-100%)

    // === Advanced Options ===
    PARAM_ANAMORPHIC,           // Anamorphic stretch amount (0-100%)
    PARAM_ANAMORPHIC_ANGLE,     // Anamorphic direction angle
    PARAM_COMPOSITE_MODE,       // Composite mode (Add/Screen/Overlay)
    PARAM_HDR_MODE,             // Enable Karis Average for HDR

    PARAM_COUNT
};

// Parameter disk IDs (for saving/loading)
enum ParamDiskID {
    DISK_ID_INTENSITY = 1,
    DISK_ID_RADIUS,
    DISK_ID_THRESHOLD,
    DISK_ID_SOFT_KNEE,
    DISK_ID_QUALITY,
    DISK_ID_FRACTIONAL_BLEND,
    DISK_ID_GLOW_COLOR,
    DISK_ID_COLOR_TEMP,
    DISK_ID_PRESERVE_COLOR,
    DISK_ID_ANAMORPHIC,
    DISK_ID_ANAMORPHIC_ANGLE,
    DISK_ID_COMPOSITE_MODE,
    DISK_ID_HDR_MODE
};

// ============================================================================
// Enumerations
// ============================================================================

// Blur quality levels (determines MIP chain depth)
enum class BlurQuality : int {
    Low = 1,        // 3 levels - fastest, lower quality
    Medium = 2,     // 4 levels - balanced
    High = 3,       // 5 levels - high quality (default)
    Ultra = 4       // 6 levels - maximum quality
};

// Composite blend modes
enum class CompositeMode : int {
    Add = 1,        // Additive blending (brightest)
    Screen = 2,     // Screen blending (natural)
    Overlay = 3     // Overlay blending (contrast)
};

// ============================================================================
// Parameter Defaults & Ranges
// ============================================================================

namespace Defaults {
    // Basic
    constexpr float Intensity       = 100.0f;   // 100%
    constexpr float Radius          = 50.0f;    // 50 pixels
    constexpr float Threshold       = 50.0f;    // 50%
    constexpr float SoftKnee        = 50.0f;    // 50%

    // Blur
    constexpr int   Quality         = static_cast<int>(BlurQuality::High);
    constexpr bool  FractionalBlend = true;

    // Color
    constexpr float ColorTemp       = 0.0f;     // Neutral
    constexpr float PreserveColor   = 100.0f;   // 100%

    // Advanced
    constexpr float Anamorphic      = 0.0f;     // Disabled
    constexpr float AnamorphicAngle = 0.0f;     // Horizontal
    constexpr int   CompositeMode   = static_cast<int>(CompositeMode::Add);
    constexpr bool  HDRMode         = true;
}

namespace Ranges {
    // Intensity
    constexpr float IntensityMin    = 0.0f;
    constexpr float IntensityMax    = 200.0f;

    // Radius
    constexpr float RadiusMin       = 0.0f;
    constexpr float RadiusMax       = 500.0f;

    // Threshold & Soft Knee
    constexpr float ThresholdMin    = 0.0f;
    constexpr float ThresholdMax    = 100.0f;

    // Color Temperature
    constexpr float ColorTempMin    = -100.0f;
    constexpr float ColorTempMax    = 100.0f;

    // Anamorphic
    constexpr float AnamorphicMin   = 0.0f;
    constexpr float AnamorphicMax   = 100.0f;

    // Anamorphic Angle
    constexpr float AngleMin        = -90.0f;
    constexpr float AngleMax        = 90.0f;
}

// ============================================================================
// GPU Data Structures
// ============================================================================

// GPU framework type
enum class GPUFrameworkType {
    None = 0,
    DirectX = 1,
    CUDA = 2
};

// GPU-specific data stored per device
struct JustGlowGPUData {
    void* renderer;             // Pointer to renderer (DirectX or CUDA)
    GPUFrameworkType framework;
    bool initialized;
};

// Pre-render data passed to SmartRender
struct JustGlowPreRenderData {
    // Extracted parameters
    float intensity;
    float radius;
    float threshold;
    float softKnee;

    BlurQuality quality;
    bool fractionalBlend;

    float glowColorR, glowColorG, glowColorB;
    float colorTemp;
    float preserveColor;

    float anamorphic;
    float anamorphicAngle;
    CompositeMode compositeMode;
    bool hdrMode;

    // Computed values
    int mipLevels;
    float fractionalAmount;
};

// ============================================================================
// Entry Point
// ============================================================================

extern "C" {
    DllExport PF_Err PluginDataEntryFunction(
        PF_PluginDataPtr    inPtr,
        PF_PluginDataCB     inPluginDataCallBackPtr,
        SPBasicSuite*       inSPBasicSuitePtr,
        const char*         inHostName,
        const char*         inHostVersion);

    DllExport PF_Err EffectMain(
        PF_Cmd              cmd,
        PF_InData*          in_data,
        PF_OutData*         out_data,
        PF_ParamDef*        params[],
        PF_LayerDef*        output,
        void*               extra);
}

// ============================================================================
// Command Handlers
// ============================================================================

PF_Err About(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output);

PF_Err GlobalSetup(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output);

PF_Err GlobalSetdown(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output);

PF_Err ParamsSetup(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output);

PF_Err GPUDeviceSetup(
    PF_InData*              in_data,
    PF_OutData*             out_data,
    PF_GPUDeviceSetupExtra* extra);

PF_Err GPUDeviceSetdown(
    PF_InData*                  in_data,
    PF_OutData*                 out_data,
    PF_GPUDeviceSetdownExtra*   extra);

PF_Err PreRender(
    PF_InData*          in_data,
    PF_OutData*         out_data,
    PF_PreRenderExtra*  extra);

PF_Err SmartRender(
    PF_InData*              in_data,
    PF_OutData*             out_data,
    PF_SmartRenderExtra*    extra,
    bool                    isGPU);

// ============================================================================
// Utility Functions
// ============================================================================

// Calculate MIP levels from radius and quality
int CalculateMipLevels(float radius, BlurQuality quality);

// Calculate fractional blend amount for smooth radius transitions
float CalculateFractionalAmount(float radius, int mipLevels);

// Get quality level count (number of MIP levels)
inline int GetQualityLevelCount(BlurQuality quality) {
    return static_cast<int>(quality) + 2; // Low=3, Medium=4, High=5, Ultra=6
}

// Color temperature to RGB multipliers
void ColorTempToRGB(float temp, float& rMult, float& gMult, float& bMult);

#endif // JUSTGLOW_H
