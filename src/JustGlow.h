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

// Forward declaration for MAX_MIP_LEVELS
constexpr int PRERENDER_MAX_MIP_LEVELS = 12;

// GPU framework availability (can be overridden by CMake)
#ifdef _WIN32
    #ifndef HAS_DIRECTX
        #define HAS_DIRECTX 1
    #endif
    #ifndef HAS_CUDA
        #define HAS_CUDA 0  // Set by CMake if CUDA toolkit found
    #endif
    #ifndef HAS_METAL
        #define HAS_METAL 0
    #endif
#else
    #ifndef HAS_DIRECTX
        #define HAS_DIRECTX 0
    #endif
    #ifndef HAS_CUDA
        #define HAS_CUDA 0
    #endif
    #ifndef HAS_METAL
        #define HAS_METAL 1
    #endif
#endif

// ============================================================================
// Plugin Metadata
// ============================================================================

#define PLUGIN_NAME             "JustGlow"
#define PLUGIN_MATCH_NAME       "com.justglow.effect"
#define PLUGIN_CATEGORY         "Stylize"
#define PLUGIN_DESCRIPTION      "High-quality GPU Glow Effect with Dual Kawase Blur"

#define MAJOR_VERSION           1
#define MINOR_VERSION           3
#define BUG_VERSION             0
#define STAGE_VERSION           PF_Stage_DEVELOP
#define BUILD_VERSION           1

// ============================================================================
// Parameter Definitions
// ============================================================================

enum ParamID {
    PARAM_INPUT = 0,

    // === Core 4 Parameters (The Big Four) ===
    PARAM_RADIUS,               // Glow reach distance (0-100) → controls active MIP levels
    PARAM_SPREAD,               // Blur softness (0-100) → controls blur offset (max 3.5px)
    PARAM_FALLOFF,              // Decay slope (0-100) → controls exponential decay k
    PARAM_INTENSITY,            // Glow power (0-10) → HDR exposure pow(2, intensity)

    // === Threshold ===
    PARAM_THRESHOLD,            // Brightness threshold (0-100%)
    PARAM_SOFT_KNEE,            // Soft knee width (0-100%)

    // === Blur Options ===
    PARAM_QUALITY,              // Quality level (Low/Medium/High/Ultra)
    PARAM_FALLOFF_TYPE,         // Decay curve type (Exponential/InverseSquare/Linear)

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
    DISK_ID_RADIUS = 1,
    DISK_ID_SPREAD,
    DISK_ID_FALLOFF,
    DISK_ID_INTENSITY,
    DISK_ID_THRESHOLD,
    DISK_ID_SOFT_KNEE,
    DISK_ID_QUALITY,
    DISK_ID_FALLOFF_TYPE,
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
    Low = 1,        // 4 levels - fastest
    Medium = 2,     // 6 levels - balanced
    High = 3,       // 8 levels - high quality (default)
    Ultra = 4       // 12 levels - maximum quality (Deep Glow feel)
};

// Falloff curve types (decay models)
enum class FalloffType : int {
    Exponential = 1,    // pow(0.5, i*k) - Deep Glow standard, balanced
    InverseSquare = 2,  // 1/(x^2+1) - Realistic VFX, sharp core + long tail
    Linear = 3          // 1-x*0.1 - Soft/foggy, dreamy feel
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
    // Core 4 Parameters
    constexpr float Radius          = 100.0f;   // 100% = all MIP levels active
    constexpr float Spread          = 50.0f;    // 50% = 2.25px blur offset (balanced)
    constexpr float Falloff         = 50.0f;    // 50% = k=1.6 (balanced decay)
    constexpr float Intensity       = 1.0f;     // 1.0 = 2x brightness (pow(2,1))

    // Threshold
    constexpr float Threshold       = 70.0f;    // 70% - visible effect on application
    constexpr float SoftKnee        = 50.0f;    // 50%

    // Blur Options
    constexpr int   Quality         = static_cast<int>(BlurQuality::High);
    constexpr int   FalloffType     = static_cast<int>(FalloffType::Exponential);

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
    // Core 4 Parameters
    constexpr float RadiusMin       = 0.0f;
    constexpr float RadiusMax       = 200.0f;   // Extended range for wider glow

    constexpr float SpreadMin       = 0.0f;
    constexpr float SpreadMax       = 100.0f;   // Maps to 1.0-3.5px offset

    constexpr float FalloffMin      = 0.0f;
    constexpr float FalloffMax      = 100.0f;   // Maps to k=0.2-3.0

    constexpr float IntensityMin    = 0.0f;
    constexpr float IntensityMax    = 10.0f;    // pow(2, 10) = 1024x max

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
    // Core 4 Parameters
    float radius;       // 0-100: controls active MIP levels
    float spread;       // 0-100: controls blur offset (1.0-3.5px)
    float falloff;      // 0-100: controls decay k (0.2-3.0)
    float intensity;    // 0-10: HDR exposure pow(2, intensity)

    // Threshold
    float threshold;
    float softKnee;

    // Blur Options
    BlurQuality quality;
    FalloffType falloffType;

    // Color
    float glowColorR, glowColorG, glowColorB;
    float colorTemp;
    float preserveColor;

    // Advanced
    float anamorphic;
    float anamorphicAngle;
    CompositeMode compositeMode;
    bool hdrMode;

    // Computed values
    int mipLevels;          // Based on quality setting
    float activeLimit;      // Radius mapped to MIP level limit
    float blurOffsets[PRERENDER_MAX_MIP_LEVELS]; // Spread -> per-level offset (decays to 1.5px)
    float decayK;           // Falloff mapped to decay constant (0.2-3.0)
    float exposure;         // Intensity mapped to pow(2, intensity)
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

// Get quality level count (number of MIP levels)
// New system: Quality determines MIP depth, not Radius
inline int GetQualityLevelCount(BlurQuality quality) {
    switch (quality) {
        case BlurQuality::Low:    return 4;   // Fast
        case BlurQuality::Medium: return 6;   // Balanced
        case BlurQuality::High:   return 8;   // High quality
        case BlurQuality::Ultra:  return 12;  // Deep Glow feel
        default:                  return 8;
    }
}

// Color temperature to RGB multipliers
void ColorTempToRGB(float temp, float& rMult, float& gMult, float& bMult);

#endif // JUSTGLOW_H
