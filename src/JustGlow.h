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
#define MINOR_VERSION           6
#define BUG_VERSION             0
#define STAGE_VERSION           PF_Stage_DEVELOP
#define BUILD_VERSION           1

// ============================================================================
// Parameter Definitions
// ============================================================================

enum ParamID {
    PARAM_INPUT = 0,

    // === Core Parameters (New Order) ===
    PARAM_INTENSITY,            // Level 1 starting weight (0-100%) → core concentration
    PARAM_EXPOSURE,             // Brightness multiplier (0-50) → final glow brightness
    PARAM_RADIUS,               // Glow reach distance (0-200) → controls active MIP levels
    PARAM_SPREAD_DOWN,          // Downsample spread (-10 to 10) → added to offset at max MIP level
    PARAM_SPREAD_UP,            // Upsample spread (-10 to 10) → added to offset at max MIP level
    PARAM_OFFSET_DOWN,          // Downsample base offset (0-10) → base sampling distance
    PARAM_OFFSET_UP,            // Upsample base offset (0-10) → base sampling distance
    PARAM_OFFSET_PREFILTER,     // Prefilter offset (0-10) → prefilter sampling distance
    PARAM_PREFILTER_QUALITY,    // Prefilter quality (13-tap/25-tap/Sep5/Sep9)
    PARAM_FALLOFF,              // Decay rate per level (0-100) → weight decay
    PARAM_THRESHOLD,            // Brightness threshold (0-100%)
    PARAM_SOFT_KNEE,            // Soft knee width (0-100%)

    // === Blur Options ===
    PARAM_QUALITY,              // Quality level (Low/Medium/High/Ultra)
    PARAM_FALLOFF_TYPE,         // Decay curve type (Exponential/InverseSquare/Linear)
    PARAM_BLUR_MODE,            // Upsample blur mode (3x3/5x5 Gaussian)

    // === Color Options ===
    PARAM_GLOW_COLOR,           // Glow tint color
    PARAM_COLOR_TEMP,           // Color temperature shift (-100 to +100)
    PARAM_PRESERVE_COLOR,       // Original color preservation (0-100%)

    // === Advanced Options ===
    PARAM_ANAMORPHIC,           // Anamorphic stretch amount (0-100%)
    PARAM_ANAMORPHIC_ANGLE,     // Anamorphic direction angle
    PARAM_CHROMATIC_ABERRATION, // Chromatic aberration amount (0-100)
    PARAM_CA_TINT_R,            // CA Red channel tint color
    PARAM_CA_TINT_B,            // CA Blue channel tint color
    PARAM_COMPOSITE_MODE,       // Composite mode (Add/Screen/Overlay)
    PARAM_HDR_MODE,             // Enable Karis Average for HDR
    PARAM_LINEARIZE,            // Enable Linear conversion
    PARAM_INPUT_PROFILE,        // Input color profile (sRGB/Rec709/Gamma2.2)
    PARAM_DITHER,               // Dithering amount (0-100%) for banding prevention

    // === Debug Options ===
    PARAM_DEBUG_VIEW,           // Debug view mode (Final/Prefilter/Down0-6/Up0-6/GlowOnly)
    PARAM_SOURCE_OPACITY,       // Source layer opacity (0-100%)
    PARAM_GLOW_OPACITY,         // Glow opacity (0-200%)
    PARAM_PADDING_THRESHOLD,    // Padding clipping threshold (0-1%)

    PARAM_COUNT
};

// Parameter disk IDs (for saving/loading)
enum ParamDiskID {
    DISK_ID_INTENSITY = 1,
    DISK_ID_EXPOSURE,
    DISK_ID_RADIUS,
    DISK_ID_SPREAD_DOWN,
    DISK_ID_SPREAD_UP,
    DISK_ID_OFFSET_DOWN,
    DISK_ID_OFFSET_UP,
    DISK_ID_OFFSET_PREFILTER,
    DISK_ID_PREFILTER_QUALITY,
    DISK_ID_FALLOFF,
    DISK_ID_THRESHOLD,
    DISK_ID_SOFT_KNEE,
    DISK_ID_QUALITY,
    DISK_ID_FALLOFF_TYPE,
    DISK_ID_BLUR_MODE,
    DISK_ID_GLOW_COLOR,
    DISK_ID_COLOR_TEMP,
    DISK_ID_PRESERVE_COLOR,
    DISK_ID_ANAMORPHIC,
    DISK_ID_ANAMORPHIC_ANGLE,
    DISK_ID_CHROMATIC_ABERRATION,
    DISK_ID_CA_TINT_R,
    DISK_ID_CA_TINT_B,
    DISK_ID_COMPOSITE_MODE,
    DISK_ID_HDR_MODE,
    DISK_ID_LINEARIZE,
    DISK_ID_INPUT_PROFILE,
    DISK_ID_DITHER,
    DISK_ID_DEBUG_VIEW,
    DISK_ID_SOURCE_OPACITY,
    DISK_ID_GLOW_OPACITY,
    DISK_ID_PADDING_THRESHOLD
};

// ============================================================================
// Enumerations
// ============================================================================

// Quality: MIP chain depth (6-12 levels)
// Replaced enum with direct integer for finer control
// 6 = fast, 8 = balanced (default), 12 = maximum quality

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

// Input color profile (gamma curve for linearization)
enum class InputProfile : int {
    sRGB = 1,       // Standard sRGB (gamma ~2.2 with linear toe)
    Rec709 = 2,     // Rec.709 (gamma 2.4)
    Gamma22 = 3     // Pure gamma 2.2
};

// Prefilter quality modes
enum class PrefilterQuality : int {
    Star13 = 1,     // 13-tap star pattern (current, fast)
    Grid25 = 2,     // 25-tap 5x5 discrete Gaussian (quality)
    Sep5 = 3,       // Separable 5+5 tap (10 total, balanced)
    Sep9 = 4        // Separable 9+9 tap (18 total, high quality)
};

// Upsample blur modes
enum class BlurMode : int {
    Gaussian3x3 = 1,  // 9-tap 3x3 Gaussian (fast, default)
    Gaussian5x5 = 2   // 25-tap 5x5 Gaussian (smoother, better diagonal coverage)
};

// Debug view modes for visualizing pipeline stages
// Down0 removed (same as Prefilter, which shows MIP[0])
enum class DebugViewMode : int {
    Final = 1,      // Normal output (source + glow composite)
    Prefilter = 2,  // Prefilter result (MIP[0], threshold applied)
    Down1 = 3,      // Downsample level 1 (MIP[1], half res)
    Down2 = 4,      // Downsample level 2 (MIP[2], quarter res)
    Down3 = 5,      // Downsample level 3
    Down4 = 6,      // Downsample level 4
    Down5 = 7,      // Downsample level 5
    Down6 = 8,      // Downsample level 6
    Up0 = 9,        // Upsample level 0 (full res glow)
    Up1 = 10,       // Upsample level 1
    Up2 = 11,       // Upsample level 2
    Up3 = 12,       // Upsample level 3
    Up4 = 13,       // Upsample level 4
    Up5 = 14,       // Upsample level 5
    Up6 = 15,       // Upsample level 6
    GlowOnly = 16   // Final glow without source
};

// ============================================================================
// Parameter Defaults & Ranges
// ============================================================================

namespace Defaults {
    // Core Parameters (New Order)
    constexpr float Intensity       = 75.0f;    // 75% = balanced core vs spread
    constexpr float Exposure        = 1.0f;     // 1x brightness multiplier
    constexpr float Radius          = 75.0f;    // 75% = balanced glow reach
    constexpr float SpreadDown      = 1.0f;     // 1.0 = minimal blur offset
    constexpr float SpreadUp        = 1.0f;     // 1.0 = minimal blur offset
    constexpr float OffsetDown      = 1.0f;     // 1.0 = standard base offset
    constexpr float OffsetUp        = 1.0f;     // 1.0 = standard base offset
    constexpr float OffsetPrefilter = 1.0f;     // 1.0 = standard prefilter offset
    constexpr int   PrefilterQuality = static_cast<int>(::PrefilterQuality::Star13);  // Fast default
    constexpr float Falloff         = 50.0f;    // 50% = neutral (0%=boost outer, 100%=decay)
    constexpr float Threshold       = 25.0f;    // 25% - lower threshold for more glow
    constexpr float SoftKnee        = 75.0f;    // 75% - softer threshold transition

    // Blur Options
    constexpr int   Quality         = 8;  // MIP levels (6-12), 8 = balanced default
    constexpr int   FalloffType     = static_cast<int>(::FalloffType::Exponential);
    constexpr int   BlurMode        = static_cast<int>(::BlurMode::Gaussian3x3);  // Fast default

    // Color
    constexpr float ColorTemp       = 0.0f;     // Neutral
    constexpr float PreserveColor   = 100.0f;   // 100%

    // Advanced
    constexpr float Anamorphic      = 0.0f;     // Disabled
    constexpr float AnamorphicAngle = 0.0f;     // Horizontal
    constexpr float ChromaticAberration = 0.0f; // Disabled
    constexpr int   CompositeMode   = static_cast<int>(::CompositeMode::Add);
    constexpr bool  HDRMode         = true;
    constexpr bool  Linearize       = false;  // OFF by default (simpler, no alpha issues)
    constexpr int   InputProfile    = static_cast<int>(::InputProfile::sRGB);
    constexpr float Dither          = 50.0f;  // 50% dithering for banding prevention

    // Debug
    constexpr int   DebugView       = static_cast<int>(DebugViewMode::Final);
    constexpr float SourceOpacity   = 100.0f;   // 100% = full source visibility
    constexpr float GlowOpacity     = 100.0f;   // 100% = normal glow, up to 200%
    constexpr float PaddingThreshold = 0.3f;    // 0.3% = clip very dark values for padding optimization
}

namespace Ranges {
    // Core Parameters (New Order)
    constexpr float IntensityMin    = 0.0f;
    constexpr float IntensityMax    = 100.0f;   // Level 1 weight: 0% = spread, 100% = core

    constexpr float ExposureMin     = 0.0f;
    constexpr float ExposureMax     = 50.0f;    // Linear: 0 = no glow, 50 = 50x

    constexpr float RadiusMin       = 0.0f;
    constexpr float RadiusMax       = 100.0f;   // 100% = all MIP levels active

    constexpr float SpreadMin       = -10.0f;
    constexpr float SpreadMax       = 10.0f;    // Spread at max MIP level (-10 to 10)

    constexpr float OffsetMin       = 0.0f;
    constexpr float OffsetMax       = 10.0f;    // Base offset (0-10)

    constexpr float FalloffMin      = 0.0f;
    constexpr float FalloffMax      = 100.0f;   // 0=boost, 50=neutral, 100=decay

    // Quality (MIP levels)
    constexpr int   QualityMin      = 6;
    constexpr int   QualityMax      = 12;

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

    // Chromatic Aberration
    constexpr float ChromaticAberrationMin = 0.0f;
    constexpr float ChromaticAberrationMax = 100.0f;

    // Dither
    constexpr float DitherMin       = 0.0f;
    constexpr float DitherMax       = 100.0f;

    // Debug: Glow Opacity
    constexpr float GlowOpacityMin  = 0.0f;
    constexpr float GlowOpacityMax  = 200.0f;   // Up to 200% for boosted glow

    // Debug: Padding Threshold
    constexpr float PaddingThresholdMin = 0.0f;
    constexpr float PaddingThresholdMax = 1.0f;  // 0-1% (very small values)
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
    // Core Parameters (New Order)
    float intensity;    // 0-100: Level 1 starting weight (core concentration)
    float exposure;     // 0-50: Brightness multiplier (final glow brightness)
    float radius;       // 0-200: controls active MIP levels
    float spreadDown;   // -10 to 10: downsample spread at max MIP level
    float spreadUp;     // -10 to 10: upsample spread at max MIP level
    float offsetDown;   // 0-10: downsample base offset
    float offsetUp;     // 0-10: upsample base offset
    float offsetPrefilter; // 0-10: prefilter sampling offset
    PrefilterQuality prefilterQuality;  // Prefilter blur quality mode
    float falloff;      // 0-100: decay rate per level
    float threshold;    // 0-100: brightness threshold
    float softKnee;     // 0-100: soft knee width

    // Blur Options
    int quality;  // MIP levels (6-12)
    FalloffType falloffType;
    BlurMode blurMode;  // Upsample Gaussian kernel size (3x3/5x5)

    // Color
    float glowColorR, glowColorG, glowColorB;
    float colorTemp;
    float preserveColor;

    // Advanced
    float anamorphic;
    float anamorphicAngle;
    float chromaticAberration;
    float caTintR[3];           // CA Red channel tint (RGB)
    float caTintB[3];           // CA Blue channel tint (RGB)
    CompositeMode compositeMode;
    bool hdrMode;
    bool linearize;         // Enable sRGB to Linear conversion
    InputProfile inputProfile;  // Input color profile (sRGB/Rec709/Gamma2.2)
    float dither;           // Dithering amount (0-100%)

    // Debug
    DebugViewMode debugView;
    float sourceOpacity;        // 0-100%
    float glowOpacity;          // 0-200%
    float paddingThreshold;     // 0-1%: clip dark values for padding optimization

    // Computed values
    int mipLevels;          // Based on quality setting
    float activeLimit;      // Radius mapped to MIP level limit
    float blurOffsets[PRERENDER_MAX_MIP_LEVELS]; // Spread -> per-level offset (scaled by downsample)
    float decayK;           // Falloff value (0-100, 50=neutral)
    float level1Weight;     // Intensity mapped to Level 1 starting weight (0-1)
    float downsampleFactor; // Preview resolution factor (1.0=full, 0.5=half, etc.)
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
// Quality is now a direct integer (6-12)
inline int GetQualityLevelCount(int quality) {
    // Clamp to valid range
    if (quality < Ranges::QualityMin) return Ranges::QualityMin;
    if (quality > Ranges::QualityMax) return Ranges::QualityMax;
    return quality;
}

// Color temperature to RGB multipliers
void ColorTempToRGB(float temp, float& rMult, float& gMult, float& bMult);

#endif // JUSTGLOW_H
