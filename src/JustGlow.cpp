/**
 * JustGlow - High Quality GPU Glow Effect for After Effects
 *
 * Main plugin implementation file.
 */

#include "JustGlow.h"
#include "JustGlowParams.h"
#include <AE_Macros.h>
#include <Util/Param_Utils.h>
#include <algorithm>  // for std::max, std::min

#if HAS_DIRECTX
#include "JustGlowGPURenderer.h"
#endif

#if HAS_CUDA
#include "JustGlowCUDARenderer.h"
#include <cuda.h>
#endif

#if HAS_DIRECTX || HAS_CUDA
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdarg>

// ============================================================================
// Debug Logging (shared with GPU renderer)
// ============================================================================

static std::wstring GetLogFilePath() {
    wchar_t tempPath[MAX_PATH];
    GetTempPathW(MAX_PATH, tempPath);
    return std::wstring(tempPath) + L"JustGlow_debug.log";
}

static void LogMsg(const char* format, ...) {
    static std::ofstream logFile;
    static bool initialized = false;

    if (!initialized) {
        logFile.open(GetLogFilePath(), std::ios::out | std::ios::trunc);  // Clear on each AE session
        initialized = true;
        if (logFile.is_open()) {
            logFile << "\n========== JustGlow Plugin Loaded ==========\n";
            logFile.flush();
        }
    }

    if (logFile.is_open()) {
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        logFile << std::put_time(&tm, "[%H:%M:%S] ");

        char buffer[1024];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);

        logFile << buffer << std::endl;
        logFile.flush();
    }
}

#define PLUGIN_LOG(fmt, ...) LogMsg(fmt, ##__VA_ARGS__)
#else
#define PLUGIN_LOG(fmt, ...) ((void)0)
#endif // HAS_DIRECTX || HAS_CUDA

// ============================================================================
// Plugin Data Entry
// ============================================================================

PF_Err PluginDataEntryFunction(
    PF_PluginDataPtr        inPtr,
    PF_PluginDataCB         inPluginDataCallBackPtr,
    SPBasicSuite*           inSPBasicSuitePtr,
    const char*             inHostName,
    const char*             inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;

    result = PF_REGISTER_EFFECT(
        inPtr,
        inPluginDataCallBackPtr,
        PLUGIN_NAME,
        PLUGIN_MATCH_NAME,
        PLUGIN_CATEGORY,
        AE_RESERVED_INFO);

    return result;
}

// ============================================================================
// Main Entry Point
// ============================================================================

PF_Err EffectMain(
    PF_Cmd          cmd,
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output,
    void*           extra)
{
    PF_Err err = PF_Err_NONE;

    try {
        switch (cmd) {
            case PF_Cmd_ABOUT:
                err = About(in_data, out_data, params, output);
                break;

            case PF_Cmd_GLOBAL_SETUP:
                err = GlobalSetup(in_data, out_data, params, output);
                break;

            case PF_Cmd_GLOBAL_SETDOWN:
                err = GlobalSetdown(in_data, out_data, params, output);
                break;

            case PF_Cmd_PARAMS_SETUP:
                err = ParamsSetup(in_data, out_data, params, output);
                break;

            case PF_Cmd_GPU_DEVICE_SETUP:
                err = GPUDeviceSetup(in_data, out_data,
                    reinterpret_cast<PF_GPUDeviceSetupExtra*>(extra));
                break;

            case PF_Cmd_GPU_DEVICE_SETDOWN:
                err = GPUDeviceSetdown(in_data, out_data,
                    reinterpret_cast<PF_GPUDeviceSetdownExtra*>(extra));
                break;

            case PF_Cmd_SMART_PRE_RENDER:
                err = PreRender(in_data, out_data,
                    reinterpret_cast<PF_PreRenderExtra*>(extra));
                break;

            case PF_Cmd_SMART_RENDER:
                err = SmartRender(in_data, out_data,
                    reinterpret_cast<PF_SmartRenderExtra*>(extra), false);
                break;

            case PF_Cmd_SMART_RENDER_GPU:
                err = SmartRender(in_data, out_data,
                    reinterpret_cast<PF_SmartRenderExtra*>(extra), true);
                break;

            default:
                break;
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}

// ============================================================================
// About
// ============================================================================

PF_Err About(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output)
{
    PF_SPRINTF(out_data->return_msg,
        "%s v%d.%d.%d\r\n"
        "%s\r\n\r\n"
        "High-quality GPU glow effect using Dual Kawase blur\r\n"
        "with 13-tap downsample and Karis average for HDR.",
        PLUGIN_NAME,
        MAJOR_VERSION,
        MINOR_VERSION,
        BUG_VERSION,
        PLUGIN_DESCRIPTION);

    return PF_Err_NONE;
}

// ============================================================================
// Global Setup
// ============================================================================

PF_Err GlobalSetup(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output)
{
    PF_Err err = PF_Err_NONE;

    PLUGIN_LOG("=== GlobalSetup ===");

    // Set plugin version
    out_data->my_version = PF_VERSION(
        MAJOR_VERSION,
        MINOR_VERSION,
        BUG_VERSION,
        STAGE_VERSION,
        BUILD_VERSION);

    // Output flags
    out_data->out_flags =
        PF_OutFlag_DEEP_COLOR_AWARE |           // 16bpc support
        PF_OutFlag_PIX_INDEPENDENT |            // Pixels processed independently
        PF_OutFlag_I_EXPAND_BUFFER |            // May need larger output
        PF_OutFlag_SEND_UPDATE_PARAMS_UI;       // Update UI on param change

    // Output flags 2 - GPU support
    out_data->out_flags2 =
        PF_OutFlag2_FLOAT_COLOR_AWARE |         // 32bpc float support
        PF_OutFlag2_SUPPORTS_SMART_RENDER |     // SmartFX support
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING | // Thread-safe
        PF_OutFlag2_REVEALS_ZERO_ALPHA          // Glow extends into transparent areas
#if HAS_CUDA || HAS_DIRECTX
        | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32   // GPU rendering (float32)
#endif
#if HAS_DIRECTX
        | PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING // DirectX 12 support
#endif
        ;

    PLUGIN_LOG("GlobalSetup complete, flags=0x%X, flags2=0x%X", out_data->out_flags, out_data->out_flags2);
    PLUGIN_LOG("  HAS_CUDA=%d, HAS_DIRECTX=%d", HAS_CUDA, HAS_DIRECTX);
    PLUGIN_LOG("  GPU_RENDER_F32=%d, DIRECTX_RENDERING=%d",
        (out_data->out_flags2 & PF_OutFlag2_SUPPORTS_GPU_RENDER_F32) ? 1 : 0,
        (out_data->out_flags2 & PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING) ? 1 : 0);

    return err;
}

// ============================================================================
// Global Setdown
// ============================================================================

PF_Err GlobalSetdown(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output)
{
    // Nothing to clean up at global level
    return PF_Err_NONE;
}

// ============================================================================
// Params Setup
// ============================================================================

PF_Err ParamsSetup(
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    // ===========================================
    // Core Parameters (New Order)
    // ===========================================

    // Intensity (0-100%) - Level 1 starting weight (core concentration)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Intensity",
        Ranges::IntensityMin,
        Ranges::IntensityMax,
        Ranges::IntensityMin,
        Ranges::IntensityMax,
        Defaults::Intensity,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_INTENSITY);

    // Exposure (0-50) - Brightness multiplier
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Exposure",
        Ranges::ExposureMin,
        Ranges::ExposureMax,
        Ranges::ExposureMin,
        Ranges::ExposureMax,
        Defaults::Exposure,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_EXPOSURE);

    // Radius (0-200) - Controls how far glow reaches (active MIP levels)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Radius",
        Ranges::RadiusMin,
        Ranges::RadiusMax,
        Ranges::RadiusMin,
        Ranges::RadiusMax,
        Defaults::Radius,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_RADIUS);

    // Spread Down (1-5) - Downsample offset at max MIP level
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Spread Down",
        Ranges::SpreadMin,
        Ranges::SpreadMax,
        Ranges::SpreadMin,
        Ranges::SpreadMax,
        Defaults::SpreadDown,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_SPREAD_DOWN);

    // Spread Up (0-10) - Upsample offset at max MIP level
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Spread Up",
        Ranges::SpreadMin,
        Ranges::SpreadMax,
        Ranges::SpreadMin,
        Ranges::SpreadMax,
        Defaults::SpreadUp,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_SPREAD_UP);

    // Offset Down (0-3) - Downsample base offset
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Offset Down",
        Ranges::OffsetMin,
        Ranges::OffsetMax,
        Ranges::OffsetMin,
        Ranges::OffsetMax,
        Defaults::OffsetDown,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_OFFSET_DOWN);

    // Offset Up (0-3) - Upsample base offset
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Offset Up",
        Ranges::OffsetMin,
        Ranges::OffsetMax,
        Ranges::OffsetMin,
        Ranges::OffsetMax,
        Defaults::OffsetUp,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_OFFSET_UP);

    // Offset Prefilter (0-10) - Prefilter sampling offset
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Offset Prefilter",
        Ranges::OffsetMin,
        Ranges::OffsetMax,
        Ranges::OffsetMin,
        Ranges::OffsetMax,
        Defaults::OffsetPrefilter,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_OFFSET_PREFILTER);

    // Prefilter Quality - blur algorithm selection
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        "Prefilter Quality",
        4,  // Number of choices
        Defaults::PrefilterQuality,
        "13-tap Star (Fast)|25-tap Grid|Separable 5+5|Separable 9+9 (HQ)",
        DISK_ID_PREFILTER_QUALITY);

    // Falloff (0-100) - Decay rate per level
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Falloff",
        Ranges::FalloffMin,
        Ranges::FalloffMax,
        Ranges::FalloffMin,
        Ranges::FalloffMax,
        Defaults::Falloff,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_FALLOFF);

    // Threshold (0-100%)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Threshold",
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Defaults::Threshold,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_THRESHOLD);

    // Threshold Softness (0-100%)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Threshold Softness",
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Defaults::SoftKnee,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_SOFT_KNEE);

    // ===========================================
    // Blur Options
    // ===========================================

    // Quality (MIP levels, 6-12)
    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER(
        "Quality",
        Ranges::QualityMin,    // 6
        Ranges::QualityMax,    // 12
        Ranges::QualityMin,    // slider min
        Ranges::QualityMax,    // slider max
        Defaults::Quality,     // 8
        DISK_ID_QUALITY);

    // Falloff Type (decay curve shape)
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        "Falloff Type",
        3,  // Number of choices
        Defaults::FalloffType,
        "Exponential|Inverse Square|Linear",
        DISK_ID_FALLOFF_TYPE);

    // Blur Mode (upsample Gaussian kernel size)
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        "Blur Mode",
        2,  // Number of choices
        Defaults::BlurMode,
        "3x3 Gaussian|5x5 Gaussian",
        DISK_ID_BLUR_MODE);

    // ===========================================
    // Color Options
    // ===========================================

    // Glow Color
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(
        "Glow Color",
        255,  // Red
        255,  // Green
        255,  // Blue
        DISK_ID_GLOW_COLOR);

    // Color Temperature (-100 to +100)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Color Temperature",
        Ranges::ColorTempMin,
        Ranges::ColorTempMax,
        Ranges::ColorTempMin,
        Ranges::ColorTempMax,
        Defaults::ColorTemp,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_COLOR_TEMP);

    // Preserve Color (0-100%)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Preserve Color",
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Defaults::PreserveColor,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_PRESERVE_COLOR);

    // Desaturation (0-100%) - Max-based, adds only
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Desaturation",
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Defaults::Desaturation,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_DESATURATION);

    // ===========================================
    // Advanced Options
    // ===========================================

    // Anamorphic (0-100%)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Anamorphic",
        Ranges::AnamorphicMin,
        Ranges::AnamorphicMax,
        Ranges::AnamorphicMin,
        Ranges::AnamorphicMax,
        Defaults::Anamorphic,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_ANAMORPHIC);

    // Anamorphic Angle
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE(
        "Anamorphic Angle",
        static_cast<PF_Fixed>(Defaults::AnamorphicAngle * 65536),
        DISK_ID_ANAMORPHIC_ANGLE);

    // Chromatic Aberration
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Chromatic Aberration",
        Ranges::ChromaticAberrationMin, Ranges::ChromaticAberrationMax,
        Ranges::ChromaticAberrationMin, Ranges::ChromaticAberrationMax,
        Defaults::ChromaticAberration,
        PF_Precision_TENTHS, 0, 0,
        DISK_ID_CHROMATIC_ABERRATION);

    // CA Tint Red (color for R channel shift)
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(
        "CA Tint Red",
        255,  // Red
        0,    // Green
        0,    // Blue
        DISK_ID_CA_TINT_R);

    // CA Tint Blue (color for B channel shift)
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(
        "CA Tint Blue",
        0,    // Red
        0,    // Green
        255,  // Blue
        DISK_ID_CA_TINT_B);

    // Composite Mode
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        "Composite Mode",
        3,  // Number of choices
        Defaults::CompositeMode,
        "Add|Screen|Overlay",
        DISK_ID_COMPOSITE_MODE);

    // HDR Mode
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "HDR Mode",
        "Anti-firefly (Karis Average)",
        Defaults::HDRMode ? 1 : 0,
        0,
        DISK_ID_HDR_MODE);

    // Linearize
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "Linearize",
        "Convert input to Linear (enable when AE Linearize is OFF)",
        Defaults::Linearize ? 1 : 0,
        0,
        DISK_ID_LINEARIZE);

    // Input Profile (appears when Linearize is ON)
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        "Input Profile",
        3,  // Number of choices
        Defaults::InputProfile,
        "sRGB|Rec.709|Gamma 2.2",
        DISK_ID_INPUT_PROFILE);

    // Dither (banding prevention)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Dither",
        Ranges::DitherMin, Ranges::DitherMax,
        Ranges::DitherMin, Ranges::DitherMax,
        Defaults::Dither,
        PF_Precision_TENTHS, 0, 0,
        DISK_ID_DITHER);

    // ===========================================
    // Debug Options
    // ===========================================

    // Debug View (Down0 removed - same as Prefilter)
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        "Debug View",
        16,  // Number of choices (removed Down0)
        Defaults::DebugView,
        "Final|Prefilter|Down 1|Down 2|Down 3|Down 4|Down 5|Down 6|Up 0|Up 1|Up 2|Up 3|Up 4|Up 5|Up 6|Glow Only",
        DISK_ID_DEBUG_VIEW);

    // Source Opacity (0-100%)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Source Opacity",
        Ranges::ThresholdMin,   // 0
        Ranges::ThresholdMax,   // 100
        Ranges::ThresholdMin,
        Ranges::ThresholdMax,
        Defaults::SourceOpacity,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_SOURCE_OPACITY);

    // Glow Opacity (0-200%)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Glow Opacity",
        Ranges::GlowOpacityMin,
        Ranges::GlowOpacityMax,
        Ranges::GlowOpacityMin,
        Ranges::GlowOpacityMax,
        Defaults::GlowOpacity,
        PF_Precision_TENTHS,
        0,
        0,
        DISK_ID_GLOW_OPACITY);

    // Padding Threshold (0-1%) - clip dark values for padding optimization
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Padding Threshold",
        Ranges::PaddingThresholdMin,
        Ranges::PaddingThresholdMax,
        Ranges::PaddingThresholdMin,
        Ranges::PaddingThresholdMax,
        Defaults::PaddingThreshold,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        DISK_ID_PADDING_THRESHOLD);

    // Unpremultiply (for composite testing)
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "Unpremultiply",
        "Unpremultiply glow before composite",
        0,  // Default: OFF
        0,
        DISK_ID_UNPREMULTIPLY);

    out_data->num_params = PARAM_COUNT;

    return err;
}

// ============================================================================
// GPU Device Setup
// ============================================================================

PF_Err GPUDeviceSetup(
    PF_InData*              in_data,
    PF_OutData*             out_data,
    PF_GPUDeviceSetupExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    PLUGIN_LOG("=== GPUDeviceSetup ===");
    PLUGIN_LOG("Requested GPU framework: %d (CUDA=%d, DIRECTX=%d)",
        extra->input->what_gpu, PF_GPU_Framework_CUDA, PF_GPU_Framework_DIRECTX);

#if HAS_CUDA || HAS_DIRECTX
    // Allocate GPU data
    JustGlowGPUData* gpuData = new JustGlowGPUData();
    gpuData->initialized = false;
    gpuData->renderer = nullptr;
    gpuData->framework = GPUFrameworkType::None;
    PLUGIN_LOG("GPU data allocated");

    try {
        // Get device info
        const void* suiteP = nullptr;
        PF_GPUDeviceSuite1* gpuSuite = nullptr;

        err = in_data->pica_basicP->AcquireSuite(
            kPFGPUDeviceSuite,
            kPFGPUDeviceSuiteVersion1,
            &suiteP);

        PLUGIN_LOG("AcquireSuite result: err=%d, suiteP=%p", err, suiteP);

        if (!err && suiteP) {
            gpuSuite = const_cast<PF_GPUDeviceSuite1*>(
                static_cast<const PF_GPUDeviceSuite1*>(suiteP));
            PF_GPUDeviceInfo deviceInfo;
            err = gpuSuite->GetDeviceInfo(
                in_data->effect_ref,
                extra->input->device_index,
                &deviceInfo);

            PLUGIN_LOG("GetDeviceInfo: err=%d, framework=%d, device=%p, context=%p, queue=%p",
                err, deviceInfo.device_framework, deviceInfo.devicePV,
                deviceInfo.contextPV, deviceInfo.command_queuePV);

#if HAS_CUDA
            // Try CUDA first
            if (!err && extra->input->what_gpu == PF_GPU_Framework_CUDA) {
                PLUGIN_LOG("Creating JustGlowCUDARenderer...");
                JustGlowCUDARenderer* renderer = new JustGlowCUDARenderer();

                // Initialize with CUDA context and stream
                PLUGIN_LOG("Initializing CUDA renderer with context=%p, stream=%p",
                    deviceInfo.contextPV, deviceInfo.command_queuePV);

                if (renderer->Initialize(
                    static_cast<CUcontext>(deviceInfo.contextPV),
                    static_cast<CUstream>(deviceInfo.command_queuePV)))
                {
                    gpuData->renderer = renderer;
                    gpuData->framework = GPUFrameworkType::CUDA;
                    gpuData->initialized = true;
                    PLUGIN_LOG("CUDA Renderer initialized successfully!");
                }
                else {
                    PLUGIN_LOG("ERROR: CUDA Renderer initialization failed!");
                    delete renderer;
                    err = PF_Err_OUT_OF_MEMORY;
                }
            }
#endif

#if HAS_DIRECTX
            // Try DirectX if CUDA not requested or not available
            if (!gpuData->initialized && !err &&
                extra->input->what_gpu == PF_GPU_Framework_DIRECTX) {
                PLUGIN_LOG("Creating JustGlowGPURenderer (DirectX)...");
                JustGlowGPURenderer* renderer = new JustGlowGPURenderer();

                // Initialize with DirectX device and command queue
                PLUGIN_LOG("Initializing DirectX renderer...");
                if (renderer->Initialize(
                    static_cast<ID3D12Device*>(deviceInfo.devicePV),
                    static_cast<ID3D12CommandQueue*>(deviceInfo.command_queuePV)))
                {
                    gpuData->renderer = renderer;
                    gpuData->framework = GPUFrameworkType::DirectX;
                    gpuData->initialized = true;
                    PLUGIN_LOG("DirectX Renderer initialized successfully!");
                }
                else {
                    PLUGIN_LOG("ERROR: DirectX Renderer initialization failed!");
                    delete renderer;
                    err = PF_Err_OUT_OF_MEMORY;
                }
            }
#endif

            // If no renderer was initialized, return error
            if (!gpuData->initialized) {
                PLUGIN_LOG("ERROR: No renderer initialized for framework %d", extra->input->what_gpu);
                err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
            }

            in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
        }
    }
    catch (...) {
        PLUGIN_LOG("EXCEPTION in GPUDeviceSetup!");
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    // Store GPU data
    extra->output->gpu_data = gpuData;

    // Set GPU support flags (required at both GLOBAL_SETUP and GPU_DEVICE_SETUP per SDK)
    if (gpuData->initialized && !err) {
        out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
#if HAS_DIRECTX
        if (gpuData->framework == GPUFrameworkType::DirectX) {
            out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;
        }
#endif
        PLUGIN_LOG("GPUDeviceSetup: out_flags2 updated to 0x%X", out_data->out_flags2);
    }

    PLUGIN_LOG("GPUDeviceSetup complete, err=%d, initialized=%d, framework=%d",
        err, gpuData->initialized, static_cast<int>(gpuData->framework));
#else
    PLUGIN_LOG("ERROR: No GPU support compiled!");
    err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
#endif

    return err;
}

// ============================================================================
// GPU Device Setdown
// ============================================================================

PF_Err GPUDeviceSetdown(
    PF_InData*                  in_data,
    PF_OutData*                 out_data,
    PF_GPUDeviceSetdownExtra*   extra)
{
    PF_Err err = PF_Err_NONE;

#if HAS_CUDA || HAS_DIRECTX
    JustGlowGPUData* gpuData = reinterpret_cast<JustGlowGPUData*>(extra->input->gpu_data);

    if (gpuData) {
        if (gpuData->renderer) {
            switch (gpuData->framework) {
#if HAS_CUDA
                case GPUFrameworkType::CUDA: {
                    JustGlowCUDARenderer* renderer =
                        static_cast<JustGlowCUDARenderer*>(gpuData->renderer);
                    renderer->Shutdown();
                    delete renderer;
                    break;
                }
#endif
#if HAS_DIRECTX
                case GPUFrameworkType::DirectX: {
                    JustGlowGPURenderer* renderer =
                        static_cast<JustGlowGPURenderer*>(gpuData->renderer);
                    renderer->Shutdown();
                    delete renderer;
                    break;
                }
#endif
                default:
                    break;
            }
        }
        delete gpuData;
    }
#endif

    return err;
}

// ============================================================================
// Pre Render Data Cleanup
// ============================================================================

static void DeletePreRenderData(void* pre_render_data)
{
    PLUGIN_LOG("DeletePreRenderData called");
    if (pre_render_data) {
        delete reinterpret_cast<JustGlowPreRenderData*>(pre_render_data);
    }
}

// ============================================================================
// Pre Render
// ============================================================================

PF_Err PreRender(
    PF_InData*          in_data,
    PF_OutData*         out_data,
    PF_PreRenderExtra*  extra)
{
    PF_Err err = PF_Err_NONE;
    PF_RenderRequest req = extra->input->output_request;
    PF_CheckoutResult in_result;

    PLUGIN_LOG("=== PreRender ===");
    PLUGIN_LOG("PreRender input: bitdepth=%d, what_gpu=%d, gpu_data=%p",
        extra->input->bitdepth,
        extra->input->what_gpu,
        extra->input->gpu_data);

    // Get downsample factor for preview resolution support
    // Full resolution = 1.0, Half = 0.5, Quarter = 0.25, etc.
    float downsampleX = static_cast<float>(in_data->downsample_x.num) /
                        static_cast<float>(in_data->downsample_x.den);
    float downsampleY = static_cast<float>(in_data->downsample_y.num) /
                        static_cast<float>(in_data->downsample_y.den);
    float downsampleFactor = (downsampleX + downsampleY) * 0.5f;  // Average

    PLUGIN_LOG("PreRender: downsample factor = %.3f (X=%.3f, Y=%.3f)",
        downsampleFactor, downsampleX, downsampleY);

    // Allocate pre-render data
    JustGlowPreRenderData* preRenderData = new JustGlowPreRenderData();

    // Get quality parameter early for extent calculation
    PF_ParamDef qualityParam;
    AEFX_CLR_STRUCT(qualityParam);
    PF_CHECKOUT_PARAM(in_data, PARAM_QUALITY, in_data->current_time,
        in_data->time_step, in_data->time_scale, &qualityParam);
    int quality = qualityParam.u.sd.value;  // Direct integer (6-12)
    int mipLevels = GetQualityLevelCount(quality);

    // Get spread values for more accurate expansion
    PF_ParamDef spreadDownParam, spreadUpParam;
    AEFX_CLR_STRUCT(spreadDownParam);
    AEFX_CLR_STRUCT(spreadUpParam);
    PF_CHECKOUT_PARAM(in_data, PARAM_SPREAD_DOWN, in_data->current_time,
        in_data->time_step, in_data->time_scale, &spreadDownParam);
    PF_CHECKOUT_PARAM(in_data, PARAM_SPREAD_UP, in_data->current_time,
        in_data->time_step, in_data->time_scale, &spreadUpParam);
    float spreadDown = spreadDownParam.u.fs_d.value;
    float spreadUp = spreadUpParam.u.fs_d.value;

    // Get offset parameters for accurate padding calculation
    PF_ParamDef offsetDownParam, offsetUpParam, offsetPrefilterParam, prefilterQualityParam;
    AEFX_CLR_STRUCT(offsetDownParam);
    AEFX_CLR_STRUCT(offsetUpParam);
    AEFX_CLR_STRUCT(offsetPrefilterParam);
    AEFX_CLR_STRUCT(prefilterQualityParam);
    PF_CHECKOUT_PARAM(in_data, PARAM_OFFSET_DOWN, in_data->current_time,
        in_data->time_step, in_data->time_scale, &offsetDownParam);
    PF_CHECKOUT_PARAM(in_data, PARAM_OFFSET_UP, in_data->current_time,
        in_data->time_step, in_data->time_scale, &offsetUpParam);
    PF_CHECKOUT_PARAM(in_data, PARAM_OFFSET_PREFILTER, in_data->current_time,
        in_data->time_step, in_data->time_scale, &offsetPrefilterParam);
    PF_CHECKOUT_PARAM(in_data, PARAM_PREFILTER_QUALITY, in_data->current_time,
        in_data->time_step, in_data->time_scale, &prefilterQualityParam);
    float offsetDown = offsetDownParam.u.fs_d.value;
    float offsetUp = offsetUpParam.u.fs_d.value;
    float offsetPrefilter = offsetPrefilterParam.u.fs_d.value;
    PrefilterQuality prefilterQuality = static_cast<PrefilterQuality>(prefilterQualityParam.u.pd.value);

    // Calculate glow expansion based on actual sampling offsets
    // Formula: ceil(offset) × 2 × 2^level for each level, accumulated
    //
    // Prefilter padding: ceil(maxOffset × offsetPrefilter) × 2
    //   - Sep9: maxOffset = 4.0
    //   - Others: maxOffset = 2.0
    //
    // Downsample/Upsample padding per level:
    //   offset = offsetDown + spreadDown × levelRatio
    //   padding = ceil(offset) × 2 × 2^level (converted to mip[0] resolution)

    // Prefilter padding (13-tap Star pattern)
    // 13-tap samples at outerOffset = 2.0 * offsetPrefilter
    // Diagonal reach = sqrt(2) * outerOffset = 2.83 * offsetPrefilter
    // Apply 3x margin for proper glow spread
    constexpr float SQRT2 = 1.414f;
    constexpr float PREFILTER_MARGIN = 3.0f;  // Safety margin for glow spread
    float prefilterOuterOffset = 2.0f * offsetPrefilter;
    float prefilterDiagonalReach = prefilterOuterOffset * SQRT2 * PREFILTER_MARGIN;
    int prefilterPadding = static_cast<int>(std::ceil(prefilterDiagonalReach)) * 2;

    // Downsample padding (accumulated across all levels)
    int totalDownPadding = 0;
    for (int level = 0; level < mipLevels; level++) {
        float levelRatio = (mipLevels > 1) ? (float)level / (float)(mipLevels - 1) : 0.0f;
        float offset = offsetDown + spreadDown * levelRatio;
        if (offset > 0.0f) {
            int levelPadding = static_cast<int>(std::ceil(offset)) * 2;
            totalDownPadding += levelPadding * (1 << level);
        }
    }

    // Upsample padding (accumulated across all levels)
    int totalUpPadding = 0;
    for (int level = mipLevels - 1; level >= 0; level--) {
        float levelRatio = (mipLevels > 1) ? (float)level / (float)(mipLevels - 1) : 0.0f;
        float offset = offsetUp + spreadUp * levelRatio;
        if (offset > 0.0f) {
            int levelPadding = static_cast<int>(std::ceil(offset)) * 2;
            totalUpPadding += levelPadding * (1 << level);
        }
    }

    // Total expansion: prefilter + max(downsample, upsample)
    int glowExpansion = prefilterPadding + (std::max)(totalDownPadding, totalUpPadding);

    // Scale by downsample factor for preview resolution support
    glowExpansion = static_cast<int>(glowExpansion * downsampleFactor);

    // Clamp to reasonable range (64 minimum for preview, 8192 maximum)
    glowExpansion = (std::max)(64, (std::min)(glowExpansion, 8192));

    PLUGIN_LOG("PreRender: Glow expansion = %d pixels (prefilter=%d, down=%d, up=%d)",
        glowExpansion, prefilterPadding, totalDownPadding, totalUpPadding);
    PLUGIN_LOG("PreRender: Original req.rect = (%d,%d,%d,%d) size=%dx%d",
        req.rect.left, req.rect.top, req.rect.right, req.rect.bottom,
        req.rect.right - req.rect.left, req.rect.bottom - req.rect.top);

    // Expand the request rect to get extra pixels for glow spread
    req.rect.left -= glowExpansion;
    req.rect.top -= glowExpansion;
    req.rect.right += glowExpansion;
    req.rect.bottom += glowExpansion;

    PLUGIN_LOG("PreRender: Expanded req.rect = (%d,%d,%d,%d) size=%dx%d",
        req.rect.left, req.rect.top, req.rect.right, req.rect.bottom,
        req.rect.right - req.rect.left, req.rect.bottom - req.rect.top);

    // Checkout input layer at current time (with expanded request)
    err = extra->cb->checkout_layer(
        in_data->effect_ref,
        PARAM_INPUT,
        PARAM_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &in_result);

    PLUGIN_LOG("PreRender: in_result.result_rect = (%d,%d,%d,%d) size=%dx%d",
        in_result.result_rect.left, in_result.result_rect.top,
        in_result.result_rect.right, in_result.result_rect.bottom,
        in_result.result_rect.right - in_result.result_rect.left,
        in_result.result_rect.bottom - in_result.result_rect.top);

    if (!err) {
        // Get parameter values
        PF_ParamDef param;

        // ===========================================
        // Core Parameters (New Order)
        // ===========================================

        // Intensity (0-100%) - Level 1 starting weight
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_INTENSITY, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->intensity = param.u.fs_d.value;

        // Exposure (0-50) - Brightness multiplier
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_EXPOSURE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->exposure = param.u.fs_d.value;

        // Radius (0-200)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_RADIUS, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->radius = param.u.fs_d.value;

        // Spread Down (1-5)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_SPREAD_DOWN, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->spreadDown = param.u.fs_d.value;

        // Spread Up (0-10)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_SPREAD_UP, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->spreadUp = param.u.fs_d.value;

        // Offset Down (0-3)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_OFFSET_DOWN, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->offsetDown = param.u.fs_d.value;

        // Offset Up (0-10)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_OFFSET_UP, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->offsetUp = param.u.fs_d.value;

        // Offset Prefilter (0-10)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_OFFSET_PREFILTER, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->offsetPrefilter = param.u.fs_d.value;

        // Prefilter Quality
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_PREFILTER_QUALITY, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->prefilterQuality = static_cast<PrefilterQuality>(param.u.pd.value);
        PLUGIN_LOG("PreRender: prefilterQuality param.u.pd.value = %d", param.u.pd.value);

        // Falloff (0-100)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_FALLOFF, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->falloff = param.u.fs_d.value;

        // ===========================================
        // Threshold
        // ===========================================

        // Threshold - non-linear mapping for better control
        // Threshold: UI 0-100 -> internal 0.0-1.0 (linear mapping)
        // UI 70 (default) -> 0.7 threshold
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_THRESHOLD, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->threshold = param.u.fs_d.value / 100.0f;

        // Threshold Softness: UI 0-100% -> internal 0.0-1.0
        // UI 50 (default) -> 0.5 knee width
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_SOFT_KNEE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->softKnee = param.u.fs_d.value / 100.0f;

        // ===========================================
        // Blur Options
        // ===========================================

        // Quality (MIP levels 6-12)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_QUALITY, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->quality = param.u.sd.value;  // Direct integer

        // Falloff Type
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_FALLOFF_TYPE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->falloffType = static_cast<FalloffType>(param.u.pd.value);

        // Blur Mode (upsample kernel size)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_BLUR_MODE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->blurMode = static_cast<BlurMode>(param.u.pd.value);

        // ===========================================
        // Color Options
        // ===========================================

        // Glow Color
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_GLOW_COLOR, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->glowColorR = param.u.cd.value.red / 255.0f;
        preRenderData->glowColorG = param.u.cd.value.green / 255.0f;
        preRenderData->glowColorB = param.u.cd.value.blue / 255.0f;

        // Color Temperature
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_COLOR_TEMP, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->colorTemp = param.u.fs_d.value;

        // Preserve Color
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_PRESERVE_COLOR, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->preserveColor = param.u.fs_d.value;

        // Desaturation (Max-based)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_DESATURATION, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->desaturation = param.u.fs_d.value;

        // ===========================================
        // Advanced Options
        // ===========================================

        // Anamorphic
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_ANAMORPHIC, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->anamorphic = param.u.fs_d.value;

        // Anamorphic Angle
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_ANAMORPHIC_ANGLE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->anamorphicAngle = FIX_2_FLOAT(param.u.ad.value);

        // Chromatic Aberration
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_CHROMATIC_ABERRATION, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->chromaticAberration = static_cast<float>(param.u.fs_d.value);

        // CA Tint Red
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_CA_TINT_R, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->caTintR[0] = param.u.cd.value.red / 65535.0f;
        preRenderData->caTintR[1] = param.u.cd.value.green / 65535.0f;
        preRenderData->caTintR[2] = param.u.cd.value.blue / 65535.0f;

        // CA Tint Blue
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_CA_TINT_B, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->caTintB[0] = param.u.cd.value.red / 65535.0f;
        preRenderData->caTintB[1] = param.u.cd.value.green / 65535.0f;
        preRenderData->caTintB[2] = param.u.cd.value.blue / 65535.0f;

        // Composite Mode
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_COMPOSITE_MODE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->compositeMode = static_cast<CompositeMode>(param.u.pd.value);

        // HDR Mode
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_HDR_MODE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->hdrMode = (param.u.bd.value != 0);

        // Linearize
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_LINEARIZE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->linearize = (param.u.bd.value != 0);

        // Input Profile (only used when Linearize is ON)
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_INPUT_PROFILE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->inputProfile = static_cast<InputProfile>(param.u.pd.value);

        // Dither
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_DITHER, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->dither = static_cast<float>(param.u.fs_d.value);

        // ===========================================
        // Debug Options
        // ===========================================

        // Debug View
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_DEBUG_VIEW, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->debugView = static_cast<DebugViewMode>(param.u.pd.value);

        // Source Opacity
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_SOURCE_OPACITY, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->sourceOpacity = param.u.fs_d.value;

        // Glow Opacity
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_GLOW_OPACITY, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->glowOpacity = param.u.fs_d.value;

        // Padding Threshold
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_PADDING_THRESHOLD, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->paddingThreshold = param.u.fs_d.value / 100.0f;  // Convert % to 0-0.1

        // Unpremultiply
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_UNPREMULTIPLY, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->unpremultiply = (param.u.bd.value != 0);

        // ===========================================
        // Computed Values (The Secret Sauce)
        // ===========================================

        // MIP levels from quality setting
        preRenderData->mipLevels = GetQualityLevelCount(preRenderData->quality);
        PLUGIN_LOG("PreRender: Quality=%d, MipLevels=%d", preRenderData->quality, preRenderData->mipLevels);

        // activeLimit: Radius -> soft threshold factor (0-1)
        // Radius 100% = no threshold (all glow passes)
        // Radius 0% = max threshold (only bright glow core)
        // Now uses per-level soft threshold instead of hard MIP cutoff
        preRenderData->activeLimit = preRenderData->radius / 100.0f;

        // blurOffsets: Legacy, not used in current kernels
        // Now using spreadDown/spreadUp directly in kernels with level-based interpolation
        // offset = 1.0 + spread * (level / maxLevel)
        for (int i = 0; i < preRenderData->mipLevels && i < PRERENDER_MAX_MIP_LEVELS; ++i) {
            preRenderData->blurOffsets[i] = 1.0f;  // Legacy placeholder
        }

        // decayK: Pass falloff directly (0-100), kernel calculates decayRate
        // 0% = boost outer, 50% = neutral, 100% = decay
        preRenderData->decayK = preRenderData->falloff;

        // level1Weight: Intensity -> Level 1 starting weight (0-1)
        // Intensity 0% = 0 (no outer glow), Intensity 100% = 1.0 (full)
        preRenderData->level1Weight = preRenderData->intensity / 100.0f;
    }

    // Set up output with expanded rect for glow spread
    // The glow extends beyond the original layer bounds
    extra->output->result_rect = in_result.result_rect;
    extra->output->max_result_rect = in_result.max_result_rect;

    // Expand output rect by glow expansion amount
    // This tells AE that our output is larger than the input
    extra->output->result_rect.left -= glowExpansion;
    extra->output->result_rect.top -= glowExpansion;
    extra->output->result_rect.right += glowExpansion;
    extra->output->result_rect.bottom += glowExpansion;

    extra->output->max_result_rect.left -= glowExpansion;
    extra->output->max_result_rect.top -= glowExpansion;
    extra->output->max_result_rect.right += glowExpansion;
    extra->output->max_result_rect.bottom += glowExpansion;

    PLUGIN_LOG("PreRender: Final output->result_rect = (%d,%d,%d,%d) size=%dx%d",
        extra->output->result_rect.left, extra->output->result_rect.top,
        extra->output->result_rect.right, extra->output->result_rect.bottom,
        extra->output->result_rect.right - extra->output->result_rect.left,
        extra->output->result_rect.bottom - extra->output->result_rect.top);

    extra->output->solid = FALSE;
    extra->output->pre_render_data = preRenderData;
    extra->output->delete_pre_render_data_func = DeletePreRenderData;

    // Set RETURNS_EXTRA_PIXELS flag since we expand result_rect beyond request_rect
    // This is required for glow effects that extend beyond layer bounds
    extra->output->flags = PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;

    // Flag GPU rendering as possible (only if GPU is actually available)
#if HAS_CUDA || HAS_DIRECTX
    bool gpuAvailable = (extra->input->gpu_data != nullptr) &&
                        (extra->input->what_gpu != PF_GPU_Framework_NONE);

    if (gpuAvailable) {
        extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
        PLUGIN_LOG("PreRender: GPU available (what_gpu=%d), flags set (0x%X)",
            extra->input->what_gpu, extra->output->flags);
    } else {
        PLUGIN_LOG("PreRender: GPU NOT available (gpu_data=%p, what_gpu=%d) - CPU fallback",
            extra->input->gpu_data, extra->input->what_gpu);
    }
#else
    PLUGIN_LOG("PreRender: GPU support not compiled in (HAS_CUDA=%d, HAS_DIRECTX=%d)", HAS_CUDA, HAS_DIRECTX);
#endif

    PLUGIN_LOG("PreRender complete, mipLevels=%d, err=%d", preRenderData->mipLevels, err);
    return err;
}

// ============================================================================
// Smart Render
// ============================================================================

PF_Err SmartRender(
    PF_InData*              in_data,
    PF_OutData*             out_data,
    PF_SmartRenderExtra*    extra,
    bool                    isGPU)
{
    PF_Err err = PF_Err_NONE;

    PLUGIN_LOG("=== SmartRender (isGPU=%d) ===", isGPU);

    JustGlowPreRenderData* preRenderData =
        reinterpret_cast<JustGlowPreRenderData*>(extra->input->pre_render_data);

    if (!preRenderData) {
        PLUGIN_LOG("ERROR: No preRenderData!");
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    // Checkout input and output
    PF_EffectWorld* input_worldP = nullptr;
    PF_EffectWorld* output_worldP = nullptr;

    err = extra->cb->checkout_layer_pixels(in_data->effect_ref, PARAM_INPUT, &input_worldP);
    if (err) return err;

    err = extra->cb->checkout_output(in_data->effect_ref, &output_worldP);
    if (err) return err;

    if (isGPU) {
#if HAS_CUDA || HAS_DIRECTX
        PLUGIN_LOG("GPU Rendering path");
        // GPU Rendering path
        JustGlowGPUData* gpuData =
            reinterpret_cast<JustGlowGPUData*>(const_cast<void*>(extra->input->gpu_data));

        PLUGIN_LOG("gpuData=%p", gpuData);
        if (gpuData) {
            PLUGIN_LOG("gpuData->initialized=%d, gpuData->renderer=%p, framework=%d",
                gpuData->initialized, gpuData->renderer, static_cast<int>(gpuData->framework));
        }

        if (gpuData && gpuData->initialized && gpuData->renderer) {
            // Get GPU buffer pointers
            const void* suiteP = nullptr;
            PF_GPUDeviceSuite1* gpuSuite = nullptr;

            err = in_data->pica_basicP->AcquireSuite(
                kPFGPUDeviceSuite,
                kPFGPUDeviceSuiteVersion1,
                &suiteP);

            PLUGIN_LOG("AcquireSuite in SmartRender: err=%d, suiteP=%p", err, suiteP);

            if (!err && suiteP) {
                gpuSuite = const_cast<PF_GPUDeviceSuite1*>(
                    static_cast<const PF_GPUDeviceSuite1*>(suiteP));
                void* inputData = nullptr;
                void* outputData = nullptr;

                gpuSuite->GetGPUWorldData(in_data->effect_ref, input_worldP, &inputData);
                gpuSuite->GetGPUWorldData(in_data->effect_ref, output_worldP, &outputData);

                PLUGIN_LOG("GetGPUWorldData: input=%p, output=%p", inputData, outputData);

                if (inputData && outputData) {
                    // Build render parameters
                    RenderParams rp = {};

                    // Core 4 computed values
                    rp.activeLimit = preRenderData->activeLimit;
                    for (int i = 0; i < MAX_MIP_LEVELS; ++i) {
                        rp.blurOffsets[i] = preRenderData->blurOffsets[i];
                    }
                    rp.decayK = preRenderData->decayK;
                    rp.exposure = preRenderData->exposure;
                    rp.level1Weight = preRenderData->level1Weight;
                    rp.falloffType = static_cast<int>(preRenderData->falloffType);
                    rp.spreadDown = preRenderData->spreadDown;  // 0-10 direct
                    rp.spreadUp = preRenderData->spreadUp;      // 0-10 direct
                    rp.offsetDown = preRenderData->offsetDown;  // 0-10 direct
                    rp.offsetUp = preRenderData->offsetUp;      // 0-10 direct
                    rp.offsetPrefilter = preRenderData->offsetPrefilter;  // 0-10 direct
                    rp.prefilterQuality = static_cast<int>(preRenderData->prefilterQuality);
                    PLUGIN_LOG("SmartRender: prefilterQuality = %d", rp.prefilterQuality);

                    // Threshold
                    rp.threshold = preRenderData->threshold;
                    rp.softKnee = preRenderData->softKnee;

                    // Quality (MIP levels)
                    rp.quality = preRenderData->quality;
                    rp.blurMode = static_cast<int>(preRenderData->blurMode);

                    // Color
                    rp.glowColor[0] = preRenderData->glowColorR;
                    rp.glowColor[1] = preRenderData->glowColorG;
                    rp.glowColor[2] = preRenderData->glowColorB;
                    rp.colorTemp = preRenderData->colorTemp;
                    rp.preserveColor = preRenderData->preserveColor;
                    rp.desaturation = preRenderData->desaturation / 100.0f;  // 0-1

                    // Advanced
                    rp.anamorphic = preRenderData->anamorphic;
                    rp.anamorphicAngle = preRenderData->anamorphicAngle;
                    rp.chromaticAberration = preRenderData->chromaticAberration;
                    for (int i = 0; i < 3; ++i) {
                        rp.caTintR[i] = preRenderData->caTintR[i];
                        rp.caTintB[i] = preRenderData->caTintB[i];
                    }
                    rp.compositeMode = static_cast<int>(preRenderData->compositeMode);
                    rp.hdrMode = preRenderData->hdrMode;
                    rp.linearize = preRenderData->linearize;
                    rp.inputProfile = static_cast<int>(preRenderData->inputProfile);
                    rp.dither = preRenderData->dither / 100.0f;  // 0-1

                    // Debug
                    rp.debugView = static_cast<int>(preRenderData->debugView);
                    rp.sourceOpacity = preRenderData->sourceOpacity / 100.0f;  // 0-1
                    rp.glowOpacity = preRenderData->glowOpacity / 100.0f;      // 0-2
                    rp.paddingThreshold = preRenderData->paddingThreshold;     // Already 0-0.01

                    // Image info
                    rp.width = output_worldP->width;
                    rp.height = output_worldP->height;
                    rp.srcPitch = input_worldP->rowbytes / sizeof(float) / 4;
                    rp.dstPitch = output_worldP->rowbytes / sizeof(float) / 4;
                    rp.inputWidth = input_worldP->width;
                    rp.inputHeight = input_worldP->height;
                    rp.mipLevels = preRenderData->mipLevels;

                    PLUGIN_LOG("SmartRender: output=%dx%d, input=%dx%d, srcPitch=%d, dstPitch=%d, debugView=%d",
                        rp.width, rp.height, rp.inputWidth, rp.inputHeight, rp.srcPitch, rp.dstPitch, rp.debugView);
                    PLUGIN_LOG("SmartRender DEBUG: input_world=%dx%d (rowbytes=%d), output_world=%dx%d (rowbytes=%d)",
                        input_worldP->width, input_worldP->height, input_worldP->rowbytes,
                        output_worldP->width, output_worldP->height, output_worldP->rowbytes);

                    // Execute GPU rendering based on framework
                    bool renderSuccess = false;

                    switch (gpuData->framework) {
#if HAS_CUDA
                        case GPUFrameworkType::CUDA: {
                            PLUGIN_LOG("Calling CUDA renderer->Render...");
                            JustGlowCUDARenderer* renderer =
                                static_cast<JustGlowCUDARenderer*>(gpuData->renderer);
                            renderSuccess = renderer->Render(rp,
                                reinterpret_cast<CUdeviceptr>(inputData),
                                reinterpret_cast<CUdeviceptr>(outputData));
                            break;
                        }
#endif
#if HAS_DIRECTX
                        case GPUFrameworkType::DirectX: {
                            PLUGIN_LOG("Calling DirectX renderer->Render...");
                            JustGlowGPURenderer* renderer =
                                static_cast<JustGlowGPURenderer*>(gpuData->renderer);
                            renderSuccess = renderer->Render(rp,
                                static_cast<ID3D12Resource*>(inputData),
                                static_cast<ID3D12Resource*>(outputData));
                            break;
                        }
#endif
                        default:
                            PLUGIN_LOG("ERROR: Unknown framework type!");
                            break;
                    }

                    if (!renderSuccess) {
                        PLUGIN_LOG("ERROR: Render failed!");
                        strcpy(out_data->return_msg,
                            "GPU render failed. Check %TEMP%\\JustGlow_CUDA_debug.log");
                        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
                    }
                    else {
                        PLUGIN_LOG("Render succeeded!");
                    }
                }
                else {
                    PLUGIN_LOG("ERROR: inputData or outputData is null!");
                }

                in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
            }
        }
        else {
            PLUGIN_LOG("ERROR: GPU data not initialized!");
            strcpy(out_data->return_msg,
                "GPU renderer not initialized. Restart After Effects.");
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }
#else
        PLUGIN_LOG("ERROR: No GPU support in SmartRender!");
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
#endif
    }
    else {
        // CPU Fallback - GPU is required for JustGlow
        PLUGIN_LOG("CPU Fallback path - GPU required, showing error");
        strcpy(out_data->return_msg,
            "JustGlow requires GPU acceleration. "
            "Enable GPU in Preferences > Display or check GPU drivers.");
        // Copy source as-is so user can at least see something
        PF_COPY(input_worldP, output_worldP, nullptr, nullptr);
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }

    // NOTE: Do NOT delete preRenderData here!
    // AE will call DeletePreRenderData via delete_pre_render_data_func

    return err;
}

// ============================================================================
// Utility Functions
// ============================================================================

// Note: CalculateMipLevels is now handled by GetQualityLevelCount() in the header
// The new system uses Quality to determine MIP depth, not Radius

void ColorTempToRGB(float temp, float& rMult, float& gMult, float& bMult) {
    // Approximate Planckian locus for color temperature
    // temp: -100 (cool) to +100 (warm)
    float t = temp / 100.0f;

    if (t >= 0.0f) {
        // Warm: increase red, decrease blue
        rMult = 1.0f + t * 0.3f;
        gMult = 1.0f;
        bMult = 1.0f - t * 0.3f;
    } else {
        // Cool: decrease red, increase blue
        rMult = 1.0f + t * 0.3f;
        gMult = 1.0f;
        bMult = 1.0f - t * 0.3f;
    }
}
