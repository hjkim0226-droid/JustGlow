/**
 * JustGlow - High Quality GPU Glow Effect for After Effects
 *
 * Main plugin implementation file.
 */

#include "JustGlow.h"
#include "JustGlowParams.h"
#include <AE_Macros.h>
#include <Util/Param_Utils.h>

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
        logFile.open(GetLogFilePath(), std::ios::out | std::ios::app);
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
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING // Thread-safe
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
    // Basic Parameters
    // ===========================================

    // Intensity (0-200%)
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

    // Radius (0-500 pixels)
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

    // Soft Knee (0-100%)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Soft Knee",
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

    // Quality
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        "Quality",
        4,  // Number of choices
        Defaults::Quality,
        "Low|Medium|High|Ultra",
        DISK_ID_QUALITY);

    // Fractional Blend
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "Smooth Radius",
        "Enable smooth radius transitions",
        Defaults::FractionalBlend ? 1 : 0,
        0,
        DISK_ID_FRACTIONAL_BLEND);

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

    // Allocate pre-render data
    JustGlowPreRenderData* preRenderData = new JustGlowPreRenderData();

    // Checkout input layer at current time
    err = extra->cb->checkout_layer(
        in_data->effect_ref,
        PARAM_INPUT,
        PARAM_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &in_result);

    if (!err) {
        // Get parameter values
        PF_ParamDef param;

        // Intensity
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_INTENSITY, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->intensity = param.u.fs_d.value;

        // Radius
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_RADIUS, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->radius = param.u.fs_d.value;

        // Threshold
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_THRESHOLD, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->threshold = param.u.fs_d.value;

        // Soft Knee
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_SOFT_KNEE, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->softKnee = param.u.fs_d.value;

        // Quality
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_QUALITY, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->quality = static_cast<BlurQuality>(param.u.pd.value);

        // Fractional Blend
        AEFX_CLR_STRUCT(param);
        PF_CHECKOUT_PARAM(in_data, PARAM_FRACTIONAL_BLEND, in_data->current_time,
            in_data->time_step, in_data->time_scale, &param);
        preRenderData->fractionalBlend = (param.u.bd.value != 0);

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

        // Calculate MIP levels
        preRenderData->mipLevels = CalculateMipLevels(
            preRenderData->radius, preRenderData->quality);
        preRenderData->fractionalAmount = CalculateFractionalAmount(
            preRenderData->radius, preRenderData->mipLevels);
    }

    // Set up output
    extra->output->result_rect = in_result.result_rect;
    extra->output->max_result_rect = in_result.max_result_rect;
    extra->output->solid = FALSE;
    extra->output->pre_render_data = preRenderData;
    extra->output->delete_pre_render_data_func = DeletePreRenderData;

    // Flag GPU rendering as possible
#if HAS_CUDA || HAS_DIRECTX
    extra->output->flags = PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    PLUGIN_LOG("PreRender: GPU_RENDER_POSSIBLE flag set (0x%X)", extra->output->flags);
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
                    rp.intensity = preRenderData->intensity;
                    rp.radius = preRenderData->radius;
                    rp.threshold = preRenderData->threshold;
                    rp.softKnee = preRenderData->softKnee;
                    rp.quality = static_cast<int>(preRenderData->quality);
                    rp.fractionalBlend = preRenderData->fractionalBlend;
                    rp.glowColor[0] = preRenderData->glowColorR;
                    rp.glowColor[1] = preRenderData->glowColorG;
                    rp.glowColor[2] = preRenderData->glowColorB;
                    rp.colorTemp = preRenderData->colorTemp;
                    rp.preserveColor = preRenderData->preserveColor;
                    rp.anamorphic = preRenderData->anamorphic;
                    rp.anamorphicAngle = preRenderData->anamorphicAngle;
                    rp.compositeMode = static_cast<int>(preRenderData->compositeMode);
                    rp.hdrMode = preRenderData->hdrMode;
                    rp.width = output_worldP->width;
                    rp.height = output_worldP->height;
                    rp.srcPitch = input_worldP->rowbytes / sizeof(float) / 4;
                    rp.dstPitch = output_worldP->rowbytes / sizeof(float) / 4;
                    rp.mipLevels = preRenderData->mipLevels;
                    rp.fractionalAmount = preRenderData->fractionalAmount;
                    rp.mipChain = CalculateMipChain(rp.width, rp.height, rp.mipLevels);

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
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }
#else
        PLUGIN_LOG("ERROR: No GPU support in SmartRender!");
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
#endif
    }
    else {
        PLUGIN_LOG("CPU Fallback path");
        // CPU Fallback - simple copy for now
        // TODO: Implement CPU-based glow if needed
        PF_COPY(input_worldP, output_worldP, nullptr, nullptr);
    }

    // Cleanup pre-render data
    delete preRenderData;

    return err;
}

// ============================================================================
// Utility Functions
// ============================================================================

int CalculateMipLevels(float radius, BlurQuality quality) {
    // Base levels from quality setting
    int maxLevels = GetQualityLevelCount(quality);

    // Calculate required levels from radius
    // Each level doubles the effective blur radius
    int requiredLevels = 1;
    float currentRadius = 2.0f;  // Base radius at level 1

    while (currentRadius < radius && requiredLevels < maxLevels) {
        currentRadius *= 2.0f;
        requiredLevels++;
    }

    return requiredLevels;
}

float CalculateFractionalAmount(float radius, int mipLevels) {
    if (mipLevels <= 1) return 0.0f;

    // Calculate the fractional part for smooth transitions
    float baseRadius = 2.0f * static_cast<float>(1 << (mipLevels - 1));
    float prevRadius = baseRadius / 2.0f;

    if (radius <= prevRadius) return 0.0f;
    if (radius >= baseRadius) return 1.0f;

    return (radius - prevRadius) / (baseRadius - prevRadius);
}

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
