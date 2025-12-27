/**
 * JustGlow CUDA Renderer Implementation
 *
 * CUDA Driver API based rendering pipeline.
 * Loads PTX kernels and executes the glow effect pipeline.
 */

#ifdef _WIN32

// Prevent Windows.h from defining min/max macros (conflicts with std::min/max)
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include "JustGlowCUDARenderer.h"
#include "JustGlowInterop.h"  // For InteropTexture struct
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <cstdarg>
#include <algorithm>  // for std::min, std::max

// ============================================================================
// Debug Logging
// ============================================================================

#define JUSTGLOW_CUDA_LOGGING 1

#if JUSTGLOW_CUDA_LOGGING
static std::wstring GetCUDALogFilePath() {
    wchar_t tempPath[MAX_PATH];
    GetTempPathW(MAX_PATH, tempPath);
    return std::wstring(tempPath) + L"JustGlow_CUDA_debug.log";
}

static void CUDALogMessage(const char* format, ...) {
    static std::ofstream logFile;
    static bool initialized = false;

    if (!initialized) {
        logFile.open(GetCUDALogFilePath(), std::ios::out | std::ios::trunc);
        initialized = true;
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

#define CUDA_LOG(fmt, ...) CUDALogMessage(fmt, ##__VA_ARGS__)
#else
#define CUDA_LOG(fmt, ...) ((void)0)
#endif

// Thread group size (must match CUDA kernels)
constexpr int THREAD_BLOCK_SIZE = 16;

// Note: BASE_BLUR_SIGMA is defined in JustGlowInterop.h

// ============================================================================
// Helper to get DLL module handle
// ============================================================================

static HMODULE GetCurrentModuleCUDA() {
    HMODULE hModule = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&GetCurrentModuleCUDA),
        &hModule);
    return hModule;
}

static std::wstring GetPTXPath() {
    HMODULE hModule = GetCurrentModuleCUDA();
    wchar_t modulePath[MAX_PATH];
    GetModuleFileNameW(hModule, modulePath, MAX_PATH);

    std::wstring path(modulePath);
    size_t lastSlash = path.find_last_of(L"\\/");
    if (lastSlash != std::wstring::npos) {
        path = path.substr(0, lastSlash + 1);
    }

    return path + L"CUDA_Assets\\JustGlowKernels.ptx";
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

JustGlowCUDARenderer::JustGlowCUDARenderer()
    : m_context(nullptr)
    , m_stream(nullptr)
    , m_syncEvent(nullptr)
    , m_module(nullptr)
    , m_prefilterKernel(nullptr)
    , m_downsampleKernel(nullptr)
    , m_upsampleKernel(nullptr)
    , m_compositeKernel(nullptr)
    , m_horizontalBlurKernel(nullptr)
    , m_gaussian2DDownsampleKernel(nullptr)
    , m_gaussianDownsampleHKernel(nullptr)
    , m_gaussianDownsampleVKernel(nullptr)
    , m_debugOutputKernel(nullptr)
    , m_desaturationKernel(nullptr)
    , m_refineKernel(nullptr)
    // Interop kernels
    , m_unmultSurfaceKernel(nullptr)
    , m_prefilterSurfaceKernel(nullptr)
    , m_downsampleSurfaceKernel(nullptr)
    , m_logTransPreblurHKernel(nullptr)
    , m_logTransPreblurVKernel(nullptr)
    , m_clearSurfaceKernel(nullptr)
    , m_streamsInitialized(false)
    , m_refineBoundsGPU(0)
    , m_horizontalTemp{}
    , m_gaussianDownsampleTemp{}
    , m_prefilterSepTemp{}
    , m_currentMipLevels(0)
    , m_currentWidth(0)
    , m_currentHeight(0)
    , m_initialized(false)
{
}

JustGlowCUDARenderer::~JustGlowCUDARenderer() {
    Shutdown();
}

// ============================================================================
// Error Handling
// ============================================================================

bool JustGlowCUDARenderer::CheckCUDAError(CUresult err, const char* context) {
    if (err != CUDA_SUCCESS) {
        const char* errName = nullptr;
        const char* errStr = nullptr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        CUDA_LOG("CUDA ERROR in %s: %s (%s)", context, errName ? errName : "Unknown", errStr ? errStr : "");
        return false;
    }
    return true;
}

// ============================================================================
// Initialize
// ============================================================================

bool JustGlowCUDARenderer::Initialize(CUcontext context, CUstream stream) {
    CUDA_LOG("=== JustGlow CUDA Renderer Initialize ===");

    if (!context) {
        CUDA_LOG("ERROR: Invalid CUDA context");
        return false;
    }

    m_context = context;
    m_stream = stream;
    CUDA_LOG("CUDA context and stream set: context=%p, stream=%p", context, stream);

    // Push context for this thread
    CUresult err = cuCtxPushCurrent(m_context);
    if (!CheckCUDAError(err, "cuCtxPushCurrent")) {
        return false;
    }

    // Load kernels from PTX
    if (!LoadKernels()) {
        CUDA_LOG("ERROR: Failed to load CUDA kernels");
        cuCtxPopCurrent(nullptr);
        return false;
    }

    // Create synchronization event for inter-kernel dependencies
    CUresult eventErr = cuEventCreate(&m_syncEvent, CU_EVENT_DEFAULT);
    if (!CheckCUDAError(eventErr, "cuEventCreate")) {
        CUDA_LOG("ERROR: Failed to create sync event");
        cuCtxPopCurrent(nullptr);
        return false;
    }
    CUDA_LOG("Sync event created");

    // Allocate GPU buffer for RefineKernel bounds (4 ints: minX, maxX, minY, maxY)
    err = cuMemAlloc(&m_refineBoundsGPU, 4 * sizeof(int));
    if (!CheckCUDAError(err, "cuMemAlloc(refineBoundsGPU)")) {
        CUDA_LOG("ERROR: Failed to allocate refine bounds buffer");
        cuCtxPopCurrent(nullptr);
        return false;
    }
    CUDA_LOG("Refine bounds GPU buffer allocated");

    // Initialize parallel streams for Pre-blur
    if (!InitializeStreams()) {
        CUDA_LOG("WARNING: Failed to initialize parallel streams (non-critical)");
        // Continue without parallel streams - they're optional
    }

    // Pop context
    cuCtxPopCurrent(nullptr);

    m_initialized = true;
    CUDA_LOG("=== CUDA Initialize Complete ===");
    return true;
}

// ============================================================================
// Load Kernels
// ============================================================================

bool JustGlowCUDARenderer::LoadKernels() {
    std::wstring ptxPathW = GetPTXPath();

    // Convert to narrow string
    char ptxPath[MAX_PATH];
    WideCharToMultiByte(CP_UTF8, 0, ptxPathW.c_str(), -1, ptxPath, MAX_PATH, nullptr, nullptr);
    CUDA_LOG("Loading PTX from: %s", ptxPath);

    // Load the PTX module
    CUresult err = cuModuleLoad(&m_module, ptxPath);
    if (!CheckCUDAError(err, "cuModuleLoad")) {
        CUDA_LOG("ERROR: Failed to load PTX module from %s", ptxPath);
        return false;
    }
    CUDA_LOG("PTX module loaded successfully");

    // Get kernel functions
    err = cuModuleGetFunction(&m_prefilterKernel, m_module, "PrefilterKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(PrefilterKernel)")) {
        return false;
    }
    CUDA_LOG("PrefilterKernel loaded");

    err = cuModuleGetFunction(&m_prefilter25TapKernel, m_module, "Prefilter25TapKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(Prefilter25TapKernel)")) {
        return false;
    }
    CUDA_LOG("Prefilter25TapKernel loaded");

    err = cuModuleGetFunction(&m_prefilterSep5HKernel, m_module, "PrefilterSep5HKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(PrefilterSep5HKernel)")) {
        return false;
    }
    CUDA_LOG("PrefilterSep5HKernel loaded");

    err = cuModuleGetFunction(&m_prefilterSep5VKernel, m_module, "PrefilterSep5VKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(PrefilterSep5VKernel)")) {
        return false;
    }
    CUDA_LOG("PrefilterSep5VKernel loaded");

    err = cuModuleGetFunction(&m_prefilterSep9HKernel, m_module, "PrefilterSep9HKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(PrefilterSep9HKernel)")) {
        return false;
    }
    CUDA_LOG("PrefilterSep9HKernel loaded");

    err = cuModuleGetFunction(&m_prefilterSep9VKernel, m_module, "PrefilterSep9VKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(PrefilterSep9VKernel)")) {
        return false;
    }
    CUDA_LOG("PrefilterSep9VKernel loaded");

    err = cuModuleGetFunction(&m_downsampleKernel, m_module, "DownsampleKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(DownsampleKernel)")) {
        return false;
    }
    CUDA_LOG("DownsampleKernel loaded");

    err = cuModuleGetFunction(&m_upsampleKernel, m_module, "UpsampleKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(UpsampleKernel)")) {
        return false;
    }
    CUDA_LOG("UpsampleKernel loaded");

    err = cuModuleGetFunction(&m_compositeKernel, m_module, "CompositeKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(CompositeKernel)")) {
        return false;
    }
    CUDA_LOG("CompositeKernel loaded");

    err = cuModuleGetFunction(&m_horizontalBlurKernel, m_module, "HorizontalBlurKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(HorizontalBlurKernel)")) {
        return false;
    }
    CUDA_LOG("HorizontalBlurKernel loaded");

    err = cuModuleGetFunction(&m_gaussian2DDownsampleKernel, m_module, "Gaussian2DDownsampleKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(Gaussian2DDownsampleKernel)")) {
        return false;
    }
    CUDA_LOG("Gaussian2DDownsampleKernel loaded");

    // Load deprecated kernels for fallback compatibility
    err = cuModuleGetFunction(&m_gaussianDownsampleHKernel, m_module, "GaussianDownsampleHKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(GaussianDownsampleHKernel)")) {
        return false;
    }
    CUDA_LOG("GaussianDownsampleHKernel loaded (deprecated)");

    err = cuModuleGetFunction(&m_gaussianDownsampleVKernel, m_module, "GaussianDownsampleVKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(GaussianDownsampleVKernel)")) {
        return false;
    }
    CUDA_LOG("GaussianDownsampleVKernel loaded (deprecated)");

    err = cuModuleGetFunction(&m_debugOutputKernel, m_module, "DebugOutputKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(DebugOutputKernel)")) {
        return false;
    }
    CUDA_LOG("DebugOutputKernel loaded");

    err = cuModuleGetFunction(&m_desaturationKernel, m_module, "DesaturationKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(DesaturationKernel)")) {
        return false;
    }
    CUDA_LOG("DesaturationKernel loaded");

    err = cuModuleGetFunction(&m_refineKernel, m_module, "RefineKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(RefineKernel)")) {
        return false;
    }
    CUDA_LOG("RefineKernel loaded");

    // ========================================
    // Load Interop kernels (optional - may not exist in older PTX)
    // ========================================

    // Legacy linear buffer kernels
    err = cuModuleGetFunction(&m_unmultSurfaceKernel, m_module, "UnmultKernel");
    if (err == CUDA_SUCCESS) {
        CUDA_LOG("UnmultKernel loaded (Interop)");
    } else {
        m_unmultSurfaceKernel = nullptr;
        CUDA_LOG("UnmultKernel not found (optional)");
    }

    err = cuModuleGetFunction(&m_logTransPreblurHKernel, m_module, "LogTransmittancePreblurHKernel");
    if (err == CUDA_SUCCESS) {
        CUDA_LOG("LogTransmittancePreblurHKernel loaded (Interop)");
    } else {
        m_logTransPreblurHKernel = nullptr;
        CUDA_LOG("LogTransmittancePreblurHKernel not found (optional)");
    }

    err = cuModuleGetFunction(&m_logTransPreblurVKernel, m_module, "LogTransmittancePreblurVKernel");
    if (err == CUDA_SUCCESS) {
        CUDA_LOG("LogTransmittancePreblurVKernel loaded (Interop)");
    } else {
        m_logTransPreblurVKernel = nullptr;
        CUDA_LOG("LogTransmittancePreblurVKernel not found (optional)");
    }

    // ========================================
    // Load Surface-based kernels (zero-copy Interop)
    // ========================================

    // Note: These kernels use cudaSurfaceObject_t for zero-copy access
    // They are optional and will be loaded if available in the PTX

    err = cuModuleGetFunction(&m_unmultSurfaceKernel, m_module, "UnmultSurfaceKernel");
    if (err == CUDA_SUCCESS) {
        CUDA_LOG("UnmultSurfaceKernel loaded (Surface Interop)");
    } else {
        // Keep previous m_unmultSurfaceKernel if UnmultSurfaceKernel not found
        CUDA_LOG("UnmultSurfaceKernel not found, using linear version");
    }

    err = cuModuleGetFunction(&m_prefilterSurfaceKernel, m_module, "PrefilterSurfaceKernel");
    if (err == CUDA_SUCCESS) {
        CUDA_LOG("PrefilterSurfaceKernel loaded (Surface Interop)");
    } else {
        m_prefilterSurfaceKernel = nullptr;
        CUDA_LOG("PrefilterSurfaceKernel not found (optional)");
    }

    err = cuModuleGetFunction(&m_downsampleSurfaceKernel, m_module, "DownsampleSurfaceKernel");
    if (err == CUDA_SUCCESS) {
        CUDA_LOG("DownsampleSurfaceKernel loaded (Surface Interop)");
    } else {
        m_downsampleSurfaceKernel = nullptr;
        CUDA_LOG("DownsampleSurfaceKernel not found (optional)");
    }

    // Surface-based Pre-blur H/V kernels
    CUfunction surfacePreblurH = nullptr, surfacePreblurV = nullptr;
    err = cuModuleGetFunction(&surfacePreblurH, m_module, "LogTransPreblurHSurfaceKernel");
    if (err == CUDA_SUCCESS) {
        m_logTransPreblurHKernel = surfacePreblurH;  // Override with surface version
        CUDA_LOG("LogTransPreblurHSurfaceKernel loaded (Surface Interop)");
    }

    err = cuModuleGetFunction(&surfacePreblurV, m_module, "LogTransPreblurVSurfaceKernel");
    if (err == CUDA_SUCCESS) {
        m_logTransPreblurVKernel = surfacePreblurV;  // Override with surface version
        CUDA_LOG("LogTransPreblurVSurfaceKernel loaded (Surface Interop)");
    }

    // Clear surface kernel
    err = cuModuleGetFunction(&m_clearSurfaceKernel, m_module, "ClearSurfaceKernel");
    if (err == CUDA_SUCCESS) {
        CUDA_LOG("ClearSurfaceKernel loaded (Surface Interop)");
    } else {
        m_clearSurfaceKernel = nullptr;
        CUDA_LOG("ClearSurfaceKernel not found (optional)");
    }

    return true;
}

// ============================================================================
// Shutdown
// ============================================================================

void JustGlowCUDARenderer::Shutdown() {
    CUDA_LOG("=== CUDA Renderer Shutdown ===");

    if (m_context) {
        cuCtxPushCurrent(m_context);

        ReleaseMipChain();

        // Destroy parallel streams
        DestroyStreams();

        // Free refine bounds GPU buffer
        if (m_refineBoundsGPU) {
            cuMemFree(m_refineBoundsGPU);
            m_refineBoundsGPU = 0;
            CUDA_LOG("Refine bounds GPU buffer freed");
        }

        // Destroy sync event
        if (m_syncEvent) {
            cuEventDestroy(m_syncEvent);
            m_syncEvent = nullptr;
            CUDA_LOG("Sync event destroyed");
        }

        if (m_module) {
            cuModuleUnload(m_module);
            m_module = nullptr;
        }

        cuCtxPopCurrent(nullptr);
    }

    m_prefilterKernel = nullptr;
    m_downsampleKernel = nullptr;
    m_upsampleKernel = nullptr;
    m_compositeKernel = nullptr;
    m_horizontalBlurKernel = nullptr;
    m_gaussian2DDownsampleKernel = nullptr;
    m_gaussianDownsampleHKernel = nullptr;
    m_gaussianDownsampleVKernel = nullptr;
    m_debugOutputKernel = nullptr;
    m_desaturationKernel = nullptr;
    // Interop kernels
    m_unmultSurfaceKernel = nullptr;
    m_prefilterSurfaceKernel = nullptr;
    m_downsampleSurfaceKernel = nullptr;
    m_logTransPreblurHKernel = nullptr;
    m_logTransPreblurVKernel = nullptr;
    m_clearSurfaceKernel = nullptr;
    m_context = nullptr;
    m_stream = nullptr;
    m_initialized = false;
}

// ============================================================================
// Allocate MIP Chain
// ============================================================================

bool JustGlowCUDARenderer::AllocateMipChain(int width, int height, int levels) {
    // Check if we can reuse existing chain
    if (m_currentMipLevels == levels && !m_mipChain.empty()) {
        if (m_mipChain[0].width == width && m_mipChain[0].height == height) {
            CUDA_LOG("AllocateMipChain: REUSING existing chain %dx%d, %d levels", width, height, levels);
            return true;
        }
        CUDA_LOG("AllocateMipChain: Size changed %dx%d -> %dx%d, REALLOCATING",
            m_mipChain[0].width, m_mipChain[0].height, width, height);
    } else {
        CUDA_LOG("AllocateMipChain: NEW allocation %dx%d, %d levels", width, height, levels);
    }

    ReleaseMipChain();

    m_mipChain.resize(levels);
    m_upsampleChain.resize(levels);  // Separate buffers for upsample results
    m_currentMipLevels = levels;
    m_currentWidth = width;
    m_currentHeight = height;

    int w = width;
    int h = height;

    for (int i = 0; i < levels; ++i) {
        // MIP chain for downsampled textures
        auto& mip = m_mipChain[i];
        mip.width = w;
        mip.height = h;
        mip.pitchBytes = w * 4 * sizeof(float);  // RGBA float (width * 4 channels * 4 bytes)
        mip.sizeBytes = mip.pitchBytes * h;

        CUresult err = cuMemAlloc(&mip.devicePtr, mip.sizeBytes);
        if (!CheckCUDAError(err, "cuMemAlloc for MIP")) {
            CUDA_LOG("ERROR: Failed to allocate MIP level %d (%dx%d)", i, w, h);
            ReleaseMipChain();
            return false;
        }
        CUDA_LOG("MIP[%d] allocated: %dx%d, %zu bytes", i, w, h, mip.sizeBytes);

        // Upsample chain for upsample results (same dimensions)
        auto& upsample = m_upsampleChain[i];
        upsample.width = w;
        upsample.height = h;
        upsample.pitchBytes = mip.pitchBytes;
        upsample.sizeBytes = mip.sizeBytes;

        err = cuMemAlloc(&upsample.devicePtr, upsample.sizeBytes);
        if (!CheckCUDAError(err, "cuMemAlloc for Upsample")) {
            CUDA_LOG("ERROR: Failed to allocate Upsample level %d (%dx%d)", i, w, h);
            ReleaseMipChain();
            return false;
        }
        CUDA_LOG("Upsample[%d] allocated: %dx%d, %zu bytes", i, w, h, upsample.sizeBytes);

        // Next level is half size
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
    }

    // Allocate horizontal temp buffer for separable Gaussian blur during upsample
    // Size matches level 1 (largest buffer needed for horizontal blur of prevLevel)
    if (levels > 1) {
        m_horizontalTemp.width = m_upsampleChain[1].width;
        m_horizontalTemp.height = m_upsampleChain[1].height;
        m_horizontalTemp.pitchBytes = m_upsampleChain[1].pitchBytes;
        m_horizontalTemp.sizeBytes = m_upsampleChain[1].sizeBytes;

        CUresult err = cuMemAlloc(&m_horizontalTemp.devicePtr, m_horizontalTemp.sizeBytes);
        if (!CheckCUDAError(err, "cuMemAlloc for HorizontalTemp")) {
            CUDA_LOG("ERROR: Failed to allocate HorizontalTemp buffer");
            ReleaseMipChain();
            return false;
        }
        CUDA_LOG("HorizontalTemp allocated: %dx%d, %zu bytes",
            m_horizontalTemp.width, m_horizontalTemp.height, m_horizontalTemp.sizeBytes);
    }

    // Allocate Gaussian downsample temp buffer (for H-blur at source resolution)
    // Size matches level 0 (largest source resolution for Gaussian downsample)
    {
        m_gaussianDownsampleTemp.width = m_mipChain[0].width;
        m_gaussianDownsampleTemp.height = m_mipChain[0].height;
        m_gaussianDownsampleTemp.pitchBytes = m_mipChain[0].pitchBytes;
        m_gaussianDownsampleTemp.sizeBytes = m_mipChain[0].sizeBytes;

        CUresult err = cuMemAlloc(&m_gaussianDownsampleTemp.devicePtr, m_gaussianDownsampleTemp.sizeBytes);
        if (!CheckCUDAError(err, "cuMemAlloc for GaussianDownsampleTemp")) {
            CUDA_LOG("ERROR: Failed to allocate GaussianDownsampleTemp buffer");
            ReleaseMipChain();
            return false;
        }
        CUDA_LOG("GaussianDownsampleTemp allocated: %dx%d, %zu bytes",
            m_gaussianDownsampleTemp.width, m_gaussianDownsampleTemp.height, m_gaussianDownsampleTemp.sizeBytes);
    }

    // Allocate separable prefilter temp buffer (for H-pass at MIP[0] resolution)
    {
        m_prefilterSepTemp.width = m_mipChain[0].width;
        m_prefilterSepTemp.height = m_mipChain[0].height;
        m_prefilterSepTemp.pitchBytes = m_mipChain[0].pitchBytes;
        m_prefilterSepTemp.sizeBytes = m_mipChain[0].sizeBytes;

        CUresult err = cuMemAlloc(&m_prefilterSepTemp.devicePtr, m_prefilterSepTemp.sizeBytes);
        if (!CheckCUDAError(err, "cuMemAlloc for PrefilterSepTemp")) {
            CUDA_LOG("ERROR: Failed to allocate PrefilterSepTemp buffer");
            ReleaseMipChain();
            return false;
        }
        CUDA_LOG("PrefilterSepTemp allocated: %dx%d, %zu bytes",
            m_prefilterSepTemp.width, m_prefilterSepTemp.height, m_prefilterSepTemp.sizeBytes);
    }

    return true;
}

void JustGlowCUDARenderer::ReleaseMipChain() {
    for (auto& mip : m_mipChain) {
        if (mip.devicePtr) {
            cuMemFree(mip.devicePtr);
            mip.devicePtr = 0;
        }
    }
    m_mipChain.clear();

    for (auto& upsample : m_upsampleChain) {
        if (upsample.devicePtr) {
            cuMemFree(upsample.devicePtr);
            upsample.devicePtr = 0;
        }
    }
    m_upsampleChain.clear();

    // Free horizontal temp buffer
    if (m_horizontalTemp.devicePtr) {
        cuMemFree(m_horizontalTemp.devicePtr);
        m_horizontalTemp.devicePtr = 0;
    }

    // Free Gaussian downsample temp buffer
    if (m_gaussianDownsampleTemp.devicePtr) {
        cuMemFree(m_gaussianDownsampleTemp.devicePtr);
        m_gaussianDownsampleTemp.devicePtr = 0;
    }

    // Free separable prefilter temp buffer
    if (m_prefilterSepTemp.devicePtr) {
        cuMemFree(m_prefilterSepTemp.devicePtr);
        m_prefilterSepTemp.devicePtr = 0;
    }

    m_currentMipLevels = 0;
}

// ============================================================================
// Render
// ============================================================================

bool JustGlowCUDARenderer::Render(
    const RenderParams& params,
    CUdeviceptr inputBuffer,
    CUdeviceptr outputBuffer)
{
    CUDA_LOG("=== CUDA Render Begin ===");
    CUDA_LOG("Size: %dx%d, MipLevels: %d, Exposure: %.2f, Threshold: %.2f",
        params.width, params.height, params.mipLevels, params.exposure, params.threshold);

    if (!m_initialized) {
        CUDA_LOG("ERROR: Renderer not initialized");
        return false;
    }

    // Push context
    CUresult err = cuCtxPushCurrent(m_context);
    if (!CheckCUDAError(err, "cuCtxPushCurrent in Render")) {
        return false;
    }

    bool success = true;

    // Allocate MIP chain
    if (!AllocateMipChain(params.width, params.height, params.mipLevels)) {
        CUDA_LOG("ERROR: Failed to allocate MIP chain");
        cuCtxPopCurrent(nullptr);
        return false;
    }

    // Execute pipeline

    // Step 0: Refine - Calculate BoundingBox for input
    // Use offsetPrefilter as blur radius margin
    // NOTE: inputBuffer is the original input size (params.inputWidth x inputHeight),
    //       NOT the expanded output size (params.width x height).
    //       Prefilter will transform the bounds to output coordinates.
    int blurRadiusForPrefilter = static_cast<int>(params.offsetPrefilter * 10.0f + 0.5f);
    CUDA_LOG("--- Refine (Input) ---");
    if (!ExecuteRefine(inputBuffer, params.inputWidth, params.inputHeight, params.srcPitch,
                       params.threshold, blurRadiusForPrefilter, 0)) {
        success = false;
    }

    CUDA_LOG("--- Prefilter ---");
    if (success && !ExecutePrefilter(params, inputBuffer)) {
        success = false;
    }

    if (success) {
        CUDA_LOG("--- Downsample Chain ---");
        if (!ExecuteDownsampleChain(params)) {
            success = false;
        }
    }

    // Synchronize: Downsample must complete before Upsample reads mipChain
    if (success) {
        cuEventRecord(m_syncEvent, m_stream);
        cuStreamWaitEvent(m_stream, m_syncEvent, 0);
    }

    if (success) {
        CUDA_LOG("--- Upsample Chain ---");
        if (!ExecuteUpsampleChain(params)) {
            success = false;
        }
    }

    // Synchronize: Upsample must complete before Composite reads upsampleChain
    if (success) {
        cuEventRecord(m_syncEvent, m_stream);
        cuStreamWaitEvent(m_stream, m_syncEvent, 0);
    }

    if (success) {
        CUDA_LOG("--- Composite ---");
        if (!ExecuteComposite(params, inputBuffer, outputBuffer)) {
            success = false;
        }
    }

    // Synchronize
    err = cuStreamSynchronize(m_stream);
    CheckCUDAError(err, "cuStreamSynchronize");

    // Pop context
    cuCtxPopCurrent(nullptr);

    CUDA_LOG("=== CUDA Render %s ===", success ? "Complete" : "Failed");
    return success;
}

// ============================================================================
// Execute Refine (BoundingBox Calculation)
// ============================================================================

bool JustGlowCUDARenderer::ExecuteRefine(
    CUdeviceptr input, int width, int height, int pitchPixels,
    float threshold, int blurRadius, int mipLevel)
{
    // Initialize bounds to invalid state on GPU
    // minX = width, minY = height (max values, will be reduced by atomicMin)
    // maxX = -1, maxY = -1 (min values, will be increased by atomicMax)
    int initBounds[4] = { width, -1, height, -1 };  // minX, maxX, minY, maxY
    CUresult err = cuMemcpyHtoDAsync(m_refineBoundsGPU, initBounds, 4 * sizeof(int), m_stream);
    if (!CheckCUDAError(err, "cuMemcpyHtoD(refineBounds init)")) {
        return false;
    }

    // Calculate grid dimensions
    int gridX = (width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    int gridY = (height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    // Get pointers to each bound value in GPU buffer
    CUdeviceptr ptrMinX = m_refineBoundsGPU + 0 * sizeof(int);
    CUdeviceptr ptrMaxX = m_refineBoundsGPU + 1 * sizeof(int);
    CUdeviceptr ptrMinY = m_refineBoundsGPU + 2 * sizeof(int);
    CUdeviceptr ptrMaxY = m_refineBoundsGPU + 3 * sizeof(int);

    // Kernel parameters
    void* args[] = {
        const_cast<CUdeviceptr*>(&input),
        &width, &height, &pitchPixels,
        &threshold, &blurRadius,
        &ptrMinX, &ptrMaxX, &ptrMinY, &ptrMaxY
    };

    // Launch kernel
    err = cuLaunchKernel(
        m_refineKernel,
        gridX, gridY, 1,
        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
        0, m_stream,
        args, nullptr);

    if (!CheckCUDAError(err, "cuLaunchKernel(RefineKernel)")) {
        return false;
    }

    // Synchronize to ensure kernel completes before reading back
    err = cuStreamSynchronize(m_stream);
    if (!CheckCUDAError(err, "cuStreamSynchronize(Refine)")) {
        return false;
    }

    // Read back bounds from GPU
    int resultBounds[4];
    err = cuMemcpyDtoH(resultBounds, m_refineBoundsGPU, 4 * sizeof(int));
    if (!CheckCUDAError(err, "cuMemcpyDtoH(refineBounds)")) {
        return false;
    }

    // Store in m_mipBounds
    m_mipBounds[mipLevel].minX = resultBounds[0];
    m_mipBounds[mipLevel].maxX = resultBounds[1];
    m_mipBounds[mipLevel].minY = resultBounds[2];
    m_mipBounds[mipLevel].maxY = resultBounds[3];

    // Check if bounds are valid (at least one pixel above threshold)
    if (!m_mipBounds[mipLevel].valid()) {
        // No pixels above threshold - use full image as fallback
        m_mipBounds[mipLevel].setFull(width, height);
        CUDA_LOG("Refine MIP[%d]: No active pixels, using full image %dx%d",
            mipLevel, width, height);
    } else {
        CUDA_LOG("Refine MIP[%d]: BoundingBox [%d,%d]-[%d,%d] = %dx%d (%.1f%% of %dx%d)",
            mipLevel,
            m_mipBounds[mipLevel].minX, m_mipBounds[mipLevel].minY,
            m_mipBounds[mipLevel].maxX, m_mipBounds[mipLevel].maxY,
            m_mipBounds[mipLevel].width(), m_mipBounds[mipLevel].height(),
            100.0f * m_mipBounds[mipLevel].width() * m_mipBounds[mipLevel].height() / (width * height),
            width, height);
    }

    return true;
}

// ============================================================================
// Execute Prefilter
// ============================================================================

bool JustGlowCUDARenderer::ExecutePrefilter(const RenderParams& params, CUdeviceptr input) {
    auto& dstMip = m_mipChain[0];

    // Calculate BoundingBox in output coordinates
    // Input BoundingBox (m_mipBounds[0]) is in input coordinates
    // Need to transform to output coordinates (with padding offset)
    int offsetX = (dstMip.width - params.inputWidth) / 2;
    int offsetY = (dstMip.height - params.inputHeight) / 2;

    // Blur margin based on offsetPrefilter (sampling radius)
    int blurMargin = static_cast<int>(params.offsetPrefilter * 2.0f + 0.5f);

    // Transform input BoundingBox to output coordinates with blur margin
    const auto& inBounds = m_mipBounds[0];
    int boundMinX = std::max(0, inBounds.minX + offsetX - blurMargin);
    int boundMinY = std::max(0, inBounds.minY + offsetY - blurMargin);
    int boundMaxX = std::min(dstMip.width - 1, inBounds.maxX + offsetX + blurMargin);
    int boundMaxY = std::min(dstMip.height - 1, inBounds.maxY + offsetY + blurMargin);
    int boundWidth = boundMaxX - boundMinX + 1;
    int boundHeight = boundMaxY - boundMinY + 1;

    // Grid size based on BoundingBox
    int gridX = (boundWidth + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    int gridY = (boundHeight + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    float reduction = 100.0f * boundWidth * boundHeight / (dstMip.width * dstMip.height);
    CUDA_LOG("Prefilter: %dx%d -> %dx%d, BBox [%d,%d]-[%d,%d] = %dx%d (%.1f%% of full), quality: %d",
        params.width, params.height, dstMip.width, dstMip.height,
        boundMinX, boundMinY, boundMaxX, boundMaxY,
        boundWidth, boundHeight, reduction, params.prefilterQuality);

    // Calculate color temperature multipliers
    float colorTempR = 1.0f, colorTempG = 1.0f, colorTempB = 1.0f;
    float t = params.colorTemp / 100.0f;
    if (t >= 0.0f) {
        colorTempR = 1.0f + t * 0.3f;
        colorTempB = 1.0f - t * 0.3f;
    } else {
        colorTempR = 1.0f + t * 0.3f;
        colorTempB = 1.0f - t * 0.3f;
    }

    // Common parameters
    int dstPitchPixels = dstMip.width;
    float prefilterIntensity = 1.0f;
    int useHDR = params.hdrMode ? 1 : 0;
    int useLinear = params.linearize ? 1 : 0;
    int inputProfile = params.inputProfile;
    int inputPitch = params.srcPitch;

    CUresult err;

    // Select kernel based on prefilter quality
    // 1=Star13, 2=Grid25, 3=Sep5, 4=Sep9
    if (params.prefilterQuality == 3 || params.prefilterQuality == 4) {
        // =========================================
        // Separable mode: H pass -> V pass
        // NOTE: Separable kernels don't support BoundingBox yet, use full grid
        // =========================================
        bool isSep9 = (params.prefilterQuality == 4);
        int fullGridX = (dstMip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int fullGridY = (dstMip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        CUDA_LOG("Prefilter: Separable %s mode (full grid, no BBox optimization)", isSep9 ? "9+9" : "5+5");

        // H pass: input -> temp
        void* hParams[] = {
            &input,
            &m_prefilterSepTemp.devicePtr,
            (void*)&params.inputWidth,
            (void*)&params.inputHeight,
            (void*)&inputPitch,
            (void*)&dstMip.width,
            (void*)&dstMip.height,
            (void*)&dstPitchPixels,
            (void*)&params.inputWidth,
            (void*)&params.inputHeight,
            (void*)&params.offsetPrefilter
        };

        err = cuLaunchKernel(
            isSep9 ? m_prefilterSep9HKernel : m_prefilterSep5HKernel,
            fullGridX, fullGridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, m_stream,
            hParams, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(PrefilterSepH)")) {
            return false;
        }

        // Sync between H and V passes
        cuEventRecord(m_syncEvent, m_stream);
        cuStreamWaitEvent(m_stream, m_syncEvent, 0);

        // V pass: temp -> output (with threshold/color processing)
        int tempPitch = m_prefilterSepTemp.width;
        void* vParams[] = {
            &m_prefilterSepTemp.devicePtr,
            &dstMip.devicePtr,
            (void*)&m_prefilterSepTemp.width,
            (void*)&m_prefilterSepTemp.height,
            (void*)&tempPitch,
            (void*)&dstMip.width,
            (void*)&dstMip.height,
            (void*)&dstPitchPixels,
            (void*)&params.threshold,
            (void*)&params.softKnee,
            (void*)&params.glowColor[0],
            (void*)&params.glowColor[1],
            (void*)&params.glowColor[2],
            (void*)&colorTempR,
            (void*)&colorTempG,
            (void*)&colorTempB,
            (void*)&params.preserveColor,
            (void*)&useHDR,
            (void*)&useLinear,
            (void*)&inputProfile,
            (void*)&params.offsetPrefilter,
            (void*)&params.paddingThreshold  // Alpha threshold for unpremultiply
        };

        err = cuLaunchKernel(
            isSep9 ? m_prefilterSep9VKernel : m_prefilterSep5VKernel,
            fullGridX, fullGridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, m_stream,
            vParams, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(PrefilterSepV)")) {
            return false;
        }

    } else {
        // =========================================
        // Single-pass mode: Star13 or Grid25
        // =========================================
        bool isGrid25 = (params.prefilterQuality == 2);
        CUDA_LOG("Prefilter: %s mode", isGrid25 ? "25-tap Grid" : "13-tap Star");

        void* kernelParams[] = {
            &input,
            &dstMip.devicePtr,
            (void*)&params.inputWidth,
            (void*)&params.inputHeight,
            (void*)&inputPitch,
            (void*)&dstMip.width,
            (void*)&dstMip.height,
            (void*)&dstPitchPixels,
            (void*)&params.inputWidth,
            (void*)&params.inputHeight,
            (void*)&params.threshold,
            (void*)&params.softKnee,
            (void*)&prefilterIntensity,
            (void*)&params.glowColor[0],
            (void*)&params.glowColor[1],
            (void*)&params.glowColor[2],
            (void*)&colorTempR,
            (void*)&colorTempG,
            (void*)&colorTempB,
            (void*)&params.preserveColor,
            (void*)&useHDR,
            (void*)&useLinear,
            (void*)&inputProfile,
            (void*)&params.offsetPrefilter,
            (void*)&params.paddingThreshold,  // Alpha threshold for unpremultiply
            (void*)&boundMinX,
            (void*)&boundMinY,
            (void*)&boundWidth,
            (void*)&boundHeight
        };

        err = cuLaunchKernel(
            isGrid25 ? m_prefilter25TapKernel : m_prefilterKernel,
            gridX, gridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, m_stream,
            kernelParams, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(Prefilter)")) {
            return false;
        }
    }

    // =========================================
    // Desaturation Kernel (runs after Prefilter, before Downsample)
    // Max-based: blends toward max channel (only adds, never darkens)
    // =========================================
    if (params.desaturation > 0.001f) {
        CUDA_LOG("Desaturation: %.1f%% on mipChain[0] %dx%d",
            params.desaturation * 100.0f, dstMip.width, dstMip.height);

        void* desatParams[] = {
            &dstMip.devicePtr,
            (void*)&dstMip.width,
            (void*)&dstMip.height,
            (void*)&dstPitchPixels,
            (void*)&params.desaturation
        };

        err = cuLaunchKernel(
            m_desaturationKernel,
            gridX, gridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, m_stream,
            desatParams, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(Desaturation)")) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Execute Downsample Chain
// ============================================================================

bool JustGlowCUDARenderer::ExecuteDownsampleChain(const RenderParams& params) {
    // All levels use 2D Gaussian (9-tap, ZeroPad) for temporal stability
    // Kawase removed - caused blur flickering on subpixel movement
    // ZeroPad sampling prevents edge energy concentration across different buffer sizes

    for (int i = 0; i < params.mipLevels - 1; ++i) {
        auto& srcMip = m_mipChain[i];
        auto& dstMip = m_mipChain[i + 1];

        // Calculate blur radius for this level (based on offset + spread)
        float levelRatio = (params.mipLevels > 1) ?
            static_cast<float>(i) / (params.mipLevels - 1) : 0.0f;
        float effectiveOffset = params.offsetDown + params.spreadDown * levelRatio;
        int blurRadiusForLevel = static_cast<int>(effectiveOffset * 5.0f + 0.5f);

        // Execute Refine on source MIP to calculate BoundingBox
        if (!ExecuteRefine(srcMip.devicePtr, srcMip.width, srcMip.height, srcMip.width,
                           params.threshold, blurRadiusForLevel, i + 1)) {
            CUDA_LOG("Warning: Refine failed for MIP[%d], continuing with full size", i);
            m_mipBounds[i + 1].setFull(srcMip.width, srcMip.height);
        }

        // =========================================
        // 2D Gaussian 9-tap (single pass, ZeroPad)
        // Temporally stable - no flickering on movement
        // =========================================

        // Calculate destination BoundingBox from source BoundingBox (divide by 2)
        // Source BoundingBox is stored in m_mipBounds[i+1] from RefineKernel
        const auto& srcBounds = m_mipBounds[i + 1];
        int dstBoundMinX = srcBounds.minX / 2;
        int dstBoundMinY = srcBounds.minY / 2;
        int dstBoundMaxX = (srcBounds.maxX + 1) / 2;  // Round up to include edge pixels
        int dstBoundMaxY = (srcBounds.maxY + 1) / 2;

        // Clamp to destination dimensions
        dstBoundMinX = std::max(0, std::min(dstBoundMinX, dstMip.width - 1));
        dstBoundMinY = std::max(0, std::min(dstBoundMinY, dstMip.height - 1));
        dstBoundMaxX = std::max(0, std::min(dstBoundMaxX, dstMip.width - 1));
        dstBoundMaxY = std::max(0, std::min(dstBoundMaxY, dstMip.height - 1));

        int boundWidth = dstBoundMaxX - dstBoundMinX + 1;
        int boundHeight = dstBoundMaxY - dstBoundMinY + 1;

        // Grid size based on BoundingBox (not full image!)
        int gridX = (boundWidth + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int gridY = (boundHeight + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

        int srcPitchPixels = srcMip.width;
        int dstPitchPixels = dstMip.width;

        // Dynamic offset: offsetDown at level 0, offsetDown + spreadDown at max level
        float offsetDown = params.offsetDown;
        float spreadDown = params.spreadDown;
        int level = i;
        int maxLevels = params.mipLevels;
        float paddingThreshold = params.paddingThreshold;

        float reduction = 100.0f * boundWidth * boundHeight / (dstMip.width * dstMip.height);
        CUDA_LOG("Downsample[%d]: %dx%d -> %dx%d, BBox [%d,%d]-[%d,%d] = %dx%d (%.1f%% of full)",
            i, srcMip.width, srcMip.height, dstMip.width, dstMip.height,
            dstBoundMinX, dstBoundMinY, dstBoundMaxX, dstBoundMaxY,
            boundWidth, boundHeight, reduction);

        void* kernelParams[] = {
            &srcMip.devicePtr,
            &dstMip.devicePtr,
            (void*)&srcMip.width,
            (void*)&srcMip.height,
            (void*)&srcPitchPixels,
            (void*)&dstMip.width,
            (void*)&dstMip.height,
            (void*)&dstPitchPixels,
            (void*)&offsetDown,
            (void*)&spreadDown,
            (void*)&level,
            (void*)&maxLevels,
            (void*)&paddingThreshold,
            (void*)&dstBoundMinX,
            (void*)&dstBoundMinY,
            (void*)&boundWidth,
            (void*)&boundHeight
        };

        CUresult err = cuLaunchKernel(
            m_gaussian2DDownsampleKernel,
            gridX, gridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, m_stream,
            kernelParams, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(Gaussian2DDownsample)")) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Execute Upsample Chain
// ============================================================================

bool JustGlowCUDARenderer::ExecuteUpsampleChain(const RenderParams& params) {
    // Correct upsample logic:
    // Result = GaussianUpsample(Previous) + Current × Weight
    // - Previous = previous upsample result (from deeper level, smaller texture)
    // - Current = stored downsample at current level
    //
    // Buffer mapping:
    // - input (current downsample) = m_mipChain[i]
    // - prevLevel (previous upsample result) = m_upsampleChain[i+1] (or nullptr for deepest)
    // - output = m_upsampleChain[i]
    //
    // All levels use 9-tap Discrete Gaussian (3x3 pattern) with dynamic offset
    // Offset = 1.5 + 0.3*level to prevent center clumping at higher levels
    // NO separable/linear optimization - prevents shift artifacts

    for (int i = params.mipLevels - 1; i >= 0; --i) {
        auto& currMip = m_mipChain[i];       // Current level's stored downsample (input)
        auto& dstUpsample = m_upsampleChain[i];  // Output

        // BoundingBox for this level
        // m_mipBounds[i+1] was calculated on m_mipChain[i] during downsample
        // For level 0, use m_mipBounds[1] (calculated on MIP[0])
        // For deeper levels, use m_mipBounds[i+1]
        int boundIdx = (i + 1 < MAX_MIP_LEVELS) ? (i + 1) : i;
        const auto& bounds = m_mipBounds[boundIdx];

        int boundMinX = bounds.minX;
        int boundMinY = bounds.minY;
        int boundWidth = bounds.width();
        int boundHeight = bounds.height();

        // Clamp to destination dimensions
        boundMinX = std::max(0, std::min(boundMinX, dstUpsample.width - 1));
        boundMinY = std::max(0, std::min(boundMinY, dstUpsample.height - 1));
        int boundMaxX = std::min(boundMinX + boundWidth - 1, dstUpsample.width - 1);
        int boundMaxY = std::min(boundMinY + boundHeight - 1, dstUpsample.height - 1);
        boundWidth = boundMaxX - boundMinX + 1;
        boundHeight = boundMaxY - boundMinY + 1;

        // Grid size based on BoundingBox
        int gridX = (boundWidth + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int gridY = (boundHeight + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

        // offsetUp + spreadUp for dynamic offset calculation in kernel
        float offsetUp = params.offsetUp;
        float spreadUp = params.spreadUp;

        // Level index for weight calculation
        int levelIndex = i;

        // Previous upsample result (from deeper level)
        // For deepest level (i == mipLevels-1), there's no previous result
        CUdeviceptr prevLevel = 0;
        int prevWidth = 0, prevHeight = 0, prevPitchPixels = 0;

        if (i < params.mipLevels - 1) {
            auto& prevUpsample = m_upsampleChain[i + 1];  // Deeper level's upsample result
            prevLevel = prevUpsample.devicePtr;
            prevWidth = prevUpsample.width;
            prevHeight = prevUpsample.height;
            prevPitchPixels = prevUpsample.width;
        }

        // blurMode: 1=3x3 Gaussian (9-tap), 2=5x5 Gaussian (25-tap)
        int blurMode = params.blurMode;
        int compositeMode = params.compositeMode;  // 1=Add, 2=Screen, 3=Overlay

        float reduction = 100.0f * boundWidth * boundHeight / (dstUpsample.width * dstUpsample.height);
        CUDA_LOG("Upsample[%d]: %dx%d -> %dx%d, BBox [%d,%d] %dx%d (%.1f%% of full)",
            i, prevWidth, prevHeight, dstUpsample.width, dstUpsample.height,
            boundMinX, boundMinY, boundWidth, boundHeight, reduction);

        int srcPitchPixels = currMip.width;
        int dstPitchPixels = dstUpsample.width;

        // activeLimit is now 0-1 (radius / 100) for soft threshold
        float activeLimit = params.activeLimit;
        int maxLevels = params.mipLevels;

        void* kernelParams[] = {
            &currMip.devicePtr,      // input (current level's stored downsample)
            &prevLevel,              // prevLevel (previous upsample result, or 0)
            &dstUpsample.devicePtr,  // output
            (void*)&currMip.width,
            (void*)&currMip.height,
            (void*)&srcPitchPixels,
            (void*)&prevWidth,
            (void*)&prevHeight,
            (void*)&prevPitchPixels,
            (void*)&dstUpsample.width,
            (void*)&dstUpsample.height,
            (void*)&dstPitchPixels,
            (void*)&offsetUp,
            (void*)&spreadUp,
            (void*)&levelIndex,
            (void*)&activeLimit,
            (void*)&params.decayK,
            (void*)&params.level1Weight,
            (void*)&params.falloffType,
            (void*)&maxLevels,
            (void*)&blurMode,
            (void*)&compositeMode,
            (void*)&boundMinX,
            (void*)&boundMinY,
            (void*)&boundWidth,
            (void*)&boundHeight
        };

        CUresult err = cuLaunchKernel(
            m_upsampleKernel,
            gridX, gridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, m_stream,
            kernelParams, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(Upsample)")) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Execute Composite
// ============================================================================

bool JustGlowCUDARenderer::ExecuteComposite(
    const RenderParams& params,
    CUdeviceptr original,
    CUdeviceptr output)
{
    int gridX = (params.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    int gridY = (params.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    CUDA_LOG("Composite: %dx%d, grid: %dx%d, exposure: %.2f, debugView: %d",
        params.width, params.height, gridX, gridY, params.exposure, params.debugView);

    // Use upsample result as final glow
    CUdeviceptr glow = m_upsampleChain[0].devicePtr;
    int glowWidth = m_upsampleChain[0].width;
    int glowHeight = m_upsampleChain[0].height;
    int glowPitch = m_upsampleChain[0].width;  // Pitch in pixels

    // Critical check: glowWidth must equal params.width for 1:1 UV mapping
    CUDA_LOG("Composite GLOW CHECK: params.width=%d, glowWidth=%d, match=%s",
        params.width, glowWidth, (params.width == glowWidth) ? "YES" : "NO!!!");

    // Determine debug buffer based on debugView
    // debugView: 1=Final, 2=Prefilter, 3-8=Down1-6, 9-15=Up0-6, 16=GlowOnly
    CUdeviceptr debugBuffer = 0;
    int debugWidth = 0, debugHeight = 0, debugPitch = 0;

    if (params.debugView == 2) {
        // Prefilter = MIP[0]
        debugBuffer = m_mipChain[0].devicePtr;
        debugWidth = m_mipChain[0].width;
        debugHeight = m_mipChain[0].height;
        debugPitch = m_mipChain[0].width;
        CUDA_LOG("Debug: Prefilter (MIP[0]) %dx%d", debugWidth, debugHeight);
    }
    else if (params.debugView >= 3 && params.debugView <= 8) {
        // Down1-6 (3-8) → MIP[1-6]
        int level = params.debugView - 2;  // 3→1, 4→2, ..., 8→6
        if (level >= 0 && level < static_cast<int>(m_mipChain.size())) {
            debugBuffer = m_mipChain[level].devicePtr;
            debugWidth = m_mipChain[level].width;
            debugHeight = m_mipChain[level].height;
            debugPitch = m_mipChain[level].width;
            CUDA_LOG("Debug: Down%d (MIP[%d]) %dx%d", level, level, debugWidth, debugHeight);
        }
    }
    else if (params.debugView >= 9 && params.debugView <= 15) {
        // Up0-6 (9-15)
        int level = params.debugView - 9;  // 9→0, 10→1, ..., 15→6
        if (level >= 0 && level < static_cast<int>(m_upsampleChain.size())) {
            debugBuffer = m_upsampleChain[level].devicePtr;
            debugWidth = m_upsampleChain[level].width;
            debugHeight = m_upsampleChain[level].height;
            debugPitch = m_upsampleChain[level].width;
            CUDA_LOG("Debug: Up%d %dx%d", level, debugWidth, debugHeight);
        }
    }

    // Use DebugOutputKernel for all modes (it handles Final, GlowOnly, and debug views)
    CUDA_LOG("Composite: output=%dx%d, input=%dx%d, sourceOpacity=%.2f, glowOpacity=%.2f",
        params.width, params.height, params.inputWidth, params.inputHeight,
        params.sourceOpacity, params.glowOpacity);
    CUDA_LOG("Composite DEBUG: width=%d, height=%d, inputW=%d, inputH=%d, glowW=%d, glowH=%d, srcPitch=%d, dstPitch=%d",
        params.width, params.height, params.inputWidth, params.inputHeight,
        glowWidth, glowHeight, params.srcPitch, params.dstPitch);
    CUDA_LOG("Composite DEBUG: offsetX=%d, offsetY=%d",
        (params.width - params.inputWidth) / 2, (params.height - params.inputHeight) / 2);

    int useLinear = params.linearize ? 1 : 0;
    int inputProfile = params.inputProfile;  // 1=sRGB, 2=Rec709, 3=Gamma2.2

    // Glow tint color (from params.glowColor[3])
    float glowTintR = params.glowColor[0];
    float glowTintG = params.glowColor[1];
    float glowTintB = params.glowColor[2];

    float dither = params.dither;

    // Chromatic aberration
    float chromaticAberration = params.chromaticAberration;
    float caTintRr = params.caTintR[0];
    float caTintRg = params.caTintR[1];
    float caTintRb = params.caTintR[2];
    float caTintBr = params.caTintB[0];
    float caTintBg = params.caTintB[1];
    float caTintBb = params.caTintB[2];

    // Unpremultiply option
    int unpremultiply = params.unpremultiply ? 1 : 0;

    // BoundingBox for debug mode 17
    int boundMinX = m_mipBounds[0].minX;
    int boundMaxX = m_mipBounds[0].maxX;
    int boundMinY = m_mipBounds[0].minY;
    int boundMaxY = m_mipBounds[0].maxY;
    CUDA_LOG("Composite: BoundingBox [%d,%d]-[%d,%d]", boundMinX, boundMinY, boundMaxX, boundMaxY);

    void* kernelParams[] = {
        &original,
        &debugBuffer,
        &glow,
        &output,
        (void*)&params.width,
        (void*)&params.height,
        (void*)&params.inputWidth,
        (void*)&params.inputHeight,
        (void*)&params.srcPitch,
        (void*)&debugWidth,
        (void*)&debugHeight,
        (void*)&debugPitch,
        (void*)&glowWidth,
        (void*)&glowHeight,
        (void*)&glowPitch,
        (void*)&params.dstPitch,
        (void*)&params.debugView,
        (void*)&params.exposure,
        (void*)&params.sourceOpacity,
        (void*)&params.glowOpacity,
        (void*)&params.compositeMode,
        (void*)&useLinear,
        (void*)&inputProfile,
        (void*)&glowTintR,
        (void*)&glowTintG,
        (void*)&glowTintB,
        (void*)&dither,
        (void*)&chromaticAberration,
        (void*)&caTintRr,
        (void*)&caTintRg,
        (void*)&caTintRb,
        (void*)&caTintBr,
        (void*)&caTintBg,
        (void*)&caTintBb,
        (void*)&unpremultiply,
        (void*)&boundMinX,
        (void*)&boundMaxX,
        (void*)&boundMinY,
        (void*)&boundMaxY
    };

    CUresult err = cuLaunchKernel(
        m_debugOutputKernel,
        gridX, gridY, 1,
        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
        0, m_stream,
        kernelParams, nullptr);

    return CheckCUDAError(err, "cuLaunchKernel(DebugOutput)");
}

// ============================================================================
// RenderWithInterop - Hybrid DX12-CUDA rendering pipeline
// Optimized: Zero-copy Surface I/O + BoundingBox + Parallel Streams
// ============================================================================

bool JustGlowCUDARenderer::RenderWithInterop(
    const RenderParams& params,
    InteropTexture* input,
    InteropTexture** blurredOutputs,
    int numLevels)
{
    CUDA_LOG("=== RenderWithInterop Begin (Optimized) ===");
    CUDA_LOG("Input: %dx%d, numLevels: %d, parallelStreams: %s",
        input->width, input->height, numLevels,
        m_streamsInitialized ? "YES" : "NO");

    if (!m_initialized) {
        CUDA_LOG("ERROR: Renderer not initialized");
        return false;
    }

    if (!input || !input->isValid()) {
        CUDA_LOG("ERROR: Invalid input texture");
        return false;
    }

    // Check if Surface kernels are available for zero-copy path
    bool useSurfaceKernels = (m_prefilterSurfaceKernel && m_downsampleSurfaceKernel);

    // Check required kernels
    if (!m_logTransPreblurHKernel || !m_logTransPreblurVKernel) {
        CUDA_LOG("ERROR: Pre-blur kernels not loaded");
        return false;
    }

    // Push context
    CUresult err = cuCtxPushCurrent(m_context);
    if (!CheckCUDAError(err, "cuCtxPushCurrent in RenderWithInterop")) {
        return false;
    }

    bool success = true;
    int inputWidth = input->width;
    int inputHeight = input->height;

    // ========================================
    // Check for Surface-based zero-copy path
    // ========================================

    if (useSurfaceKernels) {
        CUDA_LOG("Using Surface-based zero-copy pipeline");

        // Collect surface objects and dimensions for MIP chain
        cudaSurfaceObject_t mipSurfaces[MAX_MIP_LEVELS];
        int mipWidths[MAX_MIP_LEVELS];
        int mipHeights[MAX_MIP_LEVELS];
        BoundingBox mipBounds[MAX_MIP_LEVELS];

        // Input surface = level 0
        mipSurfaces[0] = input->cudaSurface;
        mipWidths[0] = inputWidth;
        mipHeights[0] = inputHeight;
        mipBounds[0].setFull(inputWidth, inputHeight);

        // Calculate MIP dimensions
        int w = inputWidth, h = inputHeight;
        for (int i = 1; i <= numLevels && i < MAX_MIP_LEVELS; i++) {
            w = (w + 1) / 2;
            h = (h + 1) / 2;
            if (w < 1) w = 1;
            if (h < 1) h = 1;

            if (blurredOutputs[i - 1] && blurredOutputs[i - 1]->isValid()) {
                mipSurfaces[i] = blurredOutputs[i - 1]->cudaSurface;
                mipWidths[i] = blurredOutputs[i - 1]->width;
                mipHeights[i] = blurredOutputs[i - 1]->height;
            } else {
                CUDA_LOG("ERROR: Blurred output %d is invalid", i);
                success = false;
                break;
            }
        }

        if (success) {
            // ========================================
            // Step 1: Unmult + Prefilter on input surface
            // ========================================

            CUDA_LOG("--- Surface Prefilter (includes Unmult) ---");
            // Prefilter reads from input surface and writes to output[0] surface
            // It includes threshold + color processing
            if (!ExecutePrefilterInterop(params,
                    input->cudaSurface,
                    blurredOutputs[0]->cudaSurface,
                    inputWidth, inputHeight,
                    &mipBounds[0])) {
                CUDA_LOG("ERROR: Prefilter failed");
                success = false;
            }
        }

        if (success) {
            // ========================================
            // Step 2: Downsample chain (Surface-based)
            // ========================================

            CUDA_LOG("--- Surface Downsample Chain ---");
            // Use blurredOutputs[0] as level 0 for downsample
            mipSurfaces[0] = blurredOutputs[0]->cudaSurface;
            mipWidths[0] = blurredOutputs[0]->width;
            mipHeights[0] = blurredOutputs[0]->height;

            if (!ExecuteDownsampleChainInterop(params,
                    mipSurfaces, mipWidths, mipHeights,
                    numLevels + 1, mipBounds)) {
                CUDA_LOG("ERROR: Downsample chain failed");
                success = false;
            }
        }

        // Sync before parallel Pre-blur
        cuEventRecord(m_syncEvent, m_stream);
        cuStreamWaitEvent(m_stream, m_syncEvent, 0);

        if (success) {
            // ========================================
            // Step 3: Parallel Pre-blur (CUDA Streams)
            // ========================================

            CUDA_LOG("--- Parallel Log-Transmittance Pre-blur ---");

            // Use parallel streams if available
            bool useParallel = m_streamsInitialized && (numLevels > 1);
            CUDA_LOG("Pre-blur mode: %s", useParallel ? "PARALLEL" : "SEQUENTIAL");

            // We need temp surfaces for H-pass output
            // Use m_upsampleChain as temp buffers (already allocated)

            for (int level = 1; level <= numLevels; level++) {
                if (level - 1 >= (int)m_upsampleChain.size()) {
                    CUDA_LOG("ERROR: No temp buffer for level %d", level);
                    continue;
                }

                InteropTexture* output = blurredOutputs[level - 1];
                if (!output || !output->isValid()) continue;

                // Select stream for parallel execution
                CUstream execStream = m_stream;
                if (useParallel && level - 1 < MAX_PARALLEL_STREAMS) {
                    execStream = m_preblurStreams[level - 1];
                }

                // Get bounds for this level
                const BoundingBox* levelBounds = (level < MAX_MIP_LEVELS) ? &mipBounds[level] : nullptr;

                // Create temp surface for H-pass output
                // For now, use in-place processing with the output surface
                // H-pass writes to a temp position, V-pass reads it

                // Pre-blur: output surface → temp → output surface
                // Since we need a temp buffer, use m_upsampleChain[level]
                auto& tempBuffer = m_upsampleChain[level - 1];

                int gridX = (output->width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
                int gridY = (output->height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

                float baseSigma = BASE_BLUR_SIGMA;
                int width = output->width;
                int height = output->height;

                // BoundingBox parameters
                int boundMinX = levelBounds ? levelBounds->minX : 0;
                int boundMinY = levelBounds ? levelBounds->minY : 0;
                int boundWidth = levelBounds ? levelBounds->width() : width;
                int boundHeight = levelBounds ? levelBounds->height() : height;

                CUDA_LOG("Pre-blur level %d: %dx%d, σ=%.1f×√%d=%.1f, stream=%p, BBox [%d,%d] %dx%d",
                    level, width, height, baseSigma, level, baseSigma * sqrtf((float)level),
                    execStream, boundMinX, boundMinY, boundWidth, boundHeight);

                // H-pass: output surface → linear temp buffer
                {
                    // Copy surface to temp buffer for processing
                    cudaError_t cudaErr = cudaMemcpy2DFromArrayAsync(
                        (void*)tempBuffer.devicePtr,
                        tempBuffer.pitchBytes,
                        output->cudaArray,
                        0, 0,
                        width * 4 * sizeof(float),
                        height,
                        cudaMemcpyDeviceToDevice,
                        (cudaStream_t)execStream);

                    if (cudaErr != cudaSuccess) {
                        CUDA_LOG("WARNING: Copy from surface failed for level %d", level);
                        continue;
                    }

                    // H-pass on linear buffer
                    int srcPitch = width;
                    int dstPitch = tempBuffer.width;

                    void* hParams[] = {
                        &tempBuffer.devicePtr,
                        &m_mipChain[level - 1].devicePtr,  // Use mipChain as temp
                        &width, &height,
                        &srcPitch, &dstPitch,
                        &level, &baseSigma
                    };

                    err = cuLaunchKernel(
                        m_logTransPreblurHKernel,
                        gridX, gridY, 1,
                        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
                        0, execStream,
                        hParams, nullptr);

                    if (!CheckCUDAError(err, "cuLaunchKernel(PreblurH)")) {
                        success = false;
                    }
                }

                // V-pass: mipChain → tempBuffer
                if (success) {
                    int srcPitch = m_mipChain[level - 1].width;
                    int dstPitch = tempBuffer.width;

                    void* vParams[] = {
                        &m_mipChain[level - 1].devicePtr,
                        &tempBuffer.devicePtr,
                        &width, &height,
                        &srcPitch, &dstPitch,
                        &level, &baseSigma
                    };

                    err = cuLaunchKernel(
                        m_logTransPreblurVKernel,
                        gridX, gridY, 1,
                        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
                        0, execStream,
                        vParams, nullptr);

                    if (!CheckCUDAError(err, "cuLaunchKernel(PreblurV)")) {
                        success = false;
                    }
                }

                // Copy result back to output surface
                if (success) {
                    cudaError_t cudaErr = cudaMemcpy2DToArrayAsync(
                        output->cudaArray,
                        0, 0,
                        (void*)tempBuffer.devicePtr,
                        tempBuffer.pitchBytes,
                        width * 4 * sizeof(float),
                        height,
                        cudaMemcpyDeviceToDevice,
                        (cudaStream_t)execStream);

                    if (cudaErr != cudaSuccess) {
                        CUDA_LOG("WARNING: Copy to surface failed for level %d", level);
                    }
                }
            }

            // Wait for all parallel streams to complete
            if (useParallel) {
                for (int i = 0; i < std::min(numLevels, MAX_PARALLEL_STREAMS); i++) {
                    cuStreamSynchronize(m_preblurStreams[i]);
                }
            }
        }

    } else {
        // ========================================
        // Fallback: Copy-based pipeline (legacy)
        // ========================================

        CUDA_LOG("Using copy-based fallback pipeline");

        size_t inputSize = inputWidth * inputHeight * 4 * sizeof(float);
        CUdeviceptr inputBuffer = 0;
        err = cuMemAlloc(&inputBuffer, inputSize);
        if (!CheckCUDAError(err, "cuMemAlloc(inputBuffer)")) {
            cuCtxPopCurrent(nullptr);
            return false;
        }

        int processLevels = (numLevels < 2) ? 2 : numLevels;
        if (!AllocateMipChain(inputWidth, inputHeight, processLevels + 1)) {
            cuMemFree(inputBuffer);
            cuCtxPopCurrent(nullptr);
            return false;
        }

        // Copy input surface → linear buffer
        cudaError_t cudaErr = cudaMemcpy2DFromArray(
            (void*)inputBuffer,
            inputWidth * 4 * sizeof(float),
            input->cudaArray,
            0, 0,
            inputWidth * 4 * sizeof(float),
            inputHeight,
            cudaMemcpyDeviceToDevice);

        if (cudaErr != cudaSuccess) {
            CUDA_LOG("ERROR: cudaMemcpy2DFromArray failed");
            cuMemFree(inputBuffer);
            cuCtxPopCurrent(nullptr);
            return false;
        }

        // Prefilter
        if (!ExecutePrefilter(params, inputBuffer)) {
            success = false;
        }

        // Downsample chain
        if (success && !ExecuteDownsampleChain(params)) {
            success = false;
        }

        cuEventRecord(m_syncEvent, m_stream);
        cuStreamWaitEvent(m_stream, m_syncEvent, 0);

        // Pre-blur and copy to output surfaces
        if (success) {
            for (int level = 1; level <= numLevels && level < processLevels + 1; level++) {
                if (!blurredOutputs[level - 1] || !blurredOutputs[level - 1]->isValid()) continue;

                auto& mip = m_mipChain[level];
                auto& upsample = m_upsampleChain[level];

                int gridX = (mip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
                int gridY = (mip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

                float baseSigma = BASE_BLUR_SIGMA;
                int srcPitch = mip.width;
                int dstPitch = upsample.width;

                // H-pass
                void* hParams[] = {
                    &mip.devicePtr, &upsample.devicePtr,
                    &mip.width, &mip.height,
                    &srcPitch, &dstPitch,
                    &level, &baseSigma
                };
                cuLaunchKernel(m_logTransPreblurHKernel, gridX, gridY, 1,
                    THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1, 0, m_stream, hParams, nullptr);

                cuEventRecord(m_syncEvent, m_stream);
                cuStreamWaitEvent(m_stream, m_syncEvent, 0);

                // V-pass
                void* vParams[] = {
                    &upsample.devicePtr, &mip.devicePtr,
                    &mip.width, &mip.height,
                    &dstPitch, &srcPitch,
                    &level, &baseSigma
                };
                cuLaunchKernel(m_logTransPreblurVKernel, gridX, gridY, 1,
                    THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1, 0, m_stream, vParams, nullptr);

                cuStreamSynchronize(m_stream);

                // Copy to output surface
                InteropTexture* output = blurredOutputs[level - 1];
                cudaMemcpy2DToArray(output->cudaArray, 0, 0,
                    (void*)mip.devicePtr,
                    mip.width * 4 * sizeof(float),
                    output->width * 4 * sizeof(float),
                    output->height,
                    cudaMemcpyDeviceToDevice);
            }
        }

        cuMemFree(inputBuffer);
    }

    // Final synchronize
    err = cuStreamSynchronize(m_stream);
    CheckCUDAError(err, "cuStreamSynchronize");

    // Pop context
    cuCtxPopCurrent(nullptr);

    CUDA_LOG("=== RenderWithInterop %s ===", success ? "Complete" : "Failed");
    return success;
}

// ============================================================================
// Initialize/Destroy Parallel Streams
// ============================================================================

bool JustGlowCUDARenderer::InitializeStreams() {
    if (m_streamsInitialized) {
        return true;
    }

    CUDA_LOG("Initializing %d parallel streams for Pre-blur", MAX_PARALLEL_STREAMS);

    for (int i = 0; i < MAX_PARALLEL_STREAMS; i++) {
        CUresult err = cuStreamCreate(&m_preblurStreams[i], CU_STREAM_NON_BLOCKING);
        if (err != CUDA_SUCCESS) {
            CUDA_LOG("ERROR: Failed to create stream %d", i);
            // Clean up already created streams
            for (int j = 0; j < i; j++) {
                cuStreamDestroy(m_preblurStreams[j]);
                m_preblurStreams[j] = nullptr;
            }
            return false;
        }
    }

    m_streamsInitialized = true;
    CUDA_LOG("Parallel streams initialized successfully");
    return true;
}

void JustGlowCUDARenderer::DestroyStreams() {
    if (!m_streamsInitialized) {
        return;
    }

    CUDA_LOG("Destroying parallel streams");
    for (int i = 0; i < MAX_PARALLEL_STREAMS; i++) {
        if (m_preblurStreams[i]) {
            cuStreamDestroy(m_preblurStreams[i]);
            m_preblurStreams[i] = nullptr;
        }
    }

    m_streamsInitialized = false;
}

// ============================================================================
// Execute Clear Surface
// ============================================================================

bool JustGlowCUDARenderer::ExecuteClearSurface(
    cudaSurfaceObject_t surface,
    int width, int height)
{
    if (!m_clearSurfaceKernel) {
        CUDA_LOG("WARNING: ClearSurfaceKernel not loaded");
        return false;
    }

    int gridX = (width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    int gridY = (height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    void* args[] = {
        &surface,
        &width, &height
    };

    CUresult err = cuLaunchKernel(
        m_clearSurfaceKernel,
        gridX, gridY, 1,
        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
        0, m_stream,
        args, nullptr);

    return CheckCUDAError(err, "cuLaunchKernel(ClearSurface)");
}

// ============================================================================
// Interop Helper Methods (Surface-based with BoundingBox)
// ============================================================================

bool JustGlowCUDARenderer::ExecuteUnmultInterop(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    int width, int height,
    const BoundingBox* bounds)
{
    if (!m_unmultSurfaceKernel) {
        CUDA_LOG("ERROR: UnmultSurfaceKernel not loaded");
        return false;
    }

    // Use BoundingBox if provided, otherwise full image
    int boundMinX = bounds ? bounds->minX : 0;
    int boundMinY = bounds ? bounds->minY : 0;
    int boundWidth = bounds ? bounds->width() : width;
    int boundHeight = bounds ? bounds->height() : height;

    int gridX = (boundWidth + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    int gridY = (boundHeight + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    CUDA_LOG("ExecuteUnmultInterop: %dx%d, BBox [%d,%d] %dx%d (%.1f%%)",
        width, height, boundMinX, boundMinY, boundWidth, boundHeight,
        100.0f * boundWidth * boundHeight / (width * height));

    void* args[] = {
        &input, &output,
        &width, &height,
        &boundMinX, &boundMinY, &boundWidth, &boundHeight
    };

    CUresult err = cuLaunchKernel(
        m_unmultSurfaceKernel,
        gridX, gridY, 1,
        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
        0, m_stream,
        args, nullptr);

    return CheckCUDAError(err, "cuLaunchKernel(UnmultSurface)");
}

bool JustGlowCUDARenderer::ExecutePrefilterInterop(
    const RenderParams& params,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    int width, int height,
    const BoundingBox* bounds)
{
    if (!m_prefilterSurfaceKernel) {
        CUDA_LOG("ERROR: PrefilterSurfaceKernel not loaded");
        return false;
    }

    // Use BoundingBox if provided, otherwise full image
    int boundMinX = bounds ? bounds->minX : 0;
    int boundMinY = bounds ? bounds->minY : 0;
    int boundWidth = bounds ? bounds->width() : width;
    int boundHeight = bounds ? bounds->height() : height;

    int gridX = (boundWidth + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    int gridY = (boundHeight + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    CUDA_LOG("ExecutePrefilterInterop: %dx%d, threshold=%.2f, softKnee=%.2f, BBox %dx%d",
        width, height, params.threshold, params.softKnee, boundWidth, boundHeight);

    float threshold = params.threshold;
    float softKnee = params.softKnee;
    float glowR = params.glowColor[0];
    float glowG = params.glowColor[1];
    float glowB = params.glowColor[2];
    float preserveColor = params.preserveColor;

    void* args[] = {
        &input, &output,
        &width, &height,
        &threshold, &softKnee,
        &glowR, &glowG, &glowB, &preserveColor,
        &boundMinX, &boundMinY, &boundWidth, &boundHeight
    };

    CUresult err = cuLaunchKernel(
        m_prefilterSurfaceKernel,
        gridX, gridY, 1,
        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
        0, m_stream,
        args, nullptr);

    return CheckCUDAError(err, "cuLaunchKernel(PrefilterSurface)");
}

bool JustGlowCUDARenderer::ExecuteDownsampleChainInterop(
    const RenderParams& params,
    cudaSurfaceObject_t* mipSurfaces,
    int* mipWidths, int* mipHeights,
    int numLevels,
    BoundingBox* mipBounds)
{
    if (!m_downsampleSurfaceKernel) {
        CUDA_LOG("ERROR: DownsampleSurfaceKernel not loaded");
        return false;
    }

    // Initialize first level bounds to full image if not provided
    if (mipBounds) {
        mipBounds[0].setFull(mipWidths[0], mipHeights[0]);
    }

    for (int i = 0; i < numLevels - 1; i++) {
        int srcWidth = mipWidths[i];
        int srcHeight = mipHeights[i];
        int dstWidth = mipWidths[i + 1];
        int dstHeight = mipHeights[i + 1];

        // Calculate destination BoundingBox from source BoundingBox
        int boundMinX = 0, boundMinY = 0, boundWidth = dstWidth, boundHeight = dstHeight;
        if (mipBounds) {
            const auto& srcBounds = mipBounds[i];
            boundMinX = srcBounds.minX / 2;
            boundMinY = srcBounds.minY / 2;
            int boundMaxX = (srcBounds.maxX + 1) / 2;
            int boundMaxY = (srcBounds.maxY + 1) / 2;
            boundMinX = std::max(0, std::min(boundMinX, dstWidth - 1));
            boundMinY = std::max(0, std::min(boundMinY, dstHeight - 1));
            boundMaxX = std::max(0, std::min(boundMaxX, dstWidth - 1));
            boundMaxY = std::max(0, std::min(boundMaxY, dstHeight - 1));
            boundWidth = boundMaxX - boundMinX + 1;
            boundHeight = boundMaxY - boundMinY + 1;

            // Store destination bounds for next level
            mipBounds[i + 1].minX = boundMinX;
            mipBounds[i + 1].minY = boundMinY;
            mipBounds[i + 1].maxX = boundMaxX;
            mipBounds[i + 1].maxY = boundMaxY;
        }

        int gridX = (boundWidth + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int gridY = (boundHeight + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

        float offset = params.offsetDown;
        cudaSurfaceObject_t srcSurf = mipSurfaces[i];
        cudaSurfaceObject_t dstSurf = mipSurfaces[i + 1];

        CUDA_LOG("DownsampleInterop[%d]: %dx%d -> %dx%d, BBox [%d,%d] %dx%d",
            i, srcWidth, srcHeight, dstWidth, dstHeight,
            boundMinX, boundMinY, boundWidth, boundHeight);

        void* args[] = {
            &srcSurf, &dstSurf,
            &srcWidth, &srcHeight, &dstWidth, &dstHeight,
            &offset,
            &boundMinX, &boundMinY, &boundWidth, &boundHeight
        };

        CUresult err = cuLaunchKernel(
            m_downsampleSurfaceKernel,
            gridX, gridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, m_stream,
            args, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(DownsampleSurface)")) {
            return false;
        }
    }

    return true;
}

bool JustGlowCUDARenderer::ExecuteLogTransPreblurInterop(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t tempBuffer,
    cudaSurfaceObject_t output,
    int width, int height,
    int level,
    float baseSigma,
    const BoundingBox* bounds,
    CUstream stream)
{
    if (!m_logTransPreblurHKernel || !m_logTransPreblurVKernel) {
        CUDA_LOG("ERROR: Pre-blur kernels not loaded");
        return false;
    }

    // Use provided stream or default
    CUstream execStream = stream ? stream : m_stream;

    // Use BoundingBox if provided, otherwise full image
    int boundMinX = bounds ? bounds->minX : 0;
    int boundMinY = bounds ? bounds->minY : 0;
    int boundWidth = bounds ? bounds->width() : width;
    int boundHeight = bounds ? bounds->height() : height;

    int gridX = (boundWidth + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    int gridY = (boundHeight + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    CUDA_LOG("PreblurInterop level %d: %dx%d, σ=%.1f×√%d=%.1f, BBox [%d,%d] %dx%d",
        level, width, height, baseSigma, level, baseSigma * sqrtf((float)level),
        boundMinX, boundMinY, boundWidth, boundHeight);

    // H-pass: input → tempBuffer
    {
        void* hArgs[] = {
            &input, &tempBuffer,
            &width, &height, &level, &baseSigma,
            &boundMinX, &boundMinY, &boundWidth, &boundHeight
        };

        CUresult err = cuLaunchKernel(
            m_logTransPreblurHKernel,
            gridX, gridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, execStream,
            hArgs, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(PreblurH Surface)")) {
            return false;
        }
    }

    // Sync between H and V passes (within this stream)
    if (execStream == m_stream) {
        cuEventRecord(m_syncEvent, execStream);
        cuStreamWaitEvent(execStream, m_syncEvent, 0);
    } else {
        // For parallel streams, use stream synchronization
        cuStreamSynchronize(execStream);
    }

    // V-pass: tempBuffer → output
    {
        void* vArgs[] = {
            &tempBuffer, &output,
            &width, &height, &level, &baseSigma,
            &boundMinX, &boundMinY, &boundWidth, &boundHeight
        };

        CUresult err = cuLaunchKernel(
            m_logTransPreblurVKernel,
            gridX, gridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, execStream,
            vArgs, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(PreblurV Surface)")) {
            return false;
        }
    }

    return true;
}

#endif // _WIN32
