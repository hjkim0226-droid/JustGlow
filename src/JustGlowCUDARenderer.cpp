/**
 * JustGlow CUDA Renderer Implementation
 *
 * CUDA Driver API based rendering pipeline.
 * Loads PTX kernels and executes the glow effect pipeline.
 */

#ifdef _WIN32

#include <windows.h>
#include "JustGlowCUDARenderer.h"
#include <fstream>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <cstdarg>

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
    CUDA_LOG("--- Prefilter ---");
    if (!ExecutePrefilter(params, inputBuffer)) {
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
// Execute Prefilter
// ============================================================================

bool JustGlowCUDARenderer::ExecutePrefilter(const RenderParams& params, CUdeviceptr input) {
    auto& dstMip = m_mipChain[0];

    // Calculate grid dimensions
    int gridX = (dstMip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
    int gridY = (dstMip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    CUDA_LOG("Prefilter: %dx%d -> %dx%d, grid: %dx%d, quality: %d",
        params.width, params.height, dstMip.width, dstMip.height, gridX, gridY, params.prefilterQuality);

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
        // =========================================
        bool isSep9 = (params.prefilterQuality == 4);
        CUDA_LOG("Prefilter: Separable %s mode", isSep9 ? "9+9" : "5+5");

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
            gridX, gridY, 1,
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
            gridX, gridY, 1,
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
            (void*)&params.paddingThreshold  // Alpha threshold for unpremultiply
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

        // =========================================
        // 2D Gaussian 9-tap (single pass, ZeroPad)
        // Temporally stable - no flickering on movement
        // =========================================
        int gridX = (dstMip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int gridY = (dstMip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

        int srcPitchPixels = srcMip.width;
        int dstPitchPixels = dstMip.width;

        // Dynamic offset: offsetDown at level 0, offsetDown + spreadDown at max level
        float offsetDown = params.offsetDown;
        float spreadDown = params.spreadDown;
        int level = i;
        int maxLevels = params.mipLevels;
        float paddingThreshold = params.paddingThreshold;

        CUDA_LOG("Downsample[%d]: 2D Gaussian %dx%d -> %dx%d (ZeroPad, offset=%.2f, spread=%.2f, padThresh=%.4f)",
            i, srcMip.width, srcMip.height, dstMip.width, dstMip.height, offsetDown, spreadDown, paddingThreshold);

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
            (void*)&paddingThreshold
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

        int gridX = (dstUpsample.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int gridY = (dstUpsample.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

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

        CUDA_LOG("Upsample[%d]: %s %dx%d -> %dx%d (offset=%.2f, spread=%.2f, blend=%d)",
            i, blurMode == 2 ? "5x5 Gaussian" : "3x3 Gaussian",
            prevWidth, prevHeight, dstUpsample.width, dstUpsample.height, offsetUp, spreadUp, compositeMode);

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
            (void*)&compositeMode
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
        (void*)&unpremultiply
    };

    CUresult err = cuLaunchKernel(
        m_debugOutputKernel,
        gridX, gridY, 1,
        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
        0, m_stream,
        kernelParams, nullptr);

    return CheckCUDAError(err, "cuLaunchKernel(DebugOutput)");
}

#endif // _WIN32
