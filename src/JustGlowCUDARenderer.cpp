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
    , m_module(nullptr)
    , m_prefilterKernel(nullptr)
    , m_downsampleKernel(nullptr)
    , m_upsampleKernel(nullptr)
    , m_compositeKernel(nullptr)
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
            return true;
        }
    }

    ReleaseMipChain();

    m_mipChain.resize(levels);
    m_currentMipLevels = levels;
    m_currentWidth = width;
    m_currentHeight = height;

    int w = width;
    int h = height;

    for (int i = 0; i < levels; ++i) {
        auto& mip = m_mipChain[i];
        mip.width = w;
        mip.height = h;
        mip.pitch = w * 4 * sizeof(float);  // RGBA float
        mip.sizeBytes = mip.pitch * h;

        CUresult err = cuMemAlloc(&mip.devicePtr, mip.sizeBytes);
        if (!CheckCUDAError(err, "cuMemAlloc for MIP")) {
            CUDA_LOG("ERROR: Failed to allocate MIP level %d (%dx%d)", i, w, h);
            ReleaseMipChain();
            return false;
        }
        CUDA_LOG("MIP[%d] allocated: %dx%d, %zu bytes", i, w, h, mip.sizeBytes);

        // Next level is half size
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
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
    CUDA_LOG("Size: %dx%d, MipLevels: %d, Intensity: %.2f, Threshold: %.2f",
        params.width, params.height, params.mipLevels, params.intensity, params.threshold);

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

    if (success) {
        CUDA_LOG("--- Upsample Chain ---");
        if (!ExecuteUpsampleChain(params)) {
            success = false;
        }
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

    CUDA_LOG("Prefilter: %dx%d -> %dx%d, grid: %dx%d",
        params.width, params.height, dstMip.width, dstMip.height, gridX, gridY);

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

    // Kernel parameters
    int dstPitchPixels = dstMip.width;  // Pitch in pixels, not bytes
    void* kernelParams[] = {
        &input,
        &dstMip.devicePtr,
        (void*)&params.width,
        (void*)&params.height,
        (void*)&params.srcPitch,
        (void*)&dstMip.width,
        (void*)&dstMip.height,
        (void*)&dstPitchPixels,  // Fixed: was dstMip.pitch (bytes), now pixels
        (void*)&params.threshold,
        (void*)&params.softKnee,
        (void*)&params.intensity,
        (void*)&params.glowColor[0],
        (void*)&params.glowColor[1],
        (void*)&params.glowColor[2],
        (void*)&colorTempR,
        (void*)&colorTempG,
        (void*)&colorTempB,
        (void*)&params.preserveColor,
        (void*)&params.hdrMode
    };

    CUresult err = cuLaunchKernel(
        m_prefilterKernel,
        gridX, gridY, 1,
        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
        0, m_stream,
        kernelParams, nullptr);

    return CheckCUDAError(err, "cuLaunchKernel(Prefilter)");
}

// ============================================================================
// Execute Downsample Chain
// ============================================================================

bool JustGlowCUDARenderer::ExecuteDownsampleChain(const RenderParams& params) {
    for (int i = 0; i < params.mipLevels - 1; ++i) {
        auto& srcMip = m_mipChain[i];
        auto& dstMip = m_mipChain[i + 1];

        int gridX = (dstMip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int gridY = (dstMip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

        // Alternate between X (diagonal) and + (cross) patterns
        // This breaks up boxy artifacts -> rounder glow
        int rotationMode = i % 2;  // 0=X, 1=+

        // Use fixed blurOffset from Spread parameter (1.0-3.5px, prevents ghosting)
        float blurOffset = params.blurOffset;

        CUDA_LOG("Downsample[%d]: %dx%d -> %dx%d, rotation=%s, blurOffset=%.2f",
            i, srcMip.width, srcMip.height, dstMip.width, dstMip.height,
            rotationMode == 0 ? "X" : "+", blurOffset);

        int srcPitchPixels = srcMip.width;  // Pitch in pixels, not floats
        int dstPitchPixels = dstMip.width;  // Pitch in pixels, not floats

        void* kernelParams[] = {
            &srcMip.devicePtr,
            &dstMip.devicePtr,
            (void*)&srcMip.width,
            (void*)&srcMip.height,
            (void*)&srcPitchPixels,
            (void*)&dstMip.width,
            (void*)&dstMip.height,
            (void*)&dstPitchPixels,
            (void*)&blurOffset,
            (void*)&rotationMode
        };

        CUresult err = cuLaunchKernel(
            m_downsampleKernel,
            gridX, gridY, 1,
            THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
            0, m_stream,
            kernelParams, nullptr);

        if (!CheckCUDAError(err, "cuLaunchKernel(Downsample)")) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// Execute Upsample Chain
// ============================================================================

bool JustGlowCUDARenderer::ExecuteUpsampleChain(const RenderParams& params) {
    for (int i = params.mipLevels - 2; i >= 0; --i) {
        auto& srcMip = m_mipChain[i + 1];  // Smaller
        auto& dstMip = m_mipChain[i];      // Larger

        int gridX = (dstMip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int gridY = (dstMip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

        // Use fixed blurOffset from Spread parameter (1.0-3.5px, prevents ghosting)
        float blurOffset = params.blurOffset;

        // Level index for weight calculation (inverted: deepest level first)
        int levelIndex = i;

        CUDA_LOG("Upsample[%d]: %dx%d -> %dx%d, activeLimit=%.2f, decayK=%.2f, exposure=%.2f, falloffType=%d",
            i, srcMip.width, srcMip.height, dstMip.width, dstMip.height,
            params.activeLimit, params.decayK, params.exposure, params.falloffType);

        int srcPitchPixels = srcMip.width;  // Pitch in pixels, not floats
        int dstPitchPixels = dstMip.width;  // Pitch in pixels, not floats

        // For upsample, prevLevel is the current destination (for blending)
        CUdeviceptr prevLevel = dstMip.devicePtr;

        // New kernel parameters for advanced falloff system
        void* kernelParams[] = {
            &srcMip.devicePtr,
            &prevLevel,
            &dstMip.devicePtr,
            (void*)&srcMip.width,
            (void*)&srcMip.height,
            (void*)&srcPitchPixels,
            (void*)&dstMip.width,
            (void*)&dstMip.height,
            (void*)&dstPitchPixels,
            (void*)&blurOffset,
            (void*)&levelIndex,
            (void*)&params.activeLimit,
            (void*)&params.decayK,
            (void*)&params.exposure,
            (void*)&params.falloffType
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

    CUDA_LOG("Composite: %dx%d, grid: %dx%d", params.width, params.height, gridX, gridY);

    CUdeviceptr glow = m_mipChain[0].devicePtr;
    int glowPitch = m_mipChain[0].width;  // Pitch in pixels, not floats

    // Intensity is now applied in UpsampleKernel as exposure
    // CompositeKernel just needs 1.0 to preserve the already-scaled glow
    float intensity = 1.0f;

    void* kernelParams[] = {
        &original,
        &glow,
        &output,
        (void*)&params.width,
        (void*)&params.height,
        (void*)&params.srcPitch,
        (void*)&glowPitch,
        (void*)&params.dstPitch,
        (void*)&intensity,
        (void*)&params.compositeMode
    };

    CUresult err = cuLaunchKernel(
        m_compositeKernel,
        gridX, gridY, 1,
        THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
        0, m_stream,
        kernelParams, nullptr);

    return CheckCUDAError(err, "cuLaunchKernel(Composite)");
}

#endif // _WIN32
