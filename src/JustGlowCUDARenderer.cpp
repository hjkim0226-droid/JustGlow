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
    , m_horizontalBlurKernel(nullptr)
    , m_gaussianDownsampleHKernel(nullptr)
    , m_gaussianDownsampleVKernel(nullptr)
    , m_debugOutputKernel(nullptr)
    , m_horizontalTemp{}
    , m_gaussianDownsampleTemp{}
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

    err = cuModuleGetFunction(&m_horizontalBlurKernel, m_module, "HorizontalBlurKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(HorizontalBlurKernel)")) {
        return false;
    }
    CUDA_LOG("HorizontalBlurKernel loaded");

    err = cuModuleGetFunction(&m_gaussianDownsampleHKernel, m_module, "GaussianDownsampleHKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(GaussianDownsampleHKernel)")) {
        return false;
    }
    CUDA_LOG("GaussianDownsampleHKernel loaded");

    err = cuModuleGetFunction(&m_gaussianDownsampleVKernel, m_module, "GaussianDownsampleVKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(GaussianDownsampleVKernel)")) {
        return false;
    }
    CUDA_LOG("GaussianDownsampleVKernel loaded");

    err = cuModuleGetFunction(&m_debugOutputKernel, m_module, "DebugOutputKernel");
    if (!CheckCUDAError(err, "cuModuleGetFunction(DebugOutputKernel)")) {
        return false;
    }
    CUDA_LOG("DebugOutputKernel loaded");

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
    m_horizontalBlurKernel = nullptr;
    m_gaussianDownsampleHKernel = nullptr;
    m_gaussianDownsampleVKernel = nullptr;
    m_debugOutputKernel = nullptr;
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
        mip.pitch = w * 4 * sizeof(float);  // RGBA float
        mip.sizeBytes = mip.pitch * h;

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
        upsample.pitch = mip.pitch;
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
        m_horizontalTemp.pitch = m_upsampleChain[1].pitch;
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
        m_gaussianDownsampleTemp.pitch = m_mipChain[0].pitch;
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
    // Prefilter doesn't need exposure boost - it's applied in upsample
    float prefilterIntensity = 1.0f;
    // Convert bool to int for CUDA kernel (bool pointer would be wrong size)
    int useHDR = params.hdrMode ? 1 : 0;

    // srcPitch is for the input buffer (AE's input layer)
    // Use actual srcPitch from AE (may include padding)
    int inputPitch = params.srcPitch;

    void* kernelParams[] = {
        &input,
        &dstMip.devicePtr,
        (void*)&params.inputWidth,   // srcWidth = actual input size
        (void*)&params.inputHeight,  // srcHeight = actual input size
        (void*)&inputPitch,          // srcPitch = input width in pixels
        (void*)&dstMip.width,
        (void*)&dstMip.height,
        (void*)&dstPitchPixels,
        (void*)&params.inputWidth,   // inputWidth for offset calculation
        (void*)&params.inputHeight,  // inputHeight for offset calculation
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
        (void*)&useHDR
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
    // Hybrid downsample: Gaussian (9-tap separable) for Level 0-2, Kawase for 3+
    // Gaussian preserves near-glow detail, Kawase is faster for deep levels
    constexpr int GAUSSIAN_LEVELS = 3;  // Use Gaussian for levels 0, 1, 2

    for (int i = 0; i < params.mipLevels - 1; ++i) {
        auto& srcMip = m_mipChain[i];
        auto& dstMip = m_mipChain[i + 1];

        // Use per-level blurOffset (decays from spread to 1.5px for deeper levels)
        float blurOffset = params.blurOffsets[i];

        if (i < GAUSSIAN_LEVELS) {
            // =========================================
            // Gaussian 9-tap Separable (2 passes)
            // Pass 1: Horizontal blur at source resolution
            // Pass 2: Vertical blur + 2x downsample
            // =========================================

            // Pass 1: Horizontal Gaussian blur (src -> temp at src resolution)
            int hGridX = (srcMip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
            int hGridY = (srcMip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
            int srcPitchPixels = srcMip.width;

            CUDA_LOG("Downsample[%d]: Gaussian H-blur %dx%d, blurOffset=%.2f",
                i, srcMip.width, srcMip.height, blurOffset);

            void* hBlurParams[] = {
                &srcMip.devicePtr,
                &m_gaussianDownsampleTemp.devicePtr,
                (void*)&srcMip.width,
                (void*)&srcMip.height,
                (void*)&srcPitchPixels,
                (void*)&blurOffset
            };

            CUresult err = cuLaunchKernel(
                m_gaussianDownsampleHKernel,
                hGridX, hGridY, 1,
                THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
                0, m_stream,
                hBlurParams, nullptr);

            if (!CheckCUDAError(err, "cuLaunchKernel(GaussianDownsampleH)")) {
                return false;
            }

            // Pass 2: Vertical Gaussian blur + 2x downsample (temp -> dst)
            int vGridX = (dstMip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
            int vGridY = (dstMip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
            int dstPitchPixels = dstMip.width;

            CUDA_LOG("Downsample[%d]: Gaussian V-blur+downsample %dx%d -> %dx%d",
                i, srcMip.width, srcMip.height, dstMip.width, dstMip.height);

            void* vBlurParams[] = {
                &m_gaussianDownsampleTemp.devicePtr,
                &dstMip.devicePtr,
                (void*)&srcMip.width,
                (void*)&srcMip.height,
                (void*)&srcPitchPixels,
                (void*)&dstMip.width,
                (void*)&dstMip.height,
                (void*)&dstPitchPixels,
                (void*)&blurOffset
            };

            err = cuLaunchKernel(
                m_gaussianDownsampleVKernel,
                vGridX, vGridY, 1,
                THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
                0, m_stream,
                vBlurParams, nullptr);

            if (!CheckCUDAError(err, "cuLaunchKernel(GaussianDownsampleV)")) {
                return false;
            }
        } else {
            // =========================================
            // Kawase 5-tap (single pass, faster)
            // =========================================
            int gridX = (dstMip.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
            int gridY = (dstMip.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

            // Alternate between X (diagonal) and + (cross) patterns
            // This breaks up boxy artifacts -> rounder glow
            int rotationMode = i % 2;  // 0=X, 1=+

            CUDA_LOG("Downsample[%d]: Kawase %dx%d -> %dx%d, rotation=%s, blurOffset=%.2f",
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
    }

    return true;
}

// ============================================================================
// Execute Upsample Chain
// ============================================================================

bool JustGlowCUDARenderer::ExecuteUpsampleChain(const RenderParams& params) {
    // Correct upsample logic (from Gemini correction):
    // Result = BlurUpsample(Previous) + Current × Weight
    // - Previous = previous upsample result (from deeper level, smaller texture)
    // - Current = stored downsample at current level
    //
    // Buffer mapping:
    // - input (current downsample) = m_mipChain[i]
    // - prevLevel (previous upsample result) = m_upsampleChain[i+1] (or nullptr for deepest)
    // - output = m_upsampleChain[i]
    //
    // Blur modes:
    // - Level 0-4: 2-pass Separable Gaussian (HorizontalBlur + VerticalUpsample)
    // - Level 5+:  Single-pass Tent filter (3x3)

    for (int i = params.mipLevels - 1; i >= 0; --i) {
        auto& currMip = m_mipChain[i];       // Current level's stored downsample (input)
        auto& dstUpsample = m_upsampleChain[i];  // Output

        int gridX = (dstUpsample.width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
        int gridY = (dstUpsample.height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

        // Use per-level blurOffset
        float blurOffset = params.blurOffsets[i];

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

        // Determine blur mode: 0=Tent, 1=Vertical Gaussian (separable)
        int blurMode = (i <= 4 && prevLevel != 0) ? 1 : 0;

        // For levels 0-4 with prevLevel: run 2-pass separable Gaussian
        if (blurMode == 1) {
            // Pass 1: Horizontal Gaussian blur on prevLevel → m_horizontalTemp
            int hGridX = (prevWidth + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
            int hGridY = (prevHeight + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

            void* hBlurParams[] = {
                &prevLevel,                      // input
                &m_horizontalTemp.devicePtr,     // output
                (void*)&prevWidth,
                (void*)&prevHeight,
                (void*)&prevPitchPixels,
                (void*)&blurOffset
            };

            CUresult err = cuLaunchKernel(
                m_horizontalBlurKernel,
                hGridX, hGridY, 1,
                THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
                0, m_stream,
                hBlurParams, nullptr);

            if (!CheckCUDAError(err, "cuLaunchKernel(HorizontalBlur)")) {
                return false;
            }

            CUDA_LOG("Upsample[%d]: 2-pass Separable Gaussian, H-blur %dx%d", i, prevWidth, prevHeight);

            // Pass 2: Vertical Gaussian upsample from horizontalTemp
            // Update prevLevel pointer to use the horizontally blurred result
            CUdeviceptr blurredPrev = m_horizontalTemp.devicePtr;

            int srcPitchPixels = currMip.width;
            int dstPitchPixels = dstUpsample.width;

            void* kernelParams[] = {
                &currMip.devicePtr,      // input (current level's stored downsample)
                &blurredPrev,            // prevLevel (horizontally blurred)
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
                (void*)&blurOffset,
                (void*)&levelIndex,
                (void*)&params.activeLimit,
                (void*)&params.decayK,
                (void*)&params.level1Weight,
                (void*)&params.falloffType,
                (void*)&blurMode
            };

            err = cuLaunchKernel(
                m_upsampleKernel,
                gridX, gridY, 1,
                THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
                0, m_stream,
                kernelParams, nullptr);

            if (!CheckCUDAError(err, "cuLaunchKernel(Upsample-VerticalGaussian)")) {
                return false;
            }
        } else {
            // Single-pass Tent filter for levels 5+ or when no prevLevel
            CUDA_LOG("Upsample[%d]: Tent filter %dx%d -> %dx%d",
                i, prevWidth, prevHeight, dstUpsample.width, dstUpsample.height);

            int srcPitchPixels = currMip.width;
            int dstPitchPixels = dstUpsample.width;

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
                (void*)&blurOffset,
                (void*)&levelIndex,
                (void*)&params.activeLimit,
                (void*)&params.decayK,
                (void*)&params.level1Weight,
                (void*)&params.falloffType,
                (void*)&blurMode
            };

            CUresult err = cuLaunchKernel(
                m_upsampleKernel,
                gridX, gridY, 1,
                THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,
                0, m_stream,
                kernelParams, nullptr);

            if (!CheckCUDAError(err, "cuLaunchKernel(Upsample-Tent)")) {
                return false;
            }
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

    // Determine debug buffer based on debugView
    // debugView: 1=Final, 2=Prefilter, 3-9=Down0-6, 10-16=Up0-6, 17=GlowOnly
    CUdeviceptr debugBuffer = 0;
    int debugWidth = 0, debugHeight = 0, debugPitch = 0;

    if (params.debugView >= 2 && params.debugView <= 9) {
        // Prefilter (2) or Down0-6 (3-9)
        // Prefilter result is in m_mipChain[0]
        // Down0 is also m_mipChain[0], Down1 is m_mipChain[1], etc.
        int level = (params.debugView == 2) ? 0 : (params.debugView - 3);
        if (level >= 0 && level < static_cast<int>(m_mipChain.size())) {
            debugBuffer = m_mipChain[level].devicePtr;
            debugWidth = m_mipChain[level].width;
            debugHeight = m_mipChain[level].height;
            debugPitch = m_mipChain[level].width;
            CUDA_LOG("Debug: Down level %d (%dx%d)", level, debugWidth, debugHeight);
        }
    }
    else if (params.debugView >= 10 && params.debugView <= 16) {
        // Up0-6 (10-16)
        int level = params.debugView - 10;
        if (level >= 0 && level < static_cast<int>(m_upsampleChain.size())) {
            debugBuffer = m_upsampleChain[level].devicePtr;
            debugWidth = m_upsampleChain[level].width;
            debugHeight = m_upsampleChain[level].height;
            debugPitch = m_upsampleChain[level].width;
            CUDA_LOG("Debug: Up level %d (%dx%d)", level, debugWidth, debugHeight);
        }
    }

    // Use DebugOutputKernel for all modes (it handles Final, GlowOnly, and debug views)
    CUDA_LOG("Composite: output=%dx%d, input=%dx%d, sourceOpacity=%.2f, glowOpacity=%.2f",
        params.width, params.height, params.inputWidth, params.inputHeight,
        params.sourceOpacity, params.glowOpacity);

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
        (void*)&params.compositeMode
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
