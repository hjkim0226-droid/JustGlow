# JustGlow Session Notes (2025-12-18)

## Recent Fixes

### 1. Windows min/max Macro Conflict (C2589, C2059)
- **File:** `src/JustGlow.cpp` line 716
- **Problem:** Windows headers define `min` and `max` as macros, interfering with `std::min`/`std::max`
- **Error messages:**
  - `error C2589: '(': illegal token on right side of '::'`
  - `error C2059: syntax error: ')'`
- **Fix:** Wrapped function names in parentheses to prevent macro expansion
```cpp
// Before (broken)
glowExpansion = std::max(64, std::min(glowExpansion, 1024));

// After (fixed)
glowExpansion = (std::max)(64, (std::min)(glowExpansion, 1024));
```

### 2. Bool to Int Type Mismatch for CUDA Kernel
- **File:** `src/JustGlowCUDARenderer.cpp` ExecutePrefilter function
- **Problem:** CUDA kernel `PrefilterKernel` expects `int useHDR` but was passed pointer to `bool params.hdrMode`
- **Fix:** Created local int variable with explicit conversion
```cpp
// Before (broken)
void* kernelParams[] = {
    // ... other params ...
    (void*)&params.hdrMode  // bool - WRONG for CUDA kernel expecting int
};

// After (fixed)
int useHDR = params.hdrMode ? 1 : 0;
void* kernelParams[] = {
    // ... other params ...
    (void*)&useHDR
};
```

## Commit
- **Message:** "fix: Windows build errors - min/max macro and bool type"
- **Branch:** main
- **Status:** Pushed, waiting for GitHub Actions build result

## Key Files

| File | Description |
|------|-------------|
| `src/JustGlow.cpp` | Main plugin implementation, PreRender function |
| `src/JustGlowCUDARenderer.cpp` | CUDA rendering pipeline |
| `src/JustGlowKernels.cu` | CUDA kernels (Prefilter, Downsample, Upsample, Composite) |
| `src/JustGlowParams.h` | Parameter structures, MIP chain configuration |

## Pending Tasks
- [ ] Verify GitHub Actions build success
- [ ] Test alpha channel handling on Windows
- [ ] Test boundary expansion on Windows

## Algorithm Overview
- **Dual Kawase blur** with MIP pyramid
- 13-tap prefilter (Karis Average for HDR)
- 5-tap downsample with X/+ rotation alternation
- 9-tap tent upsample with falloff-based blending
- Dynamic MIP levels (up to 12, until min dimension < 16px)
