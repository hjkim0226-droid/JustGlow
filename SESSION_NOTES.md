# JustGlow Session Notes (2025-12-18)

## Current Version
- **Version:** 1.0.0 (defined in `src/JustGlow.h`)
- **Next push:** Bump to 1.1.0 for algorithm improvements

## Recent Changes (This Session)

### 1. Distance-Based Weight Calculation
- **File:** `src/JustGlowKernels.cu`
- Changed from level-based to actual pixel distance
- `distance = pow(2, level + 1)` gives 2, 4, 8, 16... pixels
- More physically accurate light falloff

### 2. Separable Gaussian for Detail Levels
- **File:** `src/JustGlowKernels.cu` UpsampleKernel
- Levels 0-4: Separable Gaussian (cross pattern, weights [1,4,6,4,1]/16)
- Levels 5+: Tent filter (3x3, 9-tap)
- Smoother text/edge handling on detail levels

### 3. Exposure Moved to Composite
- **File:** `src/JustGlowCUDARenderer.cpp`
- Removed exposure from UpsampleKernel
- Applied in CompositeKernel to prevent accumulation explosion

### 4. Non-linear Threshold Mapping
- **File:** `src/JustGlow.cpp`
- UI 0-100 maps to actual 0-70 with quadratic curve
- `threshold = (ui/100)^2 * 70`
- Better control in practical range

### 5. Radius Factor in glowExpansion
- **File:** `src/JustGlow.cpp`
- `radiusFactor = 0.5 + (radius / 200.0)`
- Max increased from 1024 to 2048
- Fixes buffer overflow at radius=100

## Build Status
- All commits pushed and building successfully on GitHub Actions
- Latest: "feat: Add separable Gaussian for levels 0-4, tent for 5+"

## Key Files

| File | Description |
|------|-------------|
| `src/JustGlow.cpp` | Main plugin, PreRender, parameter mapping |
| `src/JustGlow.h` | Version definitions, parameter IDs |
| `src/JustGlowCUDARenderer.cpp` | CUDA rendering pipeline |
| `src/JustGlowKernels.cu` | CUDA kernels (weight calc, Gaussian/Tent upsample) |
| `src/JustGlowParams.h` | Parameter structures, MIP chain config |

## Algorithm Overview
- **Dual Kawase blur** with MIP pyramid
- 13-tap prefilter (Karis Average for HDR)
- 5-tap downsample with X/+ rotation alternation
- **Hybrid upsample:** Separable Gaussian (0-4) + Tent (5+)
- Distance-based falloff: `pow(0.5, distance * decayK * 0.02)`
- Dynamic MIP levels (up to 12, until min dimension < 16px)

## Weight Formula
```cpp
float distance = powf(2.0f, level + 1.0f);  // 2, 4, 8, 16...
float k = decayK * 0.02f;
// Exponential falloff
return powf(0.5f, distance * k);
```
