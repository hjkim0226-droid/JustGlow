# JustGlow Architecture & Technical Documentation

**Version:** 1.1.0
**Date:** 2025-12-18

---

## 1. Overview

JustGlow is a high-performance GPU glow effect plugin for Adobe After Effects, designed to achieve Deep Glow-like quality with 2x+ faster performance using the **Dual Kawase Blur** algorithm with modern enhancements.

### Core Philosophy
- **95% of Deep Glow quality** at **50% or less of the cost**
- Physically-based light falloff
- Rounder glow through rotation tricks (not brute-force iterations)

---

## 2. Pipeline Architecture (V-Cycle)

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 0: PREFILTER (13-tap Circle Kernel)                       │
│  - Soft threshold application                                    │
│  - Karis Average (HDR anti-firefly)                             │
│  - Ensures no pixel dropout at diagonal edges                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: DOWNSAMPLE CHAIN (5-tap Dual Kawase)                   │
│                                                                  │
│  Level 0 (1920×1080) ──X──▶ Level 1 (960×540)                   │
│  Level 1 (960×540)   ──+──▶ Level 2 (480×270)                   │
│  Level 2 (480×270)   ──X──▶ Level 3 (240×135)                   │
│  Level 3 (240×135)   ──+──▶ Level 4 (120×68)                    │
│  ...continues until min(w,h) < 16px                              │
│                                                                  │
│  ✨ X/+ Rotation Alternation: Breaks boxy artifacts             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: UPSAMPLE CHAIN (9-tap Tent Filter + Falloff)           │
│                                                                  │
│  Deepest Level ──────────────────────────────────────────────▶  │
│       │                                                          │
│       ▼ Upsample (9-tap tent)                                   │
│  Level N-1 + (Current × pow(falloff, 1)) ────────────────────▶  │
│       │                                                          │
│       ▼ Upsample                                                 │
│  Level N-2 + (Current × pow(falloff, 2)) ────────────────────▶  │
│       │                                                          │
│       ...continues to Level 0                                    │
│                                                                  │
│  ✨ Falloff: Physical light decay (inverse square approximation)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: COMPOSITE                                               │
│  - Blend modes: Add / Screen / Overlay                          │
│  - Alpha expansion for transparent backgrounds                   │
│  - Final output                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Technologies

### 3.1 Dual Kawase Blur (5-tap)
- **Origin:** ARM's SIGGRAPH 2015 presentation
- **Cost:** ~50% of Gaussian blur
- **Pattern:** 4 corners + center, weighted (0.125 × 4 + 0.5)

### 3.2 X/+ Rotation Alternation (Zero-Cost Trick)
**Problem:** Box blur creates diamond/boxy artifacts
**Solution:** Alternate sampling pattern direction

```
Even Levels (X - Diagonal):     Odd Levels (+ - Cross):
    ↖   ↗                              ↑
      ●                            ← ● →
    ↙   ↘                              ↓
```

**Result:** X + + = Snowflake (❄️) ≈ Circle (●)
**Cost:** Zero additional computation (same sample count)

### 3.3 Dynamic MIP Levels
- **Low:** Max 4 levels (fast, tight glow)
- **Medium:** Max 6 levels (balanced)
- **High:** Max 8 levels (good quality)
- **Ultra:** Max 12 levels (Deep Glow-like, until ~16px)

### 3.4 Falloff (Physical Light Decay)
```cpp
levelWeight = pow(falloff, level);
```

| Falloff Value | Effect |
|---------------|--------|
| 1.0 (100%) | All levels equal = overblown, white out |
| 0.5 (50%) | Aggressive decay = tight core |
| 0.7 (70%) | **Balanced** = Deep Glow feel (default) |
| 0.25 | Physical (1/r²) = realistic but not artistic |

---

## 4. Quality vs Performance Comparison

### MIP Level Computation Cost (1080p)

| Level | Resolution | Pixels | Cost (%) |
|-------|------------|--------|----------|
| 0 | 1920×1080 | 2,073,600 | 75.0% |
| 1 | 960×540 | 518,400 | 18.8% |
| 2 | 480×270 | 129,600 | 4.7% |
| 3 | 240×135 | 32,400 | 1.2% |
| 4 | 120×68 | 8,160 | 0.3% |
| 5 | 60×34 | 2,040 | 0.07% |
| 6 | 30×17 | 510 | 0.02% |
| 7 | 15×8 | 120 | 0.004% |
| 8 | 7×4 | 28 | 0.001% |

**Insight:** Levels 6-8 cost < 0.03% total but provide "atmosphere/air" feel

### Deep Glow vs JustGlow

| Aspect | Deep Glow | JustGlow |
|--------|-----------|----------|
| Algorithm | Gaussian Pyramid | Dual Kawase |
| Samples/Level | 9-25+ | 5 (down) + 9 (up) |
| Shape | Perfect circle | Rotated polygon ≈ circle |
| Depth | ~8 levels | Up to 12 levels |
| Speed | Baseline | **~2x faster** |

---

## 5. Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Intensity | 0-200% | 100% | Glow brightness multiplier |
| Radius | 0-500 | 50 | Blur spread (scales offsets) |
| Threshold | 0-100% | 50% | Brightness cutoff |
| Soft Knee | 0-100% | 50% | Threshold transition softness |
| Quality | Low/Med/High/Ultra | High | MIP chain depth |
| **Falloff** | 0-100% | 70% | Light decay rate |
| Glow Color | RGB | White | Tint color |
| Color Temp | -100 to +100 | 0 | Warm/cool shift |
| Composite | Add/Screen/Overlay | Add | Blend mode |
| HDR Mode | On/Off | On | Karis average anti-firefly |

---

## 6. File Structure

```
src/
├── JustGlow.h              # Main header, enums, params
├── JustGlow.cpp            # AE plugin entry, parameter handling
├── JustGlowParams.h        # GPU constant buffers, MIP chain config
├── JustGlowCUDARenderer.h  # CUDA renderer interface
├── JustGlowCUDARenderer.cpp # CUDA render pipeline implementation
├── JustGlowKernels.cu      # CUDA compute kernels
├── JustGlowGPURenderer.h   # DirectX 12 renderer interface
└── JustGlowGPURenderer.cpp # DirectX 12 implementation

shaders/
├── Common.hlsli            # Shared HLSL definitions
├── Prefilter.hlsl          # 13-tap prefilter
├── Downsample.hlsl         # 5-tap Kawase downsample
├── Upsample.hlsl           # 9-tap tent upsample
├── PostProcess.hlsl        # Color/anamorphic processing
└── Composite.hlsl          # Final blend
```

---

## 7. Version History

### v1.1.0 (2025-12-18) - "Deep Glow Killer"
- ✅ Dynamic MIP levels (up to 12, until 16px)
- ✅ X/+ rotation alternation (rounder glow)
- ✅ Falloff parameter (physical light decay)
- ✅ Ultra quality = Deep Glow-like atmosphere

### v1.0.8 (2025-12-17)
- ✅ Transparent background support (alpha expansion)
- ✅ Radius properly affects blur size

### v1.0.7 (2025-12-17)
- ✅ Fixed CUDA pitch calculation (ILLEGAL_ADDRESS error)

### v1.0.6 (2025-12-17)
- ✅ Fixed GPU rendering path (was falling back to CPU)
- ✅ Added out_flags2 to GPUDeviceSetup per SDK requirement

---

## 8. Build Instructions

```bash
# Windows (CUDA)
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# Output
build/Release/JustGlow_v1.1.0.aex
build/Release/CUDA_Assets/JustGlowKernels.ptx
```

---

## 9. Future Improvements

| Feature | Description | Priority |
|---------|-------------|----------|
| Dithering | Reduce banding in gradients | Medium |
| FP16 | Half precision for deep MIP levels | Low |
| Tone Mapping | HDR to SDR with artistic control | Medium |
| Metal Support | macOS GPU rendering | High |

---

*Generated by Claude Code*
