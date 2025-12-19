# JustGlow Architecture & Technical Documentation

**Version:** 1.3.0
**Date:** 2025-12-19
**Last Review:** 코드 검토 완료

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

### v1.3.0 (2025-12-19) - "Documentation Complete"
- ✅ Full code review completed
- ✅ CUDA implementation documented
- ✅ Alpha-weighted normalization (edge artifact fix)
- ✅ sampleBilinearZeroPad (boundary handling)
- ✅ sRGB→Linear conversion order fix
- 📝 Known issues documented (see CODE_REVIEW_REPORT.md)

### v1.2.0 (2025-12-18) - "Edge Fix"
- ✅ Fixed edge clipping with zero-pad sampling
- ✅ Fixed alpha channel handling (premultiplied)
- ✅ Debug view modes for pipeline inspection

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

## 8. GPU Rendering Architecture

### 8.1 Supported Frameworks

| Platform | Framework | Status |
|----------|-----------|--------|
| Windows | DirectX 12 | ✅ Production |
| Windows | CUDA | ✅ Production |
| macOS | Metal | 🔜 Planned |

### 8.2 DirectX 12 vs CUDA Comparison

| Aspect | DirectX 12 | CUDA |
|--------|------------|------|
| Shader Format | Compiled CSO | PTX (JIT) |
| Memory | D3D12 Resources | cuMemAlloc |
| Synchronization | ID3D12Fence | cuStream |
| Context | AE-managed Device | AE-managed CUcontext |
| Texture Sampling | Hardware Samplers | Manual Bilinear |

### 8.3 CUDA Buffer Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    CUDA MEMORY LAYOUT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT (from AE)                                                 │
│  └─ CUdeviceptr (RGBA F32, premultiplied alpha)                 │
│                                                                  │
│  MIP CHAIN (Downsample Results)                                  │
│  ├─ m_mipChain[0]: Level 0 prefiltered (full resolution)        │
│  ├─ m_mipChain[1]: Level 1 (1/2 × 1/2)                          │
│  ├─ m_mipChain[2]: Level 2 (1/4 × 1/4)                          │
│  └─ ...up to m_mipChain[11] for Ultra quality                   │
│                                                                  │
│  UPSAMPLE CHAIN (Separate from MIP to prevent race conditions)  │
│  ├─ m_upsampleChain[0]: Final upsampled result                  │
│  ├─ m_upsampleChain[1]: Upsampled from level 2                  │
│  └─ ...mirrors MIP chain depth                                   │
│                                                                  │
│  TEMP BUFFERS                                                    │
│  ├─ m_horizontalTemp: Separable Gaussian horizontal pass        │
│  └─ m_gaussianDownsampleTemp: Gaussian vertical pass            │
│                                                                  │
│  OUTPUT (to AE)                                                  │
│  └─ CUdeviceptr (RGBA F32, composite result)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.4 CUDA Kernel Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CUDA KERNEL FLOW                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. PrefilterKernel                                                   │
│     ├─ Input: AE source buffer                                       │
│     ├─ Output: m_mipChain[0]                                         │
│     └─ Operations:                                                    │
│         • 13-tap circle sampling (sampleBilinearZeroPad)             │
│         • Soft threshold application                                  │
│         • Karis Average (HDR firefly prevention)                     │
│         • Alpha-weighted normalization                                │
│         • sRGB → Linear conversion                                   │
│                                                                       │
│  2. GaussianDownsampleH/VKernel (Levels 0-4)                         │
│     ├─ Input: Previous MIP level                                     │
│     ├─ Output: m_horizontalTemp → m_mipChain[level+1]                │
│     └─ Pattern: 9-tap separable Gaussian                             │
│                                                                       │
│  3. DownsampleKernel (Levels 5+)                                     │
│     ├─ Input: Previous MIP level                                     │
│     ├─ Output: m_mipChain[level+1]                                   │
│     └─ Pattern: 5-tap Kawase (X/+ rotation)                          │
│                                                                       │
│  4. UpsampleKernel (from deepest to level 0)                         │
│     ├─ Input: Deeper level + current MIP level                       │
│     ├─ Output: m_upsampleChain[level]                                │
│     └─ Operations:                                                    │
│         • 9-tap tent filter                                           │
│         • Falloff-weighted blending                                   │
│                                                                       │
│  5. DebugOutputKernel                                                 │
│     ├─ Input: m_upsampleChain[0] + AE source                         │
│     ├─ Output: AE output buffer                                       │
│     └─ Operations:                                                    │
│         • Composite (Add/Screen/Overlay)                              │
│         • Alpha expansion                                             │
│         • Linear → sRGB conversion                                   │
│         • Debug view modes                                            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.5 Synchronization Requirements

현재 구현은 단일 스트림 직렬화에 의존하지만, 명시적 동기화가 권장됩니다:

```cpp
// 권장 패턴 (아직 미구현)
ExecutePrefilter(...);
cuEventRecord(prefilterDone, m_stream);
cuStreamWaitEvent(m_stream, prefilterDone, 0);
ExecuteDownsampleChain(...);
```

---

## 9. Build Instructions

```bash
# Windows (CUDA)
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

# Output
build/Release/JustGlow_v1.1.0.aex
build/Release/CUDA_Assets/JustGlowKernels.ptx
```

---

## 10. Future Improvements

| Feature | Description | Priority |
|---------|-------------|----------|
| Metal Support | macOS GPU rendering | High |
| Kernel Synchronization | Explicit event-based sync | High |
| CPU Fallback | Proper glow on non-GPU systems | Medium |
| Dithering | Reduce banding in gradients | Medium |
| Tone Mapping | HDR to SDR with artistic control | Medium |
| FP16 | Half precision for deep MIP levels | Low |
| Shared Memory | Cache optimization for bilinear | Low |

---

## 11. Related Documents

| Document | Description |
|----------|-------------|
| `docs/CODE_REVIEW_REPORT.md` | 전체 코드 검토 보고서 |
| `docs/CUDA_IMPLEMENTATION.md` | CUDA 구현 상세 문서 |
| `docs/AE_GPU_SDK_REFERENCE.md` | AE GPU SDK 참조 |
| `docs/AE_GPU_CUDA_TROUBLESHOOTING.md` | CUDA 트러블슈팅 |
| `CLAUDE.md` | 개발 가이드 |

---

*Generated by Claude Code*
