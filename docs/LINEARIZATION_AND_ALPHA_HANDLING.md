# JustGlow Linearization & Alpha Handling Guide

## Overview

This document summarizes findings from experimenting with linearization and alpha handling in GPU glow effects.

---

## Key Concepts

### Premultiplied vs Straight Alpha

| Type | RGB Storage | Use Case |
|------|-------------|----------|
| **Premultiplied** | `RGB × Alpha` | Compositing, blending, blur operations |
| **Straight** | `RGB` (pure color) | Color space conversions, color grading |

### VFX Golden Rules

1. **Blur in Premultiplied** - Prevents edge fringing
2. **Color conversion on Straight** - sRGB↔Linear operates on pure RGB values
3. **After Effects provides Premultiplied** - All AE buffers are premultiplied

---

## Linearization Pipeline

### When Linearize is ON

```
Input (Premult sRGB)
    ↓
Unpremultiply      → Straight sRGB
    ↓
sRGB → Linear      → Straight Linear
    ↓
Premultiply        → Premult Linear  ← [BLUR HERE]
    ↓
[Blur Pipeline - Downsample/Upsample]
    ↓
Unpremultiply      → Straight Linear
    ↓
Linear → sRGB      → Straight sRGB
    ↓
Premultiply        → Premult sRGB (Output)
```

### When Linearize is OFF

```
Input (Premult sRGB)
    ↓
[Stay in Premult sRGB]  ← [BLUR HERE]
    ↓
[Blur Pipeline]
    ↓
Output (Premult sRGB)
```

---

## Threshold Application

### Correct: Apply on Premultiplied

```cuda
// After Premultiply (whether Linear or sRGB)
resR = straightR * sumA;
resG = straightG * sumA;
resB = straightB * sumA;

// Apply threshold on Premultiplied values
softThreshold(resR, resG, resB, threshold, softKnee);
```

**Reasoning:**
- Threshold determines which pixels "glow"
- Should be consistent regardless of Linearize setting
- Premultiplied values include alpha-weighted brightness

### Wrong: Threshold before Premultiply

Applying threshold on Straight values causes different behavior when Linearize is ON vs OFF.

---

## AE Linearize Working Space

### When AE Project Linearize is ON

- AE delivers **Premultiplied Linear** to plugins
- Plugin's "Linearize" option = convert **back to sRGB** if needed
- Usually keep plugin Linearize OFF (already linear)

### When AE Project Linearize is OFF

- AE delivers **Premultiplied sRGB** to plugins
- Plugin's "Linearize" option = convert **sRGB to Linear**
- Use this for HDR headroom (values > 1.0)

---

## Input Profile Options

When plugin Linearize is ON, choose input color profile:

| Profile | Gamma | Use Case |
|---------|-------|----------|
| **sRGB** | ~2.2 + linear toe | Standard web/display content |
| **Rec.709** | 2.4 | Broadcast video (BT.709) |
| **Gamma 2.2** | Pure 2.2 | Legacy content, some games |

### Conversion Functions

```cuda
// sRGB to Linear (standard)
float srgbToLinear(float c) {
    return c <= 0.04045f
        ? c / 12.92f
        : pow((c + 0.055f) / 1.055f, 2.4f);
}

// Rec.709 to Linear
float rec709ToLinear(float c) {
    return c < 0.081f
        ? c / 4.5f
        : pow((c + 0.099f) / 1.099f, 1.0f / 0.45f);
}

// Gamma 2.2 to Linear
float gamma22ToLinear(float c) {
    return pow(c, 2.2f);
}
```

---

## Experimental Observations

### AE View Options Effect

| Glow Setting | AE View | Result |
|--------------|---------|--------|
| Non-linearized | Non-linear view | Normal glow |
| Non-linearized | Linear view | Glow appears larger, alpha areas more visible |
| Linearized | Non-linear view | Less glow overall |
| Linearized | Linear view | Alpha areas visible, glow reduced |

### Key Insight

> "AE's Linear View (View LUT) only rescues 'Color' but discards 'Alpha'"

The View LUT gamma correction applies to RGB but not to alpha channel, causing apparent differences in glow extent when alpha is involved.

---

## Common Issues

### Edge Brightening (Fringing)

**Symptom:** Bright edges on transparent backgrounds

**Cause:** Dividing by near-zero alpha amplifies color

**Fix:**
```cuda
if (alpha > 0.001f) {
    straightR = premultR / alpha;
} else {
    straightR = 0.0f;  // Not premultR which causes fringing
}
```

### Glow Clipping

**Symptom:** Glow abruptly cuts off at edges

**Cause:** Out-of-bounds sampling returns garbage

**Fix:** Zero-padded bilinear sampling
```cuda
float4 sampleBilinearZeroPad(float* tex, int x, int y, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h) return {0,0,0,0};
    // ... normal sampling
}
```

### HDR Fireflies

**Symptom:** Bright pixels causing star-like artifacts

**Cause:** Single very bright pixels dominate blur

**Fix:** Karis Average - weight samples by inverse brightness
```cuda
float w = 1.0f / (1.0f + brightness);
// Apply w to all samples before averaging
```

---

## Implementation Checklist

- [x] Prefilter: Unpremult → Convert → Premult → Threshold
- [x] Downsample/Upsample: Operate on Premultiplied
- [x] Composite: Convert back (Unpremult → Inverse Convert → Premult)
- [x] Debug View: Show correct color space
- [x] Input Profile dropdown (sRGB/Rec709/Gamma2.2)
- [x] Per-profile conversion functions in CUDA kernels

---

## References

- After Effects SDK: SmartFX GPU Rendering
- VFX Best Practices: Premultiplied Alpha
- sRGB Standard: IEC 61966-2-1
- BT.709: ITU-R Recommendation BT.709
