# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JustGlow is a GPU-accelerated glow effect plugin for Adobe After Effects. It uses DirectX 12 compute shaders on Windows with planned Metal support for macOS.

**Core Algorithm:** Dual Kawase blur with Karis Average for HDR firefly prevention, 13-tap downsample, 9-tap tent upsample.

## Build Commands

```bash
# Configure (Windows)
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release --parallel

# Install to After Effects plugin directory
cmake --install build
```

The build produces `JustGlow.aex` (plugin) and `DirectX_Assets/*.cso` (compiled shaders).

## Architecture

**Plugin Entry Flow:**
1. `PluginDataEntryFunction` → registers plugin
2. `EffectMain` → dispatches AE commands
3. `GPUDeviceSetup` → creates DirectX 12 renderer
4. `SmartRender` → routes to GPU renderer

**GPU Rendering Pipeline (in JustGlowGPURenderer):**
1. Prefilter → soft threshold + Karis average
2. Downsample chain → creates MIP pyramid (3-6 levels based on quality)
3. Upsample chain → reconstructs with progressive blur blending
4. Composite → blends glow with original (Add/Screen/Overlay modes)

**Key Data Structures:**
- `GlowParams` (b0) - main constant buffer, 16-byte aligned for HLSL
- `BlurPassParams` (b1) - per-pass blur parameters
- Descriptor heap layout: slots 0-7 MIP SRVs, 8-15 MIP UAVs, 16+ pass pairs

## Critical Implementation Details

**PiPL Resource (resources/Win/JustGlow.rc):**
- Binary PiPL is embedded directly in RC file (no PiPLTool)
- Must use big-endian byte ordering for all values
- Each property requires vendorID ('ADBE'), propertyKey, propertyID, length, data
- Version must be 0, not 1
- 64-bit Windows uses '8664' code property key (not 'wx86' which is 32-bit)

**GPU Flags in PiPL (OutFlags2):**
- `0x2A001400` enables: GPU_RENDER_F32, SUPPORTS_DIRECTX_RENDERING, SMART_RENDER, THREADED_RENDERING

**Shader Loading:**
- Shaders are loaded from `DirectX_Assets/` relative to plugin DLL location
- Uses `GetModuleHandleExW` with static helper to get DLL path

## Debugging

Debug log location: `%TEMP%\JustGlow_debug.log`

Key checkpoints:
- `SmartRender: isGPU=1` confirms GPU path is active
- `GPUDeviceSetup` should appear before first render

## Testing

1. Build and copy `.aex` + `DirectX_Assets/` to AE plugins folder
2. Apply effect: Effects → Stylize → JustGlow
3. Check debug log for GPU initialization

## Parameters

13 UI parameters defined in `JustGlow.h` ParamID enum. Quality setting controls MIP chain depth (Low=3, Medium=4, High=5, Ultra=6 levels).
