# JustGlow GPU Rendering Issue Report

**Date:** 2025-12-17
**Version:** 1.0.5
**Status:** Under Investigation

---

## 1. Problem Summary

After Effects는 GPU 렌더링이 가능함에도 불구하고 `PF_Cmd_SMART_RENDER` (CPU)를 호출하며, `PF_Cmd_SMART_RENDER_GPU`를 호출하지 않습니다.

### 증상
- CUDA 렌더러가 성공적으로 초기화됨
- 모든 플래그가 올바르게 설정됨
- 32bpc 프로젝트에서도 GPU 경로가 활성화되지 않음
- 이펙트가 적용되지 않음 (CPU fallback이 단순 copy만 수행)

---

## 2. Debug Log Analysis

```
[21:58:22] GlobalSetup complete, flags=0x6000600, flags2=0x2A001400
[21:58:22]   HAS_CUDA=1, HAS_DIRECTX=1
[21:58:22]   GPU_RENDER_F32=1, DIRECTX_RENDERING=1
[21:58:23] === GPUDeviceSetup ===
[21:58:23] Requested GPU framework: 3 (CUDA=3, DIRECTX=4)
[21:58:23] CUDA Renderer initialized successfully!
[21:58:23] GPUDeviceSetup complete, err=0, initialized=1, framework=2
[21:58:23] === PreRender ===
[21:58:23] PreRender input: bitdepth=32                          <-- 32bpc OK
[21:58:23] PreRender: GPU_RENDER_POSSIBLE flag set (0x2)         <-- Flag OK
[21:58:23] PreRender complete, mipLevels=5, err=0
[21:58:23] === SmartRender (isGPU=0) ===                         <-- PROBLEM!
[21:58:23] CPU Fallback path
```

---

## 3. SDK Requirements Analysis

### 3.1 PF_OutFlag2_SUPPORTS_GPU_RENDER_F32

**SDK Header (AE_Effect.h:1007):**
```cpp
PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 = 1L << 25,
// PF_Cmd_GLOBAL_SETUP, PF_Cmd_GPU_DEVICE_SETUP.
// Must also set PF_RenderOutputFlag_GPU_RENDER_POSSIBLE at pre-render to enable GPU rendering.
```

**Required at:**
1. `PF_Cmd_GLOBAL_SETUP` - GlobalSetup()
2. `PF_Cmd_GPU_DEVICE_SETUP` - GPUDeviceSetup()

### 3.2 PF_RenderOutputFlag_GPU_RENDER_POSSIBLE

**SDK Header (AE_Effect.h:2506):**
```cpp
PF_RenderOutputFlag_GPU_RENDER_POSSIBLE = 0x2,
// if the GPU render is possible given the params and frame render context
```

**Required at:**
- `PF_Cmd_SMART_PRE_RENDER` - PreRender()

### 3.3 PreRender Input Fields

```cpp
typedef struct {
    PF_RenderRequest  output_request;
    short             bitdepth;        // 8, 16, or 32
    const void*       gpu_data;        // GPU data from GPUDeviceSetup
    PF_GPU_Framework  what_gpu;        // 0=None, 1=OpenCL, 2=Metal, 3=CUDA, 4=DirectX
    A_u_long          device_index;
} PF_PreRenderInput;
```

---

## 4. Current Implementation Status

### 4.1 GlobalSetup - OK

```cpp
out_data->out_flags2 =
    PF_OutFlag2_FLOAT_COLOR_AWARE |
    PF_OutFlag2_SUPPORTS_SMART_RENDER |
    PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 |      // <-- SET
    PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
    PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;    // <-- SET
```
**Result:** 0x2A001400 - Correct!

### 4.2 GPUDeviceSetup - MISSING FLAGS!

```cpp
PF_Err GPUDeviceSetup(...) {
    // ... CUDA initialization ...
    extra->output->gpu_data = gpuData;
    // out_data->out_flags2 NOT SET!   <-- PROBLEM
    return err;
}
```

**SDK Requirement:** `PF_OutFlag2_SUPPORTS_GPU_RENDER_F32` should also be set here.

### 4.3 PreRender - PARTIAL

```cpp
PF_Err PreRender(...) {
    // Does NOT check extra->input->what_gpu
    // Does NOT check extra->input->gpu_data

    extra->output->flags = PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;  // OK
    return err;
}
```

**Missing Checks:**
- `extra->input->what_gpu` - Should verify GPU framework is available
- `extra->input->gpu_data` - Should verify GPU was initialized

### 4.4 PiPL Resource - OK

```cpp
AE_Effect_Global_OutFlags_2 {
    0x2A001400  // Includes GPU_RENDER_F32 and DIRECTX_RENDERING
}
```

---

## 5. Potential Causes

### Priority 1: GPUDeviceSetup Missing out_flags2

SDK explicitly states the flag must be set at **both** GLOBAL_SETUP and GPU_DEVICE_SETUP.

**Fix:**
```cpp
PF_Err GPUDeviceSetup(...) {
    // ... existing code ...

    if (gpuData->initialized) {
        out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
        if (gpuData->framework == GPUFrameworkType::DirectX) {
            out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;
        }
    }

    extra->output->gpu_data = gpuData;
    return err;
}
```

### Priority 2: PreRender Should Check GPU Availability

**Fix:**
```cpp
PF_Err PreRender(...) {
    // Check if GPU is available
    bool gpuAvailable = (extra->input->gpu_data != nullptr) &&
                        (extra->input->what_gpu != PF_GPU_Framework_NONE);

    if (gpuAvailable) {
        extra->output->flags = PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    }
    // ...
}
```

### Priority 3: AE Preferences

User should verify:
- Edit > Preferences > Display > GPU Information
- Mercury GPU Acceleration is enabled
- Correct GPU is selected

---

## 6. GPU Framework Constants

```cpp
enum {
    PF_GPU_Framework_NONE = 0,
    PF_GPU_Framework_OPENCL = 1,
    PF_GPU_Framework_METAL = 2,
    PF_GPU_Framework_CUDA = 3,
    PF_GPU_Framework_DIRECTX = 4
};
```

---

## 7. Command Flow

```
1. PF_Cmd_GLOBAL_SETUP
   - Set out_flags2 with GPU flags

2. PF_Cmd_GPU_DEVICE_SETUP (called by AE if GPU flags set)
   - Initialize GPU renderer
   - Set out_flags2 again (MISSING!)
   - Store gpu_data in extra->output

3. PF_Cmd_SMART_PRE_RENDER
   - Check extra->input->what_gpu
   - Check extra->input->gpu_data
   - Set PF_RenderOutputFlag_GPU_RENDER_POSSIBLE

4. PF_Cmd_SMART_RENDER_GPU (if GPU available)
   - OR -
   PF_Cmd_SMART_RENDER (if GPU not available)
```

---

## 8. Recommended Fixes

### Fix 1: Add out_flags2 to GPUDeviceSetup

**File:** `src/JustGlow.cpp`
**Location:** GPUDeviceSetup function, before return

```cpp
// Set GPU support flags (required at both GLOBAL_SETUP and GPU_DEVICE_SETUP)
if (gpuData->initialized && !err) {
    out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
#if HAS_DIRECTX
    if (gpuData->framework == GPUFrameworkType::DirectX) {
        out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;
    }
#endif
    PLUGIN_LOG("GPUDeviceSetup: out_flags2 set to 0x%X", out_data->out_flags2);
}
```

### Fix 2: Add GPU Availability Check to PreRender

**File:** `src/JustGlow.cpp`
**Location:** PreRender function

```cpp
PLUGIN_LOG("PreRender input: bitdepth=%d, what_gpu=%d, gpu_data=%p",
    extra->input->bitdepth,
    extra->input->what_gpu,
    extra->input->gpu_data);

// Only set GPU flag if GPU is actually available
bool gpuAvailable = (extra->input->gpu_data != nullptr) &&
                    (extra->input->what_gpu != PF_GPU_Framework_NONE);

if (gpuAvailable) {
    extra->output->flags = PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    PLUGIN_LOG("PreRender: GPU available, GPU_RENDER_POSSIBLE flag set");
} else {
    PLUGIN_LOG("PreRender: GPU not available (gpu_data=%p, what_gpu=%d)",
        extra->input->gpu_data, extra->input->what_gpu);
}
```

---

## 9. Testing Checklist

- [ ] Apply Fix 1 (GPUDeviceSetup out_flags2)
- [ ] Apply Fix 2 (PreRender GPU check)
- [ ] Rebuild plugin
- [ ] Test with 32bpc project
- [ ] Check debug log for `what_gpu` and `gpu_data` values
- [ ] Verify `PF_Cmd_SMART_RENDER_GPU` is called
- [ ] Verify glow effect is rendered

---

## 10. References

- **AE_Effect.h** - Lines 1007, 2476-2482, 2493-2499, 2504-2509
- **SmartFX Documentation** - smartfx/smartfx
- **GPU Build Instructions** - intro/gpu-build-instructions

---

*Report generated by Claude Code*
