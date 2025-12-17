# After Effects GPU Plugin SDK Reference

## GPU Command Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PF_Cmd_GLOBAL_SETUP                                          │
│    └─ Set: PF_OutFlag2_SUPPORTS_GPU_RENDER_F32                  │
│    └─ Set: PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING (if DirectX)  │
│    └─ Set: PF_OutFlag2_SUPPORTS_SMART_RENDER                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. PF_Cmd_GPU_DEVICE_SETUP (called if GPU flags set)            │
│    └─ extra->input->what_gpu: Framework requested by AE         │
│    └─ Initialize renderer (CUDA/DirectX/Metal)                  │
│    └─ Set: out_data->out_flags2 (AGAIN! Required by SDK)        │
│    └─ Store: extra->output->gpu_data = your_gpu_data            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. PF_Cmd_SMART_PRE_RENDER                                      │
│    └─ extra->input->what_gpu: Current GPU framework             │
│    └─ extra->input->gpu_data: Your stored GPU data              │
│    └─ Check GPU availability                                     │
│    └─ Set: extra->output->flags = GPU_RENDER_POSSIBLE (if OK)   │
│    └─ Store pre-render data for SmartRender                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. PF_Cmd_SMART_RENDER_GPU (if GPU available)                   │
│    └─ OR PF_Cmd_SMART_RENDER (CPU fallback)                     │
│    └─ extra->input->gpu_data: Your GPU data                     │
│    └─ Get GPU world data (input/output buffers)                 │
│    └─ Execute GPU kernels                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. PF_Cmd_GPU_DEVICE_SETDOWN                                    │
│    └─ Cleanup GPU resources                                      │
│    └─ Delete renderer                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Essential Flags

### PF_OutFlag2 (out_data->out_flags2)

| Flag | Value | Required At | Description |
|------|-------|-------------|-------------|
| `PF_OutFlag2_SUPPORTS_SMART_RENDER` | `1L << 10` | GLOBAL_SETUP | Enable Smart Render |
| `PF_OutFlag2_FLOAT_COLOR_AWARE` | `1L << 11` | GLOBAL_SETUP | 32bpc support |
| `PF_OutFlag2_SUPPORTS_GPU_RENDER_F32` | `1L << 25` | GLOBAL_SETUP, **GPU_DEVICE_SETUP** | Enable GPU F32 |
| `PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING` | `1L << 27` | GLOBAL_SETUP, **GPU_DEVICE_SETUP** | DirectX support |
| `PF_OutFlag2_SUPPORTS_THREADED_RENDERING` | `1L << 12` | GLOBAL_SETUP | Multi-thread safe |

### PF_RenderOutputFlag (extra->output->flags)

| Flag | Value | Set At | Description |
|------|-------|--------|-------------|
| `PF_RenderOutputFlag_GPU_RENDER_POSSIBLE` | `0x2` | PRE_RENDER | Indicates GPU is available |

---

## GPU Framework Constants

```cpp
typedef enum {
    PF_GPU_Framework_NONE    = 0,
    PF_GPU_Framework_OPENCL  = 1,
    PF_GPU_Framework_METAL   = 2,
    PF_GPU_Framework_CUDA    = 3,
    PF_GPU_Framework_DIRECTX = 4
} PF_GPU_Framework;
```

---

## Key Structures

### PF_GPUDeviceSetupExtra

```cpp
typedef struct {
    PF_GPUDeviceSetupInput*  input;
    PF_GPUDeviceSetupOutput* output;
} PF_GPUDeviceSetupExtra;

typedef struct {
    PF_GPU_Framework what_gpu;      // Framework AE wants to use
    A_u_long         device_index;  // GPU device index
} PF_GPUDeviceSetupInput;

typedef struct {
    void* gpu_data;  // Store your GPU data here
} PF_GPUDeviceSetupOutput;
```

### PF_PreRenderInput (for GPU)

```cpp
typedef struct {
    PF_RenderRequest  output_request;
    short             bitdepth;     // 8, 16, or 32
    const void*       gpu_data;     // Your GPU data from DeviceSetup
    PF_GPU_Framework  what_gpu;     // 0=None, 1=OpenCL, 2=Metal, 3=CUDA, 4=DirectX
    A_u_long          device_index;
} PF_PreRenderInput;
```

### PF_GPUDeviceInfo

```cpp
typedef struct {
    PF_GPU_Framework device_framework;
    void*            devicePV;         // ID3D12Device* or CUdevice
    void*            contextPV;        // CUcontext (CUDA only)
    void*            command_queuePV;  // ID3D12CommandQueue* or CUstream
} PF_GPUDeviceInfo;
```

---

## Getting GPU Device Info

```cpp
PF_Err GPUDeviceSetup(PF_InData* in_data, PF_OutData* out_data,
                      PF_GPUDeviceSetupExtra* extra)
{
    PF_GPUDeviceSuite1* gpuSuite = nullptr;

    // Acquire GPU Device Suite
    in_data->pica_basicP->AcquireSuite(
        kPFGPUDeviceSuite,
        kPFGPUDeviceSuiteVersion1,
        (const void**)&gpuSuite);

    // Get device info
    PF_GPUDeviceInfo deviceInfo;
    gpuSuite->GetDeviceInfo(
        in_data->effect_ref,
        extra->input->device_index,
        &deviceInfo);

    // For CUDA:
    // deviceInfo.contextPV      = CUcontext
    // deviceInfo.command_queuePV = CUstream

    // For DirectX:
    // deviceInfo.devicePV       = ID3D12Device*
    // deviceInfo.command_queuePV = ID3D12CommandQueue*

    in_data->pica_basicP->ReleaseSuite(
        kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
}
```

---

## Getting GPU World Data (Buffers)

```cpp
PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data,
                   PF_SmartRenderExtra* extra, bool isGPU)
{
    if (isGPU) {
        PF_GPUDeviceSuite1* gpuSuite;
        in_data->pica_basicP->AcquireSuite(kPFGPUDeviceSuite,
            kPFGPUDeviceSuiteVersion1, (const void**)&gpuSuite);

        // Get input buffer
        PF_EffectWorld* input_worldP = nullptr;
        extra->cb->checkout_layer(in_data->effect_ref, PARAM_INPUT,
            ..., &input_worldP);

        // Get output buffer
        PF_EffectWorld* output_worldP = nullptr;
        extra->cb->checkout_output(in_data->effect_ref, &output_worldP);

        // Get GPU pointers
        void* inputData = nullptr;
        void* outputData = nullptr;

        gpuSuite->GetGPUWorldData(in_data->effect_ref,
            input_worldP, &inputData);
        gpuSuite->GetGPUWorldData(in_data->effect_ref,
            output_worldP, &outputData);

        // For CUDA: cast to CUdeviceptr
        CUdeviceptr cudaInput = reinterpret_cast<CUdeviceptr>(inputData);
        CUdeviceptr cudaOutput = reinterpret_cast<CUdeviceptr>(outputData);

        // Buffer info
        int width = output_worldP->width;
        int height = output_worldP->height;
        int rowbytes = output_worldP->rowbytes;  // bytes per row
        int pitch = rowbytes / sizeof(float) / 4; // pixels per row (RGBA)
    }
}
```

---

## PiPL Resource (OutFlags2 for GPU)

```cpp
// Hex value for GPU support flags
// PF_OutFlag2_SUPPORTS_GPU_RENDER_F32     = 1 << 25 = 0x02000000
// PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING  = 1 << 27 = 0x08000000
// PF_OutFlag2_SUPPORTS_SMART_RENDER       = 1 << 10 = 0x00000400
// PF_OutFlag2_FLOAT_COLOR_AWARE           = 1 << 11 = 0x00000800
// PF_OutFlag2_SUPPORTS_THREADED_RENDERING = 1 << 12 = 0x00001000

// Combined (Big Endian in .r file):
// 0x0A001C00 = basic GPU
// 0x2A001400 = with DIRECTX_RENDERING

AE_Effect_Global_OutFlags_2 { 0x2A001400 }
```

---

## CUDA Specifics

### Kernel Parameter Passing

```cpp
void* kernelParams[] = {
    &inputPtr,           // CUdeviceptr
    &outputPtr,          // CUdeviceptr
    (void*)&width,       // int (use cast for non-pointer types)
    (void*)&height,      // int
    (void*)&floatParam   // float
};

cuLaunchKernel(
    kernel,
    gridX, gridY, 1,           // Grid dimensions
    blockX, blockY, 1,         // Block dimensions
    0,                          // Shared memory bytes
    stream,                     // CUstream
    kernelParams,               // Parameters
    nullptr                     // Extra (usually nullptr)
);
```

### Error Handling

```cpp
bool CheckCUDAError(CUresult err, const char* context) {
    if (err != CUDA_SUCCESS) {
        const char* errName = nullptr;
        const char* errStr = nullptr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        // Log error
        return false;
    }
    return true;
}
```

### Common CUDA Errors

| Error | Code | Common Cause |
|-------|------|--------------|
| `CUDA_ERROR_ILLEGAL_ADDRESS` | 700 | Wrong pitch/stride calculation |
| `CUDA_ERROR_LAUNCH_FAILED` | 719 | Kernel crash, infinite loop |
| `CUDA_ERROR_OUT_OF_MEMORY` | 2 | Allocation too large |
| `CUDA_ERROR_INVALID_VALUE` | 1 | Bad parameter |
| `CUDA_ERROR_NOT_INITIALIZED` | 3 | cuInit not called |

---

## Header Files Reference

| Header | Contains |
|--------|----------|
| `AE_Effect.h` | Core effect types, flags, PF_OutFlag2 |
| `AE_EffectCB.h` | Callback functions |
| `AE_EffectGPUSuites.h` | GPU suite definitions |
| `PF_Suite_Helpers.h` | Helper macros |

---

## Debugging Tips

1. **Log Everything**
   ```cpp
   PLUGIN_LOG("GPUDeviceSetup: what_gpu=%d", extra->input->what_gpu);
   PLUGIN_LOG("PreRender: gpu_data=%p, what_gpu=%d",
       extra->input->gpu_data, extra->input->what_gpu);
   PLUGIN_LOG("SmartRender: isGPU=%d", isGPU);
   ```

2. **Check Return Values**
   - Every AE suite call
   - Every CUDA call
   - Every memory allocation

3. **Verify Data Flow**
   - gpu_data stored in DeviceSetup
   - gpu_data retrieved in PreRender
   - gpu_data retrieved in SmartRender

4. **Test Incrementally**
   - Start with smallest image (100x100)
   - Start with single MIP level
   - Add complexity gradually

---

*SDK Version: After Effects 2024*
*Last Updated: 2025-12-18*
