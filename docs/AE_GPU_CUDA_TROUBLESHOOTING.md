# After Effects GPU Plugin Development - CUDA Troubleshooting Guide

After Effects GPU 플러그인 개발 시 자주 발생하는 문제들과 해결책을 정리한 문서입니다.

---

## 1. GPU 렌더링 경로가 호출되지 않음 (isGPU=0)

### 증상
```
SmartRender (isGPU=0)  ← CPU fallback으로 처리됨
```
- `PF_Cmd_SMART_RENDER_GPU` 대신 `PF_Cmd_SMART_RENDER`가 호출됨
- GPU 초기화는 성공하지만 실제 GPU 렌더링이 실행되지 않음

### 원인
SDK 문서에 따르면 `PF_OutFlag2_SUPPORTS_GPU_RENDER_F32`는 **두 곳**에서 설정해야 함:
1. `PF_Cmd_GLOBAL_SETUP`
2. `PF_Cmd_GPU_DEVICE_SETUP` ← **이 부분을 놓치기 쉬움**

### 해결책

```cpp
// GPUDeviceSetup 함수 내부, return 전에 추가
PF_Err GPUDeviceSetup(...) {
    // ... GPU 초기화 코드 ...

    // 필수! SDK 요구사항: GLOBAL_SETUP과 GPU_DEVICE_SETUP 둘 다에서 설정
    if (gpuData->initialized && !err) {
        out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;

        // DirectX 사용 시
        #if HAS_DIRECTX
        if (gpuData->framework == GPUFrameworkType::DirectX) {
            out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;
        }
        #endif
    }

    extra->output->gpu_data = gpuData;
    return err;
}
```

### 추가 검증: PreRender에서 GPU 가용성 확인

```cpp
PF_Err PreRender(...) {
    // GPU 가용성 확인
    bool gpuAvailable = (extra->input->gpu_data != nullptr) &&
                        (extra->input->what_gpu != PF_GPU_Framework_NONE);

    if (gpuAvailable) {
        extra->output->flags = PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    }
    // ...
}
```

### SDK 레퍼런스
- `AE_Effect.h:1007` - `PF_OutFlag2_SUPPORTS_GPU_RENDER_F32`
- `AE_Effect.h:2506` - `PF_RenderOutputFlag_GPU_RENDER_POSSIBLE`

---

## 2. CUDA_ERROR_ILLEGAL_ADDRESS

### 증상
```
CUDA ERROR: CUDA_ERROR_ILLEGAL_ADDRESS (700)
```
- GPU 커널 실행 중 메모리 접근 오류
- 특히 큰 이미지나 여러 MIP 레벨에서 발생

### 원인: Pitch 단위 혼동

CUDA 커널에서 pitch를 잘못된 단위로 전달:
- **잘못됨**: `pitch`를 bytes나 floats 개수로 전달
- **올바름**: `pitch`를 **pixels** 단위로 전달 (bilinear sampling에서)

### 문제 코드 예시

```cpp
// ❌ 잘못된 코드 - pitch가 bytes 단위
int dstPitchPixels = dstMip.pitch;  // pitch는 bytes (width * 4 * sizeof(float))

// ❌ 잘못된 코드 - floats 개수로 계산
int dstPitchPixels = width * 4;  // RGBA = 4 floats
```

### 해결책

```cpp
// ✅ 올바른 코드 - pixels 단위
int srcPitchPixels = srcMip.width;  // 픽셀 단위
int dstPitchPixels = dstMip.width;  // 픽셀 단위
```

### CUDA 커널에서의 사용

```cuda
// pitch는 pixels 단위로 받음
__device__ void sampleBilinear(
    const float* src, float u, float v,
    int width, int height, int pitch,  // pitch = pixels, not bytes
    float& outR, float& outG, float& outB, float& outA)
{
    // 인덱스 계산: (y * pitch + x) * 4
    // pitch가 pixels이면: (y * width + x) * 4
    int idx = (y * pitch + x) * 4;  // RGBA 접근
    // ...
}
```

### 디버깅 팁
1. 각 MIP 레벨의 width, height, pitch 값을 로그로 출력
2. 메모리 할당 크기와 실제 접근 범위 비교
3. 가장 작은 MIP 레벨부터 테스트 (메모리가 작아서 문제 발견이 쉬움)

---

## 3. PiPL Resource 문제

### 증상
- 플러그인이 After Effects에 표시되지 않음
- "Invalid plugin" 오류

### 원인
PiPL 리소스의 바이트 오더링 또는 플래그 값 오류

### 핵심 사항

```cpp
// OutFlags2 값 (Big Endian으로 저장)
// 0x2A001400 = GPU_RENDER_F32 + DIRECTX_RENDERING + SMART_RENDER + THREADED_RENDERING

AE_Effect_Global_OutFlags_2 {
    0x2A001400  // Big Endian!
}
```

### 64비트 Windows 코드 속성
```cpp
// 64-bit Windows: '8664' (NOT 'wx86' which is 32-bit)
AE_Effect_Match_Name { "8664", "com.yourcompany.plugin" }
```

---

## 4. GPU Framework 상수

```cpp
enum {
    PF_GPU_Framework_NONE    = 0,
    PF_GPU_Framework_OPENCL  = 1,
    PF_GPU_Framework_METAL   = 2,
    PF_GPU_Framework_CUDA    = 3,
    PF_GPU_Framework_DIRECTX = 4
};
```

### GPUDeviceSetup에서 확인
```cpp
if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
    // CUDA 초기화
}
else if (extra->input->what_gpu == PF_GPU_Framework_DIRECTX) {
    // DirectX 초기화
}
```

---

## 5. CUDA PTX 로딩

### PTX 파일 위치
PTX 파일은 플러그인 DLL과 같은 디렉토리의 하위 폴더에 배치:

```
Plug-ins/
├── JustGlow.aex
└── CUDA_Assets/
    └── JustGlowKernels.ptx
```

### 경로 얻기
```cpp
static std::wstring GetPTXPath() {
    HMODULE hModule = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&GetPTXPath),  // 현재 함수 주소 사용
        &hModule);

    wchar_t modulePath[MAX_PATH];
    GetModuleFileNameW(hModule, modulePath, MAX_PATH);

    std::wstring path(modulePath);
    size_t lastSlash = path.find_last_of(L"\\/");
    path = path.substr(0, lastSlash + 1);

    return path + L"CUDA_Assets\\JustGlowKernels.ptx";
}
```

### PTX 로딩
```cpp
CUresult err = cuModuleLoad(&m_module, ptxPath);
if (err != CUDA_SUCCESS) {
    // 오류 처리
}

// 커널 함수 얻기
err = cuModuleGetFunction(&m_kernel, m_module, "KernelName");
```

---

## 6. CUDA Context 관리

### AE에서 제공하는 Context 사용
```cpp
bool Initialize(CUcontext context, CUstream stream) {
    m_context = context;
    m_stream = stream;

    // 현재 스레드에 context 바인딩
    CUresult err = cuCtxPushCurrent(m_context);
    if (err != CUDA_SUCCESS) return false;

    // 작업 수행...

    // context 해제
    cuCtxPopCurrent(nullptr);
    return true;
}
```

### Render에서도 Context Push/Pop 필요
```cpp
bool Render(...) {
    cuCtxPushCurrent(m_context);

    // 커널 실행...

    // 동기화
    cuStreamSynchronize(m_stream);

    cuCtxPopCurrent(nullptr);
    return true;
}
```

---

## 7. 투명 배경 처리

### 증상
- 투명 배경에서 글로우가 보이지 않음
- 글로우가 알파 채널을 확장하지 않음

### 원인
Composite 커널에서 원본 알파만 복사:
```cuda
output[outIdx + 3] = origA;  // 문제: 글로우 기여도 무시
```

### 해결책
```cuda
// 글로우 밝기를 알파에 반영
float glowLum = fmaxf(fmaxf(glowR, glowG), glowB);
float expandedAlpha = fmaxf(origA, clampf(glowLum, 0.0f, 1.0f));
output[outIdx + 3] = expandedAlpha;
```

---

## 8. 디버깅 로그 설정

### 로그 파일 위치
```cpp
static std::wstring GetLogFilePath() {
    wchar_t tempPath[MAX_PATH];
    GetTempPathW(MAX_PATH, tempPath);
    return std::wstring(tempPath) + L"YourPlugin_debug.log";
}
// Windows: %TEMP%\YourPlugin_debug.log
```

### 로그 매크로
```cpp
#define PLUGIN_LOG(fmt, ...) LogMessage(fmt, ##__VA_ARGS__)

// 핵심 체크포인트
PLUGIN_LOG("GPUDeviceSetup: framework=%d, initialized=%d", framework, initialized);
PLUGIN_LOG("SmartRender: isGPU=%d", isGPU);
PLUGIN_LOG("Kernel launch: grid=%dx%d, block=%dx%d", gridX, gridY, blockX, blockY);
```

---

## 9. 메모리 할당 (MIP Chain)

### 구조체
```cpp
struct MipLevel {
    CUdeviceptr devicePtr;  // GPU 메모리 포인터
    int width;
    int height;
    size_t pitch;           // bytes per row
    size_t sizeBytes;       // total allocation size
};
```

### 할당
```cpp
for (int i = 0; i < levels; ++i) {
    mip.width = w;
    mip.height = h;
    mip.pitch = w * 4 * sizeof(float);  // RGBA float
    mip.sizeBytes = mip.pitch * h;

    CUresult err = cuMemAlloc(&mip.devicePtr, mip.sizeBytes);

    // 다음 레벨
    w = (w + 1) / 2;
    h = (h + 1) / 2;
}
```

### 해제
```cpp
for (auto& mip : m_mipChain) {
    if (mip.devicePtr) {
        cuMemFree(mip.devicePtr);
        mip.devicePtr = 0;
    }
}
```

---

## 10. 체크리스트

### GPU 플러그인 디버깅 순서

1. **플러그인 로드 확인**
   - PiPL 리소스 정상?
   - AE에서 이펙트 표시?

2. **GPU 초기화 확인**
   - `GPUDeviceSetup` 호출됨?
   - `out_flags2` 설정됨?
   - Renderer 초기화 성공?

3. **GPU 경로 확인**
   - `PreRender`에서 `GPU_RENDER_POSSIBLE` 설정?
   - `SmartRender(isGPU=1)` 호출?

4. **커널 실행 확인**
   - PTX 로드 성공?
   - 커널 파라미터 올바름?
   - Pitch 단위 올바름?

5. **결과 확인**
   - 출력 버퍼에 데이터 있음?
   - 알파 채널 처리 올바름?

---

## 버전 정보

- After Effects SDK: 2024
- CUDA Toolkit: 12.x
- 테스트 환경: Windows 10/11, NVIDIA GPU

---

*이 문서는 JustGlow 플러그인 개발 경험을 바탕으로 작성되었습니다.*
