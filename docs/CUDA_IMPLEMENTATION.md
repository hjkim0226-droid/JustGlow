# JustGlow CUDA 구현 문서

**버전:** 1.3.0
**최종 수정:** 2024년 12월

---

## 1. 아키텍처 개요

### 1.1 파일 구조

```
src/
├── JustGlowCUDARenderer.h    # CUDA 렌더러 인터페이스
├── JustGlowCUDARenderer.cpp  # CUDA 렌더러 구현 (866줄)
└── JustGlowKernels.cu        # CUDA 커널 (1060줄)

build/CUDA_Assets/
└── JustGlowKernels.ptx       # 컴파일된 PTX 커널
```

### 1.2 클래스 구조

```cpp
class JustGlowCUDARenderer {
    // AE 제공 객체
    CUcontext m_context;
    CUstream m_stream;

    // PTX 모듈 및 커널
    CUmodule m_module;
    CUfunction m_prefilterKernel;
    CUfunction m_downsampleKernel;
    CUfunction m_upsampleKernel;
    CUfunction m_compositeKernel;
    CUfunction m_horizontalBlurKernel;
    CUfunction m_gaussianDownsampleHKernel;
    CUfunction m_gaussianDownsampleVKernel;
    CUfunction m_debugOutputKernel;

    // 버퍼 체인
    std::vector<CUDAMipBuffer> m_mipChain;       // 다운샘플 결과
    std::vector<CUDAMipBuffer> m_upsampleChain;  // 업샘플 결과
    CUDAMipBuffer m_horizontalTemp;              // H-blur 임시 버퍼
    CUDAMipBuffer m_gaussianDownsampleTemp;      // Gaussian 임시 버퍼
};
```

### 1.3 버퍼 구조체

```cpp
struct CUDAMipBuffer {
    CUdeviceptr devicePtr;  // GPU 메모리 포인터
    int width;              // 픽셀 단위 너비
    int height;             // 픽셀 단위 높이
    size_t pitch;           // 바이트 단위 행 간격
    size_t sizeBytes;       // 총 바이트 크기
};
```

---

## 2. 렌더링 파이프라인

### 2.1 전체 흐름

```
입력 (AE CUdeviceptr)
       │
       ▼
┌──────────────┐
│ PrefilterKernel │ ← 13-tap + Soft Threshold + Karis Average
└──────────────┘
       │
       ▼ m_mipChain[0]
┌──────────────┐
│ DownsampleChain │ ← 레벨 0-4: Gaussian, 레벨 5+: Kawase
└──────────────┘
       │
       ▼ m_mipChain[1..N]
┌──────────────┐
│ UpsampleChain │ ← 9-tap Tent + Falloff 가중치
└──────────────┘
       │
       ▼ m_upsampleChain[0]
┌──────────────┐
│ CompositeKernel │ ← 원본 + 글로우 합성
└──────────────┘
       │
       ▼
출력 (AE CUdeviceptr)
```

### 2.2 MIP 체인 구조

```
Quality: High (8 레벨)
┌─────────────────────────────────────────────┐
│ Level 0: 1920×1080 (Prefilter 출력)         │
│ Level 1:  960×540  (1/2)                    │
│ Level 2:  480×270  (1/4)                    │
│ Level 3:  240×135  (1/8)                    │
│ Level 4:  120×68   (1/16)                   │
│ Level 5:   60×34   (1/32)                   │
│ Level 6:   30×17   (1/64)                   │
│ Level 7:   15×9    (1/128)                  │
└─────────────────────────────────────────────┘
```

---

## 3. 커널 시그니처

### 3.1 PrefilterKernel

```cuda
extern "C" __global__ void PrefilterKernel(
    const float* __restrict__ input,    // 입력 버퍼 (RGBA float)
    float* __restrict__ output,         // 출력 버퍼 (MIP[0])
    int srcWidth, int srcHeight,        // 소스 크기
    int srcPitch,                       // 소스 pitch (픽셀 단위)
    int dstWidth, int dstHeight,        // 목적지 크기
    int dstPitch,                       // 목적지 pitch (픽셀 단위)
    int inputWidth, int inputHeight,    // 실제 입력 레이어 크기
    float threshold,                    // 밝기 임계값 (0-1)
    float softKnee,                     // Soft knee 폭 (0-1)
    float intensity,                    // 강도 (미사용, 1.0 고정)
    float colorR, float colorG, float colorB,  // 글로우 색상
    float colorTempR, float colorTempG, float colorTempB,  // 색온도
    float preserveColor,                // 색상 보존율
    int useHDR                          // HDR 모드 (Karis Average)
);
```

**기능:**
- 13-tap 다운샘플링 (Call of Duty 방식)
- Soft threshold 적용
- Karis Average (HDR 모드)
- sRGB → Linear 변환
- Alpha-Weighted Normalization

### 3.2 GaussianDownsampleHKernel / GaussianDownsampleVKernel

```cuda
// 수평 패스
extern "C" __global__ void GaussianDownsampleHKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height, int pitch,
    float blurOffset  // 미사용, 고정 오프셋 사용
);

// 수직 패스 + 2x 다운샘플
extern "C" __global__ void GaussianDownsampleVKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset  // 미사용
);
```

**기능:**
- 5-tap Gaussian → 3-fetch 최적화
- 레벨 0-4에서 사용 (디테일 보존)
- 가중치: [1,4,6,4,1]/16

### 3.3 DownsampleKernel (Kawase)

```cuda
extern "C" __global__ void DownsampleKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset,    // 미사용, 고정 0.5
    int rotationMode     // 0=X(대각선), 1=+(십자)
);
```

**기능:**
- Dual Kawase 5-tap (중앙 + 4코너)
- X/+ 회전 교차로 원형 글로우
- 레벨 5+에서 사용 (속도 우선)

### 3.4 UpsampleKernel

```cuda
extern "C" __global__ void UpsampleKernel(
    const float* __restrict__ input,      // 현재 레벨 다운샘플
    const float* __restrict__ prevLevel,  // 이전 업샘플 결과 (깊은 레벨)
    float* __restrict__ output,
    int srcWidth, int srcHeight, int srcPitch,
    int prevWidth, int prevHeight, int prevPitch,
    int dstWidth, int dstHeight, int dstPitch,
    float blurOffset,      // 미사용, 고정 1.5
    int levelIndex,        // 현재 레벨 (0 = 가장 상세)
    float activeLimit,     // Radius 기반 레벨 제한
    float decayK,          // Falloff 값 (0-100)
    float level1Weight,    // Level 1 시작 가중치 (Intensity)
    int falloffType,       // 0=Exponential, 1=InverseSquare, 2=Linear
    int blurMode           // 미사용
);
```

**기능:**
- 9-tap Tent 필터 (3x3)
- 물리 기반 Falloff 가중치
- Progressive 블렌딩

### 3.5 DebugOutputKernel

```cuda
extern "C" __global__ void DebugOutputKernel(
    const float* __restrict__ original,
    const float* __restrict__ debugBuffer,
    const float* __restrict__ glow,
    float* __restrict__ output,
    int width, int height,
    int inputWidth, int inputHeight,
    int originalPitch, int debugWidth, int debugHeight, int debugPitch,
    int glowWidth, int glowHeight, int glowPitch, int outputPitch,
    int debugMode,         // DebugViewMode enum
    float exposure,
    float sourceOpacity,   // 0-1
    float glowOpacity,     // 0-2
    int compositeMode      // 0=Add, 1=Screen, 2=Overlay
);
```

**Debug Mode 값:**
| 값 | 모드 | 설명 |
|----|------|------|
| 1 | Final | 최종 합성 |
| 2 | Prefilter | MIP[0] |
| 3-8 | Down1-6 | MIP[1-6] |
| 9-15 | Up0-6 | Upsample[0-6] |
| 16 | GlowOnly | 글로우만 |

---

## 4. 메모리 관리

### 4.1 버퍼 할당

```cpp
bool AllocateMipChain(int width, int height, int levels) {
    // 1. 기존 체인 재사용 가능 여부 확인
    if (canReuse) return true;

    // 2. 기존 버퍼 해제
    ReleaseMipChain();

    // 3. MIP 체인 할당
    for (int i = 0; i < levels; ++i) {
        cuMemAlloc(&mip.devicePtr, sizeBytes);
        // 다음 레벨은 절반 크기
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    // 4. Upsample 체인 할당 (별도 - race condition 방지)
    // 5. 임시 버퍼 할당
}
```

### 4.2 Pitch 규칙

**중요:** 현재 구현에서 pitch는 **픽셀 단위**로 사용됨

```cuda
// 인덱스 계산
int idx = (y * pitch + x) * 4;  // pitch는 픽셀 단위
                                // ×4는 RGBA 채널

// 메모리 레이아웃 (RGBA float)
┌─────────────────────────────────────────────┐
│ R₀₀ G₀₀ B₀₀ A₀₀ │ R₀₁ G₀₁ B₀₁ A₀₁ │ ...  │ 행 0
├─────────────────────────────────────────────┤
│ R₁₀ G₁₀ B₁₀ A₁₀ │ R₁₁ G₁₁ B₁₁ A₁₁ │ ...  │ 행 1
└─────────────────────────────────────────────┘
   └─ 4 floats ─┘
```

---

## 5. 동기화 요구사항

### 5.1 현재 구현

```cpp
// 모든 커널이 같은 스트림에서 순차 실행
ExecutePrefilter(...);
ExecuteDownsampleChain(...);
ExecuteUpsampleChain(...);
ExecuteComposite(...);
cuStreamSynchronize(m_stream);  // 마지막에만 동기화
```

### 5.2 권장 구현

```cpp
CUevent events[4];
for (int i = 0; i < 4; i++) {
    cuEventCreate(&events[i], CU_EVENT_DEFAULT);
}

ExecutePrefilter(...);
cuEventRecord(events[0], m_stream);

cuStreamWaitEvent(m_stream, events[0], 0);
ExecuteDownsampleChain(...);
cuEventRecord(events[1], m_stream);

cuStreamWaitEvent(m_stream, events[1], 0);
ExecuteUpsampleChain(...);
cuEventRecord(events[2], m_stream);

cuStreamWaitEvent(m_stream, events[2], 0);
ExecuteComposite(...);

cuStreamSynchronize(m_stream);

for (int i = 0; i < 4; i++) {
    cuEventDestroy(events[i]);
}
```

---

## 6. 가중치 계산

### 6.1 물리 기반 Falloff

```cuda
__device__ float calculatePhysicalWeight(
    float level,          // 현재 MIP 레벨
    float falloff,        // 0-100 (50=중립)
    int falloffType,      // 0=Exp, 1=InvSq, 2=Linear
    float level1Weight    // Level 1 시작 가중치 (Intensity)
) {
    // Level 0: 항상 100%
    if (level < 0.5f) return 1.0f;

    // Falloff → decayRate 변환
    // 0% → 1.25 (외곽 부스트)
    // 50% → 1.0 (중립)
    // 100% → 0.5 (강한 감쇠)
    float normalizedFalloff = (falloff - 50.0f) / 50.0f;
    float decayRate = ...;

    // weight = level1Weight × pow(decayRate, level - 1)
    return level1Weight * powf(decayRate, level - 1.0f);
}
```

### 6.2 Falloff 타입

| 타입 | 공식 | 특성 |
|------|------|------|
| Exponential | `pow(0.5, i×k)` | 균형잡힌 감쇠 (기본값) |
| InverseSquare | `1/(x²+1)` | 날카로운 코어 + 긴 꼬리 |
| Linear | `1-x×0.1` | 부드러운/안개 느낌 |

---

## 7. 색 공간 변환

### 7.1 sRGB ↔ Linear

```cuda
// sRGB → Linear (입력)
__device__ float srgbToLinear(float c) {
    if (c <= 0.0f) return 0.0f;
    return (c <= 0.04045f)
        ? c / 12.92f
        : powf((c + 0.055f) / 1.055f, 2.4f);
}

// Linear → sRGB (출력)
__device__ float linearToSrgb(float c) {
    if (c <= 0.0f) return 0.0f;
    return (c <= 0.0031308f)
        ? c * 12.92f
        : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}
```

### 7.2 파이프라인 색 공간

```
입력 (Premultiplied sRGB)
       │
       ▼
Prefilter: Alpha-Weighted Normalization
           → Unpremultiply
           → sRGB → Linear
       │
       ▼
Downsample/Upsample (Linear)
       │
       ▼
DebugOutput: Composite
             → Linear → sRGB
       │
       ▼
출력 (sRGB)
```

---

## 8. 샘플링 함수

### 8.1 표준 Bilinear

```cuda
__device__ void sampleBilinear(
    const float* src, float u, float v,
    int width, int height, int pitch,
    float& outR, float& outG, float& outB, float& outA
) {
    u = clampf(u, 0.0f, 1.0f);  // UV 클램핑
    v = clampf(v, 0.0f, 1.0f);
    // ... 4점 보간
}
```

### 8.2 Zero-Pad Bilinear

```cuda
__device__ void sampleBilinearZeroPad(
    const float* src, float u, float v,
    int width, int height, int pitch,
    float& outR, float& outG, float& outB, float& outA
) {
    // UV가 [0,1] 범위 밖이면 0 반환
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        outR = outG = outB = outA = 0.0f;
        return;
    }
    // ... 개별 샘플점 bounds 체크
}
```

**사용처:**
- `sampleBilinear`: 다운샘플, 업샘플, 합성
- `sampleBilinearZeroPad`: Prefilter (경계 확장)

---

## 9. 스레드 구성

### 9.1 기본 설정

```cpp
constexpr int THREAD_BLOCK_SIZE = 16;

// 그리드 계산
int gridX = (width + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;
int gridY = (height + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

// 커널 실행
cuLaunchKernel(
    kernel,
    gridX, gridY, 1,           // 그리드 크기
    THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE, 1,  // 블록 크기 (256 스레드)
    0, m_stream,
    kernelParams, nullptr
);
```

### 9.2 메모리 접근 패턴

```
스레드 블록 (16×16)
┌─────────────────┐
│ ● ● ● ● ● ● ● ●│  각 ●는 1 픽셀 처리
│ ● ● ● ● ● ● ● ●│
│ ● ● ● ● ● ● ● ●│  연속 메모리 접근으로
│ ● ● ● ● ● ● ● ●│  Coalesced 읽기 달성
│ ...             │
└─────────────────┘
```

---

## 10. 디버그 로그

### 10.1 로그 파일 위치

```
Windows: %TEMP%\JustGlow_CUDA_debug.log
```

### 10.2 로그 예시

```
[18:30:45] === JustGlow CUDA Renderer Initialize ===
[18:30:45] CUDA context and stream set: context=0x12345, stream=0x67890
[18:30:45] Loading PTX from: C:\...\CUDA_Assets\JustGlowKernels.ptx
[18:30:45] PTX module loaded successfully
[18:30:45] PrefilterKernel loaded
[18:30:45] DownsampleKernel loaded
...
[18:30:45] === CUDA Initialize Complete ===
[18:30:46] === CUDA Render Begin ===
[18:30:46] Size: 1920x1080, MipLevels: 8, Exposure: 10.00
[18:30:46] --- Prefilter ---
[18:30:46] Prefilter: 1920x1080 -> 1920x1080, grid: 120x68
...
```

---

## 11. 알려진 제한사항

1. **CUDA Compute Capability:** sm_50 이상 필요
2. **최대 MIP 레벨:** 12 (Ultra 품질)
3. **최소 텍스처 크기:** 16×16 픽셀
4. **지원 픽셀 포맷:** RGBA float32만

---

*문서 버전: 1.0*
*작성: Claude Code*
