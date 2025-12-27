# JustGlow 알고리즘 상세 문서

**Version**: 1.6.0
**Last Updated**: 2025-12-28
**Target**: CUDA (Primary), DirectX 12 (AMD/Intel GPU)

---

## 개요

JustGlow는 MIP Chain 기반 GPU 글로우 이펙트입니다. 핵심 아이디어는 이미지를 단계적으로 축소(Downsample)하면서 블러를 적용하고, 다시 확대(Upsample)하면서 누적 블렌딩하여 자연스러운 글로우를 생성하는 것입니다.

```
+---------------------------------------------------------------------+
|                    JustGlow Rendering Pipeline                      |
+---------------------------------------------------------------------+
|                                                                     |
|  [Input]                                                            |
|     |                                                               |
|     v                                                               |
|  +------------------+                                               |
|  |     Refine       |  BoundingBox 계산 (atomicMin/Max)            |
|  +--------+---------+                                               |
|           |                                                         |
|           v                                                         |
|  +------------------+                                               |
|  |    Prefilter     |  13-tap Gaussian + Soft Threshold + ZeroPad  |
|  |   (MIP[0])       |  -> Premultiplied RGB, Linear (optional)     |
|  +--------+---------+                                               |
|           |                                                         |
|           v                                                         |
|  +------------------+                                               |
|  |   Desaturation   |  Max-based -> highlight to white             |
|  |   (optional)     |                                               |
|  +--------+---------+                                               |
|           |                                                         |
|           v                                                         |
|  +------------------------------------------------------+           |
|  |              Downsample Chain                        |           |
|  |  MIP[0] -> MIP[1] -> MIP[2] -> ... -> MIP[N-1]      |           |
|  |  (FHD)    (960x540) (480x270)      (until 16px)     |           |
|  |                                                      |           |
|  |  Each level: 9-tap 2D Gaussian (ZeroPad)            |           |
|  +------------------------------------------------------+           |
|           |                                                         |
|           |    [Optional: Pre-blur Parallel]                        |
|           |    6개 스트림에서 레벨별 독립 Gaussian blur             |
|           v                                                         |
|  +------------------------------------------------------+           |
|  |              Upsample Chain                          |           |
|  |  MIP[N-1] -> ... -> MIP[1] -> MIP[0] (= Up[0])      |           |
|  |                                                      |           |
|  |  Each level: 9-tap Tent + Progressive Blend         |           |
|  |  Weight: level1Weight * pow(decayRate, level)       |           |
|  +------------------------------------------------------+           |
|           |                                                         |
|           v                                                         |
|  +------------------+                                               |
|  |    Composite     |  Screen/Add/Overlay + Color + Exposure       |
|  |   (Final)        |  -> Original + Glow blend                    |
|  +--------+---------+                                               |
|           |                                                         |
|           v                                                         |
|      [Output]                                                       |
|                                                                     |
+---------------------------------------------------------------------+
```

---

## 0. 커널 목록 (CUDA)

| 커널 | 파일 위치 | 목적 |
|------|-----------|------|
| `RefineKernel` | 381행 | BoundingBox 계산 (atomicMin/Max) |
| `DesaturationKernel` | 347행 | Max 채널로 채도 감소 |
| `PrefilterKernel` | 427행 | 13-tap blur + Soft Threshold |
| `Prefilter25TapKernel` | 611행 | 25-tap (5x5) 고품질 |
| `PrefilterSep5H/VKernel` | 770/828행 | Separable 5-tap |
| `PrefilterSep9H/VKernel` | 909/979행 | Separable 9-tap |
| `Gaussian2DDownsampleKernel` | 1075행 | 9-tap 2D Gaussian |
| `UpsampleKernel` | 1179행 | 9-tap + Falloff blend |
| `PreblurGaussianH/VKernel` | 1413/1487행 | 병렬 스트림용 Separable Gaussian |
| `DebugOutputKernel` | 1560행 | 최종 합성 + 디버그 |

---

## 1. Refine 단계 (BoundingBox)

### 목적
- 밝은 픽셀 영역만 계산하여 GPU 연산량 감소
- 작은 콘텐츠에서 최대 20배 성능 향상

### 알고리즘

```cuda
for each pixel (x, y):
    L = 0.299*R + 0.587*G + 0.114*B  // Rec.601 휘도
    if L >= threshold:
        atomicMin(globalMinX, x - blurRadius)
        atomicMax(globalMaxX, x + blurRadius)
        atomicMin(globalMinY, y - blurRadius)
        atomicMax(globalMaxY, y + blurRadius)
```

### 성능 효과

```
Without BoundingBox:
  1920x1080 = 2,073,600 pixels -> 100% work

With BoundingBox (작은 텍스트):
  320x320 = 102,400 pixels -> 4.9% work

성능 향상: ~20x (sparse content에서)
```

---

## 2. Prefilter 단계

### 목적
- 밝은 영역만 추출 (Threshold)
- 초기 블러 적용 (Firefly 방지)
- ZeroPad 샘플링으로 엣지 아티팩트 방지

### 2.1 샘플링 패턴

#### 13-tap Star Pattern (기본)

```
        A   B   C
          D   E
        F   G   H
          I   J
        K   L   M

가중치:
- 외곽 코너 (A,C,K,M): 0.03125 * 4 = 0.125
- 외곽 크로스 (B,F,H,L): 0.0625 * 4 = 0.25
- 내부 코너 (D,E,I,J): 0.125 * 4 = 0.5
- 중앙 (G): 0.125
총합: 1.0
```

#### 25-tap Grid Pattern (고품질)

```
■ ■ ■ ■ ■
■ ■ ■ ■ ■
■ ■ ● ■ ■   (5x5 그리드, 중앙 가중)
■ ■ ■ ■ ■
■ ■ ■ ■ ■
```

#### Separable 9-tap (최고 속도)

```
H-pass: ■ ■ ■ ● ■ ■ ■ ■ ■   (1x9)
V-pass: 동일 패턴 세로 (9x1)
```

### 2.2 Soft Threshold

```
Dynamic K: maxK = min(T, 1-T), actualK = maxK x softness

유효 범위: [T-K, T+K] (T를 중심으로 대칭)

brightness < T-K  -> contribution = 0 (제거)
brightness > T+K  -> contribution = 1 (통과)
T-K <= brightness <= T+K  -> S-curve 보간

S-curve: t = (brightness - lowerBound) / (2K)
         contribution = t^2 * (3 - 2t)
```

### 2.3 ZeroPad 샘플링

```cuda
// UV가 [0,1] 범위를 벗어나면 0 반환
if (u < 0 || u > 1 || v < 0 || v > 1) {
    return float4(0, 0, 0, 0);
}
```

**왜 ZeroPad인가?**
- Clamp 샘플링: 엣지 픽셀이 반복되어 밝기 집중
- ZeroPad 샘플링: 엣지에서 자연스럽게 페이드 아웃
- 다른 버퍼 크기 간 일관된 밝기 유지

---

## 3. Downsample Chain

### 목적
- MIP 피라미드 생성 (점진적 블러)
- 각 레벨: 이전 레벨의 1/2 해상도

### 알고리즘 (9-tap 2D Gaussian)

```
        1   2   1
        2   4   2    / 16
        1   2   1

가중치:
- Center (C):  4/16 = 0.25
- Cross (T,L,R,B): 2/16 = 0.125 each
- Diagonal (TL,TR,BL,BR): 1/16 = 0.0625 each
```

### Dynamic Offset

```cpp
levelRatio = level / (maxLevels - 1);
offset = offsetDown + spreadDown * levelRatio;

// offsetDown: 기본 오프셋 (기본값 1.0)
// spreadDown: 진행량 (기본값 0.5)
// 높은 레벨일수록 더 넓은 샘플링 -> 더 많은 블러
```

**왜 Kawase 대신 Gaussian인가?**
- Kawase 5-tap: 빠르지만 서브픽셀 이동 시 플리커링
- 9-tap 2D Gaussian: 약간 느림, 시간적으로 안정

### MIP 레벨 계산

```cpp
// Quality 파라미터 (6-12)로 최대 MIP 깊이 결정
int mipLevels = quality;  // 6 = 빠름, 12 = 최대 품질

// 실제 레벨은 해상도에 따라 제한
// 최소 차원이 16px 이하가 될 때까지
while (width > 16 && height > 16 && level < mipLevels) {
    width /= 2;
    height /= 2;
    level++;
}
```

### MIP Chain 구조

```
Level 0: 1920 x 1080 (full resolution, prefilter output)
Level 1:  960 x 540  (÷2)
Level 2:  480 x 270  (÷2)
Level 3:  240 x 135  (÷2)
Level 4:  120 x 68   (÷2)
Level 5:   60 x 34   (÷2)
Level 6:   30 x 17   (÷2)
Level 7:   15 x 9    (÷2)
Level 8:    8 x 5    (minimum size ~16px)
```

---

## 4. Pre-blur (병렬 스트림, 실험적)

### 목적
- 각 MIP 레벨에서 독립적인 Gaussian blur
- 6개 CUDA 스트림으로 병렬 실행

### 알고리즘

```cpp
// 6개 병렬 스트림으로 레벨별 독립 처리
for level in 1..mipLevels:
    streamIdx = (level - 1) % 6

    // Separable Gaussian: H-pass -> V-pass
    launchKernel(PreblurGaussianH, stream=preblurStreams[streamIdx])
    launchKernel(PreblurGaussianV, stream=preblurStreams[streamIdx])

// 모든 스트림 동기화
syncAllStreams()
```

### Progressive Blur (σ 계산)

```
σ = baseSigma × √level

Level 1: σ = 1.5 × √1 = 1.5
Level 4: σ = 1.5 × √4 = 3.0
Level 9: σ = 1.5 × √9 = 4.5

커널 반경 = 3σ (99.7% 커버리지)
최대 반경 = 15 (성능 제한)
```

### 커널 구현

```cuda
// 가우시안 가중치 계산
float twoSigmaSq = 2.0f * sigma * sigma;
for (int i = -radius; i <= radius; i++) {
    weights[i + radius] = exp(-i*i / twoSigmaSq);
}
// 정규화
weights /= sum(weights);

// ZeroPad 샘플링으로 블러 적용
for (int dx = -radius; dx <= radius; dx++) {
    if (sampleX >= 0 && sampleX < width) {
        sum += input[idx] * weights[dx + radius];
    }
    // else: ZeroPad (0 기여)
}
```

---

## 5. Upsample Chain

### 목적
- MIP 피라미드를 역으로 순회
- 각 레벨에서 블러와 가중치 블렌딩

### 알고리즘

```cpp
for level from (mipLevels - 1) down to 0:
    // Step 1: 이전 (작은) 레벨에서 업샘플
    upsampled = GaussianUpsample(previousLevel, offset)

    // Step 2: 현재 레벨의 저장된 다운샘플 가져오기
    current = mipChain[level]

    // Step 3: 물리적 가중치 계산
    weight = calculatePhysicalWeight(level, falloff, level1Weight)

    // Step 4: 블렌딩
    result = upsampled + current * weight

    // 다음 반복을 위해 저장
    upsampleChain[level] = result
```

### 9-tap Tent Filter

```
        1   2   1
        2   4   2    / 16
        1   2   1

Tent 필터: 삼각형 분포 (중앙 강조)
다운샘플과 동일한 가중치
```

### Falloff 계산

```cpp
// Falloff (0-100, 50=중립)
// 0%: 외곽 레벨 부스트 (decayRate = 1.5)
// 50%: 중립 (decayRate = 1.0)
// 100%: 코어 집중 (decayRate = 0.5)

float normalizedFalloff = (falloff - 50) / 50;  // -1 ~ 1
float decayRate = 1.0 - normalizedFalloff * 0.5;  // 0.5 ~ 1.5

// 가중치 공식
weight = level1Weight * pow(decayRate, level);

// 예시 (level1Weight = 1.0):
// Falloff 0%:   decayRate=1.5, level 3 weight = 1.0 × 1.5³ = 3.375
// Falloff 50%:  decayRate=1.0, level 3 weight = 1.0 × 1.0³ = 1.0
// Falloff 100%: decayRate=0.5, level 3 weight = 1.0 × 0.5³ = 0.125
```

### Composite Modes (내부)

```cpp
switch (compositeMode):
    Add:     result = base + glow
    Screen:  result = base + glow - base * glow
    Overlay: result = base < 0.5 ? 2*base*glow : base + 2*glow*(1-base)
```

---

## 6. Composite 단계 (최종 출력)

### 목적
- 원본 이미지와 글로우 합성
- 컬러 조정, 노출 적용, 디더링

### 최종 합성

```cpp
// 노출 및 불투명도 적용
glow = glowBuffer * exposure * glowOpacity

// 소스 불투명도 적용
source = original * sourceOpacity

// 글로우 알파 추정
glowAlpha = min(max(glowR, glowG, glowB), 1.0)

// 결과 알파
resultAlpha = max(sourceAlpha, glowAlpha)

// 합성 모드에 따른 블렌드
switch (compositeMode):
    Add:     result = source + glow
    Screen:  result = source + glow - source * glow
    Overlay: // 채널별 오버레이 블렌드
```

### 컬러 조정

```cpp
// Color Temperature (색온도)
glow.r *= colorTempR;  // warm: R+, B-
glow.g *= colorTempG;
glow.b *= colorTempB;

// Glow Color Tint
coloredGlow = lerp(glow * glowColor, glow, preserveColor);

// Exposure (노출)
coloredGlow *= exposure;
```

### Chromatic Aberration (색수차)

```cpp
// CA 양: 0-100
caAmount = chromaticAberration * 0.002;

// R 채널: 외곽에서 샘플
uR = u + dirX * caAmount;
vR = v + dirY * caAmount;

// B 채널: 내부에서 샘플
uB = u - dirX * caAmount;
vB = v - dirY * caAmount;

// G 채널: 중앙
glowR = sample(uR, vR).R;
glowG = sample(u, v).G;
glowB = sample(uB, vB).B;
```

### 디더링 (밴딩 방지)

```cpp
// 위치 기반 의사 난수
noise = fmod(sin(x * 12.9898 + y * 78.233) * 43758.5453, 1.0) - 0.5;
ditherAmount = dither * (2/255) * noise;
result += ditherAmount;
```

---

## 7. 색 공간 처리

### Premultiplied Alpha

After Effects는 Premultiplied Alpha 사용:

```
Premultiplied:  R' = R * A, G' = G * A, B' = B * A
Straight:       R, G, B, A (독립)
```

### sRGB/Linear 변환

```cpp
// sRGB -> Linear (입력 시)
if (c <= 0.04045)
    linear = c / 12.92;
else
    linear = pow((c + 0.055) / 1.055, 2.4);

// Linear -> sRGB (출력 시)
if (c <= 0.0031308)
    srgb = c * 12.92;
else
    srgb = 1.055 * pow(c, 1/2.4) - 0.055;
```

**왜 Linear 공간인가?**
- 빛 덧셈은 Linear 공간에서만 물리적으로 정확
- Gamma 공간 블렌딩은 "흐린" 색상 발생
- HDR 콘텐츠에 필수

### 추가 프로파일 지원

```cpp
// Rec.709 -> Linear
if (c < 0.081)
    linear = c / 4.5;
else
    linear = pow((c + 0.099) / 1.099, 1/0.45);

// Gamma 2.2 -> Linear
linear = pow(c, 2.2);
```

---

## 8. 파라미터 영향 매핑

| 파라미터 | 영향 단계 | 효과 |
|------------------|---------------|-------------------------------|
| Intensity        | Upsample      | Level 1 시작 가중치 (0-100%) |
| Exposure         | Composite     | 최종 밝기 배수 (0-50x)       |
| Radius           | Upsample      | 활성 MIP 레벨 제한           |
| Spread Down/Up   | Down/Upsample | 최대 레벨에서의 오프셋 증가량 |
| Offset Down/Up   | Down/Upsample | 기본 샘플링 오프셋           |
| Offset Prefilter | Prefilter     | 13-tap 샘플 간격             |
| Falloff          | Upsample      | 레벨별 가중치 감쇠율         |
| Threshold        | Prefilter     | 밝기 임계값 (0-100%)         |
| Soft Knee        | Prefilter     | 임계값 전이 부드러움         |
| Quality          | Downsample    | MIP 깊이 (6-12)              |
| Desaturation     | Post-Prefilter| Max 채널 쪽으로 채도 감소    |

---

## 9. 버퍼 구조

### MIP Chain 버퍼

```
m_mipChain[0]  = 1920x1080  (Prefilter 출력)
m_mipChain[1]  = 960x540
m_mipChain[2]  = 480x270
m_mipChain[3]  = 240x135
...
m_mipChain[N-1] = 16x16 (최소)
```

### Upsample Chain 버퍼 (CUDA)

```
m_upsampleChain[0]  = 1920x1080  (최종 글로우)
m_upsampleChain[1]  = 960x540
...
m_upsampleChain[N-1] = 16x16

별도 버퍼 사용 이유: Race condition 방지
(같은 버퍼에서 읽기/쓰기 동시 수행 방지)
```

### Pre-blur 버퍼 (CUDA)

```
m_preblurResults[0..N-1]  = 레벨별 최종 blur 결과
m_preblurTemp[0..N-1]     = H-pass 임시 버퍼

각 레벨 독립적 버퍼 -> 병렬 스트림 가능
```

---

## 10. GPU 구현 차이

### CUDA (JustGlowCUDARenderer)

```
장점:
- 더 유연한 메모리 접근
- 복잡한 알고리즘 구현 용이
- NVIDIA 최적화
- 병렬 스트림 지원

파일:
- src/JustGlowKernels.cu (커널 소스, ~2000줄)
- CUDA_Assets/*.ptx (컴파일된 PTX)
```

### DirectX 12 (JustGlowGPURenderer)

```
장점:
- DispatchIndirect 최적화 (CPU-GPU 동기화 제거)
- 하드웨어 텍스처 샘플링 (더 빠름)
- 모든 GPU 지원 (AMD, Intel, NVIDIA)

파일:
- shaders/*.hlsl (Compute Shaders)
- DirectX_Assets/*.cso (컴파일된 바이트코드)
```

### 런타임 선택

```cpp
// GPUDeviceSetup에서 결정
switch (gpu_framework) {
    case CUDA:
        renderer = new JustGlowCUDARenderer();
        break;
    case DirectX:
        renderer = new JustGlowGPURenderer();
        break;
    default:
        // CPU fallback (에러 메시지)
}
```

---

## 11. 최적화 기법

### 11.1 BoundingBox 최적화

**목적**: 콘텐츠 영역만 처리하여 GPU 연산량 감소

```
+----------------------------------------+
|         1920x1080 Full Image           |
|                                        |
|     +----------------+                 |
|     |   BoundingBox  |  <- Content     |
|     |   320x320      |     area only   |
|     +----------------+                 |
|                                        |
+----------------------------------------+

Performance: 1920x1080 -> 320x320 = 36x reduction!
```

### 11.2 DispatchIndirect (DirectX 전용)

**목적**: CPU-GPU 동기화 제거

```
[ResetBoundsCS]
      |
      v
[RefineCS] -> atomicMin/Max for BoundingBox
      |
      v
[CalcIndirectArgsCS] -> ThreadGroupCount -> IndirectArgsBuffer
      |
      v
[ExecuteIndirect] -> GPU reads buffer for dispatch size
      |
      v
[PrefilterWithBounds] -> Read offset from BoundsOutput
```

### 11.3 ZeroPad 샘플링

**문제**: Clamp 샘플링 시 엣지 픽셀 반복 -> 밝기 집중

```
Clamp:   [A][A][A][A][B][C][D][E][E][E][E]
                 ^ edge repetition

ZeroPad: [0][0][0][0][B][C][D][E][0][0][0]
                 ^ natural fade out
```

### 11.4 Separable Gaussian

**원리**: 2D Gaussian = H-pass × V-pass

```
N×N 2D blur = O(N²) 연산
1×N H-pass + N×1 V-pass = O(2N) 연산

예: 31-tap (σ=5) blur
  - 2D: 31×31 = 961 샘플
  - Separable: 31+31 = 62 샘플 (15배 빠름!)
```

---

## 12. 디버그 뷰 모드

| 모드 | 설명 | 색상 표시 |
|------|------|-----------|
| 1 (Final) | 최종 합성 결과 | - |
| 2 (Prefilter) | MIP[0] (Threshold 적용 후) | 노랑 |
| 3-7 (Down1-5) | 다운샘플 레벨 1-5 | 초록 |
| 8 (Down6+) | 다운샘플 레벨 6+ | 파랑 |
| 9-15 (Up0-6) | 업샘플 레벨 0-6 | 시안 |
| 16 (GlowOnly) | 원본 없이 글로우만 | - |
| 17 (BoundingBox) | BoundingBox 오버레이 표시 | 빨강 테두리 |

---

## 13. 성능 특성

| 단계 | 복잡도 | 병목 |
|------|--------|------|
| Refine | O(N) | 메모리 대역폭 |
| Prefilter | O(N×k) | 연산 (k taps) |
| Downsample | O(N×9) per level | 메모리 대역폭 |
| Pre-blur | O(N×2r) per level | 연산 (r = 3σ) |
| Upsample | O(N×9) per level | 메모리 대역폭 |
| Composite | O(N) | 메모리 대역폭 |

**총**: O(N × log₂(minDim) × 9) ≈ O(N × 72) for 8 MIP levels

---

## 14. 알려진 제한사항

1. **CUDA는 NVIDIA 전용**: AMD/Intel GPU는 DirectX만 사용
2. **DispatchIndirect는 DirectX 전용**: CUDA는 CPU-GPU 동기화 필요
3. **32-bit float 버퍼**: 메모리 사용량 높음 (HDR 지원 위해 필수)
4. **최소 16px까지만 다운샘플**: 더 작으면 품질 저하

---

## 15. 버전 히스토리

| 버전 | 변경 사항 |
|------|----------|
| 1.6.0 | Pre-blur 병렬 스트림, PreblurGaussianH/VKernel 추가 |
| 1.5.4 | All-CUDA 리팩토링, Interop 코드 제거 |
| 1.5.3 | ZeroPad 샘플링, BoundingBox 최적화 |
| 1.5.0 | Kawase 제거 (temporal flickering 수정) |
| 1.4.0 | Karis Average 제거 (아티팩트 수정) |

---

## 참고 자료

- [SIGGRAPH 2015] Bandwidth-Efficient Rendering (ARM)
- [GDC 2014] Next-Gen Post Processing (Call of Duty)
- [Unity] High-Quality Bloom
- [NVIDIA] CUDA Programming Guide
- [Microsoft] DirectX 12 ExecuteIndirect
