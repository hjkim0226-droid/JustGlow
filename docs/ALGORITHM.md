# JustGlow 알고리즘 상세 문서

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
|  |  Resolution: halved per level                        |           |
|  +------------------------------------------------------+           |
|           |                                                         |
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

## 1. Prefilter 단계

### 목적
- 밝은 영역만 추출 (Threshold)
- 초기 블러 적용 (Firefly 방지)
- ZeroPad 샘플링으로 엣지 아티팩트 방지

### 알고리즘

#### 1.1 샘플링 패턴 (13-tap)

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

#### 1.2 Soft Threshold

```
Dynamic K: maxK = min(T, 1-T), actualK = maxK x softness

유효 범위: [T-K, T+K] (T를 중심으로 대칭)

brightness < T-K  -> contribution = 0 (removed)
brightness > T+K  -> contribution = 1 (pass through)
T-K <= brightness <= T+K  -> S-curve interpolation

S-curve: t = (brightness - lowerBound) / (2K)
         contribution = t^2 * (3 - 2t)
```

#### 1.3 ZeroPad 샘플링

```cpp
// UV가 [0,1] 범위를 벗어나면 0 반환
if (u < 0 || u > 1 || v < 0 || v > 1) {
    return float4(0, 0, 0, 0);
}
```

**왜 ZeroPad인가?**
- Clamp 샘플링: 엣지 픽셀이 반복되어 밝기 집중
- ZeroPad 샘플링: 엣지에서 자연스럽게 페이드 아웃

---

## 2. Downsample Chain

### 목적
- MIP 피라미드 생성 (점진적 블러)
- 각 레벨: 이전 레벨의 1/2 해상도

### 알고리즘 (9-tap 2D Gaussian)

```
        1   2   1
        2   4   2    / 16
        1   2   1

샘플 위치: ±1 texel (offset 조절 가능)
```

**왜 Kawase 대신 Gaussian인가?**
- Kawase 5-tap: 빠르지만 서브픽셀 이동 시 플리커링
- 9-tap 2D Gaussian: 시간적으로 안정 (temporal stability)

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

---

## 3. Upsample Chain

### 목적
- MIP 피라미드를 역으로 순회
- 각 레벨에서 블러와 가중치 블렌딩

### 알고리즘 (9-tap Tent)

```
        1   2   1
        2   4   2    / 16
        1   2   1

Tent 필터: 삼각형 분포 (중앙 강조)
```

### Progressive Blending

```cpp
// 각 레벨에서 누적 블렌딩
float weight = level1Weight x pow(decayRate, level);
result = lerp(lowerLevel, higherLevel + lowerLevel x weight, 0.5);
```

### Falloff 계산

```cpp
// Falloff (0-100, 50=중립)
// 0%: 외곽 레벨 부스트 (decayRate = 1.5)
// 50%: 중립 (decayRate = 1.0)
// 100%: 코어 집중 (decayRate = 0.5)

float normalizedFalloff = (falloff - 50) / 50;  // -1 ~ 1
float decayRate = 1.0 - normalizedFalloff x 0.5;  // 0.5 ~ 1.5
```

---

## 4. Composite 단계

### 목적
- 원본 이미지와 글로우 합성
- 컬러 조정, 노출 적용

### 블렌드 모드

```cpp
// Add (밝음, HDR 느낌)
result = original + glow;

// Screen (자연스러움, 기본값)
result = 1 - (1 - original) x (1 - glow);

// Overlay (대비 강조)
if (original < 0.5)
    result = 2 x original x glow;
else
    result = 1 - 2 x (1 - original) x (1 - glow);
```

### 컬러 조정

```cpp
// Color Temperature (색온도)
glow.r *= colorTempR;  // warm: R+, B-
glow.g *= colorTempG;
glow.b *= colorTempB;

// Glow Color Tint
coloredGlow = lerp(glow x glowColor, glow, preserveColor);

// Exposure (노출)
coloredGlow *= exposure;
```

---

## 5. 파라미터 영향 매핑

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

## 6. 버퍼 구조

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

---

## 7. DirectX vs CUDA 구현 차이

### DirectX 12 (JustGlowGPURenderer)

```
장점:
- DispatchIndirect 최적화 가능
- 하드웨어 텍스처 샘플링 (더 빠름)
- 모든 GPU 지원

파일:
- shaders/*.hlsl (Compute Shaders)
- *.cso (컴파일된 바이트코드)
```

### CUDA (JustGlowCUDARenderer)

```
장점:
- 더 유연한 메모리 접근
- 복잡한 알고리즘 구현 용이
- NVIDIA 최적화

파일:
- src/JustGlowKernels.cu (커널 소스)
- CUDA_Assets/*.ptx (컴파일된 PTX)
```

### 런타임 선택

```cpp
// GPUDeviceSetup에서 결정
if (CUDA 사용 가능 && NVIDIA GPU) {
    renderer = new JustGlowCUDARenderer();
} else if (DirectX 12 사용 가능) {
    renderer = new JustGlowGPURenderer();
} else {
    // CPU fallback (에러 메시지)
}
```

---

## 8. 최적화 기법

### 8.1 DispatchIndirect (DirectX 전용)

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

**파이프라인**:

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

### 8.2 ZeroPad 샘플링

**문제**: Clamp 샘플링 시 엣지 픽셀 반복 -> 밝기 집중

```
Clamp:   [A][A][A][A][B][C][D][E][E][E][E]
                 ^ edge repetition

ZeroPad: [0][0][0][0][B][C][D][E][0][0][0]
                 ^ natural fade out
```

### 8.3 9-tap 2D Gaussian (Kawase 대체)

**문제**: Kawase 5-tap은 서브픽셀 이동 시 플리커링

```
Kawase 5-tap:     빠름, 하지만 불안정
9-tap 2D Gaussian: 약간 느림, 시간적 안정
```

---

## 9. 색 공간 처리

### Premultiplied Alpha

After Effects는 Premultiplied Alpha 사용:

```
Premultiplied:  R' = R x A, G' = G x A, B' = B x A
Straight:       R, G, B, A (독립)
```

### Linear 변환 (선택적)

```cpp
// sRGB -> Linear (input)
if (c <= 0.04045)
    linear = c / 12.92;
else
    linear = pow((c + 0.055) / 1.055, 2.4);

// Linear -> sRGB (output)
if (c <= 0.0031308)
    srgb = c x 12.92;
else
    srgb = 1.055 x pow(c, 1/2.4) - 0.055;
```

---

## 10. 디버그 뷰 모드

| 모드        | 설명                          |
|-------------|-------------------------------|
| Final       | 최종 합성 결과                |
| Prefilter   | MIP[0] (Threshold 적용 후)    |
| Down1-6     | 다운샘플 레벨 1-6             |
| Up0-6       | 업샘플 레벨 0-6               |
| GlowOnly    | 원본 없이 글로우만            |
| BoundingBox | BoundingBox 오버레이 표시     |

---

## 11. 알려진 제한사항

1. **CUDA는 NVIDIA 전용**: AMD/Intel GPU는 DirectX만 사용
2. **DispatchIndirect는 DirectX 전용**: CUDA는 CPU-GPU 동기화 필요
3. **32-bit float 버퍼**: 메모리 사용량 높음 (HDR 지원 위해 필수)
4. **최소 16px까지만 다운샘플**: 더 작으면 품질 저하

---

## 12. 향후 개선 가능 방향

1. **CUDA BoundingBox 최적화**: RefineKernel -> Grid 크기 동적 조절
2. **DX12-CUDA Interop**: 하이브리드 렌더링 (블러는 CUDA, 블렌드는 DX)
3. **Metal 지원**: macOS용 렌더러 추가
4. **Separable 블러**: 2-pass (H->V)로 더 넓은 블러 가능

---

## 참고 자료

- [SIGGRAPH 2015] Bandwidth-Efficient Rendering (ARM)
- [GDC 2014] Next-Gen Post Processing (Call of Duty)
- [Unity] High-Quality Bloom
- [NVIDIA] CUDA Programming Guide
