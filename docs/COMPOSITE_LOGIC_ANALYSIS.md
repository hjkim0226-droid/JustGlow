# Composite 로직 완전 분석

## 입력 상황 가정

```
원본 텍스트 레이어:
- 텍스트 영역: (1.0, 0.8, 0.6, 1.0) - 밝은 살구색, 불투명
- 텍스트 바깥: (0.0, 0.0, 0.0, 0.0) - 완전 투명

글로우 버퍼 Up[0]:
- 텍스트 중심: (0.08, 0.06, 0.04, 1.0) - 블러된 값
- 글로우 가장자리: (0.02, 0.015, 0.01, 1.0) - 퍼진 영역
- Up[0]의 alpha는 항상 1.0 (Downsample/Upsample에서 하드코딩)

파라미터:
- exposure = 10.0
- glowOpacity = 1.0
- sourceOpacity = 1.0
- compositeMode = 1 (Add)
- useLinear = false
```

---

## Step 1: 소스 샘플링

```cuda
// 원본 이미지에서 읽기
int origIdx = (srcY * originalPitch + srcX) * 4;
origR = original[origIdx + 0];  // premultiplied R
origG = original[origIdx + 1];  // premultiplied G
origB = original[origIdx + 2];  // premultiplied B
origA = original[origIdx + 3];  // Alpha
```

**텍스트 영역:**
```
origR = 1.0, origG = 0.8, origB = 0.6, origA = 1.0
```

**텍스트 바깥 (글로우만 있는 영역):**
```
origR = 0.0, origG = 0.0, origB = 0.0, origA = 0.0
```

---

## Step 2: UV 좌표 계산

```cuda
float u = ((float)x + 0.5f) / (float)width;   // 0.0 ~ 1.0
float v = ((float)y + 0.5f) / (float)height;  // 0.0 ~ 1.0
```

출력 해상도 기준으로 정규화된 좌표.

---

## Step 3: 글로우 샘플링 (Bilinear)

```cuda
float glowR, glowG, glowB, glowA;
sampleBilinear(glow, u, v, glowWidth, glowHeight, glowPitch,
               glowR, glowG, glowB, glowA);
```

Up[0] 버퍼에서 bilinear 보간으로 샘플링.

**글로우 가장자리 영역:**
```
glowR = 0.02, glowG = 0.015, glowB = 0.01, glowA = 1.0
```

**중요:** `glowA`는 항상 1.0 (Downsample/Upsample 커널에서 하드코딩됨)

---

## Step 4: Exposure × GlowOpacity 적용

```cuda
glowR *= exposure * glowOpacity;  // 0.02 * 10 * 1.0 = 0.2
glowG *= exposure * glowOpacity;  // 0.015 * 10 * 1.0 = 0.15
glowB *= exposure * glowOpacity;  // 0.01 * 10 * 1.0 = 0.1
```

**적용 후:**
```
glowR = 0.2, glowG = 0.15, glowB = 0.1
```

**주의:** glowA는 변경 안 됨 (여전히 1.0, 하지만 사용 안 함)

---

## Step 5: GlowCoverage 계산

```cuda
float glowCoverage = fmaxf(fmaxf(glowR, glowG), glowB);
// = max(0.2, 0.15, 0.1) = 0.2
```

**특징:**
- exposure 적용 후 RGB로 계산
- glowA(=1.0)는 사용하지 않고 RGB 최대값으로 추정
- exposure가 높으면 1.0 초과 후 clamp됨

---

## Step 6: Source Opacity 적용

```cuda
float srcR = origR * sourceOpacity;  // 0.0 * 1.0 = 0.0
float srcG = origG * sourceOpacity;  // 0.0 * 1.0 = 0.0
float srcB = origB * sourceOpacity;  // 0.0 * 1.0 = 0.0
float srcA = origA * sourceOpacity;  // 0.0 * 1.0 = 0.0
```

**글로우만 있는 영역 (텍스트 바깥):**
```
srcR = 0, srcG = 0, srcB = 0, srcA = 0
```

---

## Step 7: Final Alpha 계산

```cuda
float finalAlpha = fmaxf(srcA, clampf(glowCoverage, 0.0f, 1.0f));
// = max(0.0, clamp(0.2, 0, 1))
// = max(0.0, 0.2)
// = 0.2
```

**결과:** `finalAlpha = 0.2`

---

## Step 8: Composite 블렌딩

### Add 모드 (case 1):
```cuda
resR = srcR + glowR;  // 0.0 + 0.2 = 0.2
resG = srcG + glowG;  // 0.0 + 0.15 = 0.15
resB = srcB + glowB;  // 0.0 + 0.1 = 0.1
```

### Screen 모드 (case 2): ⚠️ 문제 발생!
```cuda
// Unpremultiply source
float straightSrcR = (srcA > 0.001f) ? srcR / srcA : 0.0f;
// srcA = 0 이므로 straightSrcR = 0.0

// Screen blend
float blendR = 1.0f - (1.0f - straightSrcR) * (1.0f - glowR);
// = 1.0 - (1.0 - 0.0) * (1.0 - 0.2)
// = 1.0 - 1.0 * 0.8
// = 0.2

// ⚠️ Repremultiply with finalAlpha
resR = blendR * finalAlpha;
// = 0.2 * 0.2
// = 0.04  ← 매우 어두워짐!
```

### Overlay 모드 (case 3): ⚠️ 더 심각!
```cuda
float straightSrcR = 0.0f;  // srcA = 0

// straightSrcR < 0.5 이므로 Multiply 분기
float blendR = 2.0f * straightSrcR * glowR;
// = 2.0 * 0.0 * 0.2
// = 0.0  ← 완전히 검정!

resR = blendR * finalAlpha;
// = 0.0 * 0.2
// = 0.0
```

---

## Step 9: Alpha 할당

```cuda
resA = finalAlpha;  // = 0.2
```

---

## Step 10: 최종 출력

### Add 모드:
```
output = (0.2, 0.15, 0.1, 0.2)
```

### Screen 모드:
```
output = (0.04, 0.03, 0.02, 0.2)  ← 매우 어두움!
```

### Overlay 모드:
```
output = (0.0, 0.0, 0.0, 0.2)  ← 검정!
```

---

## AE에서 흰 배경 위에 합성

AE는 premultiplied 합성:
```
result = premultRGB + background × (1 - alpha)
```

### Add 모드 출력: (0.2, 0.15, 0.1, 0.2)
```
resultR = 0.2 + 1.0 × (1 - 0.2)
        = 0.2 + 0.8
        = 1.0 (흰색) ✓
```

### Screen 모드 출력: (0.04, 0.03, 0.02, 0.2)
```
resultR = 0.04 + 1.0 × (1 - 0.2)
        = 0.04 + 0.8
        = 0.84 (약간 어두운 회색) ← 그림자처럼!
```

### Overlay 모드 출력: (0.0, 0.0, 0.0, 0.2)
```
resultR = 0.0 + 1.0 × (1 - 0.2)
        = 0.0 + 0.8
        = 0.8 (회색) ← 완전 그림자!
```

---

## 문제 요약

| 문제 | 위치 | 설명 |
|------|------|------|
| **1** | Step 5 | glowCoverage를 exposure 적용 후 RGB에서 계산 → 값이 작으면 alpha도 작아짐 |
| **2** | Step 8 (Screen/Overlay) | `blendR * finalAlpha` 이중 적용으로 어두워짐 |
| **3** | Step 8 (Overlay) | srcA=0일 때 straightSrc=0 → Multiply 분기 → 결과 0 |

---

## 해결 방안

### Option A: Screen/Overlay에서 repremult 제거
```cuda
case 2: // Screen
    resR = blendR;  // finalAlpha 곱하지 않음
    break;
```

### Option B: glowCoverage 대신 1.0 사용
```cuda
// 글로우는 발광이라 항상 불투명하게 처리
float finalAlpha = fmaxf(srcA, 1.0f);  // 또는 그냥 1.0f
```

### Option C: 원본 방식으로 롤백
```cuda
// Screen/Overlay도 premult 상태에서 직접 계산
resR = 1.0f - (1.0f - srcR) * (1.0f - glowR);
```

---

## Alpha 파이프라인 참고

```
Prefilter → sumA (가변)
Downsample → 1.0 (하드코딩)
Upsample → 1.0 (하드코딩)
─────────────────────────
Up[0].alpha = 항상 1.0
```

glowA를 사용해도 항상 1.0이라 의미 없음.
glowCoverage = max(RGB)로 추정하는 것이 현재 방식.
