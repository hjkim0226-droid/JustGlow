# Unmult Plugin Reference

## 개요

Unmult는 "matted with black" (without Alpha) 상태의 이미지에서 alpha를 추정하고 premultiplied 형태로 복원하는 기법.

## 용어 정리

| 용어 | 설명 |
|------|------|
| **Straight Alpha** | RGB와 Alpha가 분리된 상태. RGB는 순수 색상 |
| **Premultiplied Alpha** | RGB = Straight RGB × Alpha. RGB에 alpha가 곱해진 상태 |
| **Without Alpha (Matted with Black)** | Premultiplied RGB만 저장, Alpha 채널 없음. 검정 배경에 합성된 결과 |

## 핵심 공식

### Premultiplied → Without Alpha
```
withoutAlpha.R = straight.R × alpha
withoutAlpha.G = straight.G × alpha
withoutAlpha.B = straight.B × alpha
(alpha 채널은 버림)
```

### Without Alpha → Premultiplied (Unmult)
```
estimatedAlpha = max(R, G, B)
premult.R = withoutAlpha.R / estimatedAlpha
premult.G = withoutAlpha.G / estimatedAlpha
premult.B = withoutAlpha.B / estimatedAlpha
premult.A = estimatedAlpha
```

## 계산 예시

### 예시 1: 균일한 RGB

**원본:**
```
Straight: (0.2, 0.2, 0.2), Alpha: 0.8
```

**Without Alpha로 저장:**
```
R = 0.2 × 0.8 = 0.16
G = 0.2 × 0.8 = 0.16
B = 0.2 × 0.8 = 0.16
→ (0.16, 0.16, 0.16)
```

**Unmult:**
```
estimatedAlpha = max(0.16, 0.16, 0.16) = 0.16

premult.R = 0.16 / 0.16 = 1.0
premult.G = 0.16 / 0.16 = 1.0
premult.B = 0.16 / 0.16 = 1.0
premult.A = 0.16

→ (1.0, 1.0, 1.0, 0.16)
```

**검증 (실제 밝기):**
```
원래: 0.2 × 0.8 = 0.16
복원: 1.0 × 0.16 = 0.16 ✓
```

### 예시 2: 다른 RGB 값

**원본:**
```
Straight: (0.5, 0.4, 0.3), Alpha: 0.8
```

**Without Alpha로 저장:**
```
R = 0.5 × 0.8 = 0.4
G = 0.4 × 0.8 = 0.32
B = 0.3 × 0.8 = 0.24
→ (0.4, 0.32, 0.24)
```

**Unmult:**
```
estimatedAlpha = max(0.4, 0.32, 0.24) = 0.4

premult.R = 0.4 / 0.4 = 1.0
premult.G = 0.32 / 0.4 = 0.8
premult.B = 0.24 / 0.4 = 0.6
premult.A = 0.4

→ (1.0, 0.8, 0.6, 0.4)
```

**검증 (실제 밝기 보존):**
```
R: 원래 0.5 × 0.8 = 0.4, 복원 1.0 × 0.4 = 0.4 ✓
G: 원래 0.4 × 0.8 = 0.32, 복원 0.8 × 0.4 = 0.32 ✓
B: 원래 0.3 × 0.8 = 0.24, 복원 0.6 × 0.4 = 0.24 ✓
```

## 중요 특성

### 1. 실제 밝기 보존
- Straight와 Alpha 값은 원본과 달라질 수 있음
- 하지만 **실제 밝기 (Straight × Alpha)는 항상 보존**
- 화면에 표시되는 결과는 동일

### 2. 정보 손실
- Without Alpha로 저장 시 원래 alpha 정보 손실
- Unmult로 복원해도 원래 straight/alpha 조합은 복원 불가
- 단, 시각적 결과는 동일

### 3. Max 채널이 1.0이 됨
- `estimatedAlpha = max(R, G, B)`로 나누므로
- 가장 밝은 채널이 항상 1.0이 됨
- 색상 비율은 유지됨

## CUDA 구현

```cuda
// Alpha 추정
float estimatedAlpha = fmaxf(fmaxf(R, G, B), 0.001f);

// Unmult (정규화된 premultiplied)
float premultR = R / estimatedAlpha;
float premultG = G / estimatedAlpha;
float premultB = B / estimatedAlpha;
float premultA = estimatedAlpha;
```

## 참고

- After Effects의 "Unmult" 플러그인과 동일한 원리
- 검정 배경에 합성된 영상에서 alpha를 복원할 때 사용
- Glow, 화염, 폭발 등 검정 배경 소스에 유용
