# JustGlow Alpha Blend Formula

## 핵심 개념

### 용어 정리
- **glowR (matted with black)**: 블러된 값, "검정 위에 합성된 결과", alpha 채널 없이 저장된 premultiplied 값
- **glowR / glowAlpha**: 정규화된 premultiplied 값 (straight가 아님!)
- **glowR / glowAlpha²**: straight 값

> "Matted with black" = premultiplied RGB가 검정 배경에 미리 합성된 상태. Alpha 정보는 RGB에 녹아있음.

### 블렌드 공식 (Add 모드)

```cuda
// 1. 알파 계산
float glowAlpha = fminf(fmaxf(fmaxf(glowR, glowG), glowB), 1.0f);
float resA = fmaxf(srcA, glowAlpha);

// 2. alpha=1로 가정하고 블렌드 (검정 배경처럼)
float tempR = srcR + glowR;
float tempG = srcG + glowG;
float tempB = srcB + glowB;

// 3. unmult (알파추정치로 나눔) → premultiplied RGB
resR = (resA > 0.001f) ? tempR / resA : 0.0f;
resG = (resA > 0.001f) ? tempG / resA : 0.0f;
resB = (resA > 0.001f) ? tempB / resA : 0.0f;
// resA 그대로 출력
```

### Screen 모드

```cuda
// 1. 알파 계산
float resA = fmaxf(srcA, glowAlpha);

// 2. alpha=1로 가정하고 Screen 블렌드
float tempR = srcR + glowR - srcR * glowR;

// 3. unmult
resR = (resA > 0.001f) ? tempR / resA : 0.0f;
```

## 검증

### 예시: glowR = 0.1, glowAlpha = 0.1

**투명 배경 (srcR=0, srcA=0):**
```
tempR = 0 + 0.1 = 0.1
resA = max(0, 0.1) = 0.1
resR = 0.1 / 0.1 = 1.0
output: (1.0, 0.1) premultiplied
화면 표시: 0.1 ✓
```

**검정 배경 (srcR=0, srcA=1):**
```
tempR = 0 + 0.1 = 0.1
resA = max(1, 0.1) = 1.0
resR = 0.1 / 1.0 = 0.1
output: (0.1, 1.0) premultiplied
화면 표시: 0.1 ✓
```

**결과: 둘 다 화면에 0.1로 동일하게 표시!**

## 왜 이렇게 해야 하는가?

1. **glowR은 스칼라**: 블러 과정에서 alpha 정보가 사라짐 (하드코딩 1)
2. **alpha 추정 필요**: `max(R, G, B)`로 추정
3. **블렌드는 검정 기준**: 둘 다 "검정 위에 합성된 값"으로 취급
4. **unmult로 정규화**: `resA`로 나눠서 premultiplied RGB 생성
5. **결과 일관성**: 투명/검정 배경에서 동일한 밝기

## 참고

- Add/Screen의 alpha는 원래 Over 공식을 따름 (검색 결과)
- 하지만 우리 경우 `max(srcA, glowAlpha)` 사용
- 이유: glow가 발광체이므로 배경의 alpha와 glow의 alpha 중 큰 값 사용
