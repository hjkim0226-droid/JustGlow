# JustGlow Interop Reference Archive

**보관 날짜**: 2025-12-28
**목적**: DX12-CUDA Interop 아키텍처 참조용

## 보관 이유

AE의 GPU 프레임워크 선택 메커니즘:
- NVIDIA GPU → CUDA 선택 (Pure CUDA 경로 사용)
- AMD/Intel GPU → DirectX 선택 (CUDA 없음, Interop 불가)

**결론**: DX12-CUDA Interop은 AMD/Intel 사용자에게 무의미 (CUDA 없음).
따라서 All-CUDA와 All-DX12를 별도 구현하는 것이 올바른 접근.

## 아키텍처 가치

이 코드에서 가져온 핵심 개념:
1. **병렬 스트림 (6개)**: MIP 레벨별 독립 처리
2. **BoundingBox 최적화**: 활성 영역만 처리
3. **Pre-blur 분리 전략**: Downsample 후 독립적 blur 실행

## 파일 목록

### JustGlowInterop_Reference.cpp
Interop 렌더링 파이프라인:
- `RenderWithInterop()` - Surface-based 렌더링 진입점
- `ExecutePrefilterInterop()` - 13-tap Prefilter (Surface I/O)
- `ExecuteDownsampleChainInterop()` - 9-tap Downsample (Surface I/O)
- `ExecuteLogTransPreblurInterop()` - Separable Gaussian (병렬 스트림)

### JustGlowKernels_Surface.cu
Surface 기반 CUDA 커널:
- `UnmultSurfaceKernel` - √max 언멀티플라이
- `PrefilterSurfaceKernel` - 13-tap blur + threshold
- `DownsampleSurfaceKernel` - 9-tap 2D Gaussian
- `LogTransPreblurHSurfaceKernel` - 수평 Gaussian blur
- `LogTransPreblurVSurfaceKernel` - 수직 Gaussian blur
- `ClearSurfaceKernel` - 표면 초기화

### JustGlowKernels_Experimental.cu
실험적 커널 (비활성화 상태):
- `LogTransmittancePreblurKernel` - Log-Transmittance 풀 커널
- `LogTransmittancePreblurHKernel` - 분리형 수평
- `LogTransmittancePreblurVKernel` - 분리형 수직

**비활성화 이유**: Karis Average v1.4.0에서 아티팩트 발생으로 제거됨

## 참조 방법

### 병렬 스트림 초기화
```cpp
static const int MAX_PARALLEL_STREAMS = 6;
CUstream m_parallelStreams[MAX_PARALLEL_STREAMS];

bool InitializeStreams() {
    for (int i = 0; i < MAX_PARALLEL_STREAMS; i++) {
        CUresult err = cuStreamCreate(&m_parallelStreams[i], CU_STREAM_NON_BLOCKING);
        if (err != CUDA_SUCCESS) return false;
    }
    return true;
}
```

### Pre-blur 병렬 실행 패턴
```cpp
for (int level = 1; level <= numLevels; level++) {
    int streamIdx = (level - 1) % MAX_PARALLEL_STREAMS;
    CUstream execStream = m_parallelStreams[streamIdx];

    // H-blur
    cuLaunchKernel(m_blurHKernel, ..., execStream, ...);
    // V-blur
    cuLaunchKernel(m_blurVKernel, ..., execStream, ...);
}

// 모든 스트림 동기화
for (int i = 0; i < MAX_PARALLEL_STREAMS; i++) {
    cuStreamSynchronize(m_parallelStreams[i]);
}
```

### BoundingBox 전파
```cpp
// Source → Destination 좌표 변환 (다운샘플)
int dstMinX = srcBounds.minX / 2;
int dstMinY = srcBounds.minY / 2;
int dstMaxX = (srcBounds.maxX + 1) / 2;
int dstMaxY = (srcBounds.maxY + 1) / 2;
```

## 향후 계획

1. **All-CUDA**: 이 아키텍처를 CUdeviceptr 기반으로 재구현 (현재 진행 중)
2. **All-DX12**: AMD/Intel GPU용 순수 DirectX 12 구현
3. **Metal**: macOS용 Metal 구현
