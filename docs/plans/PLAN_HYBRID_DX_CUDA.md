# 하이브리드 DirectX + CUDA 아키텍처 계획

## 개요

DirectX의 **DispatchIndirect**와 CUDA의 **고급 연산**을 결합하여 CPU 동기화 없이 동적 BoundingBox 최적화 구현

## 현재 구조 vs 하이브리드 구조

### 현재 (CUDA only)
```
[CPU] cuLaunchKernel(Refine)
[CPU] cuStreamSynchronize() ← ★ 병목!
[CPU] cuMemcpyDtoH(bounds)  ← ★ 병목!
[CPU] Grid 크기 계산
[CPU] cuLaunchKernel(Prefilter, grid)
```

### 하이브리드 (DX + CUDA)
```
[DX]  Dispatch(Refine) → IndirectArgsBuffer에 Grid 크기 저장
[DX]  DispatchIndirect(Prefilter) ← GPU가 Grid 결정!
      ↓
[CUDA-DX Interop] 텍스처 공유
      ↓
[CUDA] 복잡한 처리 (선택적)
      ↓
[DX]  Composite
```

**CPU 동기화: 0회!**

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    JustGlow Hybrid Renderer                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              DirectX 12 Pipeline                      │   │
│  │                                                       │   │
│  │  [RefineCS] → IndirectArgsBuffer (Grid 크기)          │   │
│  │       ↓                                               │   │
│  │  [DispatchIndirect] → PrefilterCS                     │   │
│  │       ↓                                               │   │
│  │  [DispatchIndirect] → DownsampleCS (반복)             │   │
│  │       ↓                                               │   │
│  │  SharedTexture (DX↔CUDA)                              │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              CUDA Pipeline (선택적)                    │   │
│  │                                                       │   │
│  │  cudaGraphicsD3D12RegisterResource()                  │   │
│  │       ↓                                               │   │
│  │  [UpsampleKernel] - 복잡한 Falloff 계산               │   │
│  │       ↓                                               │   │
│  │  SharedTexture 반환                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              DirectX 12 Composite                     │   │
│  │                                                       │   │
│  │  [CompositeCS] → 최종 출력                            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: DirectX Refine + DispatchIndirect

### 1.1 IndirectArgs 버퍼 구조

```cpp
struct IndirectArgs {
    UINT ThreadGroupCountX;  // BoundingBox 기반
    UINT ThreadGroupCountY;  // BoundingBox 기반
    UINT ThreadGroupCountZ;  // 항상 1
};
```

### 1.2 Refine Compute Shader

```hlsl
// Refine.hlsl
RWStructuredBuffer<uint> g_atomicBounds : register(u1);  // [minX, maxX, minY, maxY]
RWStructuredBuffer<IndirectArgs> g_indirectArgs : register(u2);

[numthreads(16, 16, 1)]
void RefineCS(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_width || id.y >= g_height) return;

    float4 pixel = g_inputTex[id.xy];
    float L = dot(pixel.rgb, float3(0.299, 0.587, 0.114));

    if (L >= g_threshold) {
        int expandedMinX = max(0, (int)id.x - g_blurRadius);
        int expandedMaxX = min(g_width - 1, (int)id.x + g_blurRadius);
        int expandedMinY = max(0, (int)id.y - g_blurRadius);
        int expandedMaxY = min(g_height - 1, (int)id.y + g_blurRadius);

        InterlockedMin(g_atomicBounds[0], expandedMinX);
        InterlockedMax(g_atomicBounds[1], expandedMaxX);
        InterlockedMin(g_atomicBounds[2], expandedMinY);
        InterlockedMax(g_atomicBounds[3], expandedMaxY);
    }
}

// 두 번째 패스: IndirectArgs 계산
[numthreads(1, 1, 1)]
void CalcIndirectArgs(uint3 id : SV_DispatchThreadID)
{
    uint minX = g_atomicBounds[0];
    uint maxX = g_atomicBounds[1];
    uint minY = g_atomicBounds[2];
    uint maxY = g_atomicBounds[3];

    uint width = (maxX >= minX) ? (maxX - minX + 1) : g_width;
    uint height = (maxY >= minY) ? (maxY - minY + 1) : g_height;

    g_indirectArgs[0].ThreadGroupCountX = (width + 15) / 16;
    g_indirectArgs[0].ThreadGroupCountY = (height + 15) / 16;
    g_indirectArgs[0].ThreadGroupCountZ = 1;
}
```

### 1.3 Prefilter 수정

```hlsl
// Prefilter.hlsl 수정
StructuredBuffer<uint> g_bounds : register(t2);  // [minX, maxX, minY, maxY]

[numthreads(16, 16, 1)]
void main(uint3 localID : SV_DispatchThreadID)
{
    // BoundingBox 오프셋 적용
    uint minX = g_bounds[0];
    uint minY = g_bounds[2];

    uint2 globalID = uint2(localID.x + minX, localID.y + minY);

    if (globalID.x >= g_dstWidth || globalID.y >= g_dstHeight)
        return;

    // 기존 Prefilter 로직...
}
```

### 1.4 C++ 파이프라인

```cpp
void JustGlowHybridRenderer::Render(...) {
    // 1. Refine (전체 이미지)
    commandList->Dispatch(fullGridX, fullGridY, 1);

    // 2. CalcIndirectArgs (1 스레드)
    commandList->Dispatch(1, 1, 1);

    // 3. UAV 배리어
    commandList->ResourceBarrier(...);

    // 4. DispatchIndirect (GPU가 Grid 결정!)
    commandList->ExecuteIndirect(
        m_commandSignature,
        1,
        m_indirectArgsBuffer,
        0,
        nullptr, 0);

    // ... Downsample, Upsample, Composite
}
```

---

## Phase 2: CUDA-DirectX Interop

### 2.1 리소스 등록 (초기화 시)

```cpp
#include <cuda_d3d12_interop.h>

bool JustGlowHybridRenderer::InitializeCUDAInterop() {
    // DX12 리소스를 CUDA에 등록
    cudaError_t err = cudaGraphicsD3D12RegisterResource(
        &m_cudaResource,
        m_sharedTexture,
        cudaGraphicsRegisterFlagsNone);

    return err == cudaSuccess;
}
```

### 2.2 렌더링 시 사용

```cpp
void JustGlowHybridRenderer::ExecuteCUDAProcessing() {
    // 1. DX → CUDA 전환
    cudaGraphicsMapResources(1, &m_cudaResource, m_cudaStream);

    cudaArray_t cudaArray;
    cudaGraphicsSubResourceGetMappedArray(&cudaArray, m_cudaResource, 0, 0);

    // 2. CUDA 커널 실행
    LaunchUpsampleKernel(cudaArray, ...);

    // 3. CUDA → DX 전환
    cudaGraphicsUnmapResources(1, &m_cudaResource, m_cudaStream);
}
```

---

## Phase 3: 각 MIP 레벨 IndirectArgs

각 다운샘플/업샘플 레벨마다 별도의 IndirectArgs 필요:

```cpp
struct MipIndirectArgs {
    IndirectArgs prefilter;
    IndirectArgs downsample[MAX_MIP_LEVELS];
    IndirectArgs upsample[MAX_MIP_LEVELS];
};
```

각 레벨 Grid 크기 = 이전 레벨 / 2 (GPU에서 계산)

---

## 파일 구조

```
src/
├── JustGlowHybridRenderer.h      ← 새 하이브리드 렌더러
├── JustGlowHybridRenderer.cpp
├── JustGlowGPURenderer.cpp       ← 기존 DX 렌더러 (유지)
├── JustGlowCUDARenderer.cpp      ← 기존 CUDA 렌더러 (유지)
└── JustGlow.cpp                  ← 렌더러 선택 로직 수정

shaders/
├── Refine.hlsl                   ← 새 Refine 셰이더
├── CalcIndirectArgs.hlsl         ← 새 IndirectArgs 계산
├── Prefilter.hlsl                ← 수정 (bounds 파라미터)
├── Downsample.hlsl               ← 수정
└── ...
```

---

## 구현 순서

### Stage 1: DirectX DispatchIndirect 파이프라인
1. [ ] Refine.hlsl 생성
2. [ ] CalcIndirectArgs.hlsl 생성
3. [ ] IndirectArgs 버퍼 생성 (C++)
4. [ ] ExecuteIndirect Command Signature 생성
5. [ ] 기존 셰이더에 bounds 파라미터 추가
6. [ ] 테스트 (DX only, CUDA 없이)

### Stage 2: CUDA Interop (선택적)
7. [ ] cudaGraphicsD3D12RegisterResource 구현
8. [ ] Map/Unmap 로직 추가
9. [ ] 복잡한 처리를 CUDA로 이동 (필요시)

### Stage 3: 통합
10. [ ] JustGlowHybridRenderer 클래스 생성
11. [ ] GPU 타입 감지 및 렌더러 선택 로직
12. [ ] 성능 비교 테스트

---

## 예상 성능 개선

| 항목 | 현재 (CUDA) | 하이브리드 |
|------|------------|-----------|
| CPU-GPU 동기화 | N+1회 | **1회** (최종만) |
| 동기화 대기 시간 | 0.8-4ms | **~0.1ms** |
| 작은 콘텐츠 (10%) | +5ms 이득 | **+7ms 이득** |

---

## 위험 및 완화

| 위험 | 확률 | 완화 |
|------|------|------|
| CUDA-DX Interop 복잡성 | 높음 | Stage 2는 선택적, Stage 1만으로도 큰 이득 |
| DX12 ExecuteIndirect 복잡성 | 중간 | 문서화 잘 되어 있음 |
| AMD GPU 호환성 | 없음 | CUDA 없이 DX만 사용 |

---

## 다음 단계

Stage 1 먼저 구현: **DirectX DispatchIndirect 파이프라인**

Refine.hlsl 작성부터 시작합니다.
