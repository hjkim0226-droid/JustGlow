# DX12-CUDA Interop 하이브리드 아키텍처 설계

## 개요

JustGlow의 새로운 GPU 렌더링 아키텍처입니다.
CUDA의 유연한 메모리 접근과 DX12의 AE 네이티브 통합을 결합합니다.

**핵심 원칙:**
- **CUDA 중심**: 블러 연산 (Prefilter, Downsample, Upsample)
- **DX12**: 입출력 및 Composite (AE 버퍼 호환)
- **Interop**: 메모리 복사 없이 GPU 버퍼 공유

## 아키텍처 다이어그램

```
+===========================================================================+
|                    JustGlow Interop Pipeline                              |
+===========================================================================+
|                                                                           |
|  [AE Input Buffer]                                                        |
|        | (GPU pointer, 내부 정보 없음)                                    |
|        v                                                                  |
|  +-----------------+                                                      |
|  | DX12: Copy      |  AE 버퍼 -> Shared Buffer 복사                      |
|  | (필수: Interop  |  (D3D12_HEAP_FLAG_SHARED 플래그 필요)                |
|  |  호환 버퍼로)   |                                                      |
|  +--------+--------+                                                      |
|           |                                                               |
|           v                                                               |
|  +------------------+                                                     |
|  | Shared Buffer    |  <-- DX12와 CUDA 양쪽에서 접근 가능                 |
|  | (D3D12 Resource) |                                                     |
|  +--------+---------+                                                     |
|           |                                                               |
|           | cudaImportExternalMemory                                      |
|           v                                                               |
|  +------------------+                                                     |
|  | CUDA: Prefilter  |  13-tap Gaussian + Soft Threshold                  |
|  +--------+---------+                                                     |
|           |                                                               |
|           v                                                               |
|  +------------------+                                                     |
|  | CUDA: Downsample |  MIP Chain 생성 (9-tap 2D Gaussian)                |
|  | (N levels)       |                                                     |
|  +--------+---------+                                                     |
|           |                                                               |
|           v                                                               |
|  +------------------+                                                     |
|  | CUDA: Upsample   |  MIP Chain 역순회 + Progressive Blend              |
|  +--------+---------+                                                     |
|           |                                                               |
|           | Fence Signal (CUDA -> DX12)                                   |
|           v                                                               |
|  +------------------+                                                     |
|  | DX12: Composite  |  Screen/Add/Overlay + Color + Exposure             |
|  +--------+---------+                                                     |
|           |                                                               |
|           v                                                               |
|  [AE Output Buffer]                                                       |
|                                                                           |
+===========================================================================+
```

## 역할 분담

| 단계 | 담당 | 이유 |
|------|------|------|
| Input Copy | DX12 | AE 버퍼가 DX12 호환, Shared로 복사 필요 |
| Prefilter | CUDA | 복잡한 조건부 샘플링, 메모리 접근 유연성 |
| Downsample | CUDA | MIP Chain 전체를 자유롭게 접근 |
| Upsample | CUDA | 여러 레벨 동시 참조, atomics 사용 가능 |
| Composite | DX12 | AE 출력 버퍼가 DX12, 직접 쓰기 가능 |

## Shared Buffer 설계

### 버퍼 생성 (DX12)

```cpp
D3D12_HEAP_PROPERTIES heapProps = {};
heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

D3D12_RESOURCE_DESC resourceDesc = {};
resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
resourceDesc.Width = width * height * 4 * sizeof(float);  // RGBA32F
resourceDesc.Height = 1;
resourceDesc.DepthOrArraySize = 1;
resourceDesc.MipLevels = 1;
resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
resourceDesc.SampleDesc.Count = 1;
resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_SHARED;  // <-- 핵심!

device->CreateCommittedResource(
    &heapProps,
    heapFlags,
    &resourceDesc,
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    IID_PPV_ARGS(&m_sharedBuffer)
);
```

### Handle 생성 및 공유

```cpp
// DX12 -> HANDLE 생성
HANDLE sharedHandle = nullptr;
device->CreateSharedHandle(
    m_sharedBuffer.Get(),
    nullptr,
    GENERIC_ALL,
    L"JustGlowSharedBuffer",
    &sharedHandle
);

// CUDA에서 Import
cudaExternalMemoryHandleDesc memDesc = {};
memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
memDesc.handle.win32.handle = sharedHandle;
memDesc.size = bufferSize;
memDesc.flags = cudaExternalMemoryDedicated;

cudaExternalMemory_t extMem;
cudaImportExternalMemory(&extMem, &memDesc);

// CUDA 포인터 획득
cudaExternalMemoryBufferDesc bufDesc = {};
bufDesc.offset = 0;
bufDesc.size = bufferSize;

void* cudaPtr;
cudaExternalMemoryGetMappedBuffer(&cudaPtr, extMem, &bufDesc);
```

## Fence 동기화

### Fence 생성

```cpp
// DX12 Fence 생성
ID3D12Fence* fence;
device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence));

// Shared Handle
HANDLE fenceHandle;
device->CreateSharedHandle(fence, nullptr, GENERIC_ALL, L"JustGlowFence", &fenceHandle);

// CUDA에서 Import
cudaExternalSemaphoreHandleDesc semDesc = {};
semDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
semDesc.handle.win32.handle = fenceHandle;

cudaExternalSemaphore_t extSem;
cudaImportExternalSemaphore(&extSem, &semDesc);
```

### 동기화 흐름

```
시간 -->

DX12:  [Copy Input]----Signal(1)--------------------Wait(2)----[Composite]
                            |                          ^
                            v                          |
CUDA:  --------Wait(1)----[Blur Pipeline]----Signal(2)----------
```

```cpp
// DX12: Copy 완료 후 Signal
commandQueue->Signal(fence, 1);

// CUDA: DX12 완료 대기
cudaExternalSemaphoreWaitParams waitParams = {};
waitParams.params.fence.value = 1;
cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, stream);

// CUDA: 블러 실행
LaunchPrefilter(...);
LaunchDownsample(...);
LaunchUpsample(...);

// CUDA: 완료 Signal
cudaExternalSemaphoreSignalParams signalParams = {};
signalParams.params.fence.value = 2;
cudaSignalExternalSemaphoresAsync(&extSem, &signalParams, 1, stream);

// DX12: CUDA 완료 대기 후 Composite
commandQueue->Wait(fence, 2);
ExecuteComposite(...);
```

## 버퍼 레이아웃

### MIP Chain (CUDA 내부)

```
+-----------------------------------------------+
| Level 0: 1920x1080 (Prefilter Output)        |
+-----------------------------------------------+
| Level 1: 960x540                              |
+-----------------------------------------------+
| Level 2: 480x270                              |
+-----------------------------------------------+
| Level 3: 240x135                              |
+-----------------------------------------------+
| ...                                           |
+-----------------------------------------------+
| Level N: 16x16 (최소)                         |
+-----------------------------------------------+

* MIP Chain은 CUDA 전용 (cuMemAlloc)
* Interop 불필요 (CUDA 내부에서만 사용)
```

### Shared Buffers (Interop)

```
+-----------------------------------------------+
| Input Buffer (Shared)                         |
| - DX12에서 생성                               |
| - CUDA에서 Import                             |
| - AE Input -> 여기로 복사                     |
+-----------------------------------------------+

+-----------------------------------------------+
| Output Buffer (Shared)                        |
| - DX12에서 생성                               |
| - CUDA에서 Import                             |
| - CUDA Upsample 결과 -> DX12 Composite 입력   |
+-----------------------------------------------+
```

## 성능 예상

### 현재 (순수 DX12)

```
AE Input -> DX12 Copy -> DX12 Blur -> DX12 Composite -> AE Output
           [~0.5ms]     [~3ms]       [~0.5ms]
           Total: ~4ms
```

### 새 아키텍처 (Interop)

```
AE Input -> DX12 Copy -> Fence -> CUDA Blur -> Fence -> DX12 Composite -> AE Output
           [~0.5ms]    [~0.01ms]  [~2ms]     [~0.01ms]   [~0.5ms]
           Total: ~3ms (25% 개선)
```

### 성능 개선 요인

1. **CUDA 블러 효율**: 메모리 접근 패턴 최적화, Shared Memory 활용
2. **Fence 오버헤드 최소**: ~0.01ms per signal/wait
3. **메모리 복사 없음**: Interop으로 버퍼 직접 공유

## 구현 단계

### Phase 1: Interop 인프라 (1주)

- [ ] Shared Buffer 생성 로직
- [ ] Handle 공유 및 CUDA Import
- [ ] Fence 생성 및 동기화 기본 흐름
- [ ] 테스트: 간단한 Copy -> CUDA -> Copy 파이프라인

### Phase 2: CUDA 블러 통합 (1주)

- [ ] 기존 CUDA 커널 Interop 버퍼로 연결
- [ ] MIP Chain 버퍼 (CUDA 내부) 유지
- [ ] Input/Output만 Shared Buffer 사용
- [ ] 테스트: 전체 블러 파이프라인

### Phase 3: DX12 Composite (3일)

- [ ] Composite 셰이더 유지 (현재 DX12 그대로)
- [ ] Upsample 결과 (Shared) -> Composite 입력
- [ ] 테스트: 전체 파이프라인 End-to-End

### Phase 4: 최적화 (1주)

- [ ] BoundingBox 최적화 (Refine 커널)
- [ ] CUDA Shared Memory 활용
- [ ] 프로파일링 및 병목 제거

## 파일 구조 변경

```
src/
├── JustGlow.cpp              # 기존 유지
├── JustGlow.h                # 기존 유지
├── JustGlowParams.h          # 기존 유지
├── JustGlowGPURenderer.h     # 수정: Interop 멤버 추가
├── JustGlowGPURenderer.cpp   # 수정: Copy + Composite만
├── JustGlowCUDARenderer.h    # 수정: Interop Import
├── JustGlowCUDARenderer.cpp  # 수정: Interop 버퍼 사용
├── JustGlowKernels.cu        # 기존 유지
└── JustGlowInterop.h         # 신규: Interop 헬퍼
```

## 위험 및 완화

| 위험 | 확률 | 영향 | 완화 |
|------|------|------|------|
| Interop 호환성 | 중간 | 높음 | NVIDIA 드라이버 버전 체크 |
| Fence 데드락 | 낮음 | 높음 | 타임아웃 추가, 디버그 로그 |
| 버퍼 크기 불일치 | 낮음 | 중간 | Alignment 검증 로직 |
| 성능 저하 (예상 외) | 낮음 | 중간 | Fallback to 순수 DX12/CUDA |

## 다음 단계

1. `feature/dx12-cuda-interop` 브랜치 생성
2. Phase 1 구현 시작: Shared Buffer 인프라
3. 간단한 테스트 케이스로 Interop 검증
