# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JustGlow is a GPU-accelerated glow effect plugin for Adobe After Effects.

**지원 GPU 프레임워크:**
- **Windows:** DirectX 12 (기본) + CUDA (선택)
- **macOS:** Metal (계획됨)

**Core Algorithm:** Multi-level Gaussian blur with:
- 13-tap prefilter with ZeroPad (HDR firefly prevention)
- 9-tap 2D Gaussian downsample (all levels, ZeroPad)
- 9-tap tent upsample with falloff-based blending (physical light decay)
- Dynamic MIP levels (up to 12, until min dimension < 16px)

## Build Commands

```bash
# Configure (Windows)
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release --parallel

# Install to After Effects plugin directory
cmake --install build
```

**빌드 산출물:**
- `JustGlow.aex` - 플러그인 DLL
- `DirectX_Assets/*.cso` - 컴파일된 HLSL 셰이더
- `CUDA_Assets/JustGlowKernels.ptx` - 컴파일된 CUDA 커널 (CUDA Toolkit 설치 시)

## Architecture

**Plugin Entry Flow:**
1. `PluginDataEntryFunction` → registers plugin
2. `EffectMain` → dispatches AE commands
3. `GPUDeviceSetup` → creates DirectX 12 or CUDA renderer
4. `SmartRender` → routes to GPU renderer

**GPU Rendering Pipeline:**
1. Prefilter → soft threshold + ZeroPad sampling + sRGB→Linear
2. Downsample chain → creates MIP pyramid (9-tap 2D Gaussian, all levels, ZeroPad)
3. Upsample chain → reconstructs with progressive blur blending
4. Composite → blends glow with original (Add/Screen/Overlay modes)

**Key Data Structures:**
- `GlowParams` (b0) - main constant buffer, 16-byte aligned for HLSL
- `BlurPassParams` (b1) - per-pass blur parameters
- `RenderParams` - CPU→GPU 파라미터 전달 구조체

## CUDA 렌더러

**파일:**
- `src/JustGlowCUDARenderer.h/cpp` - CUDA Driver API 기반 렌더러 (~1600줄)
- `src/JustGlowKernels.cu` - CUDA 커널 구현 (~1900줄)

**커널 목록:**
- `PrefilterKernel` - 13-tap + Soft Threshold + ZeroPad
- `Gaussian2DDownsampleKernel` - 9-tap 2D Gaussian + ZeroPad (all levels)
- `UpsampleKernel` - 9-tap Tent + Falloff
- `DebugOutputKernel` - 디버그 뷰 및 최종 합성
- `DesaturationKernel` - Glow 채도 조절
- `RefineKernel` - BoundingBox 계산 (atomicMin/Max)
- `PreblurGaussianH/VKernel` - 병렬 Pre-blur (Separable Gaussian)

**버퍼 구조:**
- `m_mipChain[]` - 다운샘플 결과 저장
- `m_upsampleChain[]` - 업샘플 결과 저장 (별도 버퍼로 race condition 방지)
- `m_preblurResults[]` - Pre-blur 결과 (레벨별 병렬 처리용)
- `m_preblurTemp[]` - Pre-blur H-pass 임시 버퍼

**병렬 스트림:**
- `m_preblurStreams[6]` - 6개 CUDA 스트림으로 레벨별 Pre-blur 병렬 실행
- 각 레벨 독립적 처리: `streamIdx = (level - 1) % 6`

## Critical Implementation Details

**PiPL Resource (resources/Win/JustGlow.rc):**
- Binary PiPL is embedded directly in RC file (no PiPLTool)
- Must use big-endian byte ordering for all values
- Each property requires vendorID ('ADBE'), propertyKey, propertyID, length, data
- Version must be 0, not 1
- 64-bit Windows uses '8664' code property key (not 'wx86' which is 32-bit)

**GPU Flags in PiPL (OutFlags2):**
- `0x2A001400` enables: GPU_RENDER_F32, SUPPORTS_DIRECTX_RENDERING, SMART_RENDER, THREADED_RENDERING

**Shader/Kernel Loading:**
- DirectX: `DirectX_Assets/*.cso` (DLL 경로 기준)
- CUDA: `CUDA_Assets/JustGlowKernels.ptx` (DLL 경로 기준)

## Debugging

**로그 파일 위치:**
- DirectX: `%TEMP%\JustGlow_debug.log`
- CUDA: `%TEMP%\JustGlow_CUDA_debug.log`

**Key checkpoints:**
- `SmartRender: isGPU=1` confirms GPU path is active
- `GPUDeviceSetup` should appear before first render
- CUDA: `PTX module loaded successfully`

## Testing

1. Build and copy `.aex` + `DirectX_Assets/` + `CUDA_Assets/` to AE plugins folder
2. Apply effect: Effects → Stylize → JustGlow
3. Check debug log for GPU initialization
4. Debug View 파라미터로 파이프라인 단계별 확인

## Parameters

**19개 UI 파라미터** (`JustGlow.h` ParamID enum):

| 카테고리 | 파라미터 | 설명 |
|----------|----------|------|
| Core | Intensity | Level 1 시작 가중치 (0-100%) |
| Core | Exposure | 밝기 배수 (0-50x) |
| Core | Radius | 활성 MIP 레벨 제한 (0-200) |
| Core | Spread | 블러 오프셋 (0-100%) |
| Core | Falloff | 레벨당 감쇠율 (0-100%) |
| Threshold | Threshold | 밝기 임계값 (0-100%) |
| Threshold | Soft Knee | Soft knee 폭 (0-100%) |
| Quality | Quality | MIP 깊이 슬라이더 (6-12, 기본값 8) |
| Quality | Falloff Type | 감쇠 곡선 (Exp/InvSq/Linear) |
| Color | Glow Color | 글로우 색상 |
| Color | Color Temp | 색온도 (-100~+100) |
| Color | Preserve Color | 원본 색상 보존율 |
| Advanced | Anamorphic | 방향성 스트레치 |
| Advanced | Anamorphic Angle | 스트레치 각도 |
| Advanced | Composite Mode | 합성 모드 (Add/Screen/Overlay) |
| Advanced | HDR Mode | Karis Average 사용 여부 |
| Debug | Debug View | 파이프라인 단계 시각화 |
| Debug | Source Opacity | 원본 불투명도 |
| Debug | Glow Opacity | 글로우 불투명도 (0-200%) |

## Key Techniques

1. **ZeroPad Sampling:** All downsampling uses ZeroPad (out-of-bounds = 0) for consistent glow across different buffer sizes. Prevents edge energy concentration that caused brightness differences between Adjustment and Text layers.
2. **9-tap 2D Gaussian Downsample:** Single-pass 9-tap 2D Gaussian for all levels. No temp buffer needed, no H→V sync required. Temporally stable (no flickering on subpixel movement).
3. **Dynamic MIP Levels:** Ultra quality goes to 12 levels (until 16px), providing Deep Glow-like "atmosphere" feel
4. **Falloff Blending:** `levelWeight = pow(falloff, level)` during upsample for physical light decay
5. **DispatchIndirect Optimization (DirectX):** GPU-driven BoundingBox optimization eliminates CPU-GPU synchronization

## DispatchIndirect 최적화 (DirectX 12)

**목적:** 작은 콘텐츠 영역만 처리하여 GPU 연산량 대폭 감소 (예: 1920×1080 → 320×320 = 36배 성능 향상)

**파이프라인:**
```
[RefineCS] → atomicMin/Max로 BoundingBox 계산
    ↓
[CalcIndirectArgsCS] → ThreadGroupCount 계산 → IndirectArgsBuffer에 저장
    ↓
[ExecuteIndirect] → GPU가 버퍼를 읽어 dispatch 크기 결정 (CPU 개입 없음!)
    ↓
[PrefilterWithBounds] → BoundsOutput에서 오프셋 읽어 좌표 변환
```

**관련 파일:**
- `shaders/Refine.hlsl` - RefineCS, CalcIndirectArgsCS, ResetBoundsCS
- `shaders/Prefilter.hlsl` - mainWithBounds 엔트리포인트 추가
- `src/JustGlowGPURenderer.cpp` - ExecuteRefine, ExecutePrefilterIndirect

**버퍼 구조:**
- `m_atomicBoundsBuffer` - [minX, maxX, minY, maxY] (atomic 연산용)
- `m_indirectArgsBuffer` - [ThreadGroupCountX, Y, Z] × MIP 레벨
- `m_boundsOutputBuffer` - [minX, maxX, minY, maxY] × MIP 레벨 (다음 단계에서 읽음)

**성능 개선:**
- CPU-GPU 동기화: N+1회 → **1회** (최종만)
- 동기화 대기 시간: 0.8-4ms → **~0.1ms**

상세 내용: `docs/plans/PLAN_HYBRID_DX_CUDA.md`

## 알려진 이슈

**해결됨 (v1.5.4):**
1. ~~**CUDA ExecuteRefine 버퍼 오버런**~~ → `params.inputWidth/inputHeight` 사용 (입력 버퍼 크기)
   - 원인: `params.width/height` (출력 크기)로 입력 버퍼 접근 → CUDA_ERROR_ILLEGAL_ADDRESS
   - 수정: `JustGlowCUDARenderer.cpp:685`
2. ~~**HDR Mode 미사용**~~ → UI에서 숨김 처리 (`PF_PUI_INVISIBLE`)
   - Karis Average v1.4.0에서 제거됨 (아티팩트 발생)

**해결됨 (v1.5.3):**
1. ~~**Pitch 모호성**~~ → `pitchBytes`로 명시적 명명 (`JustGlowCUDARenderer.h`)
2. ~~**CPU Fallback 미구현**~~ → GPU 미지원 시 에러 메시지 표시 (`JustGlow.cpp:1259`)
3. ~~**커널 간 동기화**~~ → `CUevent` 기반 명시적 동기화 (`JustGlowCUDARenderer.cpp:484-498`)

**Medium:**
- 임시 버퍼 과다 할당 (성능 영향 미미)

## 최적화된 디폴트 값 (v1.5.4)

| 파라미터 | 값 | 이유 |
|----------|-----|------|
| Exposure | 1.5 | 첫 적용시 글로우가 바로 보임 |
| Threshold | 50% | 밝은 영역만 잡아서 깔끔한 글로우 |
| SoftKnee | 50% | 균형잡힌 트랜지션 |
| Quality | 9 | 약간 더 좋은 품질 |
| Desaturation | 30% | 더 컬러풀한 글로우 |
| HDRMode | false | 제거된 기능 |
| Dither | 30% | 미세한 밴딩 방지 |

## Git 브랜치 구조

```
main            ← 안정 버전 (기존 Dev-New-Architecture)
develop         ← 개발용
main-legacy     ← 기존 프로토타입 백업
develop-legacy  ← 기존 develop 백업
```

상세 내용: `docs/CODE_REVIEW_REPORT.md`

## 관련 문서

- `docs/CODE_REVIEW_REPORT.md` - 전체 코드 검토 보고서
- `docs/CUDA_IMPLEMENTATION.md` - CUDA 구현 상세
- `docs/AE_GPU_SDK_REFERENCE.md` - AE GPU SDK 참조
- `docs/AE_GPU_CUDA_TROUBLESHOOTING.md` - CUDA 트러블슈팅
- `ARCHITECTURE.md` - 전체 아키텍처 문서
- `archive/` - DX12-CUDA Interop 레퍼런스 코드 (향후 참조용)
