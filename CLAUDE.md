# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JustGlow is a GPU-accelerated glow effect plugin for Adobe After Effects.

**지원 GPU 프레임워크:**
- **Windows:** DirectX 12 (기본) + CUDA (선택)
- **macOS:** Metal (계획됨)

**Core Algorithm:** Dual Kawase blur with:
- 13-tap prefilter with ZeroPad (HDR firefly prevention)
- 9-tap 2D Gaussian downsample (Level 0-4) + 5-tap Kawase (Level 5+), all ZeroPad
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
2. Downsample chain → creates MIP pyramid (Level 0-4: 2D Gaussian, 5+: Kawase, all ZeroPad)
3. Upsample chain → reconstructs with progressive blur blending
4. Composite → blends glow with original (Add/Screen/Overlay modes)

**Key Data Structures:**
- `GlowParams` (b0) - main constant buffer, 16-byte aligned for HLSL
- `BlurPassParams` (b1) - per-pass blur parameters
- `RenderParams` - CPU→GPU 파라미터 전달 구조체

## CUDA 렌더러

**파일:**
- `src/JustGlowCUDARenderer.h/cpp` - CUDA Driver API 기반 렌더러
- `src/JustGlowKernels.cu` - CUDA 커널 구현 (1060줄)

**커널 목록:**
- `PrefilterKernel` - 13-tap + Soft Threshold + ZeroPad
- `Gaussian2DDownsampleKernel` - 9-tap 2D Gaussian + ZeroPad (Level 0-4)
- `DownsampleKernel` - Dual Kawase 5-tap + ZeroPad (Level 5+)
- `UpsampleKernel` - 9-tap Tent + Falloff
- `DebugOutputKernel` - 디버그 뷰 및 최종 합성

**버퍼 구조:**
- `m_mipChain[]` - 다운샘플 결과 저장
- `m_upsampleChain[]` - 업샘플 결과 저장 (별도 버퍼로 race condition 방지)

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
| Quality | Quality | MIP 깊이 (Low=4, Med=6, High=8, Ultra=12) |
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
2. **2D Gaussian Downsample:** Single-pass 9-tap 2D Gaussian replaces separable H+V for Level 0-4. No temp buffer needed, no H→V sync required.
3. **X/+ Rotation:** Alternates diagonal (X) and cross (+) sampling patterns during Kawase downsample (Level 5+) to break boxy artifacts → rounder glow
4. **Dynamic MIP Levels:** Ultra quality goes to 12 levels (until 16px), providing Deep Glow-like "atmosphere" feel
5. **Falloff Blending:** `levelWeight = pow(falloff, level)` during upsample for physical light decay

## 알려진 이슈

**Critical:**
1. **Pitch 모호성** - `JustGlowKernels.cu:147` - pitch가 바이트/픽셀 단위 불명확
2. **CPU Fallback 미구현** - GPU 미지원 시 단순 복사만 수행
3. **커널 간 동기화** - 스테이지 간 명시적 동기화 없음 (현재는 스트림 직렬화에 의존)

**Medium:**
- 에러 발생 시 `out_data->return_msg` 미사용
- 임시 버퍼 과다 할당

상세 내용: `docs/CODE_REVIEW_REPORT.md`

## 관련 문서

- `docs/CODE_REVIEW_REPORT.md` - 전체 코드 검토 보고서
- `docs/CUDA_IMPLEMENTATION.md` - CUDA 구현 상세
- `docs/AE_GPU_SDK_REFERENCE.md` - AE GPU SDK 참조
- `docs/AE_GPU_CUDA_TROUBLESHOOTING.md` - CUDA 트러블슈팅
- `ARCHITECTURE.md` - 전체 아키텍처 문서
