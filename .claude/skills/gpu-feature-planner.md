---
name: gpu-feature-planner
description: GPU 플러그인 개발을 위한 단계별 기능 계획 생성. After Effects 플러그인, CUDA/DirectX/Metal 셰이더 개발에 최적화. 기능 계획, 작업 분해, 파이프라인 설계 시 사용. Keywords: plan, planning, phases, gpu, shader, cuda, metal, directx, pipeline, feature.
---

# GPU Feature Planner

## Purpose

GPU 그래픽스 플러그인 개발에 최적화된 단계별 계획 생성:
- 각 Phase는 시각적으로 검증 가능한 결과물 제공
- Visual Quality Gate로 GPU 렌더링 결과 검증
- 빌드 → 설치 → AE 테스트 사이클 반영
- 크로스플랫폼 (DirectX/CUDA/Metal) 고려

## GPU 개발의 특수성

### TDD 대신 VDD (Visual-Driven Development)

| 기존 TDD | GPU VDD |
|----------|---------|
| 유닛 테스트 작성 | 레퍼런스 이미지 준비 |
| 테스트 실패 확인 | 빌드 후 시각적 차이 확인 |
| 코드 구현 | 셰이더/커널 구현 |
| 테스트 통과 | 레퍼런스와 시각적 일치 |
| 리팩토링 | 성능 최적화 |

### Quality Gate 기준

**빌드 검증:**
```bash
cmake --build build --config Release --parallel
```

**셰이더 컴파일 검증:**
```bash
# DirectX CSO
ls build/Release/DirectX_Assets/*.cso

# CUDA PTX
ls build/Release/CUDA_Assets/*.ptx

# Metal (향후)
# xcrun -sdk macosx metal -c *.metal
```

**시각적 검증:**
- Debug View 파라미터로 파이프라인 단계별 확인
- 레퍼런스 이미지와 비교 (수동 또는 ImageMagick)
- 다양한 소스 (텍스트, Adjustment Layer, 투명 배경) 테스트

**성능 검증:**
- 1080p 기준 렌더링 시간 측정
- GPU 메모리 사용량 확인
- 프레임 드랍 없이 재생 확인

## Planning Workflow

### Step 1: 요구사항 분석
1. 관련 파일 읽기 (기존 렌더러 코드, 셰이더)
2. GPU 프레임워크 의존성 파악
3. 기존 파이프라인과 통합점 확인
4. 복잡도 및 리스크 평가

### Step 2: Phase 분해

기능을 3-7개 Phase로 분해, 각 Phase는:
- **시각적으로 검증 가능한** 결과물 제공
- 최대 1-4시간 소요
- 독립적으로 롤백 가능
- 명확한 성공 기준 보유

**Phase 구조:**
```
Phase N: [명확한 결과물]
├── Goal: 이 Phase가 제공하는 기능
├── Files: 수정/생성할 파일 목록
├── Tasks:
│   ├── 준비 (레퍼런스, 테스트 데이터)
│   ├── 구현 (셰이더, 렌더러 코드)
│   └── 검증 (빌드, 시각적 테스트)
├── Quality Gate: 통과 기준
└── Dependencies: 선행 조건
```

### Step 3: 계획 문서 생성

`docs/plans/PLAN_<feature-name>.md` 생성

포함 내용:
- 개요 및 목표
- 아키텍처 결정 사항
- Phase별 상세 계획 (체크박스)
- Quality Gate 체크리스트
- 리스크 평가
- 롤백 전략
- 진행 추적 섹션

### Step 4: 사용자 승인

**중요**: 구현 시작 전 AskUserQuestion으로 명시적 승인 획득

질문 예시:
- "이 Phase 분해가 적절한가요?"
- "제안된 접근 방식에 우려 사항이 있나요?"
- "계획 문서를 생성해도 될까요?"

### Step 5: 문서 생성

1. `docs/plans/` 디렉토리 확인/생성
2. 모든 체크박스 미체크 상태로 계획 문서 생성
3. Quality Gate 설명 포함
4. 다음 단계 안내

## GPU Quality Gate 표준

### 빌드 & 컴파일
- [ ] CMake 빌드 성공 (에러 없음)
- [ ] 모든 셰이더 컴파일 성공
- [ ] 플러그인 설치 성공

### 시각적 검증
- [ ] Debug View로 해당 단계 출력 확인
- [ ] 레퍼런스 이미지와 비교
- [ ] 엣지 케이스 확인 (투명 배경, 큰 Radius 등)

### 성능 검증
- [ ] 기존 대비 성능 저하 없음
- [ ] GPU 메모리 누수 없음
- [ ] 1080p 실시간 재생 가능

### 호환성
- [ ] 기존 기능 정상 동작
- [ ] 파라미터 변경 시 크래시 없음
- [ ] 다양한 컴포지션 설정에서 동작

## Phase 크기 가이드라인

**Small Scope** (2-3 phases, 3-6시간):
- 단일 셰이더 수정
- 파라미터 추가
- 버그 수정
- 예: 새 Composite 모드 추가, Falloff 곡선 변경

**Medium Scope** (4-5 phases, 8-15시간):
- 새 렌더링 패스 추가
- 기존 파이프라인 수정
- 크로스플랫폼 포팅
- 예: CPU Fallback 구현, 새 다운샘플 알고리즘

**Large Scope** (6-7 phases, 15-25시간):
- 새 GPU 프레임워크 지원
- 전체 파이프라인 재설계
- 대규모 최적화
- 예: Metal 지원, 새 이펙트 추가

## 리스크 평가

GPU 개발 특유 리스크:
- **드라이버 호환성**: 특정 GPU/드라이버에서만 발생하는 이슈
- **정밀도 차이**: DirectX vs CUDA vs Metal 간 부동소수점 차이
- **메모리 제한**: GPU 메모리 부족 상황
- **동기화 이슈**: 비동기 GPU 작업 간 race condition
- **AE SDK 제한**: After Effects GPU SDK 제약사항

## 롤백 전략

각 Phase별 롤백 방법 문서화:
- 어떤 파일 변경을 되돌려야 하는지
- 셰이더 파일 복원
- CMakeLists.txt 변경 사항
- Git 기반 롤백 명령어

## JustGlow 특화 검증

### 파이프라인 단계별 Debug View

| Debug View | 확인 내용 |
|------------|-----------|
| 1: Prefilter | 임계값 적용, 13-tap 샘플링 |
| 2: Downsample | MIP 피라미드 생성 |
| 3: Upsample | Tent 필터 업샘플 |
| 4: Composite | 최종 합성 결과 |

### 표준 테스트 케이스

1. **텍스트 레이어**: 선명한 엣지 글로우
2. **Adjustment Layer**: 동일한 결과 (ZeroPad 검증)
3. **투명 배경**: 알파 확장 검증
4. **고대비 소스**: HDR firefly 방지 검증
5. **큰 Radius**: 깊은 MIP 레벨 검증

## 템플릿 참조

- [gpu-plan-template.md](gpu-plan-template.md) - 전체 계획 문서 템플릿
