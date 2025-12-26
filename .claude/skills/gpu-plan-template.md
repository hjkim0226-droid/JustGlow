# Implementation Plan: [Feature Name]

**Status**: 🔄 In Progress
**Started**: YYYY-MM-DD
**Last Updated**: YYYY-MM-DD
**Target Platform**: DirectX / CUDA / Metal / All

---

**⚠️ GPU 개발 지침**: 각 Phase 완료 후:
1. ✅ 완료된 작업 체크박스 체크
2. 🔨 빌드 및 셰이더 컴파일 확인
3. 🎨 Debug View로 시각적 결과 검증
4. ⚡ 성능 저하 없음 확인
5. 📅 "Last Updated" 날짜 업데이트
6. 📝 Notes 섹션에 발견 사항 기록
7. ➡️ 모든 검증 통과 후에만 다음 Phase 진행

⛔ **Quality Gate를 건너뛰거나 실패한 상태로 진행 금지**

---

## 📋 개요

### 기능 설명
[이 기능이 무엇이고 왜 필요한지]

### 성공 기준
- [ ] 기준 1
- [ ] 기준 2
- [ ] 기준 3

### 영향 범위
- **수정 파일**: [영향받는 파일 목록]
- **새 파일**: [생성할 파일 목록]
- **GPU 프레임워크**: [DirectX/CUDA/Metal]

---

## 🏗️ 아키텍처 결정

| 결정 사항 | 근거 | 트레이드오프 |
|-----------|------|--------------|
| [결정 1] | [이 접근법을 선택한 이유] | [포기하는 것] |
| [결정 2] | [이 접근법을 선택한 이유] | [포기하는 것] |

---

## 📦 의존성

### 시작 전 필수 사항
- [ ] 의존성 1: [설명]
- [ ] 의존성 2: [설명]

### 외부 의존성
- CUDA Toolkit: 12.x
- DirectX SDK: Windows SDK 포함
- After Effects SDK: 2024+

---

## 🎨 시각적 검증 전략 (VDD)

### 레퍼런스 이미지 준비
| 테스트 케이스 | 소스 | 예상 결과 |
|--------------|------|-----------|
| 텍스트 글로우 | 흰색 텍스트 on 검정 | 부드러운 확산 |
| 투명 배경 | PNG with alpha | 알파 확장됨 |
| 고대비 HDR | 밝은 하이라이트 | Firefly 없음 |
| Adjustment Layer | 동영상 위 | 텍스트와 동일 결과 |

### Debug View 활용
```
파라미터: Debug View
- 0: Final (기본)
- 1: Prefilter 출력
- 2: Downsample 체인
- 3: Upsample 결과
- 4: Glow Only
```

### 자동화 검증 (선택)
```bash
# ImageMagick 비교 (RMSE < 0.01 = 통과)
magick compare -metric RMSE output.png reference.png diff.png
```

---

## 🚀 구현 Phases

### Phase 1: [Foundation Phase Name]
**Goal**: [이 Phase가 제공하는 구체적인 기능]
**예상 시간**: X 시간
**Status**: ⏳ Pending | 🔄 In Progress | ✅ Complete

#### 준비 작업
- [ ] 레퍼런스 이미지 준비
- [ ] 테스트 컴포지션 설정

#### 구현 작업

**셰이더/커널:**
- [ ] **Task 1.1**: [셰이더/커널 파일]
  - File: `shaders/[name].hlsl` 또는 `src/[name].cu`
  - 내용: [구현할 기능]

**렌더러 코드:**
- [ ] **Task 1.2**: [렌더러 수정]
  - File: `src/JustGlow[GPU]Renderer.cpp`
  - 내용: [수정 사항]

**빌드 설정:**
- [ ] **Task 1.3**: CMakeLists.txt 업데이트 (필요시)

#### Quality Gate ✋

**⚠️ 모든 항목 통과 전 Phase 2 진행 금지**

**빌드 검증:**
```bash
# 전체 빌드
cmake --build build --config Release --parallel

# 빌드 결과 확인
ls -la build/Release/*.aex
ls -la build/Release/DirectX_Assets/*.cso
ls -la build/Release/CUDA_Assets/*.ptx
```
- [ ] CMake 빌드 성공 (에러/경고 없음)
- [ ] 셰이더 컴파일 성공 (CSO 파일 생성)
- [ ] CUDA PTX 컴파일 성공 (해당시)

**설치 & 로드:**
```bash
cmake --install build
```
- [ ] 플러그인 AE에 로드됨
- [ ] Effects → Stylize → JustGlow 표시
- [ ] GPU 렌더러 활성화 (로그 확인)

**시각적 검증:**
- [ ] Debug View로 해당 단계 출력 확인
- [ ] 레퍼런스 이미지와 육안 비교
- [ ] 텍스트 레이어 테스트 통과
- [ ] Adjustment Layer 테스트 통과

**성능 검증:**
- [ ] 1080p 렌더링 시간 기존 대비 ±10% 이내
- [ ] 실시간 재생 가능 (RAM Preview)

**호환성:**
- [ ] 기존 프로젝트 파일 정상 로드
- [ ] 모든 파라미터 정상 동작

---

### Phase 2: [Core Implementation Phase]
**Goal**: [구체적인 결과물]
**예상 시간**: X 시간
**Status**: ⏳ Pending | 🔄 In Progress | ✅ Complete

#### 준비 작업
- [ ] Phase 1 완료 확인
- [ ] 추가 레퍼런스 이미지 준비

#### 구현 작업

**셰이더/커널:**
- [ ] **Task 2.1**: [셰이더/커널]
  - File: [파일 경로]
  - 내용: [구현 내용]

**렌더러 코드:**
- [ ] **Task 2.2**: [렌더러 수정]
  - File: [파일 경로]
  - 내용: [수정 내용]

**통합:**
- [ ] **Task 2.3**: 파이프라인 통합
  - 기존 단계와 연결
  - 버퍼 관리

#### Quality Gate ✋

**빌드 검증:**
- [ ] CMake 빌드 성공
- [ ] 셰이더 컴파일 성공
- [ ] 설치 성공

**시각적 검증:**
- [ ] Debug View로 새 단계 확인
- [ ] 전체 파이프라인 결과 정상
- [ ] 엣지 케이스 테스트 통과

**성능 검증:**
- [ ] 성능 저하 없음
- [ ] GPU 메모리 사용량 적정

---

### Phase 3: [Polish & Optimization Phase]
**Goal**: [최종 결과물]
**예상 시간**: X 시간
**Status**: ⏳ Pending | 🔄 In Progress | ✅ Complete

#### 구현 작업

**최적화:**
- [ ] **Task 3.1**: 성능 최적화
  - 불필요한 메모리 할당 제거
  - 셰이더 최적화

**품질 개선:**
- [ ] **Task 3.2**: 시각적 품질 개선
  - 엣지 케이스 처리
  - 수치 정밀도 개선

**문서화:**
- [ ] **Task 3.3**: 코드 문서화
  - 주요 알고리즘 주석
  - 파라미터 설명

#### Quality Gate ✋

**최종 검증:**
- [ ] 모든 테스트 케이스 통과
- [ ] 성능 목표 달성
- [ ] 메모리 누수 없음
- [ ] 문서화 완료

---

## ⚠️ 리스크 평가

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|-----------|
| GPU 드라이버 호환성 | 낮음 | 높음 | 다양한 GPU에서 테스트 |
| DirectX/CUDA 결과 차이 | 중간 | 중간 | 동일 알고리즘 사용, 비교 테스트 |
| 메모리 부족 | 낮음 | 높음 | 4K 이상에서 테스트 |
| AE SDK 제한 | 중간 | 높음 | SDK 문서 사전 확인 |

---

## 🔄 롤백 전략

### Phase 1 실패 시
```bash
git checkout HEAD -- src/[modified_files]
git checkout HEAD -- shaders/[modified_files]
cmake --build build --config Release --parallel
cmake --install build
```

### Phase 2 실패 시
- Phase 1 완료 상태로 복원
- 새로 추가된 파일 삭제
- CMakeLists.txt 롤백

### Phase 3 실패 시
- Phase 2 완료 상태로 복원
- 최적화 변경 사항만 롤백

---

## 📊 진행 추적

### 완료 상태
- **Phase 1**: ⏳ 0%
- **Phase 2**: ⏳ 0%
- **Phase 3**: ⏳ 0%

**전체 진행률**: 0%

### 시간 추적
| Phase | 예상 | 실제 | 차이 |
|-------|------|------|------|
| Phase 1 | X시간 | - | - |
| Phase 2 | X시간 | - | - |
| Phase 3 | X시간 | - | - |
| **합계** | X시간 | - | - |

---

## 📝 Notes & Learnings

### 구현 노트
- [구현 중 발견한 인사이트]
- [계획과 다르게 진행된 결정]
- [디버깅 과정에서 발견한 사항]

### 발생한 이슈
- **이슈 1**: [설명] → [해결 방법]
- **이슈 2**: [설명] → [해결 방법]

### 향후 개선 사항
- [다음에 다르게 할 것]
- [특히 잘 된 것]

---

## 📚 참고 자료

### 문서
- `ARCHITECTURE.md` - 전체 아키텍처
- `CLAUDE.md` - 개발 가이드
- `docs/CUDA_IMPLEMENTATION.md` - CUDA 상세

### 관련 코드
- `src/JustGlowGPURenderer.cpp` - DirectX 렌더러
- `src/JustGlowCUDARenderer.cpp` - CUDA 렌더러
- `shaders/*.hlsl` - DirectX 셰이더

---

## ✅ 최종 체크리스트

**완료 전 확인:**
- [ ] 모든 Phase Quality Gate 통과
- [ ] 전체 통합 테스트 완료
- [ ] 모든 테스트 케이스 통과
- [ ] 성능 목표 달성
- [ ] 코드 문서화 완료
- [ ] CLAUDE.md 업데이트 (필요시)
- [ ] ARCHITECTURE.md 업데이트 (필요시)

---

**Plan Status**: 🔄 In Progress
**Next Action**: [다음 할 일]
**Blocked By**: [차단 요소] 또는 없음
