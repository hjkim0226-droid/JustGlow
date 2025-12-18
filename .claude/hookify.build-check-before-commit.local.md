---
name: build-check-before-commit
enabled: true
event: bash
pattern: git\s+commit
action: warn
---

⚠️ **커밋 전 빌드 체크 리마인더**

커밋하기 전에 확인해주세요:

- [ ] **GitHub Actions 빌드 성공 확인** (`gh run list --limit 1`)
- [ ] **CUDA 커널 수정 시**: PTX 컴파일 가능 여부
- [ ] **파라미터 변경 시**: PiPL 버전과 코드 버전 일치

**빌드 확인 방법:**
```bash
gh run list --limit 1  # 최근 빌드 상태 확인
gh run view <run-id>   # 상세 로그 확인
```

로컬 빌드가 필요하면 Windows 환경에서 실행해주세요.
