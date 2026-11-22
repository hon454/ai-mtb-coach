# AGENTS.md

ai-mtb-coach: MTB 강습 영상(코치님 카톡 영상 + 텍스트 피드백)에서 라이더 자세·자전거·뱅크 라인을 인식해 회고하고, 코치님 모범 주행과 비교하는 프로젝트. 공개 저장소다.

## 먼저 읽을 문서

| 문서 | 내용 |
|---|---|
| [LOG.md](LOG.md) | 작업 로그(최신이 위). 시도·결과·판단·다음, 정정된 결론 포함. 작업 전 최근 항목과 "열린 과제"를 확인 |
| [docs/capture-requests.md](docs/capture-requests.md) | 다음 강습 때 확보할 촬영·자료 요청 목록. 분석에 자료가 부족하면 여기에 추가 |
| [docs/research/tools.md](docs/research/tools.md) | 외부 도구 조사와 도입 판단(촬영 환경 기준 우선순위) |

새 조사 문서는 `docs/research/`에 두고 위 표에 추가한다.

## 구조

- `tools/` 분석 스크립트
  - `paths.py` 경로 규칙(영상 위치, 산출물 위치). 새 도구도 이걸 쓴다
  - `compare_runs.py` 구간 비교 시트·동기화 영상
  - `analyze_pose.py` 라이더 추적·크롭·뼈대, 동기화 그리드
  - `analyze_lean.py` 크롭 재인식(자세·관절각)
  - `bike_sam2.py` SAM2 자전거 마스크(`--kp-prompt`: 포즈 키포인트 프롬프트). `--deshadow`는 땅 그림자를 뺀 마스크를 `masks_deshadow/`에 따로 만듦(`masks/`는 원본 그대로)
  - `bike_fit.py` 3D 자전거 실루엣 맞추기(기울기·yaw). 자전거 지오메트리는 segments.json의 `"bike"`. 사용자가 그린 핸들바 선(`sessions/<세션>/bar_hints.json`)은 `--hints`로 켤 때만 씀(기본은 끔, 그린 선을 채점 기준으로 남김). 입력은 기본 `masks/`(원본), `--masks deshadow`로 그림자 뺀 마스크. 자전거 기울기는 참고 값
  - `bar_hint_board.py` 핸들바 선 그리기 보드(`make`)와 그린 선 추출(`extract`)
  - `compare_reference.py` 코치 vs 내 런 정렬 비교
  - `berm_line.py` 뱅크 배경판·벽면 분할·타이어 접지 궤적
  - `insight_board.py` 인사이트 장면 보드(spec JSON)
  - `separation_board.py` 자전거 vs 상체 기울기 보드(`--ref`로 코치 추가). 기본은 GVHMR 손목선 vs 3D 상체 롤을 회전 단계(15·30·45°)별로 비교. `--bike fit`은 예전 bike_fit 방식
  - `leanout_board.py` 린아웃 측정 시도(현재 근사가 부정확, LOG 참고)
  - `body3d_gvhmr.py` GVHMR 중력 기준 3D 몸 자세(상체 롤·피치, 손·발 자전거 평면, 엉덩이 오프셋). infer는 WSL의 `$GVHMR_HOME` 환경에서 돌림
  - `body3d_sam3d.py` SAM 3D Body 3D 몸 각도(상체 롤·피치, 손-발 자전거 기울기). infer 단계는 `$SAM3DBODY_HOME`의 별도 환경. `body3d_board.py`는 비교 보드. `--gravity gvhmr`로 GVHMR 중력 사용
  - `body3d_vs_gvhmr.py` SAM 3D Body vs GVHMR 프레임별 비교, 그린 핸들바 선 채점, Run 5 손-발 차이 진단
- `sessions/<세션>/` 강습별 설정(git 추적): `segments.json`(구간·자전거), `insights/*.json`(보드 spec), `berm_prompts.json`·`berm_board_captions.json`(뱅크 수동 프롬프트·캡션)
- `reference/<세션>/` 원본 영상(git 제외). 파일명 `<세션>_run<N>.mp4` 등
- `outputs/<세션>/` 모든 산출물(git 제외): `<feature>_pose`, `_lean`, `_berm`, `insights/`
- `sessions/reference-coach/` 코치님 모범 영상 세션
- `models/` 모델 가중치(git 제외)

## 실행

- Python 3.12 + uv: `uv run python tools/<script>.py sessions/<세션>/segments.json`
- GPU: RTX 4090. ffmpeg 9 사용(drawtext는 `fontfile` 지정, `%`는 전각 `％`로).

## 작업 규칙

- 진행·실패·정정은 LOG.md에 남긴다.
- 인사이트는 근거 장면을 담은 시각 자료와 함께 제공하고, 만든 이미지를 직접 확인한 뒤 설명을 맞춘다. 측정값인지 시각적 인상인지 표시한다.
- 촬영은 기본이 핸드헬드 1대. 삼각대·다중 카메라는 준비가 필요하므로 도구 선택 시 이를 기준으로 한다(docs/research/tools.md).

## 공개 저장소 규칙

- 영상, 크롭, 마스크, 보드 이미지 등 라이더·코치님 모습이 담긴 파일은 커밋하지 않는다(`reference/`, `outputs/`는 git 제외).
  - 예외: README에 쓰는 이미지는 `docs/media/`에만 둔다. 라이더 본인과 코치님 동의를 받았다(2026-11-23). 새 이미지를 넣을 때는 사용자에게 먼저 확인한다.
- 코치님 메시지 원문은 커밋하지 않는다. 공개 파일에는 ASD-STE100 스타일(짧은 문장, 한 문장에 한 내용, 능동형, 지시는 명령형)로 바꾼 한국어 문장을 쓰고, 원문은 `sessions/<세션>/feedback.local.md`(git 제외)에 둔다.
- 개인 절대경로(사용자 폴더 등)를 커밋하지 않는다. `video_dir`는 저장소 기준 상대경로로 쓴다.
