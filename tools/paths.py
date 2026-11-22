"""저장소 경로 규칙 (모든 도구 공통).

- `sessions/<세션>/segments.json`  : 강습별 구간·자전거 설정 (git 추적)
- `reference/<세션>/`              : 원본 영상 (git 제외). segments.json의 video_dir가 가리킴
- `outputs/<세션>/`                : 모든 산출물 (git 제외). 도구를 다시 돌리면 만들어짐

video_dir는 데이터 루트 기준 상대경로(권장) 또는 절대경로, `~` 허용.

데이터 루트(reference/, outputs/, models/)는 기본이 저장소 루트다. git worktree처럼 영상·산출물이 없는
체크아웃에서는 환경변수 AIMTB_DATA_ROOT로 메인 체크아웃을 가리킨다(예: AIMTB_DATA_ROOT=G:/ai-mtb-coach).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent                                   # 코드 루트
DATA_ROOT = Path(os.environ.get("AIMTB_DATA_ROOT") or ROOT).expanduser()      # reference/ outputs/ models/
MODELS = DATA_ROOT / "models"


def load_cfg(seg_path):
    return json.loads(Path(seg_path).read_text(encoding="utf-8"))


def video_dir(cfg):
    p = Path(cfg["video_dir"]).expanduser()
    return p if p.is_absolute() else DATA_ROOT / p


def session_name(seg_path):
    return Path(seg_path).resolve().parent.name


def out_base(seg_path):
    """outputs/<세션>/ — 이 세션의 모든 산출물이 들어가는 폴더."""
    return DATA_ROOT / "outputs" / session_name(seg_path)


def feat_dir(seg_path, cfg, suffix):
    """outputs/<세션>/<feature>_<suffix>/ (예: left_bank_lean)."""
    return out_base(seg_path) / f"{cfg.get('feature', 'feature')}_{suffix}"
