from pathlib import Path

from utils import files


def test_create_session_dir_creates_unique_directories(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(files, "TMP_DIR", str(tmp_path))

    first_id, first_dir = files.create_session_dir()
    second_id, second_dir = files.create_session_dir()

    assert first_id != second_id
    assert first_dir != second_dir
    assert Path(first_dir).is_dir()
    assert Path(second_dir).is_dir()
    assert Path(first_dir).name == f"session_{first_id}"
