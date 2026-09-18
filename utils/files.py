import os
import uuid

from config.settings import TMP_DIR


def create_session_dir() -> tuple[str, str]:
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TMP_DIR, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    return session_id, session_dir
