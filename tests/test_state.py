import importlib
import sys
from types import ModuleType


class SessionState(dict[str, object]):
    def __getattr__(self, name: str) -> object:
        return self[name]

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


def test_init_session_state_sets_defaults_without_overwriting(monkeypatch) -> None:
    fake_streamlit = ModuleType("streamlit")
    session_state = SessionState()
    fake_streamlit.__dict__["session_state"] = session_state
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.delitem(sys.modules, "utils.state", raising=False)
    state = importlib.import_module("utils.state")

    state.init_session_state()

    assert session_state.selected_city == "Lahore"
    assert session_state.processing_results == {}

    session_state.selected_city = "Islamabad"
    state.init_session_state()

    assert session_state.selected_city == "Islamabad"
