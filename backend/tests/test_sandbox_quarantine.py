import pytest

from app.community.aio_sandbox.quarantine import SandboxQuarantine


def test_quarantine_survives_restart_and_rejects_old_generation(tmp_path):
    fence = SandboxQuarantine(tmp_path)
    fence.block("sandbox", "user", "generation-1", "cleanup failed")
    restarted = SandboxQuarantine(tmp_path)
    assert restarted.get("sandbox")["generation"] == "generation-1"
    assert not restarted.clear("sandbox", "old-generation")
    assert restarted.get("sandbox") is not None
    assert restarted.clear("sandbox", "generation-1")
    assert SandboxQuarantine(tmp_path).get("sandbox") is None


def test_invalid_quarantine_file_fails_closed(tmp_path):
    (tmp_path / "sandbox-quarantine.json").write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="无法读取沙箱隔离记录"):
        SandboxQuarantine(tmp_path)
