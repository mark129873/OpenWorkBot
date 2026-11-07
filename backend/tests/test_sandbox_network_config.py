import json
import uuid

import pytest

from app.community.aio_sandbox.local_backend import LocalContainerBackend, normalize_allow_domains
from app.community.aio_sandbox.sandbox_info import SandboxInfo


def test_allow_domains_normalizes_and_allows_an_empty_list():
    assert normalize_allow_domains(" PyPI.org.,*.pythonhosted.org,pypi.org ") == (
        "pypi.org", "*.pythonhosted.org")
    assert normalize_allow_domains("") == ()


@pytest.mark.parametrize("value", ["*", "http://pypi.org", "127.0.0.1", "local", "pypi.org:443", "pypi.org,,github.com"])
def test_allow_domains_rejects_unsafe_rules(value):
    with pytest.raises(ValueError):
        normalize_allow_domains(value)


def test_invalid_destination_does_not_hide_other_network_events(tmp_path, monkeypatch):
    backend = LocalContainerBackend(tmp_path, tmp_path, tmp_path)
    info = SandboxInfo("sandbox", "user", "generation", "container", "proxy", "network", "egress", "url", "token")
    invalid, valid = str(uuid.uuid4()), str(uuid.uuid4())
    events = [{"request_id": invalid, "host": "localhost", "port": 80, "method": "GET"},
              {"request_id": valid, "host": "example.com", "port": 443, "method": "CONNECT"}]
    denied = []
    monkeypatch.setattr(backend, "_docker", lambda *args, **kwargs: json.dumps(events))
    monkeypatch.setattr(backend, "decide_network_event", lambda *args: denied.append(args))
    assert backend.network_events(info) == [events[1]]
    assert denied == [(info, invalid, "deny")]
