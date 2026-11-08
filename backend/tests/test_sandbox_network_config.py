import pytest

from app.community.aio_sandbox.local_backend import normalize_allow_domains


def test_allow_domains_normalizes_and_allows_an_empty_list():
    assert normalize_allow_domains(" PyPI.org.,*.pythonhosted.org,pypi.org ") == (
        "pypi.org", "*.pythonhosted.org")
    assert normalize_allow_domains("") == ()


@pytest.mark.parametrize("value", ["*", "http://pypi.org", "127.0.0.1", "local", "pypi.org:443", "pypi.org,,github.com"])
def test_allow_domains_rejects_unsafe_rules(value):
    with pytest.raises(ValueError):
        normalize_allow_domains(value)
