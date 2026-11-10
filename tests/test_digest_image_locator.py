import pytest

from qumus_simlab.flows.library.stacking import _digest

H = "ab" * 32


def test_v1_recognition_chain_locator_is_bound_by_content():
    a = _digest({"recognition_chain": {"image": {"path": "/a/x.jpg", "sha256": H, "objective": 20}}})
    b = _digest({"recognition_chain": {"image": {"path": "/b/y.jpg", "sha256": H, "objective": 20}}})
    assert a == b
    c = _digest({"recognition_chain": {"image": {"path": "/a/x.jpg", "sha256": "cd" * 32, "objective": 20}}})
    assert c != a


@pytest.mark.parametrize("locator", [{"path": "", "sha256": H}, {"path": "/a.jpg", "sha256": "zz"}, {"sha256": H}])
def test_malformed_dict_locator_is_refused(locator):
    with pytest.raises(ValueError, match="Malformed image evidence locator"):
        _digest({"image": locator})
