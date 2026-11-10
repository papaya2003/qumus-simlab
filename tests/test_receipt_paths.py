"""A record that went to the brain and back may carry the brain's picture paths."""

from qumus_simlab.records.recognition import _adopt_receipt_paths


def _chain(root):
    return {"image": {"path": f"{root}/m.jpg", "sha256": "a"}, "geometry_mask": {"path": f"{root}/g.png", "sha256": "b"},
            "material": "Gr"}


def test_path_only_difference_adopts_the_receipt():
    obs = {"recognition_chain": _chain("/brain/images"), "image": "/brain/images/m.jpg"}
    receipt = _chain("/lab/images")
    _adopt_receipt_paths(obs, receipt)
    assert obs["recognition_chain"] == receipt and obs["image"] == "/lab/images/m.jpg"


def test_any_other_difference_is_kept_for_the_check():
    obs = {"recognition_chain": {**_chain("/brain/images"), "material": "hBN"}, "image": "/brain/images/m.jpg"}
    _adopt_receipt_paths(obs, _chain("/lab/images"))
    assert obs["recognition_chain"]["material"] == "hBN" and obs["image"] == "/brain/images/m.jpg"
    bad = {"recognition_chain": {**_chain("/brain/images"), "image": {"path": "/x", "sha256": "z"}}}
    _adopt_receipt_paths(bad, _chain("/lab/images"))
    assert bad["recognition_chain"]["image"]["sha256"] == "z"
