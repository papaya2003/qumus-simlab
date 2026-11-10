from __future__ import annotations

from qumus_simlab.core.materials import is_graphene_flake, normalize_material_code
from qumus_simlab.core.types import angle_diff


def test_materials():
    assert normalize_material_code(" graphene ") == "Gr"
    assert normalize_material_code("HBN") == "hBN"
    assert normalize_material_code("") is None
    assert is_graphene_flake(type_label="graphite", thickness_layers=1, thickness_label=None)
    assert not is_graphene_flake(type_label="hBN", thickness_layers=1, thickness_label=None)


def test_angle_diff():
    assert angle_diff(179, -179) == -2
    assert angle_diff(10, 350) == 20
