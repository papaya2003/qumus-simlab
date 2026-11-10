"""Full region contact support preserves material holes and observed pockets."""
from qumus_simlab.core.regions import PolygonRegion, RegionComponent, region_dict
from qumus_simlab.perception.tracking import _whole_region_cell_coverage


def test_contact_cells_keep_material_holes_and_refuse_uncovered_material():
    source = PolygonRegion(components=(RegionComponent(
        exterior=((2., 2.), (8., 2.), (8., 8.), (2., 8.)),
        holes=(((4., 4.), (6., 4.), (6., 6.), (4., 6.)),)),))
    cells = set(range(121))
    cells.remove(5*11+5)  # Unobserved cell entirely within a real material hole.
    result = _whole_region_cell_coverage(cells, 11, 11, region_dict(source), .001, .001, .0001)
    assert result["covered"] and result["holes_filled"] is False
    cells.remove(3*11+3)
    assert not _whole_region_cell_coverage(cells, 11, 11, region_dict(source), .001, .001, 0.)["covered"]


def test_contact_margin_is_measured_without_dilating_material():
    source = PolygonRegion(components=(RegionComponent(
        exterior=((2., 2.), (8., 2.), (8., 8.), (2., 8.))),))
    cells = {y*11+x for y in range(1, 10) for x in range(1, 10)}
    body = region_dict(source)
    passed = _whole_region_cell_coverage(cells, 11, 11, body, .001, .001, .001)
    failed = _whole_region_cell_coverage(cells, 11, 11, body, .001, .001, .002)
    assert passed["covered"] and not failed["covered"]
    assert passed["target_region_px"] == body == failed["target_region_px"]
