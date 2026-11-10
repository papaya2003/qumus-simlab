"""Coarse observed priority is advisory; final source geometry is unchanged."""

from hashlib import sha256
from types import SimpleNamespace

from qumus_simlab.core.regions import as_region
from qumus_simlab.flows.library.revisit_priority import rank_revisits
from qumus_simlab.flows.library.search import RevisitCandidates, RevisitRole, ScanChip
from qumus_simlab.perception.base import FlakeDetection


def rect(w, h):
    return as_region(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    ).model_dump(mode="json")


def sample(tmp_path, fid, w=None, h=None, material="Gr", label="monolayer", area=None):
    props = {
        "material": material,
        "area_um2": area if area is not None else (w * h * 1e6 if w else 500),
        "labels": {"10x": label},
    }
    if w is not None:
        image = tmp_path / "original10.png"
        image.write_bytes(b"original public pixels")
        props["coarse_geometry_10x"] = {
            "source": "v1_lab_rgb",
            "objective": 10,
            "frame": "flake_center_chip_axes",
            "units": "mm",
            "region_mm": rect(w, h),
            "image": {
                "path": str(image),
                "sha256": sha256(image.read_bytes()).hexdigest(),
                "size_px": [100, 100],
            },
            "calibration": {"image_size_px": [100, 100]},
        }
    return {"id": fid, "kind": "flake", "location": "chip", "status": "candidate", "props": props}


def target():
    return {
        "contacts": [
            {
                "id": str(i),
                "points_mm": [
                    [x - 0.0005, y - 0.0005],
                    [x + 0.0005, y - 0.0005],
                    [x + 0.0005, y + 0.0005],
                    [x - 0.0005, y + 0.0005],
                ],
            }
            for i, (x, y) in enumerate(
                [
                    (-0.005, 0.009),
                    (0.005, 0.009),
                    (-0.005, -0.009),
                    (0.005, -0.009),
                    (-0.006, 0.004),
                    (0.006, 0.004),
                    (-0.006, -0.004),
                    (0.006, -0.004),
                ]
            )
        ]
    }


def roles():
    return [
        RevisitRole(
            role="channel", material="Gr", thickness_labels=["monolayer"], min_area_um2=50, slots=3
        )
    ]


def test_medium_compatible_gr_beats_largest_and_tiny(tmp_path):
    cands = [
        sample(tmp_path, "large", 0.010, 0.100),
        sample(tmp_path, "medium", 0.018, 0.026),
        sample(tmp_path, "tiny", 0.004, 0.004),
    ]
    hbn = sample(tmp_path, "hbn", 0.026, 0.034, material="hBN", label="20-30")
    selected, reason = rank_revisits(cands, "Gr", roles(), 1, target(), [hbn])
    assert [s["id"] for s in selected] == ["medium"]
    assert reason["candidates"]["medium"]["pair_potential"] is True
    assert reason["candidates"]["large"]["target_potential"] is False
    assert reason["candidates"]["tiny"]["target_potential"] is False


def test_unknown_geometry_is_retained_in_fair_slot(tmp_path):
    cands = [
        sample(tmp_path, "good1", 0.018, 0.026),
        sample(tmp_path, "good2", 0.020, 0.028),
        sample(tmp_path, "unknown"),
    ]
    selected, reason = rank_revisits(cands, "Gr", roles(), 2, target(), [])
    assert selected[0]["id"] == "good1" and selected[1]["id"] == "unknown"
    assert reason["candidates"]["unknown"]["reason"] == "coarse_geometry_unavailable_pending"
    assert all(s["status"] == "candidate" for s in cands)


def test_coarse_label_mismatch_stays_pending_and_not_excluded(tmp_path):
    c = sample(tmp_path, "unknown_label", label="uncertain")
    selected, reason = rank_revisits([c], "Gr", roles(), 1, target(), [])
    assert selected == [c] and reason["candidates"][c["id"]]["admission"].startswith("pending")


def test_changed_original_image_makes_geometry_unknown_not_rejected(tmp_path):
    c = sample(tmp_path, "changed", 0.018, 0.026)
    (tmp_path / "original10.png").write_bytes(b"changed")
    selected, reason = rank_revisits([c], "Gr", roles(), 1, target(), [])
    assert selected == [c] and reason["candidates"][c["id"]]["coarse_geometry_available"] is False


def test_explicit_ids_bypass_priority_and_preserve_order():
    cands = [
        {"id": fid, "kind": "flake", "location": "chip", "status": "candidate", "props": {}}
        for fid in ["a", "b"]
    ]
    store = SimpleNamespace(list_samples=lambda **kw: cands)
    step = RevisitCandidates("chip", "Gr", 2, roles(), only_new=True, flake_ids=["b", "a"])
    assert [s["id"] for s in step._targets(SimpleNamespace(store=store))] == ["b", "a"]


def test_scan_persists_only_public_rgb_coarse_region(tmp_path):
    image = tmp_path / "rgb10.png"
    image.write_bytes(b"original")
    calibration = {
        "id": "observed",
        "version": "1",
        "image_size_px": [100, 100],
        "mm_per_pixel": [0.001, 0.001],
        "u_axis": "+x",
        "v_axis": "-y",
    }
    chain = {
        "image": {
            "path": str(image),
            "sha256": sha256(image.read_bytes()).hexdigest(),
            "size_px": [100, 100],
            "objective": 10,
        },
        "calibration": calibration,
        "selected_label_ids": [7],
        "derivation": {"centroid_px": [52.0, 48.0]},
    }
    detection = FlakeDetection(
        material="Gr",
        thickness_label="monolayer",
        area_um2=100,
        dx_mm=0.01,
        dy_mm=0.02,
        confidence=0.9,
        meta={
            "geometry_source": "v1_lab_rgb",
            "outline_complete": True,
            "region_mm": rect(0.01, 0.02),
            "recognition_chain": chain,
            "sim_id": "must_not_rank",
        },
    )
    capture = {"image": str(image), "calibration": calibration, "timestamp": 3.0}
    c = ScanChip._candidate(detection, 1, 2, (0, 0), 0, capture=capture)
    evidence = c["coarse_geometry_10x"]
    assert evidence["image"] == chain["image"] and evidence["selected_label_ids"] == [7]
    assert "sim_id" not in evidence and "recognition_chain" not in evidence
    assert "coarse_geometry_10x" not in ScanChip._candidate(detection, 1, 2, (0, 0), 0)


def test_ten_only_raw_images_remain_normal_revisit_candidates():
    cands = [
        {
            "id": "ten_only",
            "status": "candidate",
            "props": {
                "images": {"10x": "raw10.png"},
                "labels": {"10x": "monolayer"},
                "area_um2": 100,
            },
        },
        {
            "id": "photographed",
            "status": "candidate",
            "props": {
                "images": {"10x": "raw10.png", "20x": "actual20.png"},
                "labels": {"10x": "monolayer"},
                "area_um2": 200,
            },
        },
    ]
    store = SimpleNamespace(list_samples=lambda **kw: cands)
    selected = RevisitCandidates("chip", "Gr", 2, roles(), only_new=True)._targets(
        SimpleNamespace(store=store)
    )
    assert [s["id"] for s in selected] == ["ten_only"]


def test_hbn_target_potential_is_useful_before_gr_is_observed(tmp_path):
    cands = [
        sample(tmp_path, "large_narrow", 0.010, 0.100, material="hBN", label="20-30"),
        sample(tmp_path, "medium", 0.025, 0.030, material="hBN", label="20-30"),
    ]
    quota = [RevisitRole(role="top_hbn", material="hBN", thickness_labels=["20-30"], slots=3)]
    selected, _ = rank_revisits(cands, "hBN", quota, 1, target(), [])
    assert selected[0]["id"] == "medium"


def test_revisit_publication_preserves_original_full_ten_frame():
    flake = {
        "id": "f",
        "props": {"images": {"10x": "originalfull10.png"}, "labels": {"10x": "monolayer"}},
    }
    captured = {}

    def get_sample(fid):
        return {"props": {"rotation_deg": 0}} if fid == "chip" else flake

    store = SimpleNamespace(
        get_sample=get_sample,
        bench="sim",
        upsert_sample=lambda fid, kind, **kw: captured.update(kw),
    )
    step = RevisitCandidates("chip", "Gr", 1)
    step._results = {"f": {"found": False, "labels": {}, "images": {"20x": "new20.png"}}}
    step._publish_results(SimpleNamespace(store=store))
    assert captured["props"]["images"] == {"10x": "originalfull10.png", "20x": "new20.png"}
