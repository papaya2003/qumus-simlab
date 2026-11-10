"""Preconstruction eight-contact fixtures/import bundles; never prepared source flakes."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from qumus_simlab.core.polygons import rectangle
from qumus_simlab.records.device_design import (
    GDSMetalPrimitive,
    HallBarTarget,
    ImportedHallBarTarget,
    TargetGDSImportReceipt,
    _derived_import_contacts,
    _import_geometry,
    imported_target_directory,
    parse_hallbar_target,
    target_import_binding,
    target_metal_primitives,
)
from qumus_simlab.records.geometry import geometry_digest


def hallbar_target(*, size_mm=(10., 10.), source_drain_span_um=40.,
                   side_span_um=20., voltage_spacing_um=10., contact_size_um=4.) -> HallBarTarget:
    """Tens-of-micrometres transfer geometry, not a surveyed/fabricated Hall bar.

    All dimensions are declared synthetic inputs. Electrode rectangles describe
    intended contact tips; this model does not fabricate leads or etch graphene.
    """
    roles = ["source", "drain", "voltage_left_1", "voltage_left_2", "voltage_left_3",
             "voltage_right_1", "voltage_right_2", "voltage_right_3"]
    centers = [(0, source_drain_span_um/2), (0, -source_drain_span_um/2),
               *[(-side_span_um/2, y) for y in (-voltage_spacing_um, 0, voltage_spacing_um)],
               *[(side_span_um/2, y) for y in (-voltage_spacing_um, 0, voltage_spacing_um)]]
    return HallBarTarget(source="simulated_design", reference="Declared synthetic eight-contact transfer wafer v1; dimensions are not measured laboratory geometry",
        size_mm=size_mm, contacts=[{"id": role, "role": role,
            "points_mm": rectangle(contact_size_um/1000, contact_size_um/1000,
                (size_mm[0]/2+x/1000, size_mm[1]/2+y/1000))}
            for role, (x, y) in zip(roles, centers, strict=True)])


#: declared blank source-chip stocks; 9 is the real lab's count
STOCK_SOURCES = {4: "declared_four_blank_chip_diagnostic_v1",
                 6: "declared_six_blank_chip_common_fixture_v1",
                 9: "declared_nine_blank_chip_lab_stock_v1"}


def with_hallbar_target(config, *, chip_id="chip_e01", target: HallBarTarget | ImportedHallBarTarget | None = None,
                        raw_stock_count: int | None = None):
    """Configure the prescribed wafer before bench construction; sources stay raw.

    By default the linked bench (nine left-board slots) gets the real lab's nine
    blank source chips (Jialei, 2026-11-02); the older bench has six slots and
    keeps the six-chip fixture. Four remains for diagnostics.
    """
    result = deepcopy(config)
    if raw_stock_count is None:
        raw_stock_count = 9 if result.get("bench_geometry") else 6
    if any(s["kind"] in ("flake", "device") for s in result.get("initial_samples", [])):
        raise ValueError("the common Hall-bar fixture starts from tapes without prepared flakes or devices")
    if type(raw_stock_count) is not int or raw_stock_count not in STOCK_SOURCES:
        raise ValueError("declare the four-chip diagnostic, the six-chip fixture or the nine-chip lab raw stock")
    existing = {s["id"] for s in result["initial_samples"]}
    for index in range(5, raw_stock_count + 1):
        sample_id = f"chip_{index:03d}"
        if sample_id not in existing:
            result["initial_samples"].append({"id": sample_id, "kind": "chip",
                "location": f"left_chip_{index}", "status": "bare",
                "props": {"substrate": "SiO2/Si 285nm", "stock_source": STOCK_SOURCES[raw_stock_count]}})
    matches = [s for s in result.get("initial_samples", []) if s["id"] == chip_id and s["kind"] == "chip"]
    if len(matches) != 1:
        raise ValueError("the prescribed eight-contact wafer must be one known initial chip")
    typed = parse_hallbar_target(target) if target is not None else hallbar_target()
    props = matches[0].setdefault("props", {})
    props.update(device_design_target=typed.model_dump(mode="json"),
                 electrodes="eight existing source/drain and voltage contact tips", layers_top_first=[])
    if isinstance(typed, ImportedHallBarTarget):
        props["electrodes"] = "Complete owner-imported electrode metal with eight fixed contact ends"
        props["imported_target_binding"] = target_import_binding(typed)
    return result


def electrode_pattern_from_target(target: HallBarTarget | ImportedHallBarTarget):
    """Renderer/physical model share all original primitives; never their hull."""
    typed = parse_hallbar_target(target)
    return [{"id": p["id"], "kind": "contact", "points":
             [[x-typed.size_mm[0]/2, y-typed.size_mm[1]/2] for x, y in p["points_mm"]],
             **({"source_primitive_index": p["source_primitive_index"]}
                if "source_primitive_index" in p else {})}
            for p in target_metal_primitives(typed)]


def build_imported_hallbar_target(*, primitives, source_sha256, owner_instruction_id,
                                  owner_instruction_sha256, size_mm=(10., 10.)):
    """Pure construction from original integer semantics plus exact converted outlines.

    Paths need outline_dbu from their actual KLayout conversion; reconstructing
    a path from an axis-aligned box or hull is not an import operation.
    """
    parsed = []
    for source in primitives:
        if isinstance(source, GDSMetalPrimitive):
            parsed.append(source)
            continue
        raw = dict(source)
        raw.pop("bbox_dbu", None)  # Display string, not an authority.
        if raw.get("kind") == "box" and isinstance(raw.get("box_dbu"), str):
            match = re.fullmatch(r"\((-?\d+),(-?\d+);(-?\d+),(-?\d+)\)", raw["box_dbu"])
            if not match:
                raise ValueError("Original integer box syntax is unsupported")
            raw["box_dbu"] = [int(value) for value in match.groups()]
        if "outline_dbu" not in raw:
            if raw.get("kind") == "polygon":
                raw["outline_dbu"] = raw["points_dbu"]
            elif raw.get("kind") == "box":
                x0, y0, x1, y1 = raw["box_dbu"]
                raw["outline_dbu"] = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
            else:
                raise ValueError("Original path needs its exact converted integer polygon")
        parsed.append(GDSMetalPrimitive.model_validate(raw))
    parsed = tuple(parsed)
    if tuple(part.index for part in parsed) != tuple(range(34)):
        raise ValueError("Import needs all 34 original primitives in original index order")
    contacts = _derived_import_contacts(parsed)
    receipt = TargetGDSImportReceipt(source_sha256=source_sha256,
        owner_instruction_id=owner_instruction_id, owner_instruction_sha256=owner_instruction_sha256,
        geometry_sha256=geometry_digest(_import_geometry(parsed, contacts, size_mm)))
    return ImportedHallBarTarget(source="design", reference="saved-gds-import:"+geometry_digest(receipt.model_dump(mode="json")),
        size_mm=size_mm, contacts=contacts, metal_primitives=parsed, import_receipt=receipt)


def load_imported_hallbar_target(bundle_dir, *, workspace=None, approval_reference=None, approval_sha256=None):
    """Read one operator-selected target.json; optionally materialize its same bytes.

    Provenance comes from the existing owner/config route, not a model's approval
    boolean. Expected approval arguments verify stored identity and never replace it.
    workspace=None is strictly read-only; existing generated files are never reset.
    """
    bundle = Path(bundle_dir).resolve()
    files = {name: bundle/name for name in ("target.json", "electrode-layout.gds")}
    if any(path.is_symlink() or path.resolve().parent != bundle for path in files.values()):
        raise ValueError("Imported target bundle must contain regular sibling files")
    raw = {name: path.read_bytes() for name, path in files.items()}
    target = parse_hallbar_target(json.loads(raw["target.json"]))
    if not isinstance(target, ImportedHallBarTarget):
        raise ValueError("An imported target requires its explicit versioned contract")
    receipt = target.import_receipt
    if (hashlib.sha256(raw["electrode-layout.gds"]).hexdigest() != receipt.source_sha256
            or approval_reference is not None and approval_reference != receipt.owner_instruction_id
            or approval_sha256 is not None and approval_sha256 != receipt.owner_instruction_sha256):
        raise ValueError("Imported target bytes or expected owner instruction identity differ")
    if workspace is not None:
        directory = imported_target_directory(Path(workspace), target)
        if directory.exists():
            for name, data in raw.items():
                saved = directory/name
                if saved.is_symlink() or saved.read_bytes() != data:
                    raise ValueError("Existing imported target bundle is immutable; use its original files")
        else:
            directory.mkdir(parents=True, exist_ok=False)
            for name, data in raw.items():
                with (directory/name).open("xb") as output:
                    output.write(data)
    return target
