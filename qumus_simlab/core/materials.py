"""Material name normalisation (spec 7.1).

Ported from v1 common/material_semantics.py. Material labels are set by code
from the job and QR codes; the AI never fills them in.
"""

from __future__ import annotations

GRAPHITE_CODE = "Gr"
GRAPHENE_THICKNESS_LABELS = {"1", "1l", "monolayer"}
GRAPHITE_GATE_THICKNESS_LABELS = {"manylayer"}

_ALIASES = {
    "gr": GRAPHITE_CODE,
    "graphite": GRAPHITE_CODE,
    "graphene": GRAPHITE_CODE,
    "hbn": "hBN",
    "boron nitride": "hBN",
    "rese2": "ReSe2",
    "re se2": "ReSe2",
    "re-se2": "ReSe2",
}


#: every way a text may name a material, longest first (so "boron nitride"
#: wins over nothing shorter); codes themselves included
_NAMES = sorted({*_ALIASES, *(c.lower() for c in {*_ALIASES.values()})}, key=len, reverse=True)


def materials_in_text(text: str) -> list[str]:
    """Material codes named in free text, in order of first mention:
    "exfoliate graphene and hBN" -> ["Gr", "hBN"]. Whole words only."""
    import re

    low = text.lower()
    hits: list[tuple[int, str]] = []
    for name in _NAMES:
        m = re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", low)
        if m:
            hits.append((m.start(), normalize_material_code(name) or name))
    out: list[str] = []
    for _, code in sorted(hits):
        if code not in out:
            out.append(code)
    return out


def normalize_material_code(material: str | None) -> str | None:
    if material is None:
        return None
    normalized = material.strip()
    if not normalized:
        return None
    return _ALIASES.get(normalized.lower(), normalized)


def is_graphene_flake(
    *, type_label: str | None, thickness_layers: float | None, thickness_label: str | None
) -> bool:
    """Graphene is defined as Gr + monolayer (1L)."""
    if normalize_material_code(type_label) != GRAPHITE_CODE:
        return False
    if thickness_layers is not None and float(thickness_layers) == 1.0:
        return True
    return (
        thickness_label is not None
        and thickness_label.strip().lower() in GRAPHENE_THICKNESS_LABELS
    )


def is_graphite_gate_flake(*, type_label: str | None, thickness_label: str | None) -> bool:
    """A graphite gate uses the categorical Gr/manylayer label."""
    if normalize_material_code(type_label) != GRAPHITE_CODE:
        return False
    if thickness_label is None:
        return False
    return thickness_label.strip().lower() in GRAPHITE_GATE_THICKNESS_LABELS


#: thickness class names the perception code writes into the ledger (v1
#: segmenter/material.py, autosearch); device plans must use the same words
THICKNESS_LABELS = {
    GRAPHITE_CODE: ["monolayer", "bilayer", "trilayer", "fewlayer", "manylayer"],
    "hBN": ["0-10", "10-20", "20-30", "30-40", ">40"],
}
_LABEL_ALIASES = {
    "1": "monolayer", "1l": "monolayer", "single layer": "monolayer", "single-layer": "monolayer",
    "2": "bilayer", "2l": "bilayer", "3": "trilayer", "3l": "trilayer",
    "few layer": "fewlayer", "few-layer": "fewlayer", "many layer": "manylayer",
    "many-layer": "manylayer", "graphite": "manylayer", "gt40": ">40",
}


def normalize_thickness_label(material: str | None, label: str) -> str | None:
    """A device-plan thickness label in the ledger's words ("1L" -> "monolayer",
    "20-30 nm" -> "20-30"); None if the material has a vocabulary and the label
    is not in it. Labels of materials without a vocabulary pass through."""
    vocab = THICKNESS_LABELS.get(normalize_material_code(material) or "")
    s = " ".join(label.strip().lower().replace("–", "-").replace("nm", "").split())
    s = s.replace(" - ", "-").strip()
    s = _LABEL_ALIASES.get(s, s)
    if vocab is None:
        return label.strip()
    return s if s in vocab else None


#: ledger bins as closed-open ranges, for mapping a thickness range to labels
_HBN_BINS_NM = [("0-10", 0.0, 10.0), ("10-20", 10.0, 20.0), ("20-30", 20.0, 30.0),
                ("30-40", 30.0, 40.0), (">40", 40.0, float("inf"))]
#: layer counts are whole numbers: closed ranges
_GR_BINS_LAYERS = [("monolayer", 1, 1), ("bilayer", 2, 2), ("trilayer", 3, 3),
                   ("fewlayer", 4, 9), ("manylayer", 10, float("inf"))]
GRAPHENE_LAYER_NM = 0.335


def _overlap(lo: float, hi: float, bins: list[tuple[str, float, float]]) -> list[str]:
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:  # a single value: the bin it falls in
        return [n for n, a, b in bins if a <= lo < b]
    return [n for n, a, b in bins if lo < b and hi > a]


def labels_for_layers(material: str | None, lo: float, hi: float) -> list[str]:
    """Graphene layer counts (1..3 = mono/bi/trilayer, 4..9 fewlayer, 10+ manylayer)."""
    if normalize_material_code(material) != GRAPHITE_CODE:
        return []
    if hi < lo:
        lo, hi = hi, lo
    return [n for n, a, b in _GR_BINS_LAYERS if lo <= b and hi >= a]


def labels_for_nm(material: str | None, lo: float, hi: float) -> list[str]:
    """The ledger thickness labels a thickness range in nm overlaps: hBN bins
    (10-20, 20-30, ...), graphene by layer count (0.335 nm per layer)."""
    mat = normalize_material_code(material)
    if mat == "hBN":
        return _overlap(lo, hi, _HBN_BINS_NM)
    if mat == GRAPHITE_CODE:
        return labels_for_layers(mat, lo / GRAPHENE_LAYER_NM, hi / GRAPHENE_LAYER_NM)
    return []


def labels_from_text(material: str | None, text: str) -> list[str]:
    """Labels from a free-text thickness ("10-40 nm", "1-3 layers", "20 nm",
    "monolayer", "1L"); [] when nothing can be read from it."""
    import re

    t = (text or "").strip().lower().replace("\u2013", "-")
    if not t:
        return []
    m = re.fullmatch(r"~?\s*(\d+(?:\.\d+)?)\s*(?:-|to|\u2013)\s*(\d+(?:\.\d+)?)\s*(nm|layers?|l)\b.*", t)
    if m is None:
        m = re.fullmatch(r"~?\s*(\d+(?:\.\d+)?)()\s*(nm|layers?|l)\b.*", t)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        if m.group(3) == "nm":
            return labels_for_nm(material, lo, hi)
        return labels_for_layers(material, lo, hi)
    one = normalize_thickness_label(material, text)
    vocab = THICKNESS_LABELS.get(normalize_material_code(material) or "")
    return [one] if one and vocab and one in vocab else []
