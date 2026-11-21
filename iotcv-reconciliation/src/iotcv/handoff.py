"""IoT -> CV handoff interface (``Flag`` / ``CVVerdict``).

The IoT detectors emit ``Flag`` objects. A CV verifier consumes a flag, inspects
the named location, and returns a ``CVVerdict``. Both sides are defined here and
are JSON-serializable.

Schema note: optional ``evidence`` keys may be added; renaming or removing
top-level fields changes the schema version.

Flag JSON schema (v1):
    {
      "schema_version": "1.0",
      "flag_id":      str,    # globally unique within a run
      "tick":         int,    # simulation time the flag was raised
      "lane":         str,    # "weight" | "location" | "dwell"
      "anomaly_type": str,    # e.g. "phantom_inventory", "zone_mismatch", "stranded"
      "location":     str,    # canonical key: "bin:<id>" | "asset:<id>" | "item:<id>"
      "zone":         int,    # physical zone hint for the CV camera (-1 if N/A)
      "reason":       str,    # human-readable detector rationale
      "confidence":   float,  # detector confidence in [0, 1]
      "evidence":     {..}    # lane-specific numeric evidence (free-form, optional)
    }

CVVerdict JSON schema (v1):
    {
      "schema_version": "1.0",
      "flag_id":   str,    # echoes the Flag it answers
      "tick":      int,    # when the verdict was produced
      "verified":  bool,   # True => CV CONFIRMS a real discrepancy at the location
      "label":     str,    # e.g. "empty", "wrong_item", "correct", "occluded"
      "confidence": float, # CV confidence in [0, 1]
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

SCHEMA_VERSION = "1.0"

# Canonical anomaly-type vocabulary shared with the CV side. Kept small on purpose.
# Additive since schema 1.0: the CV verifier lanes introduced three finer-grained
# types (wrong_item_qty, stranded_dwell, missed_putaway). Adding enum members is a
# backward-compatible change (old readers ignore unknown types); renaming or
# removing a member would be breaking, so the original tokens are retained.
ANOMALY_TYPES = {
    # weight lane
    "phantom_inventory",   # system says stock present, bin lighter than expected
    "low_stock",           # bin below expected, restock signal
    "empty_bin",           # bin reads empty but system shows units
    "wrong_item",          # weight residual implies a different SKU present
    "wrong_item_qty",      # graded stock-level mismatch verified by the density lane
    # location lane
    "zone_mismatch",       # asset not in its expected zone
    "ghost_move",          # asset relocated with no matching WMS move
    # dwell lane
    "stranded",            # item dwelling far beyond expected process time
    "stranded_dwell",      # dwell-lane stranded item (On-dock verifier vocabulary)
    "missed_putaway",      # received stock never stowed within its dwell window
}

LANES = {"weight", "location", "dwell"}


@dataclass
class Flag:
    """An IoT anomaly flag handed off to CV for physical verification."""

    flag_id: str
    tick: int
    lane: str
    anomaly_type: str
    location: str
    zone: int = -1
    reason: str = ""
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.lane not in LANES:
            raise ValueError(f"unknown lane {self.lane!r}; expected one of {sorted(LANES)}")
        if self.anomaly_type not in ANOMALY_TYPES:
            raise ValueError(
                f"unknown anomaly_type {self.anomaly_type!r}; "
                f"expected one of {sorted(ANOMALY_TYPES)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} out of [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Flag":
        # Ignore unknown keys defensively so additive schema changes don't break old readers.
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "Flag":
        return cls.from_dict(json.loads(s))


@dataclass
class CVVerdict:
    """The CV model's response to a Flag (here produced by the oracle stub)."""

    flag_id: str
    tick: int
    verified: bool
    label: str = ""
    confidence: float = 0.0
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CVVerdict":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "CVVerdict":
        return cls.from_dict(json.loads(s))


def write_flags_jsonl(flags, path: str) -> None:
    """Persist a sequence of Flags as JSON Lines for the CV team to consume."""
    with open(path, "w") as f:
        for fl in flags:
            f.write(fl.to_json() + "\n")


def read_flags_jsonl(path: str):
    """Read Flags back from JSON Lines (round-trips write_flags_jsonl)."""
    flags = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                flags.append(Flag.from_json(line))
    return flags
