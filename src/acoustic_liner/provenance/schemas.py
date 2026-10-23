"""Schema identifiers of the records this package writes and reads.

A public identifier has the form `acoustic-liner/<name>/<major>`; the major version rises only for an incompatible
change. A reader accepts the current identifier and the earlier ones listed for it, and refuses anything else.
"""
from __future__ import annotations

# name -> the identifier writers emit
CURRENT = {
    "vehicle-scenario": "acoustic-liner/vehicle-scenario/1",
    "vehicle-cfd-bundle": "acoustic-liner/vehicle-cfd-bundle/1",
    "vehicle-verdict": "acoustic-liner/vehicle-verdict/2",
    "falcon-supplemental-verdict": "acoustic-liner/falcon-supplemental-verdict/2",
    "vehicle-statistical-followup": "acoustic-liner/vehicle-statistical-followup/2",
    "figure-manifest": "acoustic-liner/figure-manifest/1",
    "visualization-cases": "acoustic-liner/visualization-cases/1",
    "visualization-manifest": "acoustic-liner/visualization-manifest/1",
    "checkpoint-manifest": "acoustic-liner/checkpoint-manifest/1",
    "evidence-manifest": "acoustic-liner/evidence-manifest/1",
    "posthoc-followups": "acoustic-liner/posthoc-followups/1",
    "readme-figures": "acoustic-liner/readme-figures/1",
    # intermediate records that had no schema field while the study ran (their version 1 is identified by the
    # fields listed in UNVERSIONED_V1); version 2 drops the seal, lock and protocol hashes
    "shard-manifest": "acoustic-liner/shard-manifest/2",
    "fem-audit-seal": "acoustic-liner/fem-audit-seal/2",
    "cfd-targeted-selection": "acoustic-liner/cfd-targeted-selection/2",
    "cfd-targeted-amendment": "acoustic-liner/cfd-targeted-amendment/2",
    "cfd-targeted-result": "acoustic-liner/cfd-targeted-result/2",
    "statistical-impedance-estimate": "acoustic-liner/statistical-impedance-estimate/2",
    "fem-field-manifest": "acoustic-liner/fem-field-manifest/2",
}

# name -> the fields every version-1 record of an unversioned format carries
UNVERSIONED_V1 = {
    "shard-manifest": ("manifest_sha256", "inputs"),
    "fem-audit-seal": ("record_sha256",),
    "cfd-targeted-selection": ("prereg_sha256",),
    "cfd-targeted-amendment": ("prereg_sha256",),
    "cfd-targeted-result": ("prereg_sha256",),
    "statistical-impedance-estimate": ("protocol_sha256",),
    "fem-field-manifest": ("historical_source_sha256",),
}

# name -> earlier identifiers a reader still accepts. Version 1 of the verdict and follow-up records names the
# protocol document and its SHA-256 (records written during the study); version 2 does not. The names with the
# historical package prefix are the identifiers of the implementation that wrote the archived records.
EARLIER = {
    "vehicle-scenario": ("casef_vehicle_scenario/1",),
    "vehicle-cfd-bundle": ("casef_vehicle_cfd_bundle/1",),
    "vehicle-verdict": ("acoustic-liner/vehicle-verdict/1", "casef_vehicle_verdict/1"),
    "falcon-supplemental-verdict": ("acoustic-liner/falcon-supplemental-verdict/1",
                                    "casef_falcon_supplemental_verdict/1"),
    "vehicle-statistical-followup": ("acoustic-liner/vehicle-statistical-followup/1",
                                     "casef_vehicle_statistical_followup/1"),
    "figure-manifest": ("casef_figure_manifest/1",),
    "visualization-cases": ("casef_visualization_cases/1",),
    "visualization-manifest": ("casef_visualization_manifest/2",),   # casef_visualization_manifest/1 is refused
}

# format of the five archived evidence summaries (read by the public verifier; never written for a new record)
ARCHIVED_SUMMARY_FORMAT = "linergen-public-evidence/1"


class SchemaError(RuntimeError):
    """An identifier is not the current or an accepted earlier identifier of the schema."""


def schema(name: str) -> str:
    if name not in CURRENT:
        raise SchemaError(f"no schema named {name!r}")
    return CURRENT[name]


def accept_schema(value, name: str) -> str:
    """The current identifier of `name` if `value` is it or one of its accepted earlier identifiers."""
    if value == schema(name) or value in EARLIER.get(name, ()):
        return CURRENT[name]
    raise SchemaError(f"schema {value!r} is not {CURRENT[name]!r} or an accepted earlier identifier of it")


def is_current(value, name: str) -> bool:
    return value == schema(name)


def record_version(rec: dict, name: str) -> int:
    """1 or 2. A record with a `schema` field must carry the current identifier (version 2) or, for a format that
    had one, an accepted earlier identifier (version 1). A record without one is version 1 of an unversioned format
    and must carry that version's fields. Anything else is refused."""
    if "schema" in rec:
        if rec["schema"] == schema(name):
            return int(rec["schema"].rsplit("/", 1)[1])
        if rec["schema"] in EARLIER.get(name, ()):
            return 1
        raise SchemaError(f"schema {rec['schema']!r} is not a supported version of {name!r}")
    need = UNVERSIONED_V1.get(name)
    if need is None or any(k not in rec for k in need):
        raise SchemaError(f"record has no schema and is not a version-1 {name!r} record")
    return 1
