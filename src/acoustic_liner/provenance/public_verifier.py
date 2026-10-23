# -*- coding: utf-8 -*-
"""Fast public verifier: checks a fresh clone in seconds, without the checkpoint file.

    acoustic-liner verify public

This verifier is independent of the code that produced the results. It uses the standard library and NumPy only.
It imports no evaluation, FEM or model code of this package, no Torch, and none of the package's statistics
functions; every statistic is recomputed here from the per-cell and per-pair values, and no stored aggregate is
trusted.

Two kinds of check are printed. RECOMPUTED: a published number is recomputed from the published per-cell or
per-pair values. CONSISTENT: two published files agree (for example a file and the SHA-256 the manifest lists for
it); this shows the files belong together, not where they came from.

Checks:
  1. `evidence/manifest.json`: every summary and the checkpoint manifest match the SHA-256 and size it lists, and no
     public evidence file contains an absolute path, user or host name, or machine directory;
  2. final evaluation: the paired contrasts and bootstrap intervals recomputed from the 1,500 per-cell values equal
     the stored aggregate; the per-condition wins, feasibility, uniqueness and the CMA-ES increment quoted in the
     README are recomputed;
  3. LinerGen checkpoint: parameter count computed from the released architecture; epoch, weight, configuration and
     recorded source hashes agree between the summary, the final-test record and the checkpoint manifest;
  4. FEM-transfer audit: correlation, median and maximum absolute error and bias recomputed from the 20 pairs;
  5. vehicles: each formal verdict follows the interval rule against its target;
  5b. post-hoc follow-ups: the Atlas I interval is recomputed from the stored corrected-FEM values and allowances and
     its outcome follows the interval rule; each Falcon 9 stage is consistent with its recorded checks;
  6. figures: every README figure is listed in evidence/readme-figures.json with its current SHA-256 (the archived
     figure manifest is checked as a summary in 1, but its figures are no longer distributed);
  7. if the checkpoint has been downloaded: its file, weights and recorded hashes, read without Torch.

The original records the summaries were built from are not distributed, so a public clone cannot rebuild the
summaries from them; the summaries themselves are what these checks recompute from.

All results are simulation-based.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import re
import sys
import zipfile

import numpy as np

from .. import paths
from . import schemas as SC

ROOT = paths.ROOT
EV = ROOT / "evidence"
CHECKPOINT_MANIFEST = ROOT / "release" / "checkpoint-manifest.json"

# ---- the registered bootstrap stream, restated -------------------------------------------------------------------
# A frozen deterministic constant, not a name of this package. The study derives its random streams from one seed
# sequence whose entropy is the first 8 bytes (big-endian) of SHA-256 of this byte label; the bootstrap is child 9
# of that sequence. The label is the one the released results were computed with, so it is kept byte for byte.
# Changing it would change every bootstrap interval.
SEED_DOMAIN_LABEL = b"casef-physgen-v1-root"
BOOTSTRAP_STREAM = 9
N_BOOT = 10000


def bootstrap_rng():
    entropy = int.from_bytes(hashlib.sha256(SEED_DOMAIN_LABEL).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(BOOTSTRAP_STREAM,)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(name):
    return json.loads((EV / name).read_text(encoding="utf-8"))


# a local path, user or host name, or machine directory in a public evidence file
PRIVATE = re.compile(r"[A-Za-z]:[\\/]|/(home|Users|mnt)/|"
                     r"\"(host|hostname|prepared_on|platform|command_line|cwd)\"", re.I)


# =================================================================================================================
#  independent statistics (NumPy only)
# =================================================================================================================
def diff_table(cells, budget, x, y, kx="best_val", ky="best_val"):
    """{condition: {seed: method x minus method y}} at one budget."""
    by = {}
    for c in cells:
        if c["budget"] == budget and c[x][kx] is not None and c[y][ky] is not None:
            by.setdefault(c["cond_index"], {})[c["seed"]] = c[x][kx] - c[y][ky]
    return by


def contrast(cells, budget, x, y, kx="best_val", ky="best_val"):
    """Paired contrast: the condition is the unit (mean over its seeds); the bootstrap resamples conditions."""
    by = diff_table(cells, budget, x, y, kx, ky)
    ids = sorted(by)
    cm = np.array([np.mean(list(by[i].values())) for i in ids])
    seeds = sorted({s for i in ids for s in by[i]})
    seed_means = {str(s): float(np.mean([by[i][s] for i in ids if s in by[i]])) for s in seeds}
    boot = cm[bootstrap_rng().integers(0, len(cm), size=(N_BOOT, len(cm)))].mean(1)
    return dict(n_conditions=len(ids), mean=float(cm.mean()),
                ci95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], seed_means=seed_means,
                min_seed_mean=float(min(seed_means.values())))


TRANSFER_FAILURE_MEDIAN_DB = 1.0          # the audit's descriptive threshold, restated


def audit_summary(rows):
    """Summary of the reduced-order scorer against FEM over the audit pairs (diff = oracle - FEM)."""
    d = [r for r in rows if r.get("diff") is not None]
    absd = [abs(r["diff"]) for r in d]
    by_corner = {}
    for r in d:
        for c in r["corners"]:
            by_corner.setdefault("_".join(map(str, c["corner_id"])), []).append(r["oracle"] - c["value"])
    med = float(np.median(absd)) if absd else None
    return dict(n=len(rows), n_evaluated=len(d), median_abs_diff_dB=med,
                mean_diff_dB=float(np.mean([r["diff"] for r in d])) if d else None,
                max_abs_diff_dB=float(max(absd)) if absd else None,
                by_corner_median_diff_dB={k: float(np.median(v)) for k, v in sorted(by_corner.items())},
                corr_oracle_fem=float(np.corrcoef([r["oracle"] for r in d], [r["fem"] for r in d])[0, 1])
                if len(d) > 2 else None,
                transfer_failure=None if med is None else bool(med > TRANSFER_FAILURE_MEDIAN_DB),
                rule=f"median |oracle - FEM| > {TRANSFER_FAILURE_MEDIAN_DB} dB = transfer failure (descriptive)")


def parameter_count(arch):
    """Number of parameters of the generator, from the architecture numbers: an input layer on noise and condition,
    a two-layer condition encoder, `n_blocks` FiLM blocks (layer norm, two linear layers, a scale-and-shift layer),
    an output layer norm and the output head."""
    w, n, c, o = arch["width"], arch["noise_dim"], arch["cond_dim"], arch["out_dim"]
    linear = lambda i, j: i * j + j
    block = 2 * w + linear(w, w) + linear(w, w) + linear(w, 2 * w)
    return linear(n + c, w) + linear(c, w) + linear(w, w) + arch["n_blocks"] * block + 2 * w + linear(w, o)


# =================================================================================================================
#  the checkpoint file, read without Torch
# =================================================================================================================
class CheckpointFormatError(RuntimeError):
    """The file is not a checkpoint of the expected plain form."""


def _rebuild_tensor(storage, offset, size, stride, *_):
    size, n = tuple(size), int(np.prod(size, dtype=np.int64))
    contiguous = tuple(int(np.prod(size[i + 1:], dtype=np.int64)) for i in range(len(size)))
    if tuple(stride) != contiguous:
        raise CheckpointFormatError("a tensor is not stored contiguously")
    return storage[offset:offset + n].reshape(size)


class _Reader(pickle.Unpickler):
    """Reads the checkpoint's pickle with four allowed names and nothing else; no project or Torch code runs."""
    ALLOWED = {("collections", "OrderedDict"): collections.OrderedDict,
               ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
               ("torch", "DoubleStorage"): "float64",
               ("torch.torch_version", "TorchVersion"): str}

    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise CheckpointFormatError(f"the file names {module}.{name}, which this reader does not allow")
        return self.ALLOWED[(module, name)]

    def persistent_load(self, pid):
        kind, dtype, key, _location, numel = pid
        if kind != "storage" or dtype != "float64":
            raise CheckpointFormatError("a stored block is not float64")
        data = np.frombuffer(self.archive.read(f"{self.prefix}/data/{key}"), dtype="<f8")
        if len(data) != numel:
            raise CheckpointFormatError("a stored block has the wrong length")
        return data


def read_checkpoint(path):
    with zipfile.ZipFile(path) as z:
        pkl = [n for n in z.namelist() if n.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise CheckpointFormatError("not a single-archive checkpoint")
        prefix = pkl[0][:-len("/data.pkl")]
        if f"{prefix}/byteorder" in z.namelist() and z.read(f"{prefix}/byteorder").strip() != b"little":
            raise CheckpointFormatError("the file is not little-endian")
        with z.open(pkl[0]) as f:
            return _Reader(f, z, prefix).load()


def describe_checkpoint(path):
    ck = read_checkpoint(path)
    sd, b = ck["model"], ck["binding"]
    h = hashlib.sha256()                      # the weight hash: sorted tensor names plus raw tensor bytes
    for k, v in sorted(sd.items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v).tobytes())
    return dict(weights_sha256=h.hexdigest(), n_tensors=len(sd), n_parameters=int(sum(v.size for v in sd.values())),
                dtypes=sorted({str(v.dtype) for v in sd.values()}), selected_epoch=int(ck["epoch"]),
                architecture_sha256=b["arch_sha256"], configuration_sha256=b["config_sha256"],
                source_sha256=b["source_sha256"], scorer_sha256=b["scorer_sha256"])


def verify_checkpoint(path, cm, out=print):
    ok = True

    def check(name, cond):
        nonlocal ok
        out(("PASS  " if cond else "FAIL  ") + name)
        ok &= bool(cond)

    check("file SHA-256 matches the manifest", sha256_file(path) == cm["asset"]["sha256"])
    check("file size matches the manifest", path.stat().st_size == cm["asset"]["size_bytes"])
    if not ok:
        sys.exit("stopping: the file is not the published asset")
    d, hb = describe_checkpoint(path), cm["historical_binding"]
    check("weight hash recomputed from the tensors matches the manifest",
          d["weights_sha256"] == cm["model"]["weights_sha256"])
    for k in ("n_tensors", "n_parameters", "dtypes", "selected_epoch"):
        check(f"{k} matches the manifest", d[k] == cm["model"][k])
    check("configuration_sha256 matches the manifest", d["configuration_sha256"] == cm["binding"]["configuration_sha256"])
    for k in ("architecture_sha256", "source_sha256", "scorer_sha256"):
        check(f"historical {k} recorded in the file matches the manifest", d[k] == hb[k])
    out("checkpoint verified" if ok else "checkpoint NOT verified")
    return ok


# =================================================================================================================
#  the verifier
# =================================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(prog="acoustic-liner verify public", description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name + (f"  ({detail})" if detail and not cond else ""))
        ok &= bool(cond)

    # 1. manifest and privacy (CONSISTENT) ------------------------------------------------------------------------
    man = load("manifest.json")
    summaries = man["historical_summaries"]
    shape = man["schema"] == SC.schema("evidence-manifest") and summaries["format"] == SC.ARCHIVED_SUMMARY_FORMAT
    for name, e in summaries["files"].items():
        p = EV / name
        check(f"CONSISTENT  {p.relative_to(ROOT).as_posix()} matches the SHA-256 and size in evidence/manifest.json",
              shape and p.is_file() and sha256_file(p) == e["sha256"] and p.stat().st_size == e["bytes"])
    cm = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    e = man["checkpoint_manifest"]
    check(f"CONSISTENT  {e['path']} matches the SHA-256 and size in evidence/manifest.json",
          (ROOT / e["path"]).resolve() == CHECKPOINT_MANIFEST.resolve() and sha256_file(CHECKPOINT_MANIFEST) == e["sha256"]
          and CHECKPOINT_MANIFEST.stat().st_size == e["bytes"]
          and e["schema"] == cm["schema"] == SC.schema("checkpoint-manifest"))
    for name, e in man.get("current_summaries", {}).items():
        p = EV / name
        check(f"CONSISTENT  evidence/{name} matches the SHA-256 and size in evidence/manifest.json",
              p.is_file() and sha256_file(p) == e["sha256"] and p.stat().st_size == e["bytes"])
    for p in sorted(EV.glob("*.json")):
        m = PRIVATE.search(p.read_text(encoding="utf-8"))
        check(f"{p.name} contains no local paths, user or host names", not m, m and m.group(0))

    # 2. final evaluation -----------------------------------------------------------------------------------------
    fe = load("linergen-final-evaluation.json")
    cols = fe["cells"]["columns"]
    cells = []
    for r in fe["cells"]["rows"]:
        c = {k: {} for k in "ABCD"} | {"proposals": {}}
        for k, v in zip(cols, r):
            if "." in k:
                g, f = k.split(".")
                c[g][f] = v
            else:
                c[k] = v
        cells.append(c)
    check("1,500 cells = 100 conditions x 5 seeds x 3 budgets",
          len(cells) == 1500 and len({c["cond_index"] for c in cells}) == 100)
    for b in fe["budgets"]:
        agg = fe["aggregate"][str(b)]
        for key, x, y, kx in (("D_minus_A", "D", "A", "best_val"), ("D_minus_B", "D", "B", "best_val"),
                              ("C32_minus_B32", "C", "B", "initial32_best_val")):
            r = contrast(cells, b, x, y, kx=kx, ky=kx)
            check(f"RECOMPUTED  {key} at {b}: mean {r['mean']:+.3f} dB [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] recomputed",
                  r == agg[key])
        cb = [c for c in cells if c["budget"] == b]
        dA, dB = diff_table(cells, b, "D", "A"), contrast(cells, b, "D", "B")
        wins = sum(np.mean(list(v.values())) > 0 for v in dA.values())
        seeds_pos = all(v > 0 for v in contrast(cells, b, "D", "A")["seed_means"].values()) and \
            all(v > 0 for v in dB["seed_means"].values())
        check(f"RECOMPUTED  budget {b}: LinerGen-seeded CMA-ES beats the baseline on {wins}/100 conditions; all seed means > 0",
              wins == 100 and seeds_pos)
        check(f"RECOMPUTED  budget {b}: raw proposals 100% feasible and unique; every winner feasible and within mass",
              all(c["proposals"]["raw_feasibility"] == 1.0 and c["proposals"]["unique_fraction"] == 1.0 for c in cb)
              and all(c[m]["feasible"] and c[m]["mass_ok"] for c in cb for m in "ABCD"))
        dc = diff_table(cells, b, "D", "C")
        inc = float(np.mean([np.mean(list(v.values())) for v in dc.values()]))
        print(f"INFO  budget {b}: CMA-ES increment over the 32 proposals (D - C) = {inc:+.5f} dB")

    # 3. checkpoint (RECOMPUTED parameter count; the rest CONSISTENT) -------------------------------------------------
    ck = fe["linergen_checkpoint"]
    hb = cm["historical_binding"]
    n = parameter_count(ck["architecture"])
    check(f"RECOMPUTED  parameter count {n:,} from the architecture (summary and checkpoint manifest agree)",
          n == ck["n_parameters"] == cm["model"]["n_parameters"])
    check("CONSISTENT  architecture and its recorded hash: summary = checkpoint manifest",
          ck["architecture"] == cm["model"]["architecture"] and ck["architecture_sha256"] == hb["architecture_sha256"])
    check("CONSISTENT  recorded LinerGen source hash: final-test record = checkpoint manifest",
          fe["checkpoint"]["source_sha256"] == hb["source_sha256"])
    check(f"CONSISTENT  selected epoch {ck['selected_epoch']}: summary = checkpoint manifest",
          ck["selected_epoch"] == cm["model"]["selected_epoch"])
    check("CONSISTENT  weight hash: summary = checkpoint manifest = final-test record",
          ck["weights_sha256"] == cm["model"]["weights_sha256"] == fe["checkpoint"]["model_sha256"])
    check("CONSISTENT  configuration hash: summary = checkpoint manifest",
          ck["configuration_sha256"] == cm["binding"]["configuration_sha256"])

    # 4. FEM-transfer audit -----------------------------------------------------------------------------------------
    fa = load("fem-transfer-audit.json")
    s = audit_summary(fa["rows"])
    check(f"RECOMPUTED  FEM audit: r {s['corr_oracle_fem']:.3f}, median |err| {s['median_abs_diff_dB']:.3f}, bias "
          f"{s['mean_diff_dB']:+.3f}, max {s['max_abs_diff_dB']:.3f} dB recomputed from {len(fa['rows'])} pairs",
          s == fa["summary"] and len(fa["rows"]) == 20)
    check("RECOMPUTED  FEM audit: every diff = oracle - FEM", all(r["diff"] == r["oracle"] - r["fem"] for r in fa["rows"]))

    # 5. vehicles ---------------------------------------------------------------------------------------------------
    for f in ("falcon9-vehicle.json", "atlas-i-vehicle.json"):
        v = load(f)
        for vd in v["verdicts"]:
            iv, T = vd["interval"], vd["target_db"]
            expect = "UNRESOLVED" if iv is None else ("PASS" if iv[0] > T else "FAIL" if iv[1] < T else "UNRESOLVED")
            check(f"RECOMPUTED  {v['scenario_name']} {vd['stage']}: {vd['outcome']} follows the interval rule (T = {T} dB)",
                  T == v["target_dB"] and vd["outcome"].endswith(expect))
        bb = [x for case in v["cfd_corrected_fem"]["primary"].values() for x in case["broadband"].values()]
        print(f"INFO  {v['scenario_name']}: CFD-corrected FEM broadband {min(bb):.3f}-{max(bb):.3f} dB (not a verdict)")

    # 5b. post-hoc follow-ups ----------------------------------------------------------------------------------------
    ph = load("posthoc-followups.json")
    check("CONSISTENT  posthoc-followups.json carries the public schema", ph["schema"] == SC.schema("posthoc-followups"))
    for st in ph["stages"]:
        tag = f"post-hoc {st['stage']} ({st['vehicle']})"
        if st["stage"] == "resolution_followup" and st["vehicle"] == "atlas":
            fem = st["fem_primary"]
            vals = [{h: float(v) for h, v in fem[t][f].items()} for t in fem for f in fem[t]]
            fine = min(vals[0], key=float)
            f01, f01_hi = min(v[fine] for v in vals), max(v[fine] for v in vals)
            m = max(abs(v[h] - v[fine]) for v in vals for h in v if h != fine)
            sens = st["sensitivity"]
            sv = [sens["fem_nominal_0p1"]] + [c["fem_robust_db"] for c in sens["fem_envelope_combos_0p1"]
                                              if c["fem_robust_db"] is not None]
            s_down, s_up = max(0.0, sens["fem_nominal_0p1"] - min(sv)), max(0.0, max(sv) - sens["fem_nominal_0p1"])
            L, U, iv = f01 - m - s_down, f01_hi + m + s_up, st["interval"]
            check(f"RECOMPUTED  {tag}: L {L:.4f}, U {U:.4f} dB recomputed from {len(vals)} FEM values and the allowances",
                  len(vals) == 10 and abs(L - iv["L"]) < 1e-12 and abs(U - iv["U"]) < 1e-12)
            complete = (not st["reasons"] and not st["lost_fixed_point"] and not sens["escalation_triggers"]
                        and all(r["accepted"] for r in st["replaced_rows"].values()) and st["selection"]["match"])
            expect = "PASS" if complete and L > st["target_db"] else (
                "FAIL" if complete and U < st["target_db"] else "UNRESOLVED")
            check(f"RECOMPUTED  {tag}: {st['outcome']} follows the interval rule (T = {st['target_db']} dB)", st["outcome"] == expect)
        elif st["stage"] == "highspeed_pair":
            acc = {k: all(r["gates"].values()) and r["bound"] for k, r in st["runs"].items()}
            check(f"CONSISTENT  {tag}: acceptance follows each run's gates; {st['outcome']} because the pair is "
                  f"{'accepted' if all(acc.values()) else 'not accepted'}",
                  all(acc[k] == r["accepted"] for k, r in st["runs"].items())
                  and st["pair_accepted"] == all(acc.values()) and (st["outcome"] == "UNRESOLVED" or all(acc.values())))
        elif st["stage"] == "bounded_sensitivity":
            incomplete = any(c["fem_robust_db"] is None for c in st["sensitivity"]["fem_envelope_combos_0p1"])
            check(f"CONSISTENT  {tag}: {st['label']} (fixed-point loss: {incomplete}); formal verdict {st['outcome']}",
                  st["outcome"] == "UNRESOLVED" and (st["label"] == "NOT BOUNDED") == (incomplete or bool(st["reasons"])))
        else:
            check(f"CONSISTENT  {tag}: diagnostic only, verdict {st['outcome']}", st["outcome"] == "UNRESOLVED")

    # 6. figures ----------------------------------------------------------------------------------------------------
    listed = {x["path"]: x["sha256"] for x in load("readme-figures.json")["figures"]}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    used = set(re.findall(r'(?:\(|src="|href=")(docs/figures/[^)"\s]+\.(?:png|gif))', readme))
    check("CONSISTENT  every README figure is listed in a figure manifest", used <= set(listed), sorted(used - set(listed)))
    check("CONSISTENT  every listed figure matches its SHA-256",
          all((ROOT / p).is_file() and sha256_file(ROOT / p) == h for p, h in listed.items()))

    # 7. checkpoint file, if downloaded -----------------------------------------------------------------------------
    ckp = ROOT / cm["asset"]["save_as"]
    if ckp.is_file():
        check("downloaded checkpoint (read without Torch)", verify_checkpoint(ckp, cm))
    else:
        print(f"SKIP  checkpoint file not downloaded ({cm['asset']['save_as']}); see README")

    print("public verification passed" if ok else "public verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
