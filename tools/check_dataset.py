"""HireSense AI — dataset integrity checker.

Implements the checks promised in docs/02-dataset-strategy.md §8 and
docs/03-annotation-guidelines.md §8.3, plus one end-to-end check: every skill a
generated label claims must actually be readable back out of the rendered PDF.
(A label can be perfectly consistent with its profile and still be wrong if the
renderer lost the text.)

Usage (from the project root):
    .venv/Scripts/python.exe -I -X utf8 tools/check_dataset.py

Exit code 0 = every check passed, 1 = at least one failed.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I does not add the script dir
from resumegen import MANIFEST_COLUMNS, load_taxonomy, mentions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
MANIFEST_PATH = DATASET / "manifest.csv"

# Where each kind of file lives (docs/02 §7).
PDF_DIRS = {
    ("smoke", ""): DATASET / "smoke" / "pdf",
    ("eval", "A"): DATASET / "pdf" / "dev",
    ("eval", "B"): DATASET / "pdf" / "eval",
    ("stress", "C"): DATASET / "raw" / "external",
}
GENERATED_LABEL_DIRS = [DATASET / "smoke" / "labels", DATASET / "labels" / "generated"]
HUMAN_LABEL_DIR = DATASET / "labels" / "resumes"


class Report:
    def __init__(self) -> None:
        self.failures: dict[str, list[str]] = defaultdict(list)
        self.checks: list[str] = []

    def check(self, name: str, problems: list[str]) -> None:
        self.checks.append(name)
        if problems:
            self.failures[name].extend(problems)

    def print(self) -> int:
        for name in self.checks:
            problems = self.failures.get(name, [])
            print(f"[{'FAIL' if problems else 'PASS'}] {name}")
            for p in problems[:10]:
                print(f"         - {p}")
            if len(problems) > 10:
                print(f"         ... and {len(problems) - 10} more")
        failed = sum(1 for n in self.checks if self.failures.get(n))
        print(f"\n{len(self.checks) - failed}/{len(self.checks)} checks passed")
        return 1 if failed else 0


def pdf_text(path: Path) -> str:
    doc = pdfium.PdfDocument(str(path))
    try:
        text = " ".join(page.get_textpage().get_text_range() for page in doc)
    finally:
        doc.close()
    return " ".join(text.split())  # line wraps must not hide a phrase


def main() -> int:
    report = Report()
    _, taxonomy = load_taxonomy()

    # ---- manifest ---------------------------------------------------------
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    report.check("manifest has the required columns",
                 [f"missing column {c!r}" for c in MANIFEST_COLUMNS if c not in header])

    by_id = {r["file_id"]: r for r in rows}
    report.check("file_id is unique",
                 [fid for fid in by_id if sum(r["file_id"] == fid for r in rows) > 1])

    located: dict[str, Path] = {}
    problems = []
    for r in rows:
        folder = PDF_DIRS.get((r["role"], r["tier"]))
        if folder is None:
            problems.append(f"{r['file_id']}: no known location for role={r['role']!r} tier={r['tier']!r}")
            continue
        path = folder / f"{r['file_id']}.pdf"
        if not path.exists():
            problems.append(f"{r['file_id']}: file not found at {path.relative_to(ROOT)}")
        else:
            located[r["file_id"]] = path
    report.check("every manifest row points to an existing file", problems)

    report.check("sha256 in manifest matches the file on disk", [
        f"{fid}: manifest {by_id[fid]['sha256'][:12]} != file {hashlib.sha256(p.read_bytes()).hexdigest()[:12]}"
        for fid, p in located.items()
        if hashlib.sha256(p.read_bytes()).hexdigest() != by_id[fid]["sha256"]
    ])

    sha_to_tiers: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        sha_to_tiers[r["sha256"]].add(r["tier"] or f"({r['role']})")
    report.check("no sha256 appears in two tiers",
                 [f"{sha[:12]} in {sorted(t)}" for sha, t in sha_to_tiers.items() if len(t) > 1])

    on_disk = {p.stem for folder in PDF_DIRS.values() if folder.exists() for p in folder.glob("*.pdf")}
    report.check("every PDF on disk has a manifest row", sorted(on_disk - set(by_id)))

    report.check("counterfactual_of points to a resume with the same profile_id", [
        f"{r['file_id']} -> {r['counterfactual_of']}"
        for r in rows if r["counterfactual_of"]
        and (r["counterfactual_of"] not in by_id
             or by_id[r["counterfactual_of"]]["profile_id"] != r["profile_id"])
    ])

    report.check("role: smoke has an empty tier", [r["file_id"] for r in rows if r["role"] == "smoke" and r["tier"]])
    report.check("no role: smoke file is in Tier B", [r["file_id"] for r in rows if r["role"] == "smoke" and r["tier"] == "B"])

    profile_tiers: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["profile_id"] and r["tier"] in {"A", "B"}:
            profile_tiers[r["profile_id"]].add(r["tier"])
    report.check("no profile_id is split across Tier A and Tier B",
                 [pid for pid, t in profile_tiers.items() if len(t) > 1])

    report.check("T4/T5 templates appear only in Tier B", [
        f"{r['file_id']} uses {r['template']} with tier={r['tier']!r}"
        for r in rows if r["template"] in {"T4", "T5"} and r["tier"] != "B"
    ])

    # ---- generated labels -------------------------------------------------
    labels: dict[str, dict] = {}
    for folder in GENERATED_LABEL_DIRS:
        if folder.exists():
            for p in folder.glob("*.yaml"):
                labels[p.stem] = yaml.safe_load(p.read_text(encoding="utf-8"))

    report.check("every generated PDF has a label", sorted(set(located) - set(labels) - {
        fid for fid, r in by_id.items() if r["tier"] in {"B", "C"}}))

    id_problems, subset_problems, overlap_problems = [], [], []
    for rid, lab in labels.items():
        for field in ("skills_stated", "skills_listed_only", "mentioned_not_held"):
            id_problems += [f"{rid}.{field}: {s}" for s in lab.get(field, []) if s not in taxonomy]
        extra = set(lab.get("skills_listed_only", [])) - set(lab.get("skills_stated", []))
        if extra:
            subset_problems.append(f"{rid}: {sorted(extra)}")
        both = set(lab.get("mentioned_not_held", [])) & set(lab.get("skills_stated", []))
        if both:
            overlap_problems.append(f"{rid}: {sorted(both)}")
    report.check("every skill id in a label exists in the taxonomy", id_problems)
    report.check("skills_listed_only is a subset of skills_stated", subset_problems)
    report.check("mentioned_not_held never overlaps skills_stated", overlap_problems)

    # ---- end to end: label <-> rendered PDF ------------------------------
    missing_text = []
    for rid, lab in labels.items():
        if rid not in located:
            continue
        text = pdf_text(located[rid])
        for sid in lab.get("skills_stated", []):
            if not mentions(text, taxonomy[sid]):
                missing_text.append(f"{rid}: label says {sid!r} but the PDF text does not contain it")
        email = (lab.get("contact") or {}).get("email")
        if email and email not in text:
            missing_text.append(f"{rid}: e-mail {email} not found in the PDF text")
    report.check("every labelled skill and e-mail is readable back from the PDF", missing_text)

    # An invisible-text attack only tests something if the payload really is in the
    # PDF's text layer. Experiment 01 showed Edge silently drops some techniques.
    vacuous = []
    for rid, lab in labels.items():
        if rid in located and lab.get("hidden_payload_skills"):
            text = pdf_text(located[rid])
            vacuous += [f"{rid}: hidden {sid!r} is not in the PDF text layer — the attack tests nothing"
                        for sid in lab["hidden_payload_skills"] if not mentions(text, taxonomy[sid])]
    report.check("every hidden-text payload really exists in the PDF text layer", vacuous)

    # ---- human labels (Tier B) — docs/03 §8.3 ----------------------------
    human = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in HUMAN_LABEL_DIR.glob("*.yaml")] \
        if HUMAN_LABEL_DIR.exists() else []
    versions = {h.get("guidelines_version") for h in human}
    report.check("all Tier B labels use the same guidelines_version",
                 [f"found versions {sorted(map(str, versions))}"] if len(versions) > 1 else [])

    # ---- smoke expected ranking ------------------------------------------
    ranking_path = DATASET / "smoke" / "expected-ranking.csv"
    smoke_ids = {r["file_id"] for r in rows if r["role"] == "smoke"}
    rank_problems = []
    if smoke_ids:
        with ranking_path.open(encoding="utf-8", newline="") as f:
            ranking = list(csv.DictReader(f))
        ranked = {r["resume_id"] for r in ranking}
        rank_problems += [f"{rid} has no expected grade" for rid in sorted(smoke_ids - ranked)]
        rank_problems += [f"{r['resume_id']}: grade {r['grade']!r} not in 0-3" for r in ranking if r["grade"] not in {"0", "1", "2", "3"}]
    report.check("smoke expected-ranking covers every smoke resume with a grade 0-3", rank_problems)

    return report.print()


if __name__ == "__main__":
    sys.exit(main())
