"""HireSense AI — synthetic resume generator.

    resume = render(template, profile, identity)

* profile  — the career content (skills, experience, education). It must NOT contain
             any identity field. This is what lets us render counterfactual pairs:
             same profile, different identity (docs/02-dataset-strategy.md §6.1).
* identity — given name, family name, prefix, pronouns. The e-mail address is
             DERIVED from the name, never hand-written, so it changes with the name.
* template — the visual layout (T1 ... T5). T4–T5 are reserved for the evaluation
             set and are refused for any other role (§5.1).

Because we know exactly what went into each resume, the generator also writes the
ground-truth label for it. That label is only trustworthy if the profile is honest,
so the generator validates every profile before rendering (see `validate_profile`):
a skill may only be claimed for a bullet if that bullet's text actually mentions it.

Usage (from the project root):
    .venv/Scripts/python.exe -I -X utf8 tools/resumegen.py dataset/smoke/build.yaml
    .venv/Scripts/python.exe -I -X utf8 tools/resumegen.py dataset/smoke/build.yaml --verify-determinism
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import tempfile
import time
from datetime import date
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TAXONOMY_PATH = DATASET / "taxonomy" / "skills.v0.yaml"
IDENTITIES_PATH = DATASET / "identities" / "identities.yaml"
MANIFEST_PATH = DATASET / "manifest.csv"

MANIFEST_COLUMNS = [
    "file_id", "sha256", "tier", "template", "source", "license", "added_at",
    "profile_id", "counterfactual_of", "role", "renderer",
]

# Fields that belong to an identity. Finding any of them inside a profile is an error.
IDENTITY_FIELDS = {"name", "given_name", "family_name", "full_name", "prefix", "pronouns", "email"}

# Which templates each role may use. T4–T5 never leak into development (§5.1).
ALLOWED_TEMPLATES = {
    "smoke": {"T1", "T2", "T3"},
    "dev": {"T1", "T2", "T3"},
    "eval": {"T4", "T5"},
}

DEGREE_LEVELS = {"high_school", "vocational", "bachelor", "master", "phd"}


class ProfileError(ValueError):
    """A profile would produce a dishonest or malformed ground-truth label."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_taxonomy() -> tuple[int, dict[str, dict]]:
    data = load_yaml(TAXONOMY_PATH)
    return data["taxonomy_version"], {entry["id"]: entry for entry in data["skills"]}


def surface_forms(entry: dict) -> list[str]:
    """Every way a skill may be written: canonical name, aliases and context_aliases.

    context_aliases (e.g. "excel", which is also an English verb) are included on purpose.
    This function is used to check that a LABEL is honest, and the profile author already
    knows the word means the skill. Deciding from context whether a bare "excel" in an
    unknown resume is a skill is the parser's job (Phase 2), not the generator's.
    """
    forms = [entry["canonical"], *entry.get("aliases", []), *entry.get("context_aliases", [])]
    return sorted({f.lower() for f in forms}, key=len, reverse=True)


def mentions(text: str, entry: dict) -> bool:
    """True if `text` contains the skill as a whole word (case-insensitive)."""
    lowered = text.lower()
    for form in surface_forms(entry):
        if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", lowered):
            return True
    return False


# ---------------------------------------------------------------------------
# Validation — this is what makes the generated label trustworthy
# ---------------------------------------------------------------------------

def _find_identity_fields(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else key
            if key in IDENTITY_FIELDS:
                hits.append(here)
            hits.extend(_find_identity_fields(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(_find_identity_fields(value, f"{path}[{i}]"))
    return hits


def validate_profile(profile: dict, taxonomy: dict[str, dict]) -> None:
    pid = profile.get("profile_id", "<missing profile_id>")
    errors: list[str] = []

    leaked = _find_identity_fields(profile)
    if leaked:
        errors.append(f"identity fields must not be inside a profile: {leaked}")

    def check_ids(ids: list[str], where: str) -> None:
        for sid in ids:
            if sid not in taxonomy:
                errors.append(f"{where}: unknown skill id {sid!r}")

    # Skills list: each entry's display text must actually be a way of writing that skill.
    # An entry WITHOUT an id is an out-of-taxonomy skill (e.g. "IFRS"): allowed on purpose,
    # because real resumes are full of them; it is recorded in the label as `oov_skills`.
    for item in profile.get("skills", []):
        if item.get("id") is None:
            continue
        check_ids([item["id"]], "skills")
        if item["id"] in taxonomy and not mentions(item["text"], taxonomy[item["id"]]):
            errors.append(f"skills: text {item['text']!r} is not a surface form of {item['id']!r}")

    # Bullets and projects: a skill may only be claimed if the sentence mentions it.
    claimed_in_use: set[str] = set()
    for job in profile.get("experience", []):
        for bullet in job.get("bullets", []):
            check_ids(bullet.get("skills", []), "experience")
            for sid in bullet.get("skills", []):
                if sid in taxonomy and not mentions(bullet["text"], taxonomy[sid]):
                    errors.append(f"experience: bullet claims {sid!r} but its text does not mention it: {bullet['text']!r}")
                claimed_in_use.add(sid)
    for project in profile.get("projects", []):
        check_ids(project.get("skills", []), "projects")
        for sid in project.get("skills", []):
            if sid in taxonomy and not mentions(project["text"], taxonomy[sid]):
                errors.append(f"projects: {project['name']!r} claims {sid!r} but its text does not mention it")
            claimed_in_use.add(sid)

    # Learning goals: mentioned, but NOT held. Must not overlap with anything held.
    held = {item["id"] for item in profile.get("skills", []) if item.get("id")} | claimed_in_use
    for goal in profile.get("learning", []):
        check_ids(goal.get("not_held", []), "learning")
        for sid in goal.get("not_held", []):
            if sid in held:
                errors.append(f"learning: {sid!r} is listed as not held, but the profile also claims it")
            if sid in taxonomy and not mentions(goal["text"], taxonomy[sid]):
                errors.append(f"learning: text does not mention {sid!r}: {goal['text']!r}")

    for edu in profile.get("education", []):
        if edu.get("degree") not in DEGREE_LEVELS:
            errors.append(f"education: degree {edu.get('degree')!r} not in {sorted(DEGREE_LEVELS)}")

    # Reverse direction: every skill the visible text mentions must be declared somewhere.
    # Otherwise the label silently misses a skill the resume DOES state, and a parser that
    # extracts it correctly would be scored as a false positive.
    # A profile may waive a specific hit with `ignore_mentions: [{id, reason}]` — e.g. the
    # alias "rest" matching "the rest of the team".
    ignored = {m["id"] for m in profile.get("ignore_mentions", [])}
    visible: list[tuple[str, str]] = [("summary", profile.get("summary", ""))]
    visible += [("skills", item["text"]) for item in profile.get("skills", [])]
    visible += [("experience", b["text"]) for j in profile.get("experience", []) for b in j.get("bullets", [])]
    visible += [("projects", pr["text"]) for pr in profile.get("projects", [])]
    for where, text in visible:
        for sid, entry in taxonomy.items():
            if sid in held or sid in ignored:
                continue
            if mentions(text, entry):
                errors.append(f"{where}: text mentions {sid!r} but no part of the profile declares it: {text!r}")

    if errors:
        raise ProfileError(f"{pid}:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def email_for(identity: dict) -> str:
    """firstname.l@example.com — derived, so it always changes together with the name."""
    given, family = identity["given_name"], identity["family_name"]
    if not (given.isascii() and family.isascii()):
        raise ValueError(f"{identity['identity_id']}: names must be ASCII to derive an e-mail")
    local = re.sub(r"[^a-z]", "", given.lower()) + "." + family[0].lower()
    return f"{local}@example.com"


def display_name(identity: dict) -> str:
    parts = [identity.get("prefix"), identity["given_name"], identity["family_name"]]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Label — derived from the profile, in the shape of docs/03 §8.1
# ---------------------------------------------------------------------------

def derive_label(resume_id: str, profile: dict, identity: dict, taxonomy_version: int) -> dict:
    listed = [item["id"] for item in profile.get("skills", []) if item.get("id")]
    oov = [item["text"] for item in profile.get("skills", []) if not item.get("id")]
    in_use: list[str] = []
    for job in profile.get("experience", []):
        for bullet in job.get("bullets", []):
            in_use.extend(bullet.get("skills", []))
    for project in profile.get("projects", []):
        in_use.extend(project.get("skills", []))

    stated = sorted(set(listed) | set(in_use))
    listed_only = sorted(set(listed) - set(in_use))
    not_held = sorted({sid for goal in profile.get("learning", []) for sid in goal.get("not_held", [])})

    return {
        "resume_id": resume_id,
        "profile_id": profile["profile_id"],
        "identity_id": identity["identity_id"],
        "label_source": "generator",          # NOT human-verified — see docs/02 §4.1
        "taxonomy_version": taxonomy_version,
        "contact": {
            "name": display_name(identity),
            "email": email_for(identity),
            "phone": profile.get("phone"),
        },
        "skills_stated": stated,
        "skills_listed_only": listed_only,
        "mentioned_not_held": not_held,       # e.g. "eager to learn X" — must NOT be extracted
        "oov_skills": oov,                    # skills the taxonomy does not know
        "education": [
            {k: e.get(k) for k in ("degree", "field", "institution", "end_year")}
            for e in profile.get("education", [])
        ],
        "experience": [
            {k: (str(j[k]) if j.get(k) is not None else None) for k in ("title", "employer", "start", "end")}
            for j in profile.get("experience", [])
        ],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def month_label(value: Any) -> str:
    if value is None:
        return "Present"
    year, month = str(value).split("-")
    return date(int(year), int(month), 1).strftime("%b %Y")


def make_env() -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), undefined=StrictUndefined, autoescape=True)
    env.filters["month"] = month_label
    return env


OPTIONAL_LIST_FIELDS = ("learning", "projects", "experience", "education", "skills")


def render_html(env: Environment, template: str, profile: dict, identity: dict) -> str:
    # Templates use StrictUndefined so a misspelt field fails loudly instead of rendering
    # blank. Optional list sections therefore get an explicit empty default here.
    p = {field: [] for field in OPTIONAL_LIST_FIELDS} | profile
    return env.get_template(f"{template}.html.j2").render(
        name=display_name(identity),
        pronouns=identity.get("pronouns"),
        email=email_for(identity),
        p=p,
    )


FIXED_PDF_DATE = b"20000101000000"   # 14 digits, same length as the real YYYYMMDDHHMMSS
_PDF_DATE = re.compile(rb"(/(?:CreationDate|ModDate) ?\(D:)(\d{14})")


def normalize_pdf_timestamps(path: Path) -> None:
    """Replace the print timestamps Chromium embeds, so the same input gives the same bytes.

    Without this, re-rendering an unchanged resume one second later changes its sha256,
    and the manifest could no longer tell "same file" from "different file".
    The replacement has exactly the same length, so the PDF's byte offsets (xref table)
    stay valid. Edge writes the dates only in the Info dictionary, not in XMP
    (checked 2026-11-24, Edge 157).
    """
    data = path.read_bytes()
    fixed, count = _PDF_DATE.subn(lambda m: m.group(1) + FIXED_PDF_DATE, data)
    if count == 0:
        raise RuntimeError(f"{path.name}: no PDF timestamp found — did the renderer change?")
    assert len(fixed) == len(data)
    path.write_bytes(fixed)


def print_pdf(page: Page, html: str, out: Path) -> None:
    page.set_content(html)
    page.pdf(path=str(out), format="A4", print_background=True)
    normalize_pdf_timestamps(out)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["file_id"])
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_file", type=Path, help="YAML that maps resume_id -> profile x identity x template")
    parser.add_argument("--verify-determinism", action="store_true",
                        help="render every resume twice and report whether the bytes are identical")
    args = parser.parse_args()

    build_path = (ROOT / args.build_file).resolve() if not args.build_file.is_absolute() else args.build_file
    build = load_yaml(build_path)
    base = build_path.parent
    role, tier = build["role"], build.get("tier") or ""
    profile_dir, pdf_dir, label_dir = base / build["profiles_dir"], base / build["pdf_dir"], base / build["labels_dir"]
    pdf_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_version, taxonomy = load_taxonomy()
    identities = {i["identity_id"]: i for i in load_yaml(IDENTITIES_PATH)["identities"]}

    # Validate EVERYTHING before rendering anything.
    plan = []
    for entry in build["resumes"]:
        if entry["template"] not in ALLOWED_TEMPLATES[role]:
            raise SystemExit(f"{entry['resume_id']}: template {entry['template']} is not allowed for role {role!r}")
        profile = load_yaml(profile_dir / f"{entry['profile']}.yaml")
        validate_profile(profile, taxonomy)
        identity = identities[entry["identity"]]
        email_for(identity)  # fails early on non-ASCII names
        plan.append((entry, profile, identity))
    print(f"validated {len(plan)} profiles against taxonomy v{taxonomy_version}")

    env = make_env()
    manifest = {row["file_id"]: row for row in read_manifest()}
    nondeterministic: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        renderer = f"playwright-{package_version('playwright')}/msedge-{browser.version}"
        page = browser.new_page()

        for entry, profile, identity in plan:
            rid = entry["resume_id"]
            html = render_html(env, entry["template"], profile, identity)
            pdf_path = pdf_dir / f"{rid}.pdf"
            print_pdf(page, html, pdf_path)
            digest = sha256_of(pdf_path)

            if args.verify_determinism:
                # Wait past the next second: PDF timestamps have 1-second resolution, so two
                # renders inside the same second would match by coincidence and hide the problem.
                time.sleep(1.1)
                with tempfile.TemporaryDirectory() as tmp:
                    again = Path(tmp) / "again.pdf"
                    print_pdf(page, html, again)
                    if sha256_of(again) != digest:
                        nondeterministic.append(rid)

            label = derive_label(rid, profile, identity, taxonomy_version)
            (label_dir / f"{rid}.yaml").write_text(
                yaml.safe_dump(label, sort_keys=False, allow_unicode=True), encoding="utf-8")

            manifest[rid] = {
                "file_id": rid, "sha256": digest, "tier": tier, "template": entry["template"],
                "source": "synthetic", "license": "internal", "added_at": date.today().isoformat(),
                "profile_id": profile["profile_id"], "counterfactual_of": entry.get("counterfactual_of") or "",
                "role": role, "renderer": renderer,
            }
            print(f"  {rid}  {entry['template']}  {profile['profile_id']} x {identity['identity_id']}  "
                  f"{pdf_path.stat().st_size:>6} bytes  sha256={digest[:12]}")
        browser.close()

    write_manifest(list(manifest.values()))
    print(f"manifest: {MANIFEST_PATH.relative_to(ROOT)} ({len(manifest)} rows)")
    if args.verify_determinism:
        if nondeterministic:
            print(f"NOT deterministic: re-rendering changed the bytes of {nondeterministic}")
        else:
            print("deterministic: every resume rendered twice produced identical bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
