import argparse
import csv
import json
import re

# This file converts new_benchmark.txt into a .jsonl file

# Regex to find patient headers like: "Patient 10001234"
PATIENT_HEADER_RE = re.compile(r"(?m)^(?:\s*)Patient\s+(\d{8})\b")

# This creates a map of the leading digit -> similarity score relative to 1xxxxxxx anchor
SIM_MAP = {
    "2": 1.0,
    "3": 0.75,
    "4": 0.5,
    "5": 0.25,
    "6": 0.0,
}

def parse_patients(txt):
    patients = {}

    matches = list(PATIENT_HEADER_RE.finditer(txt))
    if not matches:
        return patients

    for i, m in enumerate(matches):
        pid = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        block = txt[start:end].strip()
        patients[pid] = block

    return patients


def build_benchmark_pairs(patients):
    records = []

    # Precompute set for existence checks
    available_ids = set(patients.keys())

    for pid in sorted(patients.keys()):
        if not pid.startswith("1"):
            continue

        suffix = pid[1:] 
        anchor_text = patients[pid]

        for lead, sim in SIM_MAP.items():
            sib_id = lead + suffix
            if sib_id in available_ids:
                pair_text = patients[sib_id]
                records.append({
                    "anchor_id": pid,
                    "pair_id": sib_id,
                    "similarity": sim,
                    "anchor_text": anchor_text,
                    "pair_text": pair_text,
                })

    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(path, records):

    fieldnames = ["anchor_id", "pair_id", "similarity"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "anchor_id": r["anchor_id"],
                "pair_id": r["pair_id"],
                "similarity": r["similarity"],
            })


def main():
    ap = argparse.ArgumentParser(description="Build benchmark pairs from patient .txt")
    ap.add_argument("input_txt", help="Path to input .txt file")
    ap.add_argument("--out-jsonl", default="benchmark_pairs.jsonl",
                    help="Output JSONL path (default: benchmark_pairs.jsonl)")
    ap.add_argument("--out-csv", default="benchmark_pairs.csv",
                    help="Output CSV path (default: benchmark_pairs.csv)")
    ap.add_argument("--include-texts-in-jsonl", action="store_true",
                    help="Include full patient blocks in JSONL (default on)")
    ap.add_argument("--no-include-texts-jsonl", action="store_true",
                    help="Strip patient texts from JSONL payloads")
    args = ap.parse_args()

    with open(args.input_txt, "r", encoding="utf-8") as f:
        raw = f.read()

    patients = parse_patients(raw)
    if not patients:
        raise SystemExit("No patients found. Check the input formatting (lines should start with 'Patient <8 digits>').")

    records = build_benchmark_pairs(patients)

    write_jsonl(args.out_jsonl, records)

    write_csv(args.out_csv, records)
    print(f"Parsed patients: {len(patients)}")


if __name__ == "__main__":
    main()
