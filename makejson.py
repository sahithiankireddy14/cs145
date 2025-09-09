import re, json, argparse
# This file is deprecated

def clean(s):
    return re.sub(r'\s+', ' ', s.strip())

def parse_bullets(section):
    out = []
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("- "):
            out.append(clean(line[2:]))
    return out

def parse_drgs(section):
    drgs = {}
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("- "):
            body = line[2:]
            if ":" in body:
                k, v = body.split(":", 1)
                drgs[clean(k)] = clean(v)
    return drgs

def parse_orders(section):
    orders = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        line = line[2:]
        if ":" in line:
            name, rest = line.split(":", 1)
            parts = [clean(p) for p in rest.split(",") if clean(p)]
        else:
            name, parts = line, []

        route = None
        notes_segments = []
        for p in parts:
            # search all
            if re.search(r'\b(PO|IV|SC|IM|IH|IN|NG|TD|Q\d+H|q\d+h|bolus|infusion|Drip|NC|HFNC|PRN|HS|BID|TID|QAM|QPM)\b', p, re.I):
                if route is None:
                    route = p
                else:
                    notes_segments.append(p)
            else:
                notes_segments.append(p)
        notes = ", ".join(notes_segments) if notes_segments else None

        obj = {"name": clean(name)}
        if route: obj["route"] = route
        if notes: obj["notes"] = notes
        orders.append(obj)
    return orders

def extract_section(block, header, stop_markers):
    pattern = re.compile(
        rf"{re.escape(header)}\s*(?:\n|\r\n)?(.*?)(?=(" + "|".join([re.escape(m) for m in stop_markers]) + r")|$)",
        re.DOTALL | re.IGNORECASE
    )
    m = re.search(pattern, block)
    return m.group(1).strip() if m else ""

def parse(text):
    text = text.replace("\r\n", "\n")

    patients = []

    # Find all patient
    p_iter = list(re.finditer(r"(?m)^Patient\s+(\d+)\s*$", text))

    for i, pm in enumerate(p_iter):
        pid = pm.group(1)
        pstart = pm.end()
        pend = p_iter[i+1].start() if i + 1 < len(p_iter) else len(text)
        pblock = text[pstart:pend]

        # get demographics info
        gender_m = re.search(r"(?m)^Gender:\s*(.+)$", pblock)
        age_m    = re.search(r"(?m)^Age:\s*(\d+)", pblock)
        race_m   = re.search(r"(?m)^Race:\s*(.+)$", pblock)

        patient_obj = {
            "patient_id": pid,
            "gender": clean(gender_m.group(1)) if gender_m else None,
            "age": int(age_m.group(1)) if age_m else None,
            "race": clean(race_m.group(1)) if race_m else None,
            "admissions": []
        }

        # admission sections
        a_iter = list(re.finditer(r"(?m)^Admission\s+(\d+)\s*$", pblock))
        for j, am in enumerate(a_iter):
            aid = am.group(1)
            astart = am.end()
            aend = a_iter[j+1].start() if j + 1 < len(a_iter) else len(pblock)
            ablock = pblock[astart:aend]

            # sections
            # stop markers
            drg_text   = extract_section(ablock, "DRGs:", ["\n    Medications:", "\n\tMedications:", "\nMedications:", "\n    Orders:", "\nOrders:", "\n    Procedures:", "\nProcedures:", "\n---", "\nAdmission", "\nPatient"])
            meds_text  = extract_section(ablock, "Medications:", ["\n    Orders:", "\nOrders:", "\n    Procedures:", "\nProcedures:", "\n---", "\nAdmission", "\nPatient"])
            orders_text= extract_section(ablock, "Orders:", ["\n    Procedures:", "\nProcedures:", "\n---", "\nAdmission", "\nPatient"])
            procs_text = extract_section(ablock, "Procedures:", ["\n---", "\nAdmission", "\nPatient"])

            admission = {
                "admission_id": aid,
                "drgs":        parse_drgs(drg_text),
                "medications": parse_bullets(meds_text),
                "orders":      parse_orders(orders_text),
                "procedures":  parse_bullets(procs_text)
            }
            patient_obj["admissions"].append(admission)

        patients.append(patient_obj)

    return {"patients": patients}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="Path to the raw text (your big pasted block).")
    ap.add_argument("-o", "--outfile", default="patients.json", help="Where to write JSON.")
    args = ap.parse_args()

    raw = open(args.infile, "r", encoding="utf-8").read()
    data = parse(raw)
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.outfile} with {len(data['patients'])} patients.")

if __name__ == "__main__":
    main()
