"""One-off script: analyse prescription coverage by drug class."""
import pandas as pd
import re
import sys

DRUG_CLASSES = {
    "antibiotic": [
        r"vancomycin", r"piperacillin", r"ceftriaxone", r"cefazolin", r"cefepime",
        r"meropenem", r"imipenem", r"azithromycin", r"levofloxacin", r"ciprofloxacin",
        r"metronidazole", r"ampicillin", r"clindamycin", r"doxycycline",
        r"trimethoprim", r"sulfamethoxazole", r"linezolid", r"daptomycin",
        r"nitrofurantoin", r"gentamicin", r"tobramycin", r"amikacin", r"rifampin",
        r"fluconazole", r"micafungin", r"caspofungin", r"anidulafungin",
        r"cephalexin", r"cefdinir", r"cefpodoxime", r"cefuroxime",
    ],
    "anticoagulant": [r"heparin", r"enoxaparin", r"warfarin", r"apixaban",
                      r"rivaroxaban", r"fondaparinux", r"dabigatran", r"argatroban"],
    "diuretic": [r"furosemide", r"torsemide", r"bumetanide", r"metolazone",
                 r"hydrochlorothiazide", r"spironolactone", r"chlorothiazide", r"acetazolamide"],
    "steroid": [r"methylprednisolone", r"prednisone", r"dexamethasone",
                r"hydrocortisone", r"prednisolone", r"fludrocortisone"],
    "insulin": [r"insulin"],
    "opioid": [r"morphine", r"hydromorphone", r"oxycodone", r"fentanyl",
               r"tramadol", r"codeine", r"methadone", r"buprenorphine",
               r"oxymorphone", r"hydrocodone", r"meperidine"],
}

compiled = {
    cls: [re.compile(p, re.IGNORECASE) for p in pats]
    for cls, pats in DRUG_CLASSES.items()
}

print("Loading prescriptions...", flush=True)
df = pd.read_csv(
    "C:/Users/ValterAdmin/Documents/VS code projects/TemporaryMLthesis/downloads/hosp/prescriptions.csv",
    usecols=["drug", "drug_type"], dtype=str
)
main = df[df["drug_type"] == "MAIN"].copy()
total = len(main)
print(f"Total MAIN rows: {total:,}", flush=True)

def classify(drug):
    for cls, pats in compiled.items():
        if any(p.search(str(drug)) for p in pats):
            return cls
    return None

print("Classifying unique drug names...", flush=True)
unique_drugs = main["drug"].unique()
drug_cls = {d: classify(d) for d in unique_drugs}
main["cls"] = main["drug"].map(drug_cls)
classified = main["cls"].notna().sum()

print(f"\nTotal MAIN rows:   {total:,}")
print(f"Classified:        {classified:,}  ({classified/total*100:.1f}%)")
print(f"Unclassified:      {total-classified:,}  ({(total-classified)/total*100:.1f}%)")
print("\nPer-class breakdown:")
print(main["cls"].value_counts(dropna=False).to_string())
print("\nTop 80 unclassified drugs:")
print(main[main["cls"].isna()]["drug"].value_counts().head(80).to_string())
sys.stdout.flush()
