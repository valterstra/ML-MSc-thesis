"""Drug classification patterns for prescription-based action flags."""

from __future__ import annotations

DRUG_CLASSES: dict[str, list[str]] = {
    "antibiotic": [
        r"vancomycin", r"piperacillin", r"ceftriaxone", r"cefazolin", r"cefepime",
        r"meropenem", r"imipenem", r"azithromycin", r"levofloxacin", r"ciprofloxacin",
        r"metronidazole", r"ampicillin", r"clindamycin", r"doxycycline",
        r"trimethoprim", r"sulfamethoxazole", r"linezolid", r"daptomycin",
        r"nitrofurantoin", r"gentamicin", r"tobramycin", r"amikacin", r"rifampin",
        r"fluconazole", r"micafungin", r"caspofungin", r"anidulafungin",
        r"cephalexin", r"cefdinir", r"cefpodoxime", r"cefuroxime",
    ],
    "anticoagulant": [
        r"heparin", r"enoxaparin", r"warfarin", r"apixaban", r"rivaroxaban",
        r"fondaparinux", r"dabigatran", r"argatroban",
    ],
    "diuretic": [
        r"furosemide", r"torsemide", r"bumetanide", r"metolazone",
        r"hydrochlorothiazide", r"spironolactone", r"chlorothiazide", r"acetazolamide",
    ],
    "steroid": [
        r"methylprednisolone", r"prednisone", r"dexamethasone",
        r"hydrocortisone", r"prednisolone", r"fludrocortisone",
    ],
    "insulin": [
        r"insulin",
    ],
    "opioid": [
        r"morphine", r"hydromorphone", r"oxycodone", r"fentanyl",
        r"tramadol", r"codeine", r"methadone", r"buprenorphine",
        r"oxymorphone", r"hydrocodone", r"meperidine",
    ],
}
