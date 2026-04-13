"""Drug classification patterns for prescription-based action flags."""

from __future__ import annotations

DRUG_CLASSES: dict[str, list[str]] = {
    "antibiotic": [
        r"vancomycin", r"piperacillin", r"ceftriaxone", r"cefazolin", r"cefepime",
        r"meropenem", r"imipenem", r"ertapenem", r"azithromycin", r"levofloxacin",
        r"ciprofloxacin", r"metronidazole", r"ampicillin", r"clindamycin", r"doxycycline",
        r"trimethoprim", r"sulfamethoxazole", r"linezolid", r"daptomycin",
        r"nitrofurantoin", r"gentamicin", r"tobramycin", r"amikacin", r"rifampin",
        r"fluconazole", r"micafungin", r"caspofungin", r"anidulafungin",
        r"cephalexin", r"cefdinir", r"cefpodoxime", r"cefuroxime",
    ],
    "anticoagulant": [
        r"heparin", r"enoxaparin", r"warfarin", r"apixaban", r"rivaroxaban",
        r"fondaparinux", r"dabigatran", r"argatroban", r"bivalirudin",
    ],
    "diuretic": [
        r"furosemide", r"torsemide", r"bumetanide", r"metolazone",
        r"hydrochlorothiazide", r"spironolactone", r"chlorothiazide", r"acetazolamide",
        r"eplerenone", r"chlorthalidone",
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
    "electrolyte": [
        r"potassium chloride", r"potassium phosphate",
        r"magnesium sulfate", r"magnesium oxide",
        r"calcium gluconate", r"calcium carbonate", r"calcium chloride",
        r"sodium bicarbonate",
        r"neutra.phos", r"sodium phosphate",
    ],
    "cardiovascular": [
        # beta-blockers
        r"metoprolol", r"carvedilol", r"atenolol", r"bisoprolol",
        r"labetalol", r"propranolol", r"nebivolol",
        # ACE inhibitors
        r"lisinopril", r"enalapril", r"captopril", r"ramipril",
        r"quinapril", r"fosinopril", r"benazepril",
        # ARBs
        r"losartan", r"valsartan", r"irbesartan", r"olmesartan",
        r"candesartan", r"telmisartan",
        # calcium channel blockers
        r"amlodipine", r"diltiazem", r"verapamil", r"nifedipine",
        r"nicardipine", r"clevidipine", r"felodipine",
        # statins
        r"atorvastatin", r"simvastatin", r"rosuvastatin", r"pravastatin",
        r"lovastatin", r"fluvastatin", r"pitavastatin",
        # antiplatelets
        r"clopidogrel", r"ticagrelor", r"prasugrel", r"aspirin",
        # antiarrhythmics
        r"amiodarone", r"digoxin", r"flecainide", r"sotalol", r"dronedarone",
        # vasodilators
        r"hydralazine", r"nitroglycerin", r"isosorbide", r"nitroprusside",
    ],
}
