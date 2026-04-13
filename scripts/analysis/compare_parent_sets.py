"""Compare NOTEARS raw parent sets vs NOTEARS-augmented parent sets for each next_* target."""
import pandas as pd

STATE_VARS = ['creatinine','bun','sodium','potassium','bicarbonate','anion_gap',
              'calcium','glucose','hemoglobin','wbc','platelets','phosphate','magnesium','is_icu']
STATIC = ['age_at_admit','charlson_score']
DRUGS = ['antibiotic_active','anticoagulant_active','diuretic_active','steroid_active','insulin_active']
NEXT_TARGETS = ['next_' + v for v in STATE_VARS]
valid_sources = set(STATE_VARS) | set(STATIC)

notears = pd.read_csv('reports/causal_v3/step_b_notears/edges_lambda_0p0100.csv')
pc = pd.read_csv('reports/causal_v3/step_b/parent_sets.csv')

# --- Strict PC parent sets ---
strict_pc = {}
for _, row in pc.iterrows():
    target = row['target']
    drug_p = [p.strip() for p in str(row['drug_parents']).split(',')
              if p.strip() and p.strip() != 'nan']
    state_p = [p.strip() for p in str(row['state_parents']).split(',')
               if p.strip() and p.strip() != 'nan']
    state_clean = [p for p in state_p if not p.startswith('next_')]
    strict_pc[target] = drug_p + state_clean

# --- Pure NOTEARS parent sets (state/static sources -> next_* targets only) ---
notears_state = {}  # target -> list of (source, weight)
for _, row in notears.iterrows():
    src, tgt = str(row['source']), str(row['target'])
    if tgt not in NEXT_TARGETS:
        continue
    if src.startswith('next_'):
        continue
    if src not in valid_sources:
        continue
    notears_state.setdefault(tgt, []).append((src, row['weight']))

# --- NOTEARS-augmented = strict PC union NOTEARS state parents ---
notears_aug = {}
for target in NEXT_TARGETS:
    existing = list(strict_pc.get(target, []))
    existing_set = set(existing)
    added = []
    for src, w in notears_state.get(target, []):
        if src not in existing_set:
            added.append((src, w))
            existing_set.add(src)
    notears_aug[target] = existing + [s for s, _ in added]

# --- Comparison ---
print()
print('=' * 80)
print('PARENT SET COMPARISON: NOTEARS-only vs NOTEARS-augmented (strict PC + NOTEARS)')
print('=' * 80)

summary_rows = []
for target in NEXT_TARGETS:
    pc_set = set(strict_pc.get(target, []))
    nt_set = {s for s, _ in notears_state.get(target, [])}
    aug_set = set(notears_aug.get(target, []))

    in_both   = sorted(pc_set & nt_set)
    pc_only   = sorted(pc_set - nt_set)
    nt_only   = sorted(nt_set - pc_set)

    # NOTEARS weights for NOTEARS-found parents
    nt_weights = {s: w for s, w in notears_state.get(target, [])}

    print()
    print(f'TARGET: {target}')
    print(f'  Strict PC parents  ({len(pc_set):2d}): {sorted(pc_set)}')
    print(f'  NOTEARS state pars ({len(nt_set):2d}): {sorted(nt_set)}')
    print(f'  Agreed (both)      ({len(in_both):2d}): {in_both}')
    print(f'  PC-only            ({len(pc_only):2d}): {pc_only}')
    if pc_only:
        print(f'    -> These are DRUG parents from PC (NOTEARS excluded because we filter drugs out of NOTEARS)')
    print(f'  NOTEARS-only       ({len(nt_only):2d}): {nt_only}')
    if nt_only:
        print(f'    -> Weights: {[(s, round(nt_weights[s],4)) for s in sorted(nt_only)]}')
    print(f'  NOTEARS-aug final  ({len(aug_set):2d}): {sorted(aug_set)}')

    summary_rows.append({
        'target': target,
        'strict_pc_n': len(pc_set),
        'notears_state_n': len(nt_set),
        'notears_aug_n': len(aug_set),
        'agreed': len(in_both),
        'pc_only_drugs': ', '.join(pc_only),
        'notears_only_additions': ', '.join(nt_only),
    })

print()
print('=' * 80)
print('SUMMARY TABLE')
print('=' * 80)
print(f"{'Target':<20} {'PC':>4} {'NT':>4} {'Aug':>4} {'Agree':>6}  NOTEARS-only additions (w/weight)")
print('-' * 80)
for row in summary_rows:
    target = row['target']
    nt_only = [s for s, _ in notears_state.get(target, []) if s not in strict_pc.get(target, [])]
    nt_weights = {s: w for s, w in notears_state.get(target, [])}
    additions = ', '.join(f"{s}({round(nt_weights[s],3):+.3f})" for s in nt_only)
    print(f"{row['target']:<20} {row['strict_pc_n']:>4} {row['notears_state_n']:>4} {row['notears_aug_n']:>4} {row['agreed']:>6}  {additions}")

print()
print('KEY DISTINCTION:')
print('  "Pure NOTEARS parent set" would include drug edges from NOTEARS (rows 14-39 in the CSV).')
print('  "NOTEARS-augmented" EXCLUDES drug edges from NOTEARS and takes drugs from PC only.')
print('  This is deliberate: drug-effect attribution stays causally clean (PC-validated).')
print()

# Show what a pure-NOTEARS parent set would look like (including drugs)
print('=' * 80)
print('BONUS: What pure-NOTEARS parent sets look like (drugs included from NOTEARS)')
print('=' * 80)
notears_full = {}
for _, row in notears.iterrows():
    src, tgt = str(row['source']), str(row['target'])
    if tgt not in NEXT_TARGETS:
        continue
    if src.startswith('next_'):
        continue
    notears_full.setdefault(tgt, []).append((src, row['weight']))

for target in NEXT_TARGETS:
    full = sorted(notears_full.get(target, []), key=lambda x: -abs(x[1]))
    aug  = sorted(notears_aug.get(target, []))
    full_names = [s for s, _ in full]
    print(f'{target:<20}: NOTEARS-full={[s for s in full_names]}'
          f' (n={len(full_names)})  vs  NOTEARS-aug={aug} (n={len(aug)})')
