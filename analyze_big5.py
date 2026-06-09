"""
analyze_big5.py

Computes Big Five trait scores from raw item responses,
then correlates them with productivity archetype features
from the student dataset.

This gives us data-derived shift weights to replace the
manually tuned values in cold_start.py.

Big Five scoring:
- EXT (Extraversion)    : EXT1-EXT10
- EST (Neuroticism)     : EST1-EST10  (emotional stability = inverse neuroticism)
- AGR (Agreeableness)   : AGR1-AGR10
- CSN (Conscientiousness): CSN1-CSN10
- OPN (Openness)        : OPN1-OPN10

Items answered on 1-5 scale (0 = missing, treated as neutral 3)
Some items are reverse-scored — standard Big Five scoring rules apply.
"""

import pandas as pd
import numpy as np

# ─── LOAD BIG FIVE DATA (50k sample) ─────────────────────────────────────────
print("Loading Big Five dataset (50k sample)...")
big5 = pd.read_csv('data/raw/big5.csv', sep='\t', nrows=50000)

# Replace 0 with NaN (0 = did not answer in this dataset)
personality_cols = [c for c in big5.columns if any(
    c.startswith(p) and not c.endswith('_E')
    for p in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
) and len(c) <= 5]

big5[personality_cols] = big5[personality_cols].replace(0, np.nan)

print(f"Personality columns found: {len(personality_cols)}")
print(f"Columns: {personality_cols}")

# ─── SCORE EACH TRAIT ────────────────────────────────────────────────────────
# Standard Big Five reverse-score items:
# EXT: reverse EXT2, EXT4, EXT6, EXT8, EXT10
# EST: reverse EST1, EST3, EST5, EST6, EST7, EST8, EST9, EST10
# AGR: reverse AGR1, AGR3, AGR5, AGR7
# CSN: reverse CSN2, CSN4, CSN6, CSN8
# OPN: reverse OPN2, OPN4, OPN6, OPN8

def reverse(series):
    return 6 - series  # 1→5, 2→4, 3→3, 4→2, 5→1

# Extraversion
ext_cols = [f'EXT{i}' for i in range(1, 11) if f'EXT{i}' in big5.columns]
big5_ext = big5[ext_cols].copy()
for col in ['EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10']:
    if col in big5_ext.columns:
        big5_ext[col] = reverse(big5_ext[col])
big5['E_score'] = big5_ext.mean(axis=1)

# Neuroticism (EST = emotional stability = inverse of neuroticism)
est_cols = [f'EST{i}' for i in range(1, 11) if f'EST{i}' in big5.columns]
big5_est = big5[est_cols].copy()
for col in ['EST1', 'EST3', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10']:
    if col in big5_est.columns:
        big5_est[col] = reverse(big5_est[col])
# EST measures stability — invert to get Neuroticism score
big5['N_score'] = 6 - big5_est.mean(axis=1)

# Agreeableness
agr_cols = [f'AGR{i}' for i in range(1, 11) if f'AGR{i}' in big5.columns]
big5_agr = big5[agr_cols].copy()
for col in ['AGR1', 'AGR3', 'AGR5', 'AGR7']:
    if col in big5_agr.columns:
        big5_agr[col] = reverse(big5_agr[col])
big5['A_score'] = big5_agr.mean(axis=1)

# Conscientiousness
csn_cols = [f'CSN{i}' for i in range(1, 11) if f'CSN{i}' in big5.columns]
big5_csn = big5[csn_cols].copy()
for col in ['CSN2', 'CSN4', 'CSN6', 'CSN8']:
    if col in big5_csn.columns:
        big5_csn[col] = reverse(big5_csn[col])
big5['C_score'] = big5_csn.mean(axis=1)

# Openness
opn_cols = [f'OPN{i}' for i in range(1, 11) if f'OPN{i}' in big5.columns]
big5_opn = big5[opn_cols].copy()
for col in ['OPN2', 'OPN4', 'OPN6', 'OPN8']:
    if col in big5_opn.columns:
        big5_opn[col] = reverse(big5_opn[col])
big5['O_score'] = big5_opn.mean(axis=1)

# Normalize all trait scores to [0, 1]
for trait in ['E_score', 'N_score', 'A_score', 'C_score', 'O_score']:
    big5[f'{trait}_norm'] = (big5[trait] - 1) / (5 - 1)

print("\n=== BIG FIVE TRAIT SCORE DISTRIBUTIONS ===")
trait_scores = ['E_score_norm', 'N_score_norm', 'A_score_norm',
                'C_score_norm', 'O_score_norm']
print(big5[trait_scores].describe().round(3).to_string())

# ─── LOAD PRODUCTIVITY DATA ───────────────────────────────────────────────────
print("\nLoading student productivity dataset...")
prod = pd.read_csv('data/raw/student_productivity.csv')

# ─── COMPUTE PRODUCTIVITY PROXY FEATURES ─────────────────────────────────────
# Map the same features we use in Focentra
prod['completion_proxy']   = prod['assignments_completed'] / 19.0
prod['focus_proxy']        = prod['study_hours_per_day'] / 10.0
prod['distraction_proxy']  = (prod['social_media_hours'] + prod['gaming_hours']) / 14.0
prod['stress_proxy']       = prod['stress_level'] / 10.0
prod['motivation_proxy']   = prod['productivity_score'] / 100.0
prod['consistency_proxy']  = prod['focus_score'] / 99.0

# ─── CROSS-DATASET CORRELATION ───────────────────────────────────────────────
# Since these are different datasets we cannot directly correlate rows.
# Instead we correlate WITHIN each dataset:
# Big Five: trait → trait correlations (internal structure)
# Productivity: feature → feature correlations
# Then we use the KNOWN psychological literature mappings:
# C → completion, focus, consistency (well established)
# N → stress, burnout (well established)
# O → variety seeking, distraction (moderate evidence)

print("\n=== BIG FIVE INTERNAL CORRELATIONS ===")
trait_corr = big5[trait_scores].corr().round(3)
print(trait_corr.to_string())

print("\n=== CONSCIENTIOUSNESS INTERNAL CONSISTENCY ===")
csn_items = [c for c in csn_cols if c in big5.columns]
csn_data = big5[csn_items].copy()
for col in ['CSN2', 'CSN4', 'CSN6', 'CSN8']:
    if col in csn_data.columns:
        csn_data[col] = reverse(csn_data[col])
csn_corr = csn_data.corr()['CSN1'].drop('CSN1').round(3)
print("CSN1 vs other CSN items:")
print(csn_corr.to_string())

print("\n=== PRODUCTIVITY FEATURE CORRELATIONS ===")
prod_features = ['completion_proxy', 'focus_proxy', 'distraction_proxy',
                 'stress_proxy', 'motivation_proxy', 'consistency_proxy']
prod_corr = prod[prod_features].corr().round(3)
print(prod_corr.to_string())

# ─── DERIVE SHIFT WEIGHTS ────────────────────────────────────────────────────
# Key correlations we care about for cold_start.py:
# C → completion (how strongly does C predict task completion?)
# N → stress (how strongly does N predict burnout signals?)
# O → distraction (how strongly does O predict distraction?)

print("\n=== DERIVED SHIFT WEIGHTS FOR cold_start.py ===")

# Within Big Five: C correlates with low N (disciplined = less neurotic)
c_n_corr = abs(big5['C_score_norm'].corr(big5['N_score_norm']))
# C correlates with low E variability (consistent schedule)
c_e_corr = abs(big5['C_score_norm'].corr(big5['E_score_norm']))
# N standalone reliability
n_self = big5['N_score_norm'].std()
# O standalone
o_self = big5['O_score_norm'].std()

# Within productivity: completion vs stress
comp_stress = abs(prod['completion_proxy'].corr(prod['stress_proxy']))
# focus vs productivity
focus_prod = prod['focus_proxy'].corr(prod['motivation_proxy'])
# distraction vs completion
dist_comp = abs(prod['distraction_proxy'].corr(prod['completion_proxy']))

print(f"\nBig Five internal:")
print(f"  C ↔ N correlation (inverse):  {c_n_corr:.4f}")
print(f"  C ↔ E correlation:            {c_e_corr:.4f}")
print(f"  N score std dev:              {n_self:.4f}")
print(f"  O score std dev:              {o_self:.4f}")

print(f"\nProductivity internal:")
print(f"  completion ↔ stress:          {comp_stress:.4f}")
print(f"  focus ↔ motivation:           {focus_prod:.4f}")
print(f"  distraction ↔ completion:     {dist_comp:.4f}")

# Derive recommended shift weights
# Logic: stronger correlation = stronger shift weight justified
c_shift  = round(min(0.6, max(0.3, focus_prod * 0.8)), 2)
n_shift  = round(min(0.7, max(0.4, comp_stress * 1.2 + 0.3)), 2)
o_shift  = round(min(0.5, max(0.25, dist_comp * 0.9)), 2)

print(f"\n=== RECOMMENDED SHIFT WEIGHTS ===")
print(f"  HIGH_C_SHIFT (→ Consistent Achiever): {c_shift}")
print(f"  HIGH_N_SHIFT (→ Burnout-Prone):        {n_shift}")
print(f"  LOW_C_SHIFT  (→ Easily Distracted):    {o_shift}")
print(f"\nPaste these values into cold_start.py _apply_onboarding_shift()")
print(f"replacing MAX_SHIFT values.")