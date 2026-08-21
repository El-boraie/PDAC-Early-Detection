# Demo Patients — Ready-to-Use Fill-Ins for the Presentation

Five synthetic demo patients, spanning the full risk spectrum, built from **real,
de-identified rows** in the Debernardi et al. urinary biomarker dataset
(`data/processed/tabular_clean.csv`) — only the name is fictional; every
biomarker value is a genuine measured value from a real (anonymized) research
participant, so the model's response to them is authentic, not made up.
Register's own caption already says "Demo data — not real patients"; these
follow that same convention.

Each row is picked to tell a specific story on screen — don't just click
through them identically; the point of showing more than one is the contrast.

**All values are pre-rounded to match each field's input precision** (2 decimals
for creatinine/LYVE1, whole numbers for REG1B/TFF1/CA19-9) so what you type
matches what's in the dataset exactly, no on-the-fly rounding needed.

---

## Quick-reference table

| # | Name | DOB | Sex | Creatinine (mg/dL) | LYVE1 (ng/mL) | REG1B (ng/mL) | TFF1 (ng/mL) | CA19-9 (U/mL) | Real diagnosis (don't say on camera*) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Maria Novak | 1976-07-22 | Female | 0.53 | 0.01 | 5 | 5 | 0 | Control (healthy) |
| 2 | Robert Klein | 1949-07-22 | Female | 1.41 | 1.04 | 91 | 566 | 37 | Benign (borderline CA19-9) |
| 3 | David Ahmadi | 1969-07-22 | Male | 1.73 | 2.63 | 41 | 530 | 10 | PDAC, stage IA (early) |
| 4 | Linda Osei | 1972-07-22 | Female | 0.55 | 6.59 | 133 | 811 | 110 | PDAC, stage IIB (moderate) |
| 5 | Thomas Becker | 1973-07-22 | Male | 0.64 | 6.78 | 110 | 435 | 18360 | PDAC, stage III (advanced) |

*The "real diagnosis" column is for **your** reference so you know what the
model *should* say — don't read it out before running the prediction, or the
live demo loses its "let's see what the model finds" moment.

---

## Patient 1 — Maria Novak — the clean baseline

**Register:** DOB `1976-07-22`, Sex `Female`.
**Biomarkers:** Creatinine `0.53`, LYVE1 `0.01`, REG1B `5`, TFF1 `5`, CA19-9 `0`.
**What to say:** "A straightforward negative case — everything at or near
baseline." Expect a clearly low fused/clinical score.

**Optional CT pairing:** upload the real NIH healthy scan —
`data/raw/NIH_Pancreas_CT/0024_pancreas_ct/images/00001_1.2.826.0.1.3680043.2.1125.1.10349856497345767271749258522763940_0000.nii.gz`
(~this is the file behind the "nih_00001" byte-exact validation mentioned in
`docs/Dashboard_documentation.md`). Pairs a clean scan with a clean biomarker
profile for a fully negative combined case.

## Patient 2 — Robert Klein — the interesting near-miss

**Register:** DOB `1949-07-22`, Sex `Female`.
**Biomarkers:** Creatinine `1.41`, LYVE1 `1.04`, REG1B `91`, TFF1 `566`, CA19-9 `37`.
**What to say:** CA19-9 of 37 sits right at the standard clinical cutoff (<37
U/mL is "normal") — a case where the single familiar marker alone is
borderline/ambiguous, but the real diagnosis is benign, not cancer. Good for
showing the SHAP panel: watch which feature the model actually leans on here
rather than just CA19-9.

## Patient 3 — David Ahmadi — the one CA19-9 alone would miss

**Register:** DOB `1969-07-22`, Sex `Male`.
**Biomarkers:** Creatinine `1.73`, LYVE1 `2.63`, REG1B `41`, TFF1 `530`, CA19-9 `10`.
**What to say:** This is genuinely early-stage PDAC (stage IA), but CA19-9 is
a normal-looking 10 U/mL — by CA19-9 alone, this would be missed entirely.
**This is the single strongest storytelling patient in the set** — it's the
concrete case for why the model uses six other features, not just the one
clinicians already know. Worth building a full sentence of narration around
this one specifically.

## Patient 4 — Linda Osei — the middle case

**Register:** DOB `1972-07-22`, Sex `Female`.
**Biomarkers:** Creatinine `0.55`, LYVE1 `6.59`, REG1B `133`, TFF1 `811`, CA19-9 `110`.
**What to say:** Stage IIB PDAC, moderately elevated markers across the board
— use this one to show a "moderate risk" band result if you want three
distinct visual risk levels on screen rather than just low/high.

## Patient 5 — Thomas Becker — the unambiguous positive

**Register:** DOB `1973-07-22`, Sex `Male`.
**Biomarkers:** Creatinine `0.64`, LYVE1 `6.78`, REG1B `110`, TFF1 `435`, CA19-9 `18360`.
**What to say:** Stage III, CA19-9 nearly 500x the normal cutoff — an
unambiguous high-risk case. Good for showing the "elevated risk" band and a
confident SHAP waterfall with CA19-9 as the dominant driver.

**Optional CT pairing:** upload a real MSD cancer scan —
`data/raw/Task07_Pancreas/imagesTr/pancreas_001.nii.gz` (~32MB) or
`pancreas_004.nii.gz` (~29MB) — both are two of the three patients the raw
upload pipeline was byte-exact validated against. Pairs a real tumour-bearing
scan with a high-risk biomarker profile for a fully positive combined case,
and lets you show the Grad-CAM overlay on an actual cancer scan.

---

## Practical notes for the day

- **These are two different real datasets** (imaging and biomarkers aren't
  paired per-patient anywhere in this project — see the handoff, Section 2d).
  Uploading a real CT file alongside one of these biomarker rows under one
  demo patient name is a presentation convenience, not a claim that they're
  the same real person. If asked directly, say so — it's exactly the kind of
  honest-disclosure standard the rest of the project already follows.
- **Test the CT uploads once, off-camera, before recording** — both files are
  25-35MB, so preprocessing (reorient/resample/HU-window) will take a few
  real seconds; know what that pause feels like so it doesn't read as the app
  hanging live.
- **Register each patient once, then don't re-register on a re-take** — if
  you re-run the same demo twice, use "Continue with this existing patient"
  on Register rather than creating `Maria Novak #2` — the dedup prompt itself
  is a real feature worth showing once, but doesn't need repeating for every
  re-take.
- **Recommended order for the video**: Patient 1 (clean baseline) →
  Patient 3 (the CA19-9-would-miss-it story — your strongest point) →
  Patient 5 (unambiguous positive, with the CT/Grad-CAM pairing). Patients 2
  and 4 are good backups if you have time or a question comes up, but 1/3/5
  alone already tell a complete, contrasting story in under the two minutes
  Section 8 of the demo script budgets.
