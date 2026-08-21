# Table 16 — Imaging Dataset Summary: NIH Pancreas-CT (Healthy Negative Cohort)

*(Renumber to match wherever this actually lands relative to Table 15 in the
IR — written as the next table in the same series.)*

| Property | NIH Pancreas-CT (Healthy Negative Cohort) |
|---|---|
| Source Institution | National Institutes of Health (NIH) Clinical Center, Bethesda, MD, USA — accessed in this project via a Hugging Face mirror (`CADS-dataset/0024_pancreas_ct`) |
| Published Reference | Roth, H., Farag, A., Turkbey, E. B., Lu, L., Liu, J., & Summers, R. M. (2016). *Data From Pancreas-CT* (Version 2) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU |
| Total 3D Volumes | 82 scans from 80 subjects (80/80 loaded and used in this project — see Section L3 of `CT_EDA_documentation.md`) |
| Estimated 2D Slices | ~18,942 |
| Annotation Type | Pancreas (manual, by a medical student, radiologist-verified) — **in the original TCIA release only**. Not used here: this project's mirror shipped a different, multi-organ auto-segmentation mask with no label-to-organ mapping, so masks were judged unusable and this pass is image-only |
| Modality | Portal venous phase CT (contrast-enhanced, ~70s post-injection) |
| Format | NIfTI (`.nii.gz`) as received via the mirror — pre-converted; the original TCIA release format is DICOM |
| Subject Condition | Healthy controls (no pancreatic or abdominal abnormality) — used as the detection head's negative class, since MSD Task07 alone is 100% cancer-positive |
| License | CC BY 3.0 |

---

**Where each fact came from, for your own citation trail:**
- License, dataset citation, DOI, contrast/phase description, body coverage,
  acquisition center, and "healthy controls" pathology label are all taken
  directly from the dataset's own bundled README:
  `data/raw/NIH_Pancreas_CT/0024_pancreas_ct/README_0024_pancreas_ct.md`.
- Subject/scan counts (82 scans / 80 subjects, 80/80 successfully loaded),
  the "manual pancreas annotation not usable here" finding, the slice
  estimate (~18,942), and the image-only-pass justification are all from
  this project's own `docs/CT_EDA_documentation.md`, Section L
  (`notebooks/CT EDA.ipynb`).

**One thing worth flagging before this goes in the IR:** the README's own
"Number of CT volumes" field says 80, while its prose description says "82
... scans from 80 subjects" — the two numbers in the source document
disagree with each other, not just with this project's count. This project
consistently used 80 (matching what was actually downloaded and loaded), so
that's the number carried into the table above — but if the discrepancy
matters for your write-up, it's a source-document inconsistency, not
something this project introduced.
