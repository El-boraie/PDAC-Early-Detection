# Table 17 — Imaging Dataset Summary: PANORAMA Challenge (External Test Set)

*(Renumber to match wherever this lands relative to Tables 15–16 in the IR —
written as the next table in the same series.)*

| Property | PANORAMA Challenge (External, Out-of-Distribution Test Set) |
|---|---|
| Source Institution | Multi-center: Radboud University Medical Center & University Medical Center Groningen (Netherlands, primary contributors), plus Ziekenhuis Groep Twente (Netherlands), Karolinska Institutet (Sweden), Haukeland University Hospital (Norway) |
| Published Reference | Alves, N., Schuurmans, M., Yakar, D., Vendittelli, P., Litjens, G., Hermans, J., & Huisman, H. (2024). *The PANORAMA Challenge: Public Training and Development Dataset (1)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13715870 |
| Download Link | https://panorama.grand-challenge.org/datasets-imaging-labels/ (challenge page) → https://zenodo.org/records/13715870 (Batch 1 of 4, the batch used in this project) |
| Total 3D Volumes | 557 volumes used in this project, drawn from Batch 1 of the official 4-batch, 2,238-case public release (49.3GB for Batch 1 alone) |
| Estimated 2D Slices | Not applicable — used only as whole-volume input through the dashboard's upload pipeline, never converted into a slice-level training/eval set |
| Annotation Type | Pancreatic tumour delineation, from the official `panorama_labels` annotation set: 108 expert/manual delineations, 449 AI-derived/automatic delineations (a re-trained version of Alves et al. 2022's method, per the challenge organizers). Used here only to derive a binary PDAC-positive/negative flag per case (**158 positive, 399 negative**) — not as training signal anywhere in this project |
| Modality | Contrast-enhanced CT (CECT) |
| Format | NIfTI (`.nii.gz`) |
| Subject Condition | Mixed — both PDAC-positive and non-PDAC cases (unlike MSD, which is 100% cancer, or NIH, which is 100% healthy) |
| License | CC BY-NC 4.0 (non-commercial) |
| Role in this project | **Not used for training or model evaluation at any point.** Used exclusively as an out-of-distribution robustness test for the dashboard's raw-upload preprocessing pipeline (`src/imaging/preprocessing.py`) — a dataset the pipeline, this project, and the trained models had never seen. 5 of the 557 volumes were processed end-to-end with no crashes and a genuine spread of predicted scores (0.09–0.99); see `docs/Dashboard_documentation.md` Section 5.2 for the exact per-file results |

---

**Verification note:** unlike the MSD and NIH tables, this project's local copy of
PANORAMA has no bundled README/license file — the facts above (citation, DOI,
license, contributing centers, batch structure, manual-vs-automatic annotation
definitions) were confirmed directly against the official challenge page
(`panorama.grand-challenge.org`) and the Zenodo record for Batch 1
(`zenodo.org/records/13715870`) via live lookup, not from local files or
memory. Counts (557 volumes, 108/449 manual/automatic split, 158/399
positive/negative split) **are** locally verified, computed directly from
`data/testing/panorama_labels_manifest.csv`.

**One overlap worth knowing about, so it doesn't get missed:** the official
PANORAMA public release also *repackages* 194 MSD cases and 80 NIH cases
inside its own combined dataset. This project's 557 test volumes use PANORAMA's
own case-numbering scheme (`100000_00001` etc.), distinct from MSD's
`pancreas_NNN` and NIH's `0000N` naming — so there is no accidental overlap
with this project's own MSD/NIH training data, but it's worth double-checking
this explicitly if any of the 557 case IDs are ever cited individually in the
write-up.
