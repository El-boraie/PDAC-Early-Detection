"""Case report PDF export, shared by the Predict page's "Generate report" action and the
Reports page's "Open PDF". Plain, honest, text-first layout -- no attempt at a polished
clinical-letterhead look, matching the project's "authenticity over polish" design brief.
"""

import io

import pypdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable,
)

PRIMARY = colors.HexColor("#17879B")
PRIMARY_DEEP = colors.HexColor("#0F5F6E")
INK = colors.HexColor("#1B2C31")
INK_SECONDARY = colors.HexColor("#5F7278")
HAIRLINE = colors.HexColor("#E0EAEC")
ROW_ALT = colors.HexColor("#F4F9F9")

# Mirrors dashboard/common.py's BIOMARKER_INFO labels -- kept as a small local copy rather
# than importing common.py (which pulls in streamlit) so this module stays a plain,
# standalone PDF generator usable outside a running Streamlit session.
FEATURE_LABELS = {
    "creatinine": "Creatinine (mg/dL)",
    "LYVE1": "LYVE1 (ng/mL)",
    "REG1B": "REG1B (ng/mL)",
    "TFF1": "TFF1 (ng/mL)",
    "plasma_CA19_9": "Plasma CA19-9 (U/mL)",
    "age": "Age",
    "sex": "Sex",
}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("PDXTitle", parent=styles["Title"], textColor=PRIMARY_DEEP, fontSize=19,
                               spaceAfter=2))
    styles.add(ParagraphStyle("PDXH2", parent=styles["Heading2"], textColor=PRIMARY_DEEP, fontSize=13,
                               spaceBefore=4, spaceAfter=4))
    styles.add(ParagraphStyle("PDXBody", parent=styles["BodyText"], fontSize=10, leading=14))
    styles.add(ParagraphStyle("PDXSmall", parent=styles["BodyText"], fontSize=8, textColor=INK_SECONDARY))
    return styles


def _section_rule(story) -> None:
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.75, color=HAIRLINE, spaceBefore=2, spaceAfter=10))


def _format_value(feat: str, val) -> str:
    if feat == "sex":
        return "Male" if val in (1, "1", "Male") else "Female"
    if feat == "age":
        return str(int(val))
    return f"{float(val):,.2f}"


def generate_case_pdf(case: dict, gradcam_png_bytes: bytes = None) -> bytes:
    """case: the same dict passed to storage.save_case() (plus case_id/patient_id/created_at
    once saved). Returns PDF file bytes."""
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    story = []

    story.append(Paragraph("PancraDX -- Case Report", styles["PDXTitle"]))
    story.append(Paragraph("Research use only. Not a diagnostic device.", styles["PDXSmall"]))
    story.append(Spacer(1, 12))

    meta_rows = [
        ["Case ID", case.get("case_id", "--")],
        ["Patient", f"{case.get('patient_name', '(unnamed)')} ({case.get('patient_id', '--')})"],
        ["Date", case.get("created_at", "--")],
        ["Modalities", case.get("modalities", "--")],
    ]
    meta_table = Table(meta_rows, colWidths=[3.2 * cm, 12.3 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), INK_SECONDARY),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    _section_rule(story)

    # --- Combined result ---
    fused_score = case.get("fused_score")
    risk_band = case.get("risk_band", "")
    if fused_score is not None:
        story.append(Paragraph("Result", styles["PDXH2"]))
        if case.get("mode") == "fused (both modalities)":
            story.append(Paragraph(
                f"<b>Combined score: {fused_score:.3f}</b> &nbsp;&nbsp; Risk band: <b>{risk_band}</b>",
                styles["PDXBody"]))
            story.append(Paragraph(
                "Combined as 0.4 &times; imaging + 0.6 &times; clinical (a fixed rule, not a "
                "trained/validated joint model -- no patient in the training data has both a "
                "CT scan and a urine sample).", styles["PDXSmall"]))
        else:
            story.append(Paragraph(
                f"<b>Score: {fused_score:.3f}</b> ({case.get('mode', '')}) &nbsp;&nbsp; Risk band: <b>{risk_band}</b>",
                styles["PDXBody"]))
            story.append(Paragraph(
                "Only one modality was provided -- this is that branch's own score, not a joint/fused score.",
                styles["PDXSmall"]))
        _section_rule(story)

    # --- Imaging panel ---
    imaging = case.get("imaging")
    if imaging:
        story.append(Paragraph("Imaging branch", styles["PDXH2"]))
        story.append(Paragraph(
            f"Calibrated score: <b>{imaging['calibrated_proba']:.3f}</b> "
            f"({imaging['granularity']}, {imaging['n_slices']} slice(s))", styles["PDXBody"]))
        if gradcam_png_bytes:
            story.append(Spacer(1, 8))
            story.append(RLImage(io.BytesIO(gradcam_png_bytes), width=6.5 * cm, height=6.5 * cm))
        _section_rule(story)

    # --- Clinical panel ---
    clinical = case.get("clinical")
    if clinical:
        story.append(Paragraph("Clinical branch", styles["PDXH2"]))
        story.append(Paragraph(f"Calibrated score: <b>{clinical['calibrated_proba']:.3f}</b>", styles["PDXBody"]))
        story.append(Spacer(1, 8))

        shap_per_feature = clinical.get("shap_per_feature", {})
        raw_features = clinical.get("raw_features", {}) or {}
        if shap_per_feature:
            story.append(Paragraph("Biomarker values &amp; SHAP contribution (log-odds scale)", styles["PDXBody"]))
            story.append(Spacer(1, 4))
            rows = [["Feature", "Value", "SHAP"]]
            row_styles = []
            for i, (feat, val) in enumerate(sorted(shap_per_feature.items(), key=lambda kv: -abs(kv[1])), start=1):
                label = FEATURE_LABELS.get(feat, feat)
                value_str = _format_value(feat, raw_features[feat]) if feat in raw_features else "--"
                rows.append([label, value_str, f"{val:+.3f}"])
                if i % 2 == 0:
                    row_styles.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
            shap_table = Table(rows, colWidths=[6.5 * cm, 4.5 * cm, 3.5 * cm])
            shap_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, PRIMARY),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                *row_styles,
            ]))
            story.append(shap_table)
        story.append(Spacer(1, 4))

    doc.build(story)
    return _strip_trailing_blank_pages(buf.getvalue())


def _strip_trailing_blank_pages(pdf_bytes: bytes) -> bytes:
    """ReportLab's auto-pagination has a known quirk where a page whose content (e.g. an
    embedded image) fits entirely on the current page can still trigger an extra, wholly
    empty trailing page -- confirmed here: a fused case's Grad-CAM image renders fine on
    page 1, yet a genuinely blank page 2 still gets emitted. Rather than fight ReportLab's
    layout-estimation heuristics, drop any trailing page with no text and no images."""
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    while len(writer.pages) > 1:
        last = writer.pages[-1]
        has_text = bool(last.extract_text().strip())
        has_images = bool(last.get("/Resources", {}).get("/XObject"))
        if has_text or has_images:
            break
        del writer.pages[-1]
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
