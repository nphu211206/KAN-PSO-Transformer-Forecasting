#!/usr/bin/env python3
"""
build_submission_docs.py
Generates COVER_LETTER_NCAA.docx (clean 2-page layout) and DECLARATIONS_NCAA.docx (clean 1-page layout)
with 100% verified Option A metrics and Springer Nature NCAA metadata.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def create_cover_letter():
    doc = docx.Document()
    doc.core_properties.author = "Phu Toan Nguyen"
    doc.core_properties.title = "Cover Letter - PSO-TCWavKAN (NCAA Submission)"

    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.95)
        section.right_margin = Inches(0.95)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    p_date = doc.add_paragraph()
    p_date.add_run("November 2026").font.size = Pt(10.5)
    p_date.paragraph_format.space_after = Pt(6)

    p_recipient = doc.add_paragraph()
    r = p_recipient.add_run(
        "Prof. John MacIntyre\n"
        "Editor-in-Chief, Neural Computing and Applications\n"
        "Springer Nature London / University of Sunderland, United Kingdom"
    )
    r.font.size = Pt(10.5)
    r.bold = True
    p_recipient.paragraph_format.space_after = Pt(8)

    p_subj = doc.add_paragraph()
    r_subj = p_subj.add_run(
        "SUBJECT: Submission of Original Research Article: \"PSO-TCWavKAN: Swarm-Optimized "
        "Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power Transformer Thermal "
        "Dynamics Forecasting\""
    )
    r_subj.bold = True
    r_subj.font.size = Pt(11)
    p_subj.paragraph_format.space_after = Pt(8)

    p_sal = doc.add_paragraph()
    p_sal.add_run("Dear Prof. John MacIntyre and Esteemed Editorial Board Members,").font.size = Pt(10.5)
    p_sal.paragraph_format.space_after = Pt(6)

    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_intro.paragraph_format.space_after = Pt(6)
    p_intro.paragraph_format.line_spacing = 1.12
    p_intro.add_run(
        "I am pleased to submit my original research manuscript entitled \"PSO-TCWavKAN: "
        "Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold Networks for Power "
        "Transformer Thermal Dynamics Forecasting\" for peer review and consideration as an "
        "Original Research Paper in Neural Computing and Applications (NCAA)."
    )

    def add_sec_heading(title):
        h = doc.add_heading(level=2)
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(3)
        h.paragraph_format.keep_with_next = True
        rh = h.add_run(title)
        rh.font.name = "Times New Roman"
        rh.font.size = Pt(11.5)
        rh.bold = True
        rh.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    # Section 1
    add_sec_heading("1. Journal Scope Alignment & Motivation")
    p1 = doc.add_paragraph()
    p1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p1.paragraph_format.space_after = Pt(4)
    p1.paragraph_format.line_spacing = 1.12
    p1.add_run(
        "Neural Computing and Applications stands as a premier international venue advancing "
        "neural architectures, metaheuristic optimization, and industrial telemetry modeling. "
        "Our work builds directly on paradigms recently published in NCAA:"
    )
    bullets_1 = [
        ("Continuous Kolmogorov–Arnold Networks (KAN): ",
         "Extending piecewise linear finite-element KANs such as P1-KAN (Warin, NCAA 2026, 38:703, "
         "DOI: 10.1007/s00521-026-12354-y) into infinitely differentiable (C\u221e) continuous Morlet wavelet edge activations."),
        ("End-to-End Wavelet Representation: ",
         "Unifying multi-octave wavelet filtering within a fully differentiable network, resolving the "
         "decoupled two-stage separation of Wavelet-LSTM (Tamilselvi et al., NCAA 2025, 37(17):10521\u201310530, "
         "DOI: 10.1007/s00521-024-10561-z)."),
        ("Distribution-Shift Neutralization: ",
         "Pairing detached-moment Invertible RevIN and a non-degenerate DLinear recovery router, avoiding "
         "multi-subnet pre-training complexity as in GCINet (Fan et al., NCAA 2025, 37(5):3227\u20133244, "
         "DOI: 10.1007/s00521-024-10692-3) and complementing frequency-decomposed transformers such as "
         "FEDAF (Yang et al., NCAA 2024, 36(26):16271\u201316288, DOI: 10.1007/s00521-024-09937-y)."),
        ("Leak-Free Swarm Pre-Optimization: ",
         "Synergizing Clerc-constrained Comprehensive Learning PSO (CLPSO) with deep neural pre-conditioning "
         "strictly inside the training partition."),
    ]
    for b_title, b_desc in bullets_1:
        bp = doc.add_paragraph(style="List Bullet")
        bp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.12
        bp.add_run(b_title).bold = True
        bp.add_run(b_desc)

    # Section 2
    add_sec_heading("2. Core Scientific Innovations & Technical Contributions")
    innovations = [
        ("Continuous Morlet WavKAN Resolution of the Grid Update Dilemma: ",
         "Discrete B-spline KANs require periodic knot re-gridding (18 explicit updates in our three-seed ablation), "
         "disrupting Adam/AdamW moments and suffering compact-support boundary gradient vanishing. Continuous Morlet "
         "wavelets with softplus-bounded dilations (\u03c3 > 0) provide smooth C\u221e analytic gradients over \u211d."),
        ("Invertible RevIN with Detached Statistics: ",
         "Detached instance moments combined with clamped safe affine scaling (\u03b5_\u03b3 = 10\u207b\u2074) prevent autograd "
         "leakage and inverse scaling singularities; omitting RevIN triggers a +59.40% MSE explosion under regime shift."),
        ("Clerc-Constrained Leak-Free CLPSO: ",
         "Governed by Maurice Clerc's constriction factor (\u03c7 = 0.72984, \u03c6 = 4.1) over an 85/15 isolated training "
         "surrogate proxy (\u2264 8,640 timestamps), CLPSO tunes the 4D vector (\u03c9\u2080, log s, log \u03b1, log\u2081\u2080 \u03b7) "
         "with an S=20 accuracy\u2013compute Pareto knee (84.4% of S=40 improvement at 49.4% runtime)."),
        ("Proposition 1 & Linear O(DN) Complexity Theorem 1: ",
         "Proposition 1 establishes non-degenerate recovery of the linear DLinear subspace as router gates vanish "
         "(verified to 0.0 max absolute deviation), while Theorem 1 proves strict linear O(BDN) backbone token-mixing complexity."),
        ("Physical Anchor in IEEE Std C57.91-2011 / IEC 60076-7: ",
         "Directly ties multi-horizon (96h\u2013720h) top-oil temperature telemetry to Arrhenius cellulose insulation "
         "depolymerization kinetics and Dynamic Thermal Rating."),
    ]
    for b_title, b_desc in innovations:
        bp = doc.add_paragraph(style="List Bullet")
        bp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.12
        bp.add_run(b_title).bold = True
        bp.add_run(b_desc)

    # Section 3
    add_sec_heading("3. Rigorous Empirical, Statistical & Edge Validation")
    p3 = doc.add_paragraph()
    p3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p3.paragraph_format.space_after = Pt(3)
    p3.paragraph_format.line_spacing = 1.12
    p3.add_run(
        "Evaluated across all 16 canonical ETT configurations (ETTh1 and ETTh2 univariate OT) over "
        "three independent seeds (2024, 2025, 2026; Bessel-corrected SD, ddof=1) on dual Tesla T4 GPUs "
        "(Python 3.13.16, PyTorch 2.11.0+cu130):"
    )
    emp_bullets = [
        ("Benchmark Accuracy: ",
         "Reduces MSE by 13.22% on average over co-evaluated DLinear across all 16 settings (up to +49.92% on "
         "ETTh1 L=96, H=720; 0.0587 \u00b1 0.0010 at L=96, H=96; 0.0548 \u00b1 0.0045 at L=336, H=96), achieving "
         "0.1312 / 0.2774 global MSE/MAE vs. DLinear (0.1568 / 0.3025), PatchTST Lite (0.1386 / 0.2853), and "
         "B-Spline KAN (0.1322 / 0.2803), while outperforming NLinear on 5/8 volatile ETTh2 configurations."),
        ("Wilcoxon\u2013Holm Significance: ",
         "Exact two-sided Wilcoxon signed-rank testing verifies significant superiority over DLinear "
         "(W\u207a = 125.0, W\u207b = 11.0, z = +2.947, p_exact = 1.68\u00d710\u207b\u00b3, p_asymp = 3.20\u00d710\u207b\u00b3, Holm p_adj = 0.0084 < 0.01; "
         "14/0/2 wins) and PatchTST Lite (12/0/4 wins, +4.14% gain, p_exact = 0.0290 < 0.05, Holm p_adj = 0.1160)."),
        ("Edge-Embedded Resource Footprint: ",
         "Contains 49.8K\u2013800.7K parameters, uses 0.050\u20130.122 MB inference VRAM (B=1; 16.041\u201334.763 MB training "
         "VRAM), and sustains 3.61\u20133.88 ms synchronized T4 GPU inference latency."),
    ]
    for b_title, b_desc in emp_bullets:
        bp = doc.add_paragraph(style="List Bullet")
        bp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.12
        bp.add_run(b_title).bold = True
        bp.add_run(b_desc)

    # Section 4
    add_sec_heading("4. Mandatory Declarations & Publication Route")
    mand = [
        ("Originality: ", "This manuscript is 100% original, unpublished, and not under review elsewhere."),
        ("Publication Route (0-APC Traditional Subscription): ",
         "Submitted under Springer Nature's Traditional Subscription Model (100% 0-APC Free Publishing, zero author fees)."),
        ("Reproducibility: ",
         "Source code, dual-GPU verification scripts, 600 DPI figure scripts, CSV logs, and all 144 NPZ prediction/ground-truth "
         "arrays are openly available at https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting and archived via "
         "Zenodo (https://doi.org/10.5281/zenodo.xxxxxxx)."),
    ]
    for m_title, m_desc in mand:
        pm = doc.add_paragraph(style="List Bullet")
        pm.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pm.paragraph_format.space_after = Pt(2)
        pm.paragraph_format.line_spacing = 1.12
        pm.add_run(m_title).bold = True
        pm.add_run(m_desc)

    p_close = doc.add_paragraph()
    p_close.paragraph_format.space_before = Pt(6)
    p_close.paragraph_format.space_after = Pt(8)
    p_close.paragraph_format.keep_with_next = True
    p_close.add_run("Thank you very much for your time, consideration, and editorial stewardship.\n\nSincerely yours,")

    p_sig = doc.add_paragraph()
    p_sig.paragraph_format.line_spacing = 1.12
    p_sig.paragraph_format.keep_together = True
    rn = p_sig.add_run("Phu Toan Nguyen\n")
    rn.bold = True
    rn.font.size = Pt(11)
    p_sig.add_run(
        "Faculty of Information Technology, Hanoi University of Business and Technology (HUBT)\n"
        "No. 29A, Lane 124, Vinh Tuy Street, Vinh Hung Ward, Hai Ba Trung District, Hanoi 100000, Vietnam\n"
        "Email: nphu211206@gmail.com | ORCID iD: 0009-0005-0529-2131 (https://orcid.org/0009-0005-0529-2131)"
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COVER_LETTER_NCAA.docx")
    doc.save(out_path)
    print(f"Successfully created {out_path}")


def create_declarations():
    doc = docx.Document()
    doc.core_properties.author = "Phu Toan Nguyen"
    doc.core_properties.title = "Official Declarations - PSO-TCWavKAN (NCAA Submission)"

    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(4)
    rt = p_title.add_run("Official Declarations for Manuscript Submission")
    rt.bold = True
    rt.font.size = Pt(16)
    rt.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(12)
    p_meta.paragraph_format.line_spacing = 1.15
    rm = p_meta.add_run(
        "Target Journal: Neural Computing and Applications (Springer Nature London, ISSN: 0941-0643 / 1433-3058)\n"
        "Manuscript Title: PSO-TCWavKAN: Swarm-Optimized Temporal Continuous Wavelet Kolmogorov–Arnold "
        "Networks for Power Transformer Thermal Dynamics Forecasting\n"
        "Author: Phu Toan Nguyen (Faculty of Information Technology, Hanoi University of Business and Technology, Hanoi, Vietnam)\n"
        "Email: nphu211206@gmail.com | ORCID: 0009-0005-0529-2131"
    )
    rm.font.size = Pt(10)
    rm.font.italic = True

    declarations_data = [
        ("Funding", "This research received no external funding from agencies in the public, commercial, or not-for-profit sectors."),
        ("Competing Interests", "The author declares no competing financial or non-financial interests related to this manuscript."),
        ("Ethics Approval and Consent to Participate", "Not applicable. This study does not involve human participants, human data, human tissue, or vertebrate animals."),
        ("Consent for Publication", "Not applicable. This manuscript does not contain any individual person's data, personal identifiers, or personal images in any form."),
        ("Availability of Data and Materials",
         "All benchmark telemetry datasets analyzed in this study are publicly accessible in the Electricity Transformer "
         "Temperature (ETT) repository hosted at https://github.com/zhouhaoyi/ETDataset. Processed ETTh1 and ETTh2 "
         "partitions follow the canonical 12/4/4-month LTSF split protocol."),
        ("Code Availability",
         "Complete PyTorch training pipelines, dual-GPU Kaggle verification scripts, 600 DPI figure generation scripts, "
         "CSV experimental tables, and 144 NPZ prediction/ground-truth arrays are openly hosted at "
         "https://github.com/nphu211206/KAN-PSO-Transformer-Forecasting and preserved for permanent archival at "
         "Zenodo (https://doi.org/10.5281/zenodo.xxxxxxx)."),
        ("Author Contributions",
         "Phu Toan Nguyen (ORCID: 0009-0005-0529-2131): Conceptualization, Methodology, Formal Proofs, Software Implementation, "
         "Statistical Analysis, Visualization, Writing \u2013 Original Draft, Writing \u2013 Review & Editing."),
    ]

    for title, text in declarations_data:
        h = doc.add_heading(level=2)
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(2)
        rh = h.add_run(title)
        rh.font.name = "Times New Roman"
        rh.font.size = Pt(12)
        rh.bold = True
        rh.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        p.add_run(text)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DECLARATIONS_NCAA.docx")
    doc.save(out_path)
    print(f"Successfully created {out_path}")


if __name__ == "__main__":
    create_cover_letter()
    create_declarations()
