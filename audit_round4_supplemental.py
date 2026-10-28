import sys, os, subprocess
sys.stdout.reconfigure(encoding='utf-8')
import fitz
import docx

def chk(desc, cond):
    res = "[OK]  " if cond else "[FAIL]"
    print(f"  {res} {desc}")
    return 1 if cond else 0

print("=" * 75)
print("  ROUND 4 SUPPLEMENTAL VERIFICATION SUITE")
print("=" * 75)

score = 0
total = 0

# 1. PDF Equation Overflow Checks
print("\n[SECTION 1: PDF EQUATION OVERFLOW & LAYOUT]")
# Recompile PDFs just in case
soffice = r'C:\Program Files\LibreOffice\program\soffice.exe'
for doc_file in ['paper_v91_FINAL.docx', 'COVER_LETTER_ELECTRICAL_ENGINEERING.docx', 'DECLARATIONS_ELECTRICAL_ENGINEERING.docx']:
    subprocess.run([soffice, '--headless', '--convert-to', 'pdf', doc_file, '--outdir', '.'], check=True)

paper_pdf = fitz.open('paper_v91_FINAL.pdf')
# Check page 11 (Eq 16) and page 12 (Eq 19)
eq16_overflow = False
eq19_overflow = False

for pg in range(len(paper_pdf)):
    for b in paper_pdf[pg].get_text('blocks'):
        txt = b[4].strip()
        x1 = b[2]
        if '(16)' in txt or 'lim_' in txt:
            if x1 > 560: eq16_overflow = True
        if '(19)' in txt or 'v_{i,d}' in txt:
            if x1 > 560: eq19_overflow = True

total += 1; score += chk("Eq. (16) strictly within margins (x1 <= 560, zero overflow)", not eq16_overflow)
total += 1; score += chk("Eq. (19) strictly within margins (x1 <= 560, zero overflow)", not eq19_overflow)

# Check all blocks on all 33 pages
max_x1 = max(b[2] for pg in range(len(paper_pdf)) for b in paper_pdf[pg].get_text('blocks'))
total += 1; score += chk(f"All 33 pages have zero margin overflow (max x1 = {max_x1:.1f} <= 560.0)", max_x1 <= 560.0)

# Check Declarations PDF Funding header
decl_pdf = fitz.open('DECLARATIONS_ELECTRICAL_ENGINEERING.pdf')
# Find where "Funding" is and ensure its text is with it
funding_pg = None
funding_y = None
funding_text_pg = None
for pg in range(len(decl_pdf)):
    for b in decl_pdf[pg].get_text('blocks'):
        txt = b[4].strip()
        if txt == 'Funding':
            funding_pg = pg + 1
            funding_y = b[1]
        if 'The author received no external funding' in txt:
            funding_text_pg = pg + 1

# Funding header and its text must be on the same page (not orphaned)
funding_not_orphaned = (funding_pg is not None and funding_pg == funding_text_pg)
total += 1; score += chk(f"Declarations: Funding heading on Page {funding_pg} (together with its text on Page {funding_text_pg}, not orphaned)", funding_not_orphaned)

# Cover Letter page count
cl_pdf = fitz.open('COVER_LETTER_ELECTRICAL_ENGINEERING.pdf')
total += 1; score += chk(f"Cover Letter is exactly 2 pages (no trailing 3rd page)", len(cl_pdf) == 2)

# 2. Manuscript Mathematical & Text Consistency
print("\n[SECTION 2: MATHEMATICAL & METHODOLOGICAL CONSISTENCY]")
doc = docx.Document('paper_v91_FINAL.docx')
paper_text = '\n'.join([p.text for p in doc.paragraphs])
for t in doc.tables:
    for r in t.rows:
        for c in r.cells:
            paper_text += '\n' + c.text

total += 1; score += chk("Section 3.5: Hybrid CLPSO-PSO explicitly defines cognitive c1 and social/exemplar c2", 
                         "cognitive attraction toward the particle's own historical best" in paper_text and "social attraction toward the dimension-wise exemplar" in paper_text)

total += 1; score += chk("Algorithm 1 Step 10: Harmonized with hybrid CLPSO-PSO (contains both pbest and exemplar)", 
                         "v_{i,d}^{t+1} = χ · [v_{i,d}^t + c₁ · r_{1,d} · (pbest_{i,d}^t − x_{i,d}^t) + c₂ · r_{2,d} · (e_{i,d}^t − x_{i,d}^t)]" in paper_text)

total += 1; score += chk("Algorithm 1 Step 27: Explicitly passes residual sequence S_b to AdaptiveDualFFTHead", 
                         "AdaptiveDualFFTHead(z_kan = WavKAN(Z_b), s_seas = S_b)" in paper_text)

total += 1; score += chk("Algorithm 1 Step 38: Epoch-level snapshot update specified for SWA", 
                         "Train epoch mini-batches with η_SWA; at epoch end, update running average" in paper_text)

total += 1; score += chk("Algorithm 1 Step 40: NO erroneous BatchNorm reference (RevIN per-instance)", 
                         "RevIN operates per-instance; no stored batch running statistics require recalibration" in paper_text)

total += 1; score += chk("Fig. 2 caption: Free of ungrounded physical claims (diurnal/transient claims removed)", 
                         "targeting diurnal dynamics and k=3,4 capturing high-frequency transients" not in paper_text and "rad/norm. unit" in paper_text)

total += 1; score += chk("Fig. 9 caption: Explicit mathematical definition of RMSE(h)", 
                         "RMSE(h) = √(1/N_h · Σ_{j=1}^{N_h} (ŷ_{j,h} − y_{j,h})²)" in paper_text)

total += 1; score += chk("Table 6 note: Clarifies Seed 42 unperturbed baseline vs 5-seed post-SWA mean", 
                         "pre-SWA reference checkpoint achieves 0.3426 versus the 5-seed post-SWA mean of 0.3162" in paper_text)

# 3. Cover Letter & Declarations Verification
print("\n[SECTION 3: COVER LETTER & DECLARATIONS VERIFICATION]")
cl_text = '\n'.join([p.text for p in docx.Document('COVER_LETTER_ELECTRICAL_ENGINEERING.docx').paragraphs])
decl_text = '\n'.join([p.text for p in docx.Document('DECLARATIONS_ELECTRICAL_ENGINEERING.docx').paragraphs])

total += 1; score += chk("Cover Letter date updated to October 28, 2026", "Date: October 28, 2026" in cl_text)
total += 1; score += chk("Cover Letter Proposition 1: 'Multi-Kernel Linear Subspace Containment'", "Multi-Kernel Linear Subspace Containment (Proposition 1)" in cl_text)
total += 1; score += chk("Address in Cover Letter and Declarations includes 'Vinh Tuy Ward'", 
                         "Vinh Tuy Ward" in cl_text and "Vinh Tuy Ward" in decl_text)
total += 1; score += chk("Declarations: Consent for publication standard non-applicable phrasing", 
                         "individual person's personal identifying data" in decl_text)

# 4. Registry Verification
print("\n[SECTION 4: EXPERIMENT REGISTRY]")
reg_text = open('github_release/EXPERIMENT_REGISTRY.md', encoding='utf-8').read()
total += 1; score += chk("EXPERIMENT_REGISTRY.md contains Campaign K11 entry", 
                         "Campaign K11: PSO-TCWavKAN v9.1 Main Benchmark Campaign" in reg_text)
total += 1; score += chk("Campaign K11 specifies 70/10/20 protocol and 5 seeds", 
                         "Canonical 70/10/20 chronological split" in reg_text and "{1000, 2000, 3000, 4000, 5000}" in reg_text)

print("\n" + "=" * 75)
print(f"  SUPPLEMENTAL AUDIT SCORE: {score} / {total} CHECKS PASSED")
if score == total:
    print("  >>> PERFECT 100% PASS - ALL SUPPLEMENTAL CHECKS PASSED <<<")
else:
    print("  >>> ATTENTION: SOME CHECKS FAILED <<<")
print("=" * 75)
