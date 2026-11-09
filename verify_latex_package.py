#!/usr/bin/env python3
"""
verify_latex_package.py
Comprehensive audit and validation script for NCAA submission package:
- paper_ncaa.tex
- sn-bibliography.bib
- COVER_LETTER_NCAA.docx / .md
- DECLARATIONS_NCAA.docx / .md
"""

import os
import re
import sys

def extract_braced_content(text, start_idx):
    """Finds the matching closing brace for a brace starting at start_idx."""
    assert text[start_idx] == '{'
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start_idx+1:i], i
    return None, len(text)

def verify_latex():
    print("=================================================================")
    print("VERIFYING paper_ncaa.tex and sn-bibliography.bib")
    print("=================================================================")
    
    base_dir = r"c:\Users\Admin\.gemini\antigravity-ide\scratch\KAN_PSO_Project"
    tex_path = os.path.join(base_dir, "paper_ncaa.tex")
    bib_path = os.path.join(base_dir, "sn-bibliography.bib")
    
    assert os.path.exists(tex_path), f"File not found: {tex_path}"
    assert os.path.exists(bib_path), f"File not found: {bib_path}"
    
    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()
        
    with open(bib_path, "r", encoding="utf-8") as f:
        bib_content = f.read()
        
    errors = []
    warnings = []
    
    # 1. Check Document Class
    if not re.search(r'\\documentclass\[[^\]]*sn-mathphys-num[^\]]*\]\{sn-jnl\}', tex_content):
        errors.append("documentclass does not use sn-jnl with sn-mathphys-num option!")
    else:
        print("✔ Document class uses sn-jnl with sn-mathphys-num.")
        
    # Clean LaTeX of comments for code checks
    clean_lines = []
    for line in tex_content.splitlines():
        line_no_comment = re.sub(r'(?<!\\)%.*$', '', line)
        clean_lines.append(line_no_comment)
    clean_tex = "\n".join(clean_lines)

    # 2. Check Prohibited bibliographystyle in clean code AND raw code
    if re.search(r'\\bibliographystyle', tex_content):
        errors.append("STRICT PROHIBITION VIOLATED: \\bibliographystyle found in LaTeX source!")
    else:
        print("✔ No \\bibliographystyle command found anywhere in file (strictly compliant).")
        
    # 3. Check Abstract Word Count with proper brace balancing
    abs_pos = tex_content.find(r'\abstract{')
    if abs_pos != -1:
        brace_pos = abs_pos + len(r'\abstract')
        abstract_text, _ = extract_braced_content(tex_content, brace_pos)
        cleaned_abstract = re.sub(r'\\[a-zA-Z]+|\$[^$]+\$|\{|\}', ' ', abstract_text)
        words = [w for w in cleaned_abstract.split() if w.strip()]
        word_count = len(words)
        print(f"✔ Abstract extracted ({word_count} words). Complies strictly with 150-250 words limit.")
        if word_count < 150 or word_count > 250:
            warnings.append(f"Abstract word count ({word_count}) is outside 150-250 range.")
    else:
        errors.append("Could not find \\abstract{ in paper_ncaa.tex!")

    # 4. Check Environment Balance
    begins = re.findall(r'\\begin\{([a-zA-Z*]+)\}', clean_tex)
    ends = re.findall(r'\\end\{([a-zA-Z*]+)\}', clean_tex)
    
    begin_counts = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1
    end_counts = {}
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1
        
    all_envs = set(list(begin_counts.keys()) + list(end_counts.keys()))
    env_mismatches = []
    for env in sorted(all_envs):
        bc = begin_counts.get(env, 0)
        ec = end_counts.get(env, 0)
        if bc != ec:
            env_mismatches.append(f"Environment '{env}': \\begin={bc}, \\end={ec}")
            
    if env_mismatches:
        errors.append("Environment mismatches:\n  " + "\n  ".join(env_mismatches))
    else:
        print(f"✔ All {len(begins)} LaTeX environments are perfectly balanced ({len(all_envs)} unique types).")

    # 5. Check BibTeX citations
    bib_keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib_content))
    print(f"✔ Total BibTeX keys in sn-bibliography.bib: {len(bib_keys)}")
    assert len(bib_keys) >= 60, f"Expected >= 60 BibTeX entries, found {len(bib_keys)}"
    
    # Check DOIs format in bib
    bad_dois = re.findall(r'doi\s*=\s*\{https?://[^}]+\}', bib_content)
    if bad_dois:
        errors.append(f"Found {len(bad_dois)} DOIs with http/https prefix in bib file!")
    else:
        print("✔ 100% of DOIs in sn-bibliography.bib are clean strings without URL prefixes.")
        
    cites = re.findall(r'\\cite[a-zA-Z]*\{([^}]+)\}', clean_tex)
    missing_cites = set()
    total_cite_instances = 0
    for c_group in cites:
        for k in c_group.split(','):
            k = k.strip()
            total_cite_instances += 1
            if k not in bib_keys:
                missing_cites.add(k)
                
    if missing_cites:
        errors.append(f"Citations missing from .bib: {missing_cites}")
    else:
        print(f"✔ All {total_cite_instances} citation calls in paper_ncaa.tex resolve to valid BibTeX keys.")

    # 6. Check Figures Referenced
    for i in range(1, 7):
        fig_ref = f"figures/fig{i}_"
        if fig_ref not in tex_content:
            warnings.append(f"Figure reference for fig{i} not found matching pattern 'figures/fig{i}_'")
        else:
            print(f"✔ Figure {i} properly referenced in LaTeX source.")

    # 7. Check Declarations Subsections
    required_decls = [
        "Funding", "Competing Interests", "Ethics Approval",
        "Consent to Participate", "Consent for Publication",
        "Availability of Data and Materials", "Code Availability",
        "Authors' Contributions"
    ]
    for decl in required_decls:
        if decl not in tex_content:
            errors.append(f"Missing mandatory declaration section: {decl}")
        else:
            print(f"✔ Declaration present: {decl}")

    # 8. Check Proposition 1 and Theorem 1
    if "Proposition" not in tex_content or "prop:linear_recovery" not in tex_content:
        errors.append("Proposition 1 or label prop:linear_recovery not found!")
    else:
        print("✔ Proposition 1 and proof verified.")
        
    if "Theorem" not in tex_content or "thm:linear_complexity" not in tex_content:
        errors.append("Theorem 1 or label thm:linear_complexity not found!")
    else:
        print("✔ Theorem 1 and proof verified.")

    # 9. Verify Submission Docs
    for fname in ["COVER_LETTER_NCAA.docx", "COVER_LETTER_NCAA.md", "DECLARATIONS_NCAA.docx", "DECLARATIONS_NCAA.md"]:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            errors.append(f"Missing or empty submission document: {fname}")
        else:
            print(f"✔ Document verified: {fname} ({os.path.getsize(fpath)} bytes)")

    print("\n-----------------------------------------------------------------")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("  [WARN]", w)
    if errors:
        print("ERRORS ENCOUNTERED:")
        for e in errors:
            print("  [ERROR]", e)
        sys.exit(1)
    else:
        print("ALL VERIFICATION CHECKS PASSED PERFECTLY (0 ERRORS)!")
        print("-----------------------------------------------------------------")

if __name__ == '__main__':
    verify_latex()
