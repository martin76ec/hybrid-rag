"""Generate synthetic test PDFs with known entities and facts.

Each PDF has a distinct domain with named entities, numerical facts, and
relationships — making it possible to measure retrieval precision and answer
correctness quantitatively.
"""

from __future__ import annotations

from pathlib import Path
from fpdf import FPDF

FIXTURES_DIR = Path(__file__).resolve().parent


def _write_pdf(path: Path, title: str, paragraphs: list[str]) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for para in paragraphs:
        pdf.multi_cell(0, 5, para)
        pdf.ln(3)
    pdf.output(str(path))
    return path


def generate_tech_company_pdf() -> Path:
    """PDF about a fictional tech company with specific metrics."""
    paragraphs = [
        "NovaMind Technologies was founded in 2018 by Dr. Elena Vasquez and Raj Patel in Austin, Texas. "
        "The company specializes in neuromorphic computing chips that mimic biological neural networks. "
        "Their flagship product, the Cortex-7 processor, achieves 120 tera-operations per second while "
        "consuming only 15 watts of power, which is 8x more efficient than competing solutions from "
        "SynapTech and BrainGrid Inc.",
        "In fiscal year 2024, NovaMind reported revenue of $340 million, representing a 62% year-over-year "
        "increase. The company employs 1,250 people across offices in Austin, Berlin, and Singapore. "
        "Dr. Vasquez serves as CEO while Patel holds the position of Chief Technology Officer. "
        "The board of directors is chaired by former DARPA director General Marcus Webb (retired).",
        "NovaMind's research division, led by Dr. Yuki Tanaka, published 47 peer-reviewed papers in 2024 "
        "on topics including spike-timing-dependent plasticity and memristor-based learning rules. "
        "The company holds 89 patents globally, with 23 additional patents pending. Their R&D budget "
        "for 2025 is set at $95 million, up from $72 million the previous year.",
        "The Cortex-7 processor is manufactured at NovaMind's fabrication facility in Dresden, Germany, "
        "using a 5-nanometer process node licensed from EuroChip Foundries. Production yield currently "
        "stands at 94.2%, up from 87.1% in the previous generation. Each wafer contains approximately "
        "1,800 die sites, and the average selling price per unit is $4,200.",
        "Key partnerships include a multi-year supply agreement with Quantum Dynamics Ltd for "
        "cryogenic memory modules, and a research collaboration with MIT's Center for Brains, Minds "
        "and Machines. NovaMind is also part of the European Neuromorphic Computing Consortium alongside "
        "Heidelberg University and ETH Zurich. The company plans to IPO in Q3 2025 at a target "
        "valuation of $5.8 billion.",
    ]
    return _write_pdf(
        FIXTURES_DIR / "tech_company.pdf",
        "NovaMind Technologies Annual Report 2024",
        paragraphs,
    )


def generate_clinical_trial_pdf() -> Path:
    """PDF about a fictional clinical trial with specific endpoints."""
    paragraphs = [
        "Study Protocol VXC-204: A Phase III Randomized Controlled Trial of Veratralimab for Advanced "
        "Pulmonary Sarcoidosis. Principal Investigator: Dr. Amara Okafor, MD, PhD, University of "
        "Michigan Medical Center. Sponsor: Veritas Biopharma Inc. Study period: March 2022 to "
        "September 2024. ClinicalTrials.gov identifier: NCT-05489231.",
        "The trial enrolled 648 participants across 42 sites in 9 countries. Eligible patients were "
        "aged 18-75 years with biopsy-confirmed Stage III or IV pulmonary sarcoidosis and a "
        "FVC (forced vital capacity) between 40% and 80% predicted. Participants were randomized "
        "1:1 to Veratralimab 300mg IV every 4 weeks or placebo, with both arms receiving standard "
        "of care including corticosteroids at stable doses.",
        "The primary endpoint was change in FVC % predicted from baseline to week 52. Veratralimab "
        "demonstrated a mean improvement of +6.8 percentage points versus -0.4 for placebo "
        "(p < 0.001, 95% CI 4.9-9.1). The key secondary endpoint, improvement in St. George's "
        "Respiratory Questionnaire (SGRQ) total score, showed a -7.5 point change versus -1.2 for "
        "placebo (p = 0.003). A responder analysis showed 58% of treatment arm patients achieved a "
        "clinically meaningful FVC improvement of at least 5%, compared to 19% in the placebo arm.",
        "Adverse events were generally manageable. The most common treatment-emergent adverse events "
        "in the Veratralimab arm were upper respiratory tract infection (12.3%), fatigue (9.8%), "
        "and mild infusion reactions (7.1%). Three patients (0.9%) discontinued due to serious "
        "adverse events: one case of pneumocystis pneumonia and two cases of grade 3 hepatotoxicity. "
        "There were no treatment-related deaths during the study period.",
        "Subgroup analysis revealed that patients with the HLA-DRB1*04 allele showed the greatest "
        "treatment benefit, with a mean FVC improvement of +9.3 percentage points. Pharmacokinetic "
        "modeling showed a half-life of 18.6 days and steady-state concentrations achieved by week 12. "
        "Veritas Biopharma plans to submit a Biologics License Application to the FDA in Q1 2025 "
        "and to the EMA in Q2 2025. If approved, Veratralimab would be the first disease-modifying "
        "therapy specifically indicated for advanced pulmonary sarcoidosis.",
    ]
    return _write_pdf(
        FIXTURES_DIR / "clinical_trial.pdf",
        "VXC-204 Clinical Trial Results",
        paragraphs,
    )


def generate_all_pdfs() -> list[Path]:
    return [generate_tech_company_pdf(), generate_clinical_trial_pdf()]
