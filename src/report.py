"""
src/report.py
-------------
Automated 8D Corrective Action Request (CAR) PDF Generator using ReportLab.
Produces a publication-quality 8D Reliability Engineering Report:
  - D1: Team & Champion
  - D2: Problem Statement & Field Failure Symptoms
  - D3: Interim Containment Actions (ICA)
  - D4: Root Cause Analysis (5-Why & Physics-of-Failure)
  - D5: Permanent Corrective Actions (PCA)
  - D6: Verification & Validation Plan
  - D7: Prevention of Recurrence & FMEA Updates
  - D8: Approval Sign-Off & Cost Avoidance Target
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
CAR_DIR = os.path.join(REPORTS_DIR, "car_reports")

# Planted root cause stories lookup
PLANTED_ROOT_CAUSES = {
    "Capacitor Aging": {
        "symptom": "Premature dry-out and electrolyte venting of DC bus intermediate capacitors leading to inverter bridge shutdown.",
        "root_cause": "Substandard rubber vent seal polymer degradation from secondary supplier lot (LOT-CAP-2024-Q3) under 65C ambient.",
        "containment": "Quarantine factory inventory of LOT-CAP-2024-Q3; issue Service Advisory SA-2025-08 for proactive field inspection.",
        "corrective_action": "Switch to automotive-grade 105C rated solid electrolyte capacitors with hermetic glass-to-metal seals (ECO-4421).",
        "prevention": "Mandate 1,000-hour accelerated thermal endurance testing on 100% of incoming capacitor lots; update DFMEA #702.",
        "five_why": [
            "Why did the UPS fail? -> Inverter tripped on DC bus overvoltage and high ripple current.",
            "Why was ripple high? -> Electrolytic capacitor bank lost >40% of rated capacitance.",
            "Why did capacitance degrade? -> Electrolyte dried out through ruptured vent seals prematurely.",
            "Why did seals rupture? -> Substandard butyl rubber blend failed thermal resistance limits.",
            "Why was substandard material used? -> Secondary supplier changed compound formulation without engineering approval notification.",
        ],
    },
    "Contact Wear": {
        "symptom": "Severe contact erosion and pitting on main silver-tungsten arcing contacts causing breaker thermal trips.",
        "root_cause": "Arc chute splitter plates clogged with conductive soot during heavy inductive switching, delaying arc quenching.",
        "containment": "Dispatch field kits with reinforced arc runners; mandate 500-cycle micro-ohm contact resistance audits.",
        "corrective_action": "Redesign contact tip geometry with Ag-W-C composite alloy providing 40% higher arc-erosion resistance (ECO-3890).",
        "prevention": "Integrate optical arc-wear telemetry sensors in new ACB trip units; update breaker maintenance standard SOP-55.",
        "five_why": [
            "Why did the ACB trip? -> Contact resistance exceeded 120 uOhm, causing localized thermal runaway.",
            "Why was resistance high? -> Main arcing contact tips were severely eroded and oxidized.",
            "Why was erosion accelerated? -> Electric arc lingered on contacts 18ms longer than specified during interruption.",
            "Why did the arc linger? -> Arc chute de-ion plates accumulated conductive copper soot, reducing dielectric recovery.",
            "Why did soot accumulate? -> Chute vent geometry restricted exhaust gas flow under heavy inductive fault interruptions.",
        ],
    },
    "Corrosion": {
        "symptom": "Heavy copper patina oxidation and galvanic corrosion on terminal busbars and auxiliary terminals.",
        "root_cause": "High relative humidity (>85% RH) combined with coastal airborne sulfur and chlorine causing copper sulfidation on unplated joints.",
        "containment": "Apply CRC industrial corrosion inhibitor spray to all coastal panels during scheduled preventive maintenance.",
        "corrective_action": "Standardize electro-tin plating (minimum 12 microns) across all copper busbars and upgrade enclosure to IP55 with Gore breathers.",
        "prevention": "Revise coastal application engineering guidelines; enforce 1,000-hr ASTM B117 salt spray qualification for marine regions.",
        "five_why": [
            "Why did terminals overheat? -> Contact resistance spiked at incoming terminal busbar splice joints.",
            "Why did resistance spike? -> Thick non-conductive copper sulfide patina formed across mating surfaces.",
            "Why did sulfidation occur? -> Atmospheric H2S and saline moisture penetrated outdoor enclosure.",
            "Why did moisture penetrate? -> Standard neoprene door gaskets degraded under coastal UV and thermal cycling.",
            "Why were standard gaskets specified? -> Environmental classification in bill-of-materials failed to account for marine industrial zoning.",
        ],
    },
}

DEFAULT_STORY = {
    "symptom": "Field failure rate exceeded statistical control thresholds requiring formal 8D corrective action investigation.",
    "root_cause": "Component manufacturing process drift combined with thermal-electrical stress cycling in field operating environments.",
    "containment": "Quarantine suspect warehouse inventory; initiate targeted field replacement campaign for affected serial numbers.",
    "corrective_action": "Implement engineering design revision ECO-5100 with upgraded component tolerances and supplier PPAP re-certification.",
    "prevention": "Update design rules, incorporate automated end-of-line functional screening, and conduct monthly supplier quality audits.",
    "five_why": [
        "Why did the unit fail in service? -> Subsystem operational parameters drifted outside allowable safety limits.",
        "Why did parameters drift? -> Component degradation under continuous thermal-electrical duty cycles.",
        "Why did degradation accelerate? -> Margin of safety between component rating and peak operating stress was insufficient.",
        "Why was margin insufficient? -> Initial worst-case design analysis did not account for harmonic distortion in industrial installations.",
        "Why was harmonic stress overlooked? -> Design verification plan lacked mandatory non-linear load profile testing.",
    ],
}


def generate_car_pdf(
    car_data: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Generates a structured, executive 8D Corrective Action Request (CAR) PDF report.

    Args:
        car_data: Dictionary containing issue metrics and details
        output_path: Destination path for the generated PDF

    Returns:
        Absolute file path to the generated PDF.
    """
    os.makedirs(CAR_DIR, exist_ok=True)
    car_id = car_data.get("car_id", "CAR-2026-001")
    if output_path is None:
        output_path = os.path.join(CAR_DIR, f"{car_id}_8D_Report.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b"),
    )
    meta_label = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#475569"),
        fontName="Helvetica-Bold",
    )
    meta_val = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # 1. Header Banner Table
    header_data = [
        [
            Paragraph("<b>ELECTRICAL RELIABILITY ENGINEERING DEPARTMENT</b><br/><font size=7 color='#64748b'>QUALITY & RELIABILITY ASSURANCE SYSTEM</font>", body_style),
            Paragraph(f"<font size=12><b>8D CORRECTIVE ACTION REPORT</b></font><br/><b>REF: {car_id}</b> | STATUS: <b>OPEN - CONTAINED</b>", body_style),
        ]
    ]
    t_header = Table(header_data, colWidths=[3.25 * inch, 4.25 * inch])
    t_header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 8))

    # Meta parameters
    prod = car_data.get("product_family", "Online UPS 10kVA")
    fm = car_data.get("failure_mode", "Capacitor Aging")
    ppit_score = car_data.get("ppit_score", 88.5)
    total_cost = car_data.get("total_warranty_cost", 125000.0)
    claims_count = car_data.get("total_claims", 320)
    recent_ppm = car_data.get("recent_ppm", 18500.0)
    baseline_ppm = car_data.get("baseline_ppm", 3200.0)
    sigma_dev = car_data.get("sigma_deviation", 4.2)

    story_data = PLANTED_ROOT_CAUSES.get(fm, DEFAULT_STORY)

    # 2. Executive Metadata Grid
    meta_grid = [
        [
            Paragraph("PRODUCT FAMILY", meta_label),
            Paragraph(str(prod), meta_val),
            Paragraph("PPIT PRIORITY SCORE", meta_label),
            Paragraph(f"<b>{ppit_score} / 100 (High Risk)</b>", meta_val),
        ],
        [
            Paragraph("FAILURE MODE", meta_label),
            Paragraph(str(fm), meta_val),
            Paragraph("WARRANTY EXPENSE", meta_label),
            Paragraph(f"${total_cost:,.2f}", meta_val),
        ],
        [
            Paragraph("TOTAL CLAIMS IN FIELD", meta_label),
            Paragraph(f"{claims_count:,} claims", meta_val),
            Paragraph("RECENT PPM / BASELINE", meta_label),
            Paragraph(f"{recent_ppm:,.0f} vs {baseline_ppm:,.0f} (+{sigma_dev:.1f}σ)", meta_val),
        ],
        [
            Paragraph("INITIATION DATE", meta_label),
            Paragraph(datetime.now().strftime("%Y-%m-%d"), meta_val),
            Paragraph("TARGET CLOSURE DATE", meta_label),
            Paragraph("2026-04-30", meta_val),
        ],
    ]
    t_meta = Table(meta_grid, colWidths=[1.8 * inch, 2.0 * inch, 1.8 * inch, 1.9 * inch])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # 3. D1: Team Formation
    story.append(Paragraph("<b>D1: CROSS-FUNCTIONAL TEAM & SPONSORSHIP</b>", h2_style))
    d1_data = [
        ["Champion:", "VP of Quality & Reliability", "Team Leader:", "Lead Reliability Engineer (Placeholder)"],
        ["SME Engineers:", "Principal Power Electronics Engineer, Mechanical Busbar Specialist", "Supplier QA:", "Senior Component Quality Manager"],
    ]
    t_d1 = Table([[Paragraph(c, body_style) for c in row] for row in d1_data], colWidths=[1.3 * inch, 2.4 * inch, 1.3 * inch, 2.5 * inch])
    t_d1.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_d1)
    story.append(Spacer(1, 6))

    # 4. D2: Problem Description
    story.append(Paragraph("<b>D2: PROBLEM STATEMENT & FIELD SYMPTOMS</b>", h2_style))
    d2_text = f"<b>Problem Statement:</b> Field warranty returns for <b>{prod}</b> exhibiting <b>{fm}</b> have accelerated significantly (+{sigma_dev:.1f} sigma above baseline), generating {claims_count:,} service claims and totaling ${total_cost:,.2f} in repair expense.<br/><b>Observed Symptoms:</b> {story_data['symptom']}"
    story.append(Paragraph(d2_text, body_style))
    story.append(Spacer(1, 6))

    # 5. D3: Interim Containment Action (ICA)
    story.append(Paragraph("<b>D3: INTERIM CONTAINMENT ACTIONS (ICA)</b>", h2_style))
    story.append(Paragraph(f"• {story_data['containment']}<br/>• 100% incoming dock quarantine placed on suspect shipment batches; field service inventory purged.<br/>• Interim verification: 0 defects escaped to field in past 14 days under containment protocol.", body_style))
    story.append(Spacer(1, 6))

    # 6. D4: Root Cause Analysis & 5-Why
    story.append(Paragraph("<b>D4: ROOT CAUSE ANALYSIS & 5-WHY INVESTIGATION</b>", h2_style))
    story.append(Paragraph(f"<b>Root Cause Verified:</b> {story_data['root_cause']}", body_style))
    story.append(Spacer(1, 4))

    # 5-Why Table
    five_why_rows = [[Paragraph(f"<b>Why {i+1}:</b>", meta_label), Paragraph(q, body_style)] for i, q in enumerate(story_data["five_why"])]
    t_why = Table(five_why_rows, colWidths=[0.8 * inch, 6.7 * inch])
    t_why.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t_why)
    story.append(Spacer(1, 6))

    # 7. D5: Permanent Corrective Action (PCA)
    story.append(Paragraph("<b>D5: PERMANENT CORRECTIVE ACTION (PCA)</b>", h2_style))
    story.append(Paragraph(f"• <b>Engineering Change:</b> {story_data['corrective_action']}<br/>• Redesigned sub-assembly tested under full HALT (Highly Accelerated Life Testing) conditions.<br/>• Supplier contract updated with mandatory PPAP Level 3 documentation and statistical Cpk $\ge$ 1.67.", body_style))
    story.append(Spacer(1, 6))

    # 8. D6: Verification & Implementation
    story.append(Paragraph("<b>D6: PCA IMPLEMENTATION & VERIFICATION RESULTS</b>", h2_style))
    d6_data = [
        ["Pilot Production Run:", "250 prototype units manufactured under revised ECO specification."],
        ["Environmental Validation:", "Passed 500-hr thermal-shock (-40C to +85C) and 1,000-hr endurance cycling with 0 degradation."],
        ["Cut-in Serial Number:", f"Implemented into production effective Serial SN-{prod[:3].upper()}-2026-CUT01."],
    ]
    t_d6 = Table([[Paragraph(r[0], meta_label), Paragraph(r[1], body_style)] for r in d6_data], colWidths=[2.0 * inch, 5.5 * inch])
    t_d6.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_d6)
    story.append(Spacer(1, 6))

    # 9. D7: Prevention of Recurrence
    story.append(Paragraph("<b>D7: SYSTEMIC ACTIONS TO PREVENT RECURRENCE</b>", h2_style))
    story.append(Paragraph(f"• {story_data['prevention']}<br/>• Updated Failure Mode and Effects Analysis (DFMEA & PFMEA) risk priority numbers (RPN reduced from 280 to 24).<br/>• Standardized quarterly supplier audits and raw material spectral analysis.", body_style))
    story.append(Spacer(1, 6))

    # 10. D8: Closure & Sign-off
    story.append(Paragraph("<b>D8: TEAM RECOGNITION & EXECUTIVE APPROVAL</b>", h2_style))
    d8_data = [
        ["Target Post-Fix PPM:", "< 800 PPM", "Est. Warranty Cost Avoided:", f"${total_cost * 0.75:,.2f}"],
        ["Reliability Lead Sign-off:", "Placeholder, CRE", "Quality Director Sign-off:", "Placeholder, VP Quality"],
    ]
    t_d8 = Table([[Paragraph(c, body_style) for c in row] for row in d8_data], colWidths=[1.8 * inch, 1.9 * inch, 1.8 * inch, 2.0 * inch])
    t_d8.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_d8)

    doc.build(story)
    print(f"Generated 8D CAR PDF report -> {output_path}")
    return output_path


def generate_all_top_car_reports():
    """Generates 8D reports for top PPIT ranked issues."""
    from src.db import load_dataset
    from src.metrics import rank_ppit_car

    claims_df = load_dataset("claims_clean")
    sales_df = load_dataset("sales")
    products_df = load_dataset("products")

    ranking = rank_ppit_car(claims_df, sales_df, products_df)
    top_issues = ranking.head(3)

    generated_files = []
    for _, issue in top_issues.iterrows():
        car_dict = issue.to_dict()
        pdf_path = generate_car_pdf(car_dict)
        generated_files.append(pdf_path)

    return generated_files


def main():
    print("=" * 60)
    print("GENERATING 8D CORRECTIVE ACTION REQUEST (CAR) PDF REPORTS")
    print("=" * 60)
    files = generate_all_top_car_reports()
    print(f"Successfully generated {len(files)} CAR PDF reports in {CAR_DIR}")


if __name__ == "__main__":
    main()
