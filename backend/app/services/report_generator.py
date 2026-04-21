"""
PDF Compliance Report Generator — powered by ReportLab
Produces a downloadable A4 audit report for legal/HR teams.
"""

from io import BytesIO
from datetime import datetime
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── Colour palette ────────────────────────────────────────────────────────
DARK  = colors.HexColor("#0f172a")
LENS  = colors.HexColor("#6366f1")
GREEN = colors.HexColor("#22c55e")
AMBER = colors.HexColor("#eab308")
RED   = colors.HexColor("#ef4444")
LIGHT = colors.HexColor("#e2e8f0")
MUTED = colors.HexColor("#94a3b8")
WHITE = colors.white


def _score_color(score: int) -> colors.HexColor:
    if score >= 80:
        return GREEN
    if score >= 60:
        return AMBER
    return RED


def _status_color(status: str) -> colors.HexColor:
    return GREEN if status == "PASS" else RED if status == "FAIL" else AMBER


def generate_pdf_report(result: dict[str, Any]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", fontSize=22, textColor=WHITE, spaceAfter=4,
                         fontName="Helvetica-Bold", alignment=TA_LEFT)
    h2 = ParagraphStyle("h2", fontSize=13, textColor=LENS, spaceAfter=6,
                         fontName="Helvetica-Bold", spaceBefore=14)
    body = ParagraphStyle("body", fontSize=9, textColor=LIGHT, spaceAfter=4,
                           fontName="Helvetica", leading=14)
    mono = ParagraphStyle("mono", fontSize=8, textColor=MUTED, spaceAfter=2,
                           fontName="Courier", leading=12)
    caption = ParagraphStyle("caption", fontSize=8, textColor=MUTED,
                              fontName="Helvetica", alignment=TA_CENTER)

    summary = result.get("summary", {})
    fair_score = result.get("fair_score", {})
    compliance_tags = result.get("compliance_tags", [])
    slice_metrics = result.get("slice_metrics", [])
    gemini = result.get("gemini_analysis", {})
    audit_id = result.get("audit_id", "—")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    story = []

    # ── Cover header ────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("FairLens™", ParagraphStyle("brand", fontSize=26, textColor=LENS,
                                               fontName="Helvetica-Bold")),
        Paragraph(f"Audit ID: {audit_id}<br/>{now}", caption),
    ]]
    header_tbl = Table(header_data, colWidths=["70%", "30%"])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (0, -1), 12),
    ]))
    story.append(header_tbl)
    story.append(HRFlowable(width="100%", thickness=2, color=LENS, spaceAfter=10))

    story.append(Paragraph("AI Bias & Fairness Compliance Report", h1))
    story.append(Paragraph(
        f"Model type: <b>{result.get('model_type', '—')}</b> &nbsp;|&nbsp; "
        f"Dataset: <b>{result.get('dataset_source', '—')}</b> &nbsp;|&nbsp; "
        f"Samples: <b>{summary.get('total_samples', '—')}</b> &nbsp;|&nbsp; "
        f"Hotspots: <b>{summary.get('hotspot_count', 0)}</b>",
        ParagraphStyle("meta", fontSize=9, textColor=MUTED, fontName="Helvetica",
                       spaceAfter=12),
    ))

    # ── FairScore ───────────────────────────────────────────────────────────
    if fair_score:
        score = fair_score.get("score", 0)
        label = fair_score.get("label", "—")
        sc = _score_color(score)
        story.append(Paragraph("Overall Fairness Score", h2))
        score_data = [[
            Paragraph(f'<font size="36" color="{sc.hexval()}">{score}</font>', caption),
            Paragraph(
                f'<font size="14" color="{sc.hexval()}"><b>{label}</b></font><br/>'
                f'<font size="9" color="{MUTED.hexval()}">out of 100 &nbsp;·&nbsp; '
                f'overall bias score {summary.get("overall_bias_score", "—")}</font>',
                body,
            ),
        ]]
        score_tbl = Table(score_data, colWidths=["20%", "80%"])
        score_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
            ("ROUNDEDCORNERS", [6, 6, 6, 6]),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(score_tbl)
        story.append(Spacer(1, 10))

    # ── Regulatory Compliance ───────────────────────────────────────────────
    if compliance_tags:
        story.append(Paragraph("Regulatory Compliance", h2))
        comp_rows = [["Regulation", "Domain", "Status", "SPD", "DI"]]
        for tag in compliance_tags:
            sc = _status_color(tag["status"])
            comp_rows.append([
                Paragraph(f"<b>{tag['label']}</b>", mono),
                Paragraph(tag["domain"], mono),
                Paragraph(f'<font color="{sc.hexval()}"><b>{tag["status"]}</b></font>', mono),
                Paragraph(str(tag.get("worst_spd", "—")), mono),
                Paragraph(str(tag.get("worst_di", "—")), mono),
            ])
        comp_tbl = Table(comp_rows, colWidths=["30%", "22%", "13%", "17%", "18%"])
        comp_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LENS),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#1e293b"), colors.HexColor("#0f172a")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#334155")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(comp_tbl)
        story.append(Spacer(1, 10))

    # ── Gemini Narrative ────────────────────────────────────────────────────
    if gemini and gemini.get("headline"):
        story.append(Paragraph("Gemini AI Analysis", h2))
        story.append(Paragraph(f'<b>{gemini.get("headline", "")}</b>', body))
        story.append(Paragraph(
            f'Severity: <b>{gemini.get("severity", "—")}</b> &nbsp;|&nbsp; '
            f'Bias type: <b>{gemini.get("bias_type", "—")}</b> &nbsp;|&nbsp; '
            f'Most affected: <b>{gemini.get("most_affected_group", "—")}</b>',
            mono,
        ))
        findings = gemini.get("key_findings", [])
        if findings:
            story.append(Spacer(1, 4))
            for f in findings:
                story.append(Paragraph(f"• {f}", body))
        if gemini.get("real_world_impact"):
            story.append(Paragraph(f'<b>Impact:</b> {gemini["real_world_impact"]}', body))
        if gemini.get("legal_risk"):
            story.append(Paragraph(f'<b>Legal risk:</b> {gemini["legal_risk"]}', body))
        if gemini.get("recommended_action"):
            story.append(Paragraph(f'<b>Recommended action:</b> {gemini["recommended_action"]}', body))
        story.append(Spacer(1, 6))

    # ── Top Bias Findings ────────────────────────────────────────────────────
    if slice_metrics:
        story.append(Paragraph("Top Bias Findings (by magnitude)", h2))
        metric_rows = [["Slice", "SPD", "DI", "Pos. Rate", "Samples", "Flagged"]]
        for m in slice_metrics[:15]:
            flagged = "⚠ YES" if m.get("flagged") else "OK"
            flagged_color = RED if m.get("flagged") else GREEN
            metric_rows.append([
                Paragraph(m["label"], mono),
                Paragraph(f'{m["statistical_parity_diff"]:+.4f}', mono),
                Paragraph(f'{m["disparate_impact"]:.4f}', mono),
                Paragraph(f'{m["positive_rate"]:.2%}', mono),
                Paragraph(str(m["size"]), mono),
                Paragraph(f'<font color="{flagged_color.hexval()}">{flagged}</font>', mono),
            ])
        metric_tbl = Table(metric_rows, colWidths=["34%", "12%", "12%", "14%", "12%", "16%"])
        metric_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LENS),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#1e293b"), colors.HexColor("#0f172a")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#334155")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(metric_tbl)
        story.append(Spacer(1, 10))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceBefore=16, spaceAfter=6))
    story.append(Paragraph(
        f"Generated by FairLens™ · Audit {audit_id} · {now} · Powered by Gemini 2.5 Flash",
        ParagraphStyle("footer", fontSize=7, textColor=MUTED, fontName="Helvetica",
                       alignment=TA_CENTER),
    ))

    doc.build(story)
    return buf.getvalue()
