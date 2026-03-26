"""
NeuroScan — Clinical AI-Assisted Radiology Report
White-background, professional radiology-style PDF.
"""

import io
import base64
import random
import string
from datetime import datetime
from PIL import Image as PILImage

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor

# ── PALETTE ───────────────────────────────────────────────────────────────────
WHITE      = HexColor('#FFFFFF')
BG         = HexColor('#F7F8FA')
NAVY       = HexColor('#1B2A4A')
MID_GRAY   = HexColor('#5A6478')
LIGHT_GRAY = HexColor('#8A919E')
RULE_GRAY  = HexColor('#D8DCE6')
TABLE_ALT  = HexColor('#F0F2F7')
RED_ACCENT   = HexColor('#C0392B')
AMBER_ACCENT = HexColor('#B8620A')
GREEN_ACCENT = HexColor('#1A7A50')
BLUE_ACCENT  = HexColor('#2563EB')

W, H   = A4
MARGIN = 44
# Reserve bottom 80 pt for footer — nothing renders below this
FOOTER_TOP = 80

CLASS_ACCENT = {
    'Glioma': RED_ACCENT, 'Meningioma': AMBER_ACCENT,
    'No Tumor': GREEN_ACCENT, 'Pituitary': BLUE_ACCENT,
}
CLASS_SEVERITY = {
    'Glioma': 'HIGH', 'Meningioma': 'MODERATE',
    'No Tumor': 'NONE — NORMAL STUDY', 'Pituitary': 'MODERATE',
}
CLASS_RECO = {
    'Glioma':
        'Urgent referral to neuro-oncology. Contrast-enhanced MRI and MR spectroscopy '
        'recommended. Neurosurgical evaluation for biopsy and resection planning.',
    'Meningioma':
        'Neurology or neurosurgery referral. Contrast-enhanced MRI for lesion '
        'characterisation and surgical planning. Observation protocol for small, '
        'asymptomatic lesions.',
    'No Tumor':
        'No immediate intervention required. Clinical correlation with presenting '
        'symptoms advised. Routine follow-up imaging if symptoms persist or worsen.',
    'Pituitary':
        'Endocrinology referral and full hormonal panel assessment. Dedicated '
        'pituitary-protocol MRI (thin-slice coronal and sagittal sequences). '
        'Ophthalmology consult if visual field deficits are reported.',
}

# Clinical prose — no "AI model" language, passive/objective voice
CLASS_FINDINGS = {
    'Glioma': (
        'Imaging features are consistent with an intra-axial neoplastic lesion of '
        'probable glial origin. The signal characteristics and spatial distribution '
        'pattern are most consistent with a primary glioma. Focal areas of abnormal '
        'signal intensity are identified, with the region of highest diagnostic '
        'significance highlighted on the attached attention map (Fig. 2). Gliomas '
        'represent the most common primary malignant brain tumour, arising from '
        'astrocytes, oligodendrocytes, or ependymal cells, and are classified '
        'Grade I–IV per the WHO 2021 CNS Tumour Classification. High-grade lesions '
        '(Grade III–IV) typically demonstrate contrast enhancement, mass effect, '
        'and surrounding vasogenic oedema on T2/FLAIR sequences. Molecular markers '
        'including IDH1/2 mutation status and MGMT promoter methylation are critical '
        'prognostic determinants and should be assessed at histopathological analysis.'
    ),
    'Meningioma': (
        'Imaging features are consistent with an extra-axial meningeal-based mass '
        'lesion. The signal pattern and anatomical distribution are most consistent '
        'with a meningioma arising from the arachnoid cap cells of the meninges, '
        'as highlighted on the attached attention map (Fig. 2). Meningiomas '
        'represent approximately 37% of all primary brain tumours, are predominantly '
        'WHO Grade I (benign), and are more prevalent in women aged 40–70. They '
        'are typically iso- to hyperdense on non-contrast CT and demonstrate avid, '
        'homogeneous enhancement post-contrast. A significant proportion are '
        'incidental findings; clinical significance is determined by lesion size, '
        'location, and symptomatology. Grade II–III variants carry higher recurrence '
        'risk and require closer follow-up.'
    ),
    'No Tumor': (
        'No focal areas of abnormal signal intensity, mass lesion, or atypical '
        'enhancement pattern consistent with intracranial neoplasia are identified '
        'within the four diagnostic categories examined (Glioma, Meningioma, '
        'Pituitary Adenoma). The attention map (Fig. 2) confirms that appropriate '
        'anatomical regions were examined in reaching this determination. This '
        'result should be correlated with clinical presentation; pathologies outside '
        'the scope of this system — including cerebral metastases, CNS lymphoma, '
        'and medulloblastoma — cannot be excluded by this analysis. Expert '
        'radiological review is recommended if clinical suspicion persists.'
    ),
    'Pituitary': (
        'Imaging features are consistent with a sellar or parasellar lesion, most '
        'consistent with a pituitary adenoma. The region of diagnostic significance '
        'is centred at the expected location of the pituitary fossa, as shown on '
        'the attached attention map (Fig. 2). Pituitary adenomas arise from '
        'secretory cells of the adenohypophysis and are classified by size — '
        'microadenoma (less than 10 mm) or macroadenoma (10 mm or greater) — and '
        'by hormonal activity. Macroadenomas may exert mass effect on the optic '
        'chiasm, producing characteristic bitemporal visual field loss. '
        'Prolactinomas are managed with dopamine agonists; other subtypes typically '
        'require transsphenoidal surgical resection.'
    ),
}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _pil_to_reader(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return ImageReader(buf)


def _b64_to_reader(b64_str):
    if b64_str.startswith('data:'):
        b64_str = b64_str.split(',', 1)[1]
    return ImageReader(io.BytesIO(base64.b64decode(b64_str)))


def _wrap(c, text, x, y, max_w, font, size, color, lh=None):
    """Word-wrap text, return y after last line."""
    if lh is None:
        lh = size * 1.55
    c.setFont(font, size)
    c.setFillColor(color)
    words = text.split()
    line, cur_y = '', y
    for word in words:
        test = (line + ' ' + word).strip()
        if c.stringWidth(test, font, size) <= max_w:
            line = test
        else:
            if line:
                c.drawString(x, cur_y, line)
                cur_y -= lh
            line = word
    if line:
        c.drawString(x, cur_y, line)
        cur_y -= lh
    return cur_y


def _estimate_wrap_height(c, text, max_w, font, size, lh=None):
    """Return how many pts the wrapped text will consume."""
    if lh is None:
        lh = size * 1.55
    words = text.split()
    line, lines = '', 0
    for word in words:
        test = (line + ' ' + word).strip()
        if c.stringWidth(test, font, size) <= max_w:
            line = test
        else:
            lines += 1
            line = word
    if line:
        lines += 1
    return lines * lh


def _hrule(c, y, x0=MARGIN, x1=None, color=RULE_GRAY, lw=0.5):
    c.setStrokeColor(color)
    c.setLineWidth(lw)
    c.line(x0, y, x1 if x1 else W - MARGIN, y)


def _sec(c, y, num, title):
    """Section label. Returns y below the label."""
    c.setFont('Helvetica-Bold', 7)
    c.setFillColor(LIGHT_GRAY)
    c.drawString(MARGIN, y, num)
    c.setFillColor(NAVY)
    c.drawString(MARGIN + 20, y, title)
    rw = c.stringWidth(title, 'Helvetica-Bold', 7)
    c.setStrokeColor(NAVY)
    c.setLineWidth(0.8)
    c.line(MARGIN + 20, y - 3, MARGIN + 20 + rw, y - 3)
    return y - 14


# ── MAIN ──────────────────────────────────────────────────────────────────────
def generate_report(scan_image_pil, predicted_class, confidence,
                    probabilities, gradcam_b64=None, report_id=None):

    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f'NeuroScan Radiology Report — {predicted_class}')
    c.setAuthor('NeuroScan Diagnostic System')

    now        = datetime.now()
    ts_display = now.strftime('%d %B %Y  %H:%M')
    ts_study   = now.strftime('%d/%m/%Y')
    rid        = report_id or ('NS-' + ''.join(
                     random.choices(string.ascii_uppercase + string.digits, k=8)))

    accent   = CLASS_ACCENT.get(predicted_class, NAVY)
    severity = CLASS_SEVERITY.get(predicted_class, 'UNKNOWN')
    is_normal = predicted_class == 'No Tumor'

    # ── BACKGROUND ────────────────────────────────────────────────────────────
    c.setFillColor(WHITE)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # ── FIXED FOOTER (drawn first so content never renders over it) ───────────
    footer_rule_y = FOOTER_TOP - 4
    _hrule(c, footer_rule_y, color=RULE_GRAY, lw=0.5)

    # Three equal columns
    col = (W - 2 * MARGIN) / 3

    # Left — Reporting system
    c.setFont('Helvetica-Bold', 7)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, footer_rule_y - 12, 'REPORTING SYSTEM')
    c.setFont('Helvetica', 7)
    c.setFillColor(MID_GRAY)
    c.drawString(MARGIN, footer_rule_y - 22, 'NeuroScan Diagnostic System v4.0')
    c.drawString(MARGIN, footer_rule_y - 32, f'Generated: {ts_display}')
    c.drawString(MARGIN, footer_rule_y - 42, f'Report ID: {rid}')

    # Centre — Referring physician
    cx = MARGIN + col + col / 2
    c.setFont('Helvetica-Bold', 7)
    c.setFillColor(NAVY)
    c.drawCentredString(cx, footer_rule_y - 12, 'REFERRING PHYSICIAN')
    c.setFont('Helvetica', 7)
    c.setFillColor(MID_GRAY)
    c.drawCentredString(cx, footer_rule_y - 26, '_________________________________')
    c.drawCentredString(cx, footer_rule_y - 36, 'Name / Signature / Date')

    # Right — Disclaimer
    dx = MARGIN + col * 2
    dw = col - 4
    c.setFont('Helvetica-Bold', 6.5)
    c.setFillColor(RED_ACCENT)
    c.drawString(dx, footer_rule_y - 12, 'MEDICAL DISCLAIMER')
    c.setFont('Helvetica', 6)
    c.setFillColor(LIGHT_GRAY)
    disc = ('For academic/research use only. Not a certified medical device. '
            'Not for clinical diagnosis or treatment. Consult a qualified '
            'healthcare professional for all medical decisions.')
    _wrap(c, disc, dx, footer_rule_y - 22, dw, 'Helvetica', 6, LIGHT_GRAY, lh=8)

    # ── HEADER ────────────────────────────────────────────────────────────────
    hdr_h = 56
    c.setFillColor(NAVY)
    c.rect(0, H - hdr_h, W, hdr_h, fill=1, stroke=0)

    # Logo / system name
    c.setFont('Helvetica-Bold', 17)
    c.setFillColor(WHITE)
    c.drawString(MARGIN, H - 32, 'NeuroScan')
    c.setFont('Helvetica', 9)
    c.setFillColor(HexColor('#8BA3C7'))
    c.drawString(MARGIN, H - 46, 'AI-Assisted Cranial MRI Analysis  |  EfficientNetB0')

    # Right meta
    c.setFont('Helvetica', 7.5)
    c.setFillColor(HexColor('#8BA3C7'))
    c.drawRightString(W - MARGIN, H - 22, f'REPORT ID: {rid}')
    c.setFont('Helvetica-Bold', 7.5)
    c.setFillColor(WHITE)
    c.drawRightString(W - MARGIN, H - 33, f'Generated: {ts_display}')

    # Accent stripe
    c.setFillColor(accent)
    c.rect(0, H - hdr_h, W, 3, fill=1, stroke=0)

    y = H - hdr_h - 18

    # ── SECTION 01: PATIENT & STUDY INFO ──────────────────────────────────────
    y = _sec(c, y, '01', 'PATIENT & STUDY INFORMATION')

    panel_h = 58
    c.setFillColor(BG); c.setStrokeColor(RULE_GRAY); c.setLineWidth(0.5)
    c.roundRect(MARGIN, y - panel_h, W - 2 * MARGIN, panel_h, 4, fill=1, stroke=1)

    cw3 = (W - 2 * MARGIN - 24) / 3
    fields = [
        ('Patient ID',    'PT-XXXXXXXX'),
        ('Date of Birth', '--/--/----'),
        ('Sex',           'Not specified'),
        ('Study Date',    ts_study),
        ('Modality',      'MRI Brain (Non-contrast)'),
        ('Accession No.', f'ACC-{rid[-6:]}'),
    ]
    for i, (lbl, val) in enumerate(fields):
        col_i, row_i = i % 3, i // 3
        fx = MARGIN + 10 + col_i * (cw3 + 8)
        fy = y - 14 - row_i * 26
        c.setFont('Helvetica', 6.5); c.setFillColor(LIGHT_GRAY); c.drawString(fx, fy, lbl.upper())
        c.setFont('Helvetica-Bold', 8.5); c.setFillColor(NAVY); c.drawString(fx, fy - 11, val)

    y -= panel_h + 14

    # ── SECTION 02: RESULT ────────────────────────────────────────────────────
    y = _sec(c, y, '02', 'ANALYSIS RESULT')

    banner_h = 52
    tr = min(255, int(accent.red   * 255 * 0.08 + 255 * 0.92))
    tg = min(255, int(accent.green * 255 * 0.08 + 255 * 0.92))
    tb = min(255, int(accent.blue  * 255 * 0.08 + 255 * 0.92))
    c.setFillColor(HexColor('#%02x%02x%02x' % (tr, tg, tb)))
    c.setStrokeColor(accent); c.setLineWidth(1)
    c.roundRect(MARGIN, y - banner_h, W - 2 * MARGIN, banner_h, 4, fill=1, stroke=1)
    c.setFillColor(accent); c.roundRect(MARGIN, y - banner_h, 4, banner_h, 2, fill=1, stroke=0)

    label = 'NO INTRACRANIAL NEOPLASM DETECTED' if is_normal else f'{predicted_class.upper()} — TUMOUR DETECTED'
    c.setFont('Helvetica-Bold', 13); c.setFillColor(accent)
    c.drawString(MARGIN + 14, y - 20, label)
    c.setFont('Helvetica', 8); c.setFillColor(MID_GRAY)
    c.drawString(MARGIN + 14, y - 34, f'Confidence: {confidence * 100:.1f}%')
    c.drawString(MARGIN + 14 + 120, y - 34, f'Risk Level: {severity}')

    # Confidence pill
    px, py = W - MARGIN - 72, y - banner_h + 10
    c.setFillColor(accent); c.roundRect(px, py, 62, 24, 12, fill=1, stroke=0)
    c.setFont('Helvetica-Bold', 12); c.setFillColor(WHITE)
    c.drawCentredString(px + 31, py + 7, f'{confidence * 100:.1f}%')

    y -= banner_h + 14

    # ── SECTION 03: IMAGING ───────────────────────────────────────────────────
    y = _sec(c, y, '03', 'IMAGING — ORIGINAL SCAN & ATTENTION MAP')

    img_sz = 175
    gap    = 14
    ix1    = MARGIN
    ix2    = MARGIN + img_sz + gap

    for ix in (ix1, ix2):
        c.setFillColor(HexColor('#F0F2F7')); c.setStrokeColor(RULE_GRAY); c.setLineWidth(0.8)
        c.roundRect(ix, y - img_sz, img_sz, img_sz, 4, fill=1, stroke=1)

    try:
        c.drawImage(_pil_to_reader(scan_image_pil.resize((300, 300))),
                    ix1 + 1, y - img_sz + 1, width=img_sz - 2, height=img_sz - 2,
                    preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    if gradcam_b64:
        try:
            c.drawImage(_b64_to_reader(gradcam_b64),
                        ix2 + 1, y - img_sz + 1, width=img_sz - 2, height=img_sz - 2,
                        preserveAspectRatio=True, mask='auto')
        except Exception:
            pass

    cap_y = y - img_sz - 10
    for ix, fig, cap1, cap2 in [
        (ix1, 'FIGURE 1', 'Original MRI Scan (Input)', ''),
        (ix2, 'FIGURE 2', 'Grad-CAM: Region of Interest', '(Gradient-weighted Class Activation Map)'),
    ]:
        c.setFont('Helvetica-Bold', 7); c.setFillColor(MID_GRAY)
        c.drawCentredString(ix + img_sz / 2, cap_y, fig)
        c.setFont('Helvetica', 7); c.setFillColor(LIGHT_GRAY)
        c.drawCentredString(ix + img_sz / 2, cap_y - 10, cap1)
        if cap2:
            c.drawCentredString(ix + img_sz / 2, cap_y - 19, cap2)

    # Grad-CAM legend
    if gradcam_b64:
        lx, ly = ix2 + 8, cap_y - 33
        lw = img_sz - 16
        for i in range(int(lw)):
            t = i / lw
            rc = int(t * 255); gc = int(max(0, 255 - abs(t - 0.5) * 510)); bc = int(255 * (1 - t))
            c.setFillColor(HexColor(f'#{rc:02x}{gc:02x}{bc:02x}'))
            c.rect(lx + i, ly, 1, 5, fill=1, stroke=0)
        c.setFont('Helvetica', 6.5); c.setFillColor(LIGHT_GRAY)
        c.drawString(lx, ly - 8, 'Low activation')
        c.drawRightString(lx + lw, ly - 8, 'High activation')

    # Probability table — right of images
    tbl_x = ix2 + img_sz + 12
    tbl_w = W - MARGIN - tbl_x
    tbl_y = y
    if tbl_w > 70:
        c.setFont('Helvetica-Bold', 7); c.setFillColor(NAVY)
        c.drawString(tbl_x, tbl_y, 'CLASS PROBABILITIES')
        _hrule(c, tbl_y - 4, x0=tbl_x, x1=W - MARGIN)
        tbl_y -= 14
        row_h = 19
        for i, cls in enumerate(['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']):
            prob = probabilities.get(cls, 0.0)
            is_p = cls == predicted_class
            ca   = CLASS_ACCENT[cls]
            c.setFillColor(TABLE_ALT if i % 2 == 0 else WHITE)
            c.rect(tbl_x, tbl_y - row_h, tbl_w, row_h, fill=1, stroke=0)
            if is_p:
                c.setFillColor(ca); c.rect(tbl_x, tbl_y - row_h, 3, row_h, fill=1, stroke=0)
            c.setFont('Helvetica-Bold' if is_p else 'Helvetica', 7.5)
            c.setFillColor(NAVY if is_p else MID_GRAY)
            c.drawString(tbl_x + 7, tbl_y - row_h + 6, cls)
            bx, by2, bmw = tbl_x + 7, tbl_y - row_h + 2, tbl_w - 30
            c.setFillColor(RULE_GRAY); c.roundRect(bx, by2, bmw, 3, 1, fill=1, stroke=0)
            c.setFillColor(ca if is_p else HexColor('#BDC3CE'))
            c.roundRect(bx, by2, max(3, bmw * prob), 3, 1, fill=1, stroke=0)
            c.setFont('Helvetica-Bold' if is_p else 'Helvetica', 7)
            c.setFillColor(NAVY if is_p else LIGHT_GRAY)
            c.drawRightString(W - MARGIN, tbl_y - row_h + 6, f'{prob * 100:.1f}%')
            tbl_y -= row_h

    y -= img_sz + 40
    _hrule(c, y + 4); y -= 12

    # ── SECTION 04: FINDINGS ──────────────────────────────────────────────────
    y = _sec(c, y, '04', 'FINDINGS')
    findings_text = CLASS_FINDINGS[predicted_class]
    text_w = W - 2 * MARGIN
    y = _wrap(c, findings_text, MARGIN, y, text_w, 'Helvetica', 9, MID_GRAY, lh=14)
    y -= 8

    # ── SECTION 05: RECOMMENDATION ────────────────────────────────────────────
    if y > FOOTER_TOP + 60:
        y = _sec(c, y, '05', 'RECOMMENDATION')
        y = _wrap(c, CLASS_RECO[predicted_class], MARGIN, y,
                  W - 2*MARGIN, 'Helvetica', 9, MID_GRAY, lh=14)
        y -= 12

    # ── SECTION 06: TECHNICAL DETAILS ─────────────────────────────────────────
    if y > FOOTER_TOP + 80:
        _hrule(c, y + 4); y -= 10
        y = _sec(c, y, '06', 'SYSTEM & TECHNICAL DETAILS')
        tech = [
            ('Classification System', 'EfficientNetB0 — Transfer Learning (ImageNet pretrained)'),
            ('Training Data',         'Brain Tumour MRI Dataset — 13,305 scans (Kaggle + Mendeley combined)'),
            ('Output Classes',        'Glioma / Meningioma / No Tumour / Pituitary Adenoma'),
            ('Performance',           'Val. Accuracy: 97.02%  |  Test Accuracy: 93.48%  |  Macro F1: 0.9388'),
            ('Explainability',        'Gradient-weighted Class Activation Mapping (Grad-CAM)'),
            ('Serving Framework',     'TensorFlow 2.13 / Keras  |  Flask REST API'),
        ]
        cw2 = (W - 2*MARGIN - 10) / 2
        for i, (lbl, val) in enumerate(tech):
            if y - (i // 2 + 1) * 20 < FOOTER_TOP + 10:
                break
            tx = MARGIN + (i % 2) * (cw2 + 10)
            ty = y - (i // 2) * 20
            c.setFont('Helvetica', 6.5); c.setFillColor(LIGHT_GRAY); c.drawString(tx, ty, lbl.upper())
            c.setFont('Helvetica', 8); c.setFillColor(MID_GRAY)
            vs = val if c.stringWidth(val, 'Helvetica', 8) <= cw2 - 4 else val[:60] + '...'
            c.drawString(tx, ty - 11, vs)

    c.save()
    return buf.getvalue()