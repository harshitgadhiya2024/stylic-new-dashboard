import html
import logging
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from urllib.parse import urlparse
from email.mime.text import MIMEText
import httpx
from fastapi import HTTPException, status
from app.config import settings

logger = logging.getLogger(__name__)


def _marketing_home() -> tuple[str, str]:
    """Returns (base_without_trailing_slash, home_href_with_slash)."""
    base = (getattr(settings, "STYLIC_MARKETING_BASE_URL", "https://stylic.ai") or "https://stylic.ai").rstrip(
        "/"
    )
    return base, f"{base}/"


def _logo_url() -> str:
    u = (getattr(settings, "STYLIC_MARKETING_LOGO_URL", None) or "").strip()
    if u:
        return u
    b, _ = _marketing_home()
    return f"{b}/favicon.ico"


def _html_email_header_logo() -> str:
    """Clickable logotype; uses remote <img> (Gmail blocks inline SVG in HTML)."""
    s = settings
    mb = (getattr(s, "STYLIC_EMAIL_LOGO_MARGIN_BOTTOM", None) or "24px").strip()
    _base, home = _marketing_home()
    href = html.escape(home, quote=True)
    src = html.escape(_logo_url(), quote=True)
    alt = html.escape(str(s.APP_NAME), quote=True)
    return f"""<div style="text-align: center; margin-bottom: {mb};">
  <a href="{href}" style="text-decoration: none; display: inline-block;" target="_blank" rel="noopener noreferrer">
    <img src="{src}" width="130" height="48" alt="{alt}" style="display: block; margin: 0 auto; border: 0; max-width: 100%; height: auto" />
  </a>
</div>"""


def _st_email_layout_open() -> str:
    """Same outer structure as email_templates_preview.html (dark wrapper + 560px column)."""
    s = settings
    bg = html.escape((s.STYLIC_EMAIL_BODY_BG or "#0f0f12").strip(), quote=True)
    tc = html.escape((s.STYLIC_EMAIL_TEXT_PRIMARY or "#e8e8ed").strip(), quote=True)
    font = s.STYLIC_EMAIL_FONT_STACK
    mx = (s.STYLIC_EMAIL_MAX_WIDTH or "560px").strip()
    pad = (s.STYLIC_EMAIL_INNER_PADDING or "40px 20px 48px").strip()
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="dark" />
  </head>
  <body style="margin:0; padding:0; background-color:{bg};">
    <div style="margin:0; padding:0; background-color:{bg}; color:{tc}; font-family: {font};">
      <div style="max-width: {mx}; margin: 0 auto; padding: {pad};">"""


def _st_email_layout_close() -> str:
    return """
      </div>
    </div>
  </body>
</html>
"""


def _st_email_card_open() -> str:
    s = settings
    g = (s.STYLIC_EMAIL_CARD_GRADIENT or "linear-gradient(180deg, #1a1a20 0%, #12121a 100%)").strip()
    sh = (s.STYLIC_EMAIL_CARD_BOX_SHADOW or "0 24px 64px rgba(0, 0, 0, 0.4)").strip()
    return f"""<div style="background: {g}; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 36px 32px; box-shadow: {sh};">"""


def _st_email_social_icon_img_style() -> str:
    """Black simple-icons SVGs → solid white on dark footer (opacity 1; -webkit- for Apple Mail)."""
    fi = (settings.STYLIC_EMAIL_SOCIAL_ICON_FILTER or "brightness(0) invert(1)").strip()
    return (
        f"display: inline-block; border: 0; width: 24px; height: 24px; opacity: 1; "
        f"filter: {fi}; -webkit-filter: {fi};"
    )


def _html_transactional_email_footer() -> str:
    """Shared footer for dark email bodies: product name, contact, company, social icons."""
    _raw_ct = (getattr(settings, "STYLIC_EMAIL_FOOTER_CONTACT", None) or "contact@stylic.ai").strip()
    contact = html.escape(_raw_ct, quote=True)
    company = html.escape(
        (getattr(settings, "STYLIC_EMAIL_FOOTER_COMPANY", None) or "Aavish AI Labs").strip(), quote=True
    )
    ig = html.escape(
        (getattr(settings, "STYLIC_SOCIAL_INSTAGRAM_URL", None) or "https://www.instagram.com/stylicai/").strip(),
        quote=True,
    )
    fb = html.escape(
        (getattr(settings, "STYLIC_SOCIAL_FACEBOOK_URL", None) or "https://www.facebook.com/stylicai/").strip(),
        quote=True,
    )
    mailto_href = html.escape(f"mailto:{_raw_ct}", quote=True)
    name = html.escape(str(settings.APP_NAME), quote=True)

    t_muted, t_bright, t_link = (
        "#9a9aa8",
        "#c8c8d0",
        "#a8b8ff",
    )
    icon_img_style = _st_email_social_icon_img_style()
    border = (getattr(settings, "STYLIC_EMAIL_FOOTER_TOP_BORDER", None) or "1px solid rgba(255, 255, 255, 0.1)").strip()
    ig_svg = (getattr(settings, "STYLIC_EMAIL_SOCIAL_INSTAGRAM_SVG", None) or "").strip() or (
        "https://cdn.jsdelivr.net/npm/simple-icons@11.11.0/icons/instagram.svg"
    )
    fb_svg = (getattr(settings, "STYLIC_EMAIL_SOCIAL_FACEBOOK_SVG", None) or "").strip() or (
        "https://cdn.jsdelivr.net/npm/simple-icons@11.11.0/icons/facebook.svg"
    )
    ig_src = html.escape(ig_svg, quote=True)
    fb_src = html.escape(fb_svg, quote=True)
    return f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top: 28px; padding-top: 20px; border-top: {border};">
  <tr>
    <td align="left" valign="top" style="width: 70%; max-width: 70%; padding: 0 12px 0 0; vertical-align: top;">
      <p style="margin: 0 0 4px; font-size: 14px; line-height: 1.4; color: {t_bright}; font-weight: 600; text-align: left;">{name}</p>
      <p style="margin: 0 0 2px; font-size: 13px; line-height: 1.5; text-align: left;">
        <a href="{mailto_href}" style="color: {t_link}; text-decoration: none;">{contact}</a>
      </p>
      <p style="margin: 0; font-size: 12px; line-height: 1.5; color: {t_muted}; text-align: left;">{company}</p>
    </td>
    <td align="right" valign="middle" style="width: 30%; max-width: 30%; white-space: nowrap; text-align: right; vertical-align: middle;">
      <a href="{ig}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; display: inline-block; margin: 0; vertical-align: middle;">
        <img src="{ig_src}" width="24" height="24" alt="Instagram" style="{icon_img_style}" />
      </a>
      <a href="{fb}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; display: inline-block; margin: 0 0 0 10px; vertical-align: middle;">
        <img src="{fb_src}" width="24" height="24" alt="Facebook" style="{icon_img_style}" />
      </a>
    </td>
  </tr>
</table>
"""


_PURPOSE_LABELS = {
    "register":    "Email Verification",
    "login":       "Login Verification",
    "forgot_password": "Password Reset",
    "change_email":   "Email Change Verification",
}


def _build_otp_email_html(otp: str, label: str) -> str:
    name = html.escape(str(settings.APP_NAME), quote=True)
    safe_otp = html.escape(otp, quote=True)
    safe_lbl = html.escape(label.lower(), quote=True)
    exp = int(settings.OTP_EXPIRE_MINUTES)
    tpc = (settings.STYLIC_EMAIL_TEXT_PRIMARY or "#e8e8ed").strip()
    return f"""{_st_email_layout_open()}{_html_email_header_logo()}{_st_email_card_open()}
        <h1 style="margin: 0 0 8px; font-size: 22px; font-weight: 600; color: {tpc}; letter-spacing: 0.02em;">{name}</h1>
        <p style="color: #a8a8b2; font-size: 15px; margin: 0 0 8px; line-height: 1.5;">Your OTP for <strong style="color: #c4c4cc;">{safe_lbl}</strong>:</p>
        <div style="font-size: 32px; font-weight: 700; letter-spacing: 10px; color: #eceaf4; text-align: center; margin: 20px 0; padding: 20px 16px; background: rgba(0, 0, 0, 0.35); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); font-family: ui-monospace, 'SF Mono', Consolas, 'Segoe UI', monospace;">{safe_otp}</div>
        <p style="color: #7a7a85; font-size: 13px; line-height: 1.55; margin: 0;">This OTP is valid for <strong style="color: #9a9aa8;">{exp} minutes</strong>.<br />
          Do not share this with anyone. If you did not request this, please ignore this email.
        </p>
      </div>
      {_html_transactional_email_footer()}{_st_email_layout_close()}"""


async def _send_via_resend(to_email: str, subject: str, html_body: str) -> None:
    if not settings.RESEND_API_KEY or not settings.RESEND_FROM_EMAIL:
        raise RuntimeError("Resend is not configured.")

    headers = {
        "Authorization": f"Bearer {settings.RESEND_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "from":    settings.RESEND_FROM_EMAIL,
        "to":      [to_email],
        "subject": subject,
        "html":    html_body,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post("https://api.resend.com/emails", headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"Resend API error ({resp.status_code}): {resp.text}")


async def _send_via_smtp(to_email: str, subject: str, html_body: str) -> None:
    if not (settings.SMTP_SERVER and settings.SMTP_EMAIL and settings.SMTP_PASSWORD):
        raise RuntimeError("SMTP is not configured.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = settings.SMTP_EMAIL
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))

    await aiosmtplib.send(
        msg,
        hostname=settings.SMTP_SERVER,
        port=settings.SMTP_PORT,
        username=settings.SMTP_EMAIL,
        password=settings.SMTP_PASSWORD,
        start_tls=True,
    )


async def send_otp_email(to_email: str, otp: str, purpose: str = "register") -> None:
    label   = _PURPOSE_LABELS.get(purpose, "Verification")
    subject = f"{settings.APP_NAME} – {label} OTP"
    html_b  = _build_otp_email_html(otp, label)

    try:
        if settings.RESEND_API_KEY and settings.RESEND_FROM_EMAIL:
            await _send_via_resend(to_email=to_email, subject=subject, html_body=html_b)
        else:
            await _send_via_smtp(to_email=to_email, subject=subject, html_body=html_b)
    except (aiosmtplib.SMTPException, httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send email: {exc}",
        )


def _build_contact_thank_you_html(safe_first_name: str) -> str:
    """`safe_first_name` must be HTML-escaped."""
    tpc = (settings.STYLIC_EMAIL_TEXT_PRIMARY or "#e8e8ed").strip()
    return f"""{_st_email_layout_open()}{_html_email_header_logo()}{_st_email_card_open()}
        <h1 style="margin: 0 0 8px; font-size: 22px; font-weight: 600; letter-spacing: 0.02em; color: {tpc};">We received your request</h1>
        <p style="margin:0 0 20px; font-size: 15px; line-height: 1.6; color: #a8a8b2;">Hi {safe_first_name},</p>
        <p style="margin:0 0 20px; font-size: 15px; line-height: 1.7; color: #c4c4cc;">Thank you for reaching out to <strong>{html.escape(str(settings.APP_NAME))}</strong>.
          A member of our team will get back to you as soon as possible, typically within one to two
          business days.
        </p>
        <div style="margin: 24px 0; height: 1px; background: rgba(255, 255, 255, 0.08)"></div>
        <p style="margin:0; font-size: 13px; line-height: 1.5; color: #7a7a85;">This message confirms we received your contact form. If you did not submit this request,
          you can safely ignore this email.
        </p>
      </div>
      {_html_transactional_email_footer()}{_st_email_layout_close()}"""


async def send_contact_thank_you_email(to_email: str, first_name: str) -> None:
    """
    Confirmation to the user after a valid contact-sales form submission.
    All user data must be pre-sanitized plain text; names are HTML-escaped here.
    """
    if not to_email or not str(to_email).strip():
        raise RuntimeError("No recipient for contact thank-you email.")
    safe = html.escape(first_name or "there", quote=True)
    subj = f"Thank you — {settings.APP_NAME} received your request."
    body = _build_contact_thank_you_html(safe)
    try:
        if settings.RESEND_API_KEY and settings.RESEND_FROM_EMAIL:
            await _send_via_resend(to_email=to_email, subject=subj, html_body=body)
        else:
            await _send_via_smtp(to_email=to_email, subject=subj, html_body=body)
    except (aiosmtplib.SMTPException, httpx.HTTPError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to send confirmation email: {exc}") from exc


def _app_base_url() -> str:
    u = (getattr(settings, "STYLIC_APP_BASE_URL", None) or "").strip().rstrip("/")
    return u or "https://app.stylic.ai"


def _build_welcome_email_html(safe_first_name: str) -> str:
    """
    New-account welcome: platform overview and CTA to the web app.
    ``safe_first_name`` is HTML-escaped. Layout matches email_templates_preview.html.
    """
    s = settings
    name = html.escape(str(s.APP_NAME), quote=True)
    tpc = (s.STYLIC_EMAIL_TEXT_PRIMARY or "#e8e8ed").strip()
    cta_g = (s.STYLIC_EMAIL_CTA_GRADIENT or "linear-gradient(90deg, #f7e6a0 0%, #f0b0c8 45%, #a8b8ff 100%)").strip()
    _m_base, m_home = _marketing_home()
    site_href = html.escape(m_home, quote=True)
    site_label = html.escape(urlparse(_m_base).netloc or "stylic.ai", quote=True)
    base = _app_base_url().rstrip("/")
    cta_href = html.escape(f"{base}/", quote=True)
    cta_lbl = html.escape("Open Stylic Studio", quote=True)
    cta_lbl_c = (s.STYLIC_EMAIL_CTA_LABEL_COLOR or "#0f0f12").strip()
    return f"""{_st_email_layout_open()}{_html_email_header_logo()}{_st_email_card_open()}
        <h1 style="margin: 0 0 12px; font-size: 24px; font-weight: 600; letter-spacing: 0.02em; color: {tpc};">Welcome to {name}</h1>
        <p style="margin:0 0 20px; font-size: 15px; line-height: 1.6; color: #a8a8b2;">Hi {safe_first_name},</p>
        <p style="margin:0 0 20px; font-size: 15px; line-height: 1.7; color: #c4c4cc;">Thank you for connecting with <strong>{name}</strong> &mdash; your virtual studio for AI fashion photography.
          Upload garments, pick models, poses, and backgrounds, and generate studio-style campaign imagery in minutes.
        </p>
        <p style="margin:0 0 10px; font-size: 14px; line-height: 1.5; color: #b8b8c0; font-weight: 600;">What you can do</p>
        <ul style="margin: 0 0 22px; padding-left: 20px; font-size: 14px; line-height: 1.65; color: #c4c4cc;">
          <li style="margin: 6px 0;">Single, multi-pose, and catalogue &mdash; consistent shots for e-commerce and lookbooks</li>
          <li style="margin: 6px 0;">Custom models, poses, and backgrounds; brand watermarks and templates</li>
          <li style="margin: 6px 0;">Color, fabric, and texture adjustments on garments; upscale, resize, and remove background</li>
          <li style="margin: 6px 0;">High-resolution output with flexible credits and plans</li>
        </ul>
        <div style="text-align: center; margin: 28px 0 8px;">
          <a href="{cta_href}" target="_blank" rel="noopener noreferrer" style="display: inline-block; padding: 14px 28px; border-radius: 999px; text-decoration: none; font-weight: 600; font-size: 15px; color: {cta_lbl_c}; background: {cta_g};">{cta_lbl}</a>
        </div>
        <p style="margin: 16px 0 0; font-size: 12px; line-height: 1.5; color: #7a7a85; text-align: center;">Or visit <a href="{site_href}" style="color: #a8b8ff; text-decoration: none;">{site_label}</a> to learn more about plans and features.
        </p>
        <div style="margin: 24px 0; height: 1px; background: rgba(255, 255, 255, 0.08)"></div>
        <p style="margin:0; font-size: 13px; line-height: 1.5; color: #7a7a85;">We are glad you are here. If you have questions, reply to this email or use the contact options below.
        </p>
      </div>
      {_html_transactional_email_footer()}{_st_email_layout_close()}"""


async def send_welcome_email(to_email: str, first_name: str = "") -> None:
    """
    Sent after successful email-password registration or new Google (Firebase) sign-in.
    Failures are logged only so auth responses are never affected.
    """
    if not to_email or not str(to_email).strip():
        return
    try:
        safe = html.escape((first_name or "").strip() or "there", quote=True)
        subj = f"Welcome to {settings.APP_NAME} — you are in"
        body = _build_welcome_email_html(safe)
        if settings.RESEND_API_KEY and settings.RESEND_FROM_EMAIL:
            await _send_via_resend(to_email=to_email.strip().lower(), subject=subj, html_body=body)
        elif settings.SMTP_SERVER and settings.SMTP_EMAIL and settings.SMTP_PASSWORD:
            await _send_via_smtp(to_email=to_email.strip().lower(), subject=subj, html_body=body)
        else:
            logger.warning("Welcome email skipped: no Resend or SMTP configured.")
    except (aiosmtplib.SMTPException, httpx.HTTPError, RuntimeError, OSError) as exc:
        logger.warning("Could not send welcome email to %s: %s", to_email, exc)
