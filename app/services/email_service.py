import html
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import httpx
from fastapi import HTTPException, status
from app.config import settings

# Brand icon images (loaded via <img> — works in Gmail; inline SVG in HTML does not).
_IG_ICON = "https://cdn.jsdelivr.net/npm/simple-icons@11.11.0/icons/instagram.svg"
_FB_ICON = "https://cdn.jsdelivr.net/npm/simple-icons@11.11.0/icons/facebook.svg"


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


def _html_email_header_logo(margin_bottom: str = "24px") -> str:
    """Clickable logotype; uses remote <img> (Gmail blocks inline SVG in HTML)."""
    _base, home = _marketing_home()
    href = html.escape(home, quote=True)
    src = html.escape(_logo_url(), quote=True)
    alt = html.escape(str(settings.APP_NAME), quote=True)
    return f"""<div style="text-align: center; margin-bottom: {margin_bottom};">
  <a href="{href}" style="text-decoration: none; display: inline-block;" target="_blank" rel="noopener noreferrer">
    <img src="{src}" width="130" height="48" alt="{alt}" style="display: block; margin: 0 auto; border: 0; max-width: 100%; height: auto;" />
  </a>
</div>"""


def _html_transactional_email_footer(*, for_dark_bg: bool) -> str:
    """
    Shared footer: product name, contact email, company, social (icon images).
    ``for_dark_bg`` adjusts text/link colors for dark vs light body backgrounds.
    """
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

    if for_dark_bg:
        t_muted, t_bright, t_link, border, icon_f = (
            "#9a9aa8",
            "#c8c8d0",
            "#a8b8ff",
            "rgba(255,255,255,0.1)",
            "opacity:0.9; filter: brightness(0) invert(1);",
        )
    else:
        t_muted, t_bright, t_link, border, icon_f = (
            "#4a4a55",
            "#1a1a1e",
            "#3d5a99",
            "rgba(0,0,0,0.08)",
            "",
        )
    ig_src = html.escape(_IG_ICON, quote=True)
    fb_src = html.escape(_FB_ICON, quote=True)
    return f"""
<div style="text-align: center; margin-top: 28px; padding-top: 20px; border-top: 1px solid {border};">
  <p style="margin: 0 0 4px; font-size: 14px; line-height: 1.4; color: {t_bright}; font-weight: 600;">{name}</p>
  <p style="margin: 0 0 2px; font-size: 13px; line-height: 1.5;">
    <a href="{mailto_href}" style="color: {t_link}; text-decoration: none;">{contact}</a>
  </p>
  <p style="margin: 0 0 14px; font-size: 12px; line-height: 1.5; color: {t_muted};">{company}</p>
  <div style="font-size: 0; line-height: 0;">
    <a href="{ig}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; display: inline-block; margin: 0 6px; vertical-align: middle;">
      <img src="{ig_src}" width="24" height="24" alt="Instagram" style="display: block; border: 0; {icon_f}" />
    </a>
    <a href="{fb}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; display: inline-block; margin: 0 6px; vertical-align: middle;">
      <img src="{fb_src}" width="24" height="24" alt="Facebook" style="display: block; border: 0; {icon_f}" />
    </a>
  </div>
</div>
"""


_PURPOSE_LABELS = {
    "register":    "Email Verification",
    "login":       "Login Verification",
    "forgot_password": "Password Reset",
    "change_email":   "Email Change Verification",
}


def _build_otp_email_html(otp: str, label: str) -> str:
    return f"""
    <html>
      <body style="font-family: 'Segoe UI', Tahoma, Arial, sans-serif; background: #f3f3f5; padding: 30px;">
        <div style="max-width: 480px; margin: 0 auto;">
          <div style="text-align: center; padding: 16px 20px 8px; background: #0f0f12; border-radius: 8px 8px 0 0;">
            {_html_email_header_logo(margin_bottom="8px")}
          </div>
          <div style="background: #fff; border-radius: 0 0 8px 8px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
          <h2 style="color: #222; margin: 0 0 4px; font-size: 18px;">{html.escape(str(settings.APP_NAME), quote=True)}</h2>
          <p style="color: #555; font-size: 15px; margin: 0 0 8px;">Your OTP for <strong>{html.escape(label.lower(), quote=True)}</strong>:</p>
          <div style="font-size: 36px; font-weight: bold; letter-spacing: 12px; color: #111; text-align: center; margin: 24px 0;">{html.escape(otp, quote=True)}</div>
          <p style="color: #888; font-size: 13px;">
            This OTP is valid for <strong>{settings.OTP_EXPIRE_MINUTES} minutes</strong>.<br>
            Do not share this with anyone. If you did not request this, please ignore this email.
          </p>
          </div>
          {_html_transactional_email_footer(for_dark_bg=False)}
        </div>
      </body>
    </html>
    """


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
    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="color-scheme" content="light dark" />
  </head>
  <body style="margin:0; padding:0; background-color:#0f0f12; color:#e8e8ed; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <div style="max-width: 560px; margin: 0 auto; padding: 40px 20px 48px;">
      {_html_email_header_logo()}
      <div style="background: linear-gradient(180deg, #1a1a20 0%, #12121a 100%); border: 1px solid rgba(255,255,255,0.08);
                  border-radius: 16px; padding: 36px 32px; box-shadow: 0 24px 64px rgba(0,0,0,0.4);">
        <h1 style="margin:0 0 8px; font-size: 22px; font-weight: 600; letter-spacing: 0.02em;">
          We received your request
        </h1>
        <p style="margin:0 0 20px; font-size: 15px; line-height: 1.6; color: #a8a8b2;">
          Hi {safe_first_name},
        </p>
        <p style="margin:0 0 20px; font-size: 15px; line-height: 1.7; color: #c4c4cc;">
          Thank you for reaching out to <strong>{html.escape(str(settings.APP_NAME))}</strong>.
          A member of our team will get back to you as soon as possible, typically within one to two
          business days.
        </p>
        <div style="margin: 24px 0; height: 1px; background: rgba(255,255,255,0.08);"></div>
        <p style="margin:0; font-size: 13px; line-height: 1.5; color: #7a7a85;">
          This message confirms we received your contact form. If you did not submit this request,
          you can safely ignore this email.
        </p>
      </div>
      {_html_transactional_email_footer(for_dark_bg=True)}
    </div>
  </body>
</html>
"""


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
