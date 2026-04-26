import html
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import httpx
from fastapi import HTTPException, status
from app.config import settings
from app.services.email_stylic_logo_svg import STYLIC_LOGO_SVG


def _marketing_home() -> tuple[str, str]:
    """Returns (base_without_trailing_slash, home_href_with_slash)."""
    base = (getattr(settings, "STYLIC_MARKETING_BASE_URL", "https://stylic.ai") or "https://stylic.ai").rstrip(
        "/"
    )
    return base, f"{base}/"


def _email_footer_with_base_url(style_light: bool = True) -> str:
    """
    Single-line footer with public base URL. ``style_light`` uses darker text on light backgrounds.
    """
    _base, home = _marketing_home()
    base_label = html.escape(_base, quote=True)
    href = html.escape(home, quote=True)
    c = "#6b6b76" if style_light else "#5c5c66"
    return f"""<p style="text-align:center; margin: 20px 0 0; font-size: 12px; line-height: 1.6; color: {c};">
  <a href="{href}" style="color: {c}; text-decoration: underline;" target="_blank" rel="noopener noreferrer">{base_label}</a>
</p>"""

_PURPOSE_LABELS = {
    "register": "Email Verification",
    "login": "Login Verification",
    "forgot_password": "Password Reset",
    "change_email": "Email Change Verification",
}


def _build_otp_email_html(otp: str, label: str) -> str:
    _b, home = _marketing_home()
    a_home = html.escape(home, quote=True)
    return f"""
    <html>
      <body style="font-family: 'Segoe UI', Tahoma, Arial, sans-serif; background: #f3f3f5; padding: 30px;">
        <div style="max-width: 480px; margin: 0 auto;">
          <div style="text-align: center; padding: 22px 20px; background: #0f0f12; border-radius: 8px 8px 0 0;">
            <a href="{a_home}" style="text-decoration: none; display: inline-block;" target="_blank" rel="noopener noreferrer">
              {STYLIC_LOGO_SVG}
            </a>
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
          {_email_footer_with_base_url(style_light=True)}
        </div>
      </body>
    </html>
    """


async def _send_via_resend(to_email: str, subject: str, html_body: str) -> None:
    if not settings.RESEND_API_KEY or not settings.RESEND_FROM_EMAIL:
        raise RuntimeError("Resend is not configured.")

    headers = {
        "Authorization": f"Bearer {settings.RESEND_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "from": settings.RESEND_FROM_EMAIL,
        "to": [to_email],
        "subject": subject,
        "html": html_body,
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
    msg["From"] = settings.SMTP_EMAIL
    msg["To"] = to_email
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
    label = _PURPOSE_LABELS.get(purpose, "Verification")
    subject = f"{settings.APP_NAME} – {label} OTP"
    html_body = _build_otp_email_html(otp, label)

    try:
        # Preferred provider: Resend. Falls back to SMTP for backward compatibility.
        if settings.RESEND_API_KEY and settings.RESEND_FROM_EMAIL:
            await _send_via_resend(to_email=to_email, subject=subject, html_body=html_body)
        else:
            await _send_via_smtp(to_email=to_email, subject=subject, html_body=html_body)
    except (aiosmtplib.SMTPException, httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send email: {exc}",
        )


def _build_contact_thank_you_html(safe_first_name: str) -> str:
    """`safe_first_name` must be HTML-escaped."""
    _base, home_href = _marketing_home()
    a_home = html.escape(home_href, quote=True)
    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="color-scheme" content="light dark" />
  </head>
  <body style="margin:0; padding:0; background-color:#0f0f12; color:#e8e8ed; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <div style="max-width: 560px; margin: 0 auto; padding: 40px 20px 48px;">
      <div style="text-align: center; margin-bottom: 24px;">
        <a href="{a_home}" style="text-decoration: none; display: inline-block;" target="_blank" rel="noopener noreferrer">
          {STYLIC_LOGO_SVG}
        </a>
        <div style="margin: 10px 0 0;">
          <a href="{a_home}" style="color: #a8b8ff; font-size: 15px; text-decoration: none; font-weight: 500;"
             target="_blank" rel="noopener noreferrer">{a_home}</a>
        </div>
      </div>
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
      {_email_footer_with_base_url(style_light=False)}
      <p style="text-align:center; margin: 8px 0 0; font-size: 11px; color: #4a4a52;">{html.escape(str(settings.APP_NAME), quote=True)}</p>
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
