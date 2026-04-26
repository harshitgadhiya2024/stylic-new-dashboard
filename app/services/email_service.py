import html
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import httpx
from fastapi import HTTPException, status
from app.config import settings

_PURPOSE_LABELS = {
    "register": "Email Verification",
    "login": "Login Verification",
    "forgot_password": "Password Reset",
    "change_email": "Email Change Verification",
}


def _build_otp_email_html(otp: str, label: str) -> str:
    return f"""
    <html>
      <body style="font-family: Arial, sans-serif; background: #f9f9f9; padding: 30px;">
        <div style="max-width: 480px; margin: auto; background: #fff; border-radius: 8px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
          <h2 style="color: #222; margin-bottom: 4px;">{settings.APP_NAME}</h2>
          <p style="color: #555; font-size: 15px;">Your OTP for <strong>{label.lower()}</strong>:</p>
          <div style="font-size: 36px; font-weight: bold; letter-spacing: 12px; color: #111; text-align: center; margin: 24px 0;">{otp}</div>
          <p style="color: #888; font-size: 13px;">
            This OTP is valid for <strong>{settings.OTP_EXPIRE_MINUTES} minutes</strong>.<br>
            Do not share this with anyone. If you didn't request this, please ignore this email.
          </p>
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
    return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="color-scheme" content="light dark" />
  </head>
  <body style="margin:0; padding:0; background-color:#0f0f12; color:#e8e8ed; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <div style="max-width: 560px; margin: 0 auto; padding: 40px 20px 48px;">
      <div style="text-align: center; margin-bottom: 28px;">
        <div style="display: inline-block; width: 72px; height: 72px; border-radius: 20px; background: linear-gradient(135deg, #f7e6a0 0%, #f0b0c8 50%, #a8b8ff 100%); box-shadow: 0 12px 32px rgba(0,0,0,0.45);"></div>
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
      <p style="text-align:center; margin-top: 28px; font-size: 12px; color: #5c5c66;">
        {html.escape(str(settings.APP_NAME))}
      </p>
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
    subj = f"Thank you — {settings.APP_NAME} received your message"
    body = _build_contact_thank_you_html(safe)
    try:
        if settings.RESEND_API_KEY and settings.RESEND_FROM_EMAIL:
            await _send_via_resend(to_email=to_email, subject=subj, html_body=body)
        else:
            await _send_via_smtp(to_email=to_email, subject=subj, html_body=body)
    except (aiosmtplib.SMTPException, httpx.HTTPError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to send confirmation email: {exc}") from exc
