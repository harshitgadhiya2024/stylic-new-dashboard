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
