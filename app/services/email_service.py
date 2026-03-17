import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import HTTPException, status
from app.config import settings

_PURPOSE_LABELS = {
    "register": "Email Verification",
    "login": "Login Verification",
    "forgot_password": "Password Reset",
}


def send_otp_email(to_email: str, otp: str, purpose: str = "register") -> None:
    label = _PURPOSE_LABELS.get(purpose, "Verification")
    subject = f"{settings.APP_NAME} – {label} OTP"

    html_body = f"""
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

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_EMAIL
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.SMTP_EMAIL, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_EMAIL, to_email, msg.as_string())
    except smtplib.SMTPException as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send email: {exc}",
        )
