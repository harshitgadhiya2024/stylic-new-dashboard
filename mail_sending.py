"""
Manual smoke-test helper for Resend.

Usage:
  export RESEND_API_KEY=...
  export RESEND_FROM_EMAIL=info@yourdomain.com
  python mail_sending.py
"""

import os
import httpx


def main() -> None:
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    from_email = os.getenv("RESEND_FROM_EMAIL", "").strip()
    to_email = os.getenv("RESEND_TEST_TO", "").strip() or "you@example.com"

    if not api_key or not from_email:
        raise RuntimeError("Missing RESEND_API_KEY or RESEND_FROM_EMAIL environment variable.")

    payload = {
        "from": from_email,
        "to": [to_email],
        "subject": "Welcome!",
        "html": "<h1>Hello!</h1><p>Thanks for signing up.</p>",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=30) as client:
        response = client.post("https://api.resend.com/emails", headers=headers, json=payload)
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()