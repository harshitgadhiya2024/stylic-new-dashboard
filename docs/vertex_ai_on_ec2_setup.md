# Vertex AI Setup on EC2 (Detailed)

This guide explains how to run `scripts/standalone_gemini_photoshoot.py` on an AWS EC2 instance using **Vertex AI**.

The script uses **Vertex API key** authentication (recommended for this flow):

```bash
pip install --upgrade google-genai
export GOOGLE_CLOUD_API_KEY="YOUR_API_KEY"
```

Put the same variable in `.env` on EC2 if you use `python-dotenv`.  
(Service-account / ADC setup below is optional if you use other tools or older flows.)

---

## 1) Prerequisites

- A Google Cloud project with billing enabled.
- Access to create service accounts and IAM roles.
- An AWS EC2 instance (Ubuntu recommended) with internet access.
- Python 3.10+ on EC2.
- Script dependencies installed:
  - `google-genai`
  - `python-dotenv`
  - `httpx`
  - `boto3`

Example:

```bash
pip install --upgrade google-genai python-dotenv httpx boto3
```

---

## 2) Enable Required Google Cloud APIs

In your Google Cloud project, enable:

- `Vertex AI API`
- `Cloud Resource Manager API` (usually already available)

Using `gcloud`:

```bash
gcloud services enable aiplatform.googleapis.com
```

---

## 3) Create Service Account for EC2

Create a dedicated service account, for example:

- Name: `ec2-vertex-photoshoot`

Assign minimum required IAM role:

- `roles/aiplatform.user` (Vertex AI User)

Optional (for broader debugging/operations):

- `roles/viewer`

Using `gcloud`:

```bash
gcloud iam service-accounts create ec2-vertex-photoshoot \
  --display-name="EC2 Vertex Photoshoot"
```

Bind role:

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:ec2-vertex-photoshoot@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

---

## 4) Create and Download Service Account Key

Generate JSON key:

```bash
gcloud iam service-accounts keys create vertex-sa-key.json \
  --iam-account=ec2-vertex-photoshoot@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

Copy this key to EC2 securely (for example with `scp`) and store it with restricted permissions.

On EC2:

```bash
mkdir -p ~/secrets
mv vertex-sa-key.json ~/secrets/vertex-sa-key.json
chmod 600 ~/secrets/vertex-sa-key.json
```

---

## 5) Configure Environment Variables on EC2

Set these variables (in `.env` or shell profile):

- `GOOGLE_CLOUD_PROJECT_ID=YOUR_PROJECT_ID`
- `GOOGLE_CLOUD_LOCATION=us-central1` (or your chosen supported region)
- `GOOGLE_APPLICATION_CREDENTIALS=/home/ubuntu/secrets/vertex-sa-key.json`

Also keep your existing app variables:

- `SEEDDREAM_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `AWS_S3_BUCKET_NAME`

Example `.env` block:

```env
GOOGLE_CLOUD_PROJECT_ID=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/home/ubuntu/secrets/vertex-sa-key.json

SEEDDREAM_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
AWS_S3_BUCKET_NAME=your-bucket
```

---

## 6) Region and Model Availability

Vertex AI model availability depends on region.  
Before running, confirm your model is available in selected region:

- `gemini-3-pro-image-preview`

If unavailable in your region, switch `GOOGLE_CLOUD_LOCATION` to a supported one.

---

## 7) Verify Vertex Auth from EC2

Quick test:

```bash
python - <<'PY'
import os
from google import genai

client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT_ID"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)
print("Vertex client initialized successfully")
PY
```

If this fails:

- Check JSON key path and file permissions.
- Check service account IAM role.
- Check project ID and region.

---

## 8) Run the Photoshoot Script

Edit input URLs and pose prompts in:

- `scripts/standalone_gemini_photoshoot.py`

Then run:

```bash
python scripts/standalone_gemini_photoshoot.py
```

---

## 9) Troubleshooting

- **`PermissionDenied` / `403`**
  - Service account missing `roles/aiplatform.user`.
  - Wrong project ID.
- **`Model not found`**
  - Model not enabled in selected region.
  - Try another supported region.
- **Auth not picked up**
  - `GOOGLE_APPLICATION_CREDENTIALS` not set correctly.
  - Wrong key file path or unreadable file.
- **Timeouts / network**
  - Ensure EC2 outbound HTTPS is allowed.

---

## 10) Security Best Practices

- Do not commit service account JSON files to git.
- Restrict key file permissions (`chmod 600`).
- Prefer short-lived credentials/workload identity where possible.
- Rotate service account keys periodically.

