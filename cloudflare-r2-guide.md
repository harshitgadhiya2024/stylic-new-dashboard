# ☁️ Cloudflare R2 — Complete Production Setup Guide

> A step-by-step guide to setting up Cloudflare R2, uploading files with Python, and serving public URLs in production.

---

## Table of Contents

1. [What is Cloudflare R2?](#1-what-is-cloudflare-r2)
2. [Account Setup](#2-account-setup)
3. [Create an R2 Bucket](#3-create-an-r2-bucket)
4. [Enable Public Access (Custom Domain or r2.dev)](#4-enable-public-access)
5. [Create API Credentials](#5-create-api-credentials)
6. [Python Setup](#6-python-setup)
7. [Upload Files & Get Public URLs](#7-upload-files--get-public-urls)
8. [Production-Ready Python Class](#8-production-ready-python-class)
9. [Environment Variables & Security](#9-environment-variables--security)
10. [CORS Configuration](#10-cors-configuration)
11. [Cost & Limits Reference](#11-cost--limits-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [This repository (FastAPI uploads)](#13-this-repository-fastapi-uploads)

---

## 1. What is Cloudflare R2?

Cloudflare R2 is an S3-compatible object storage service with **zero egress fees**. It's ideal for storing images, videos, documents, backups, and any static assets in production.

**Why R2 over AWS S3?**

| Feature | Cloudflare R2 | AWS S3 |
|---|---|---|
| Egress fees | ✅ Free | ❌ ~$0.09/GB |
| S3-compatible API | ✅ Yes | ✅ Yes |
| Free tier | 10 GB storage, 1M writes/month | 5 GB (12 months only) |
| Global CDN | ✅ Built-in via Cloudflare | ❌ Extra cost (CloudFront) |
| Storage cost | $0.015/GB/month | $0.023/GB/month |

---

## 2. Account Setup

### Step 1 — Sign up / Log in

1. Go to [https://dash.cloudflare.com](https://dash.cloudflare.com)
2. Create a free account or log in with an existing one.
3. You do **not** need a domain to use R2.

### Step 2 — Enable R2

1. In the left sidebar, click **R2 Object Storage**.
2. If prompted, click **Purchase R2** (free tier is available — no charge unless you exceed limits).
3. Add a payment method if required (Cloudflare requires this even for free-tier R2).

> ⚠️ **Note:** A valid payment method is required to activate R2, but you won't be charged within the free tier limits.

---

## 3. Create an R2 Bucket

1. Go to **R2 Object Storage** → click **Create bucket**.
2. Enter a **Bucket name** (e.g., `my-app-uploads`).
   - Use lowercase letters, numbers, and hyphens only.
   - Choose a name that reflects your app and environment (e.g., `myapp-prod-media`).
3. Select a **Location**:
   - `Automatic` (recommended) — Cloudflare picks the closest region.
   - Or choose a specific region: `ENAM` (Eastern North America), `WEUR` (Western Europe), `APAC` (Asia Pacific), etc.
4. Click **Create bucket**.

---

## 4. Enable Public Access

By default, R2 buckets are **private**. You have two options to serve files publicly:

### Option A — Use the free `r2.dev` subdomain (quick, for testing)

1. Open your bucket → go to **Settings** tab.
2. Under **Public access**, click **Allow Access** next to "R2.dev subdomain".
3. Your files will be accessible at:
   ```
   https://pub-<hash>.r2.dev/<object-key>
   ```

> ⚠️ `r2.dev` public access is **rate-limited** and not recommended for high-traffic production. Use a custom domain instead.

### Option B — Connect a Custom Domain (recommended for production)

1. You need a domain managed by Cloudflare (add your domain in Cloudflare DNS for free).
2. In your bucket → **Settings** → **Custom Domains** → click **Connect Domain**.
3. Enter your subdomain, e.g., `cdn.yourdomain.com`.
4. Cloudflare automatically creates a DNS record and SSL certificate.
5. Files will be publicly accessible at:
   ```
   https://cdn.yourdomain.com/<object-key>
   ```

> ✅ Custom domain is served via Cloudflare's global CDN with no egress cost — ideal for production.

---

## 5. Create API Credentials

1. Go to **R2 Object Storage** → click **Manage R2 API Tokens** (top right).
2. Click **Create API Token**.
3. Configure the token:
   - **Token name**: e.g., `myapp-prod-r2`
   - **Permissions**: Choose `Object Read & Write` (or `Admin Read & Write` if you need to create/delete buckets programmatically).
   - **Bucket scope**: Select specific bucket(s) for security (e.g., `my-app-uploads`).
   - **TTL**: Set an expiry or leave as no expiry for long-lived service tokens.
4. Click **Create API Token**.
5. **Copy and save** these values immediately (they won't be shown again):
   - `Access Key ID`
   - `Secret Access Key`
   - `Endpoint URL` — looks like: `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`

> 🔐 **Security tip:** Never commit credentials to Git. Use environment variables or a secrets manager.

---

## 6. Python Setup

### Install Dependencies

```bash
pip install boto3 python-dotenv
```

`boto3` is the AWS SDK for Python — it works with R2 because R2 is S3-compatible.

### Project Structure

```
my-project/
├── .env
├── r2_client.py        # R2 helper class
├── upload_example.py   # Usage example
└── requirements.txt
```

### requirements.txt

```
boto3>=1.34.0
python-dotenv>=1.0.0
```

---

## 7. Upload Files & Get Public URLs

### .env file

```env
R2_ACCOUNT_ID=your_account_id_here
R2_ACCESS_KEY_ID=your_access_key_id_here
R2_SECRET_ACCESS_KEY=your_secret_access_key_here
R2_BUCKET_NAME=my-app-uploads
R2_PUBLIC_URL=https://cdn.yourdomain.com
# or for r2.dev: https://pub-<hash>.r2.dev
```

### Basic Upload Script

```python
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize R2 client (S3-compatible)
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    region_name="auto",  # R2 uses "auto" as the region
)

BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
PUBLIC_URL = os.getenv("R2_PUBLIC_URL")


def upload_file(local_path: str, object_key: str, content_type: str = None) -> str:
    """
    Upload a local file to R2 and return its public URL.

    :param local_path: Path to the local file (e.g., "/tmp/photo.jpg")
    :param object_key: Key/path inside the bucket (e.g., "images/photo.jpg")
    :param content_type: MIME type (e.g., "image/jpeg"). Auto-detected if None.
    :return: Public URL of the uploaded file
    """
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    s3.upload_file(
        Filename=local_path,
        Bucket=BUCKET_NAME,
        Key=object_key,
        ExtraArgs=extra_args,
    )

    public_url = f"{PUBLIC_URL.rstrip('/')}/{object_key}"
    return public_url


# --- Example usage ---
if __name__ == "__main__":
    url = upload_file(
        local_path="./sample.jpg",
        object_key="images/sample.jpg",
        content_type="image/jpeg",
    )
    print(f"File uploaded! Public URL: {url}")
```

---

## 8. Production-Ready Python Class

Save this as `r2_client.py` — a reusable, production-grade R2 client.

```python
import boto3
import os
import mimetypes
import uuid
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class R2Client:
    """
    Production-ready Cloudflare R2 client.
    Supports uploading files, uploading bytes/streams,
    deleting objects, listing objects, and generating public URLs.
    """

    def __init__(self):
        account_id = os.getenv("R2_ACCOUNT_ID")
        if not account_id:
            raise ValueError("R2_ACCOUNT_ID environment variable is not set.")

        self.bucket = os.getenv("R2_BUCKET_NAME")
        self.public_url = os.getenv("R2_PUBLIC_URL", "").rstrip("/")

        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )

    # ------------------------------------------------------------------ #
    #  Upload a local file                                                  #
    # ------------------------------------------------------------------ #
    def upload_file(
        self,
        local_path: str,
        object_key: Optional[str] = None,
        folder: str = "",
        content_type: Optional[str] = None,
        make_unique: bool = False,
    ) -> str:
        """
        Upload a local file to R2.

        :param local_path:   Path to the local file.
        :param object_key:   Key inside the bucket. Auto-generated from filename if None.
        :param folder:       Optional folder prefix, e.g. "images/avatars".
        :param content_type: MIME type. Auto-detected from extension if None.
        :param make_unique:  Prepend a UUID to avoid collisions.
        :return:             Public URL of the uploaded object.
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        # Determine object key
        if object_key is None:
            filename = path.name
            if make_unique:
                filename = f"{uuid.uuid4().hex}_{filename}"
            object_key = f"{folder.strip('/')}/{filename}".lstrip("/") if folder else filename

        # Detect content type
        if content_type is None:
            content_type, _ = mimetypes.guess_type(local_path)
            content_type = content_type or "application/octet-stream"

        self.client.upload_file(
            Filename=str(path),
            Bucket=self.bucket,
            Key=object_key,
            ExtraArgs={"ContentType": content_type},
        )

        return self.get_public_url(object_key)

    # ------------------------------------------------------------------ #
    #  Upload raw bytes (e.g., in-memory image, file from a web request)  #
    # ------------------------------------------------------------------ #
    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload raw bytes directly to R2 (no temp file needed).

        :param data:         File content as bytes.
        :param object_key:   Key inside the bucket.
        :param content_type: MIME type.
        :return:             Public URL.
        """
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
        return self.get_public_url(object_key)

    # ------------------------------------------------------------------ #
    #  Upload multiple files from a directory                             #
    # ------------------------------------------------------------------ #
    def upload_directory(self, local_dir: str, prefix: str = "") -> list[dict]:
        """
        Upload all files in a local directory to R2.

        :param local_dir: Local directory path.
        :param prefix:    R2 key prefix (folder) for all uploaded files.
        :return:          List of {file, key, url} dicts.
        """
        results = []
        base = Path(local_dir)
        for file_path in base.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(base)
                key = f"{prefix.strip('/')}/{relative}".lstrip("/") if prefix else str(relative)
                url = self.upload_file(str(file_path), object_key=key)
                results.append({"file": str(file_path), "key": key, "url": url})
                print(f"  ✅ Uploaded: {key}")
        return results

    # ------------------------------------------------------------------ #
    #  Generate a pre-signed URL (for private files, time-limited access) #
    # ------------------------------------------------------------------ #
    def get_presigned_url(self, object_key: str, expires_in: int = 3600) -> str:
        """
        Generate a pre-signed URL for a private object.

        :param object_key: Key of the object in R2.
        :param expires_in: Expiry in seconds (default: 1 hour).
        :return:           Pre-signed URL string.
        """
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": object_key},
            ExpiresIn=expires_in,
        )

    # ------------------------------------------------------------------ #
    #  Get the permanent public URL                                        #
    # ------------------------------------------------------------------ #
    def get_public_url(self, object_key: str) -> str:
        if not self.public_url:
            raise ValueError("R2_PUBLIC_URL is not configured in your .env file.")
        return f"{self.public_url}/{object_key}"

    # ------------------------------------------------------------------ #
    #  Delete an object                                                    #
    # ------------------------------------------------------------------ #
    def delete_file(self, object_key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=object_key)
        print(f"🗑️  Deleted: {object_key}")

    # ------------------------------------------------------------------ #
    #  List objects in the bucket                                          #
    # ------------------------------------------------------------------ #
    def list_files(self, prefix: str = "", max_keys: int = 100) -> list[str]:
        """
        List object keys in the bucket.

        :param prefix:   Filter by prefix (folder), e.g. "images/".
        :param max_keys: Max results to return.
        :return:         List of object keys.
        """
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=max_keys,
        )
        return [obj["Key"] for obj in response.get("Contents", [])]

    # ------------------------------------------------------------------ #
    #  Check if a file exists                                              #
    # ------------------------------------------------------------------ #
    def file_exists(self, object_key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except self.client.exceptions.ClientError:
            return False
```

### Using the R2Client in your app

```python
from r2_client import R2Client

r2 = R2Client()

# 1. Upload a single file
url = r2.upload_file("./assets/logo.png", folder="images")
print(f"Logo URL: {url}")

# 2. Upload with a unique name (avoid collisions)
url = r2.upload_file("./user_photo.jpg", folder="avatars", make_unique=True)
print(f"Avatar URL: {url}")

# 3. Upload raw bytes (e.g., from a form upload in FastAPI/Flask)
with open("./document.pdf", "rb") as f:
    data = f.read()
url = r2.upload_bytes(data, "docs/document.pdf", content_type="application/pdf")
print(f"Document URL: {url}")

# 4. Upload an entire directory
results = r2.upload_directory("./static/", prefix="static")
for r in results:
    print(r["url"])

# 5. Get a time-limited private URL (15 minutes)
signed_url = r2.get_presigned_url("private/report.pdf", expires_in=900)
print(f"Signed URL: {signed_url}")

# 6. Delete a file
r2.delete_file("images/old-logo.png")

# 7. List files in a folder
files = r2.list_files(prefix="images/")
print(files)
```

---

## 9. Environment Variables & Security

### Never hard-code credentials. Use `.env` locally and secrets in production.

**.env (local development)**
```env
R2_ACCOUNT_ID=abc123def456
R2_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_SECRET_ACCESS_KEY=yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
R2_BUCKET_NAME=myapp-prod-media
R2_PUBLIC_URL=https://cdn.yourdomain.com
```

**.gitignore** — always add your `.env`:
```
.env
*.env
```

**Production environments:**

| Platform | How to set secrets |
|---|---|
| Railway / Render | Dashboard → Environment Variables |
| Heroku | `heroku config:set R2_ACCESS_KEY_ID=xxx` |
| AWS EC2 / ECS | AWS Secrets Manager or Parameter Store |
| Docker | `--env-file .env` or Docker Secrets |
| GitHub Actions | Settings → Secrets and Variables |
| Vercel / Netlify | Project settings → Environment Variables |

---

## 10. CORS Configuration

If your frontend (browser) needs to upload directly to R2, configure CORS:

1. Go to your bucket → **Settings** → **CORS Policy**.
2. Paste and customize the following JSON:

```json
[
  {
    "AllowedOrigins": ["https://yourdomain.com", "http://localhost:3000"],
    "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
    "AllowedHeaders": ["*"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

> For production, **always restrict `AllowedOrigins`** to your actual domain(s). Using `"*"` is insecure.

---

## 11. Cost & Limits Reference

### Free Tier (per month)
| Resource | Free Allowance |
|---|---|
| Storage | 10 GB |
| Class A operations (writes, lists) | 1,000,000 |
| Class B operations (reads) | 10,000,000 |
| Egress (data transfer out) | **Free (unlimited)** |

### Paid Tier (beyond free)
| Resource | Price |
|---|---|
| Storage | $0.015 / GB |
| Class A ops | $4.50 / million |
| Class B ops | $0.36 / million |
| Egress | **$0.00 (always free)** |

---

## 12. Troubleshooting

### ❌ `NoCredentialsError` or `InvalidAccessKeyId`
- Check your `.env` file has the correct values.
- Ensure `load_dotenv()` is called before accessing `os.getenv(...)`.
- Regenerate your API token in the Cloudflare dashboard if needed.

### ❌ `SignatureDoesNotMatch`
- The secret key is wrong or has extra whitespace — copy it carefully from the dashboard.

### ❌ File uploaded but URL returns 403 or 404
- Check that **Public Access is enabled** on your bucket (Section 4).
- Verify your custom domain is correctly configured in Cloudflare DNS.
- Ensure the object key in the URL matches exactly (case-sensitive).

### ❌ `NoSuchBucket`
- The bucket name in your `.env` doesn't match the actual bucket name.
- Double-check for typos (R2 bucket names are case-sensitive).

### ❌ CORS error in browser
- Update the CORS policy on your bucket (Section 10).
- Make sure your origin URL is in `AllowedOrigins`.

### ✅ Verify your connection works
```python
from r2_client import R2Client
r2 = R2Client()
print(r2.list_files())  # Should print [] or a list of keys
```

---

## Quick Reference Cheatsheet

```python
from r2_client import R2Client
r2 = R2Client()

# Upload file → get public URL
url = r2.upload_file("./photo.jpg", folder="images")

# Upload bytes (in-memory)
url = r2.upload_bytes(file_bytes, "path/file.jpg", "image/jpeg")

# Private pre-signed URL (1 hour)
url = r2.get_presigned_url("private/file.pdf", expires_in=3600)

# Delete
r2.delete_file("images/old.jpg")

# List
keys = r2.list_files(prefix="images/")
```

---

## 13. This repository (FastAPI uploads)

The app uses **`app/services/r2_service.py`** (async **aioboto3**) instead of AWS S3. Set these in **`.env`** (see also **`.env.example`**):

| Variable | Purpose |
|----------|---------|
| `R2_ACCOUNT_ID` | From R2 API token screen |
| `R2_ACCESS_KEY_ID` | S3-compatible access key |
| `R2_SECRET_ACCESS_KEY` | S3-compatible secret |
| `R2_BUCKET_NAME` | Your bucket name |
| `R2_PUBLIC_URL` | Public base URL (no trailing slash), e.g. `https://cdn.yourdomain.com` or `https://pub-xxx.r2.dev` |
| `R2_ENDPOINT_URL` | Optional; default `https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com` |

Photoshoots, user uploads, model faces, backgrounds, poses, and thumbnails all resolve public URLs as **`{R2_PUBLIC_URL}/{object_key}`**.

### Delete by public URL (API)

Authenticated clients can call:

`POST /api/v1/storage/r2/delete-by-public-url` with JSON body `{ "public_url": "https://…/users/{your-user-id}/file.jpg" }`.

Only URLs under your configured **`R2_PUBLIC_URL`** are accepted. The backend enforces ownership (your `users/` prefix, your `thumbnails/` prefix, or `photoshoots/…` rows you own in MongoDB).

---

*Guide version: April 2026 | Cloudflare R2 S3-compatible API | boto3 SDK*
