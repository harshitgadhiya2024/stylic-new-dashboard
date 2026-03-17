# Stylic AI — CI/CD Deployment Guide

**Stack:** FastAPI · Gunicorn · Nginx · EC2 (Ubuntu 22.04) · GitHub Actions

---

## Table of Contents

1. [EC2 Instance Setup](#1-ec2-instance-setup)
2. [Server Dependencies](#2-server-dependencies)
3. [Application Setup](#3-application-setup)
4. [Gunicorn Systemd Service](#4-gunicorn-systemd-service)
5. [Nginx Configuration](#5-nginx-configuration)
6. [SSL Certificate (HTTPS)](#6-ssl-certificate-https)
7. [GitHub Actions CI/CD Pipeline](#7-github-actions-cicd-pipeline)
8. [GitHub Secrets](#8-github-secrets)
9. [First Deploy Checklist](#9-first-deploy-checklist)
10. [Useful Commands](#10-useful-commands)

---

## 1. EC2 Instance Setup

### Launch Instance
- **AMI:** Ubuntu Server 22.04 LTS
- **Instance type:** t3.small (minimum) / t3.medium (recommended)
- **Storage:** 20 GB GP3
- **Key pair:** Create or use an existing `.pem` key

### Security Group — Inbound Rules

| Type  | Protocol | Port | Source    |
|-------|----------|------|-----------|
| SSH   | TCP      | 22   | Your IP   |
| HTTP  | TCP      | 80   | 0.0.0.0/0 |
| HTTPS | TCP      | 443  | 0.0.0.0/0 |

> **Do NOT expose port 8000 publicly.** Nginx proxies all traffic internally.

---

## 2. Server Dependencies

SSH into your instance and run:

```bash
sudo apt update && sudo apt upgrade -y

# Python 3.11 + pip + venv
sudo apt install -y python3.11 python3.11-venv python3-pip

# Nginx
sudo apt install -y nginx

# Git
sudo apt install -y git

# (Optional) Certbot for HTTPS
sudo apt install -y certbot python3-certbot-nginx
```

---

## 3. Application Setup

```bash
# Create app directory
sudo mkdir -p /var/www/stylicai
sudo chown ubuntu:ubuntu /var/www/stylicai

# Clone the repo
cd /var/www/stylicai
git clone https://github.com/harshitgadhiya2024/stylic-new-dashboard.git .

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Create the `.env` file

```bash
cp .env.example .env
nano .env   # fill in all real values
```

### Upload Firebase Service Account Key

Securely copy your Firebase JSON key to the server:

```bash
# From your LOCAL machine
scp -i your-key.pem stylic-ai-d1ee0-firebase-adminsdk-fbsvc-a4a36772f6.json \
    ubuntu@<EC2_PUBLIC_IP>:/var/www/stylicai/serviceAccountKey.json
```

Make sure `FIREBASE_SERVICE_ACCOUNT_KEY=./serviceAccountKey.json` in your `.env`.

---

## 4. Gunicorn Systemd Service

Create the service file:

```bash
sudo nano /etc/systemd/system/stylicai.service
```

Paste:

```ini
[Unit]
Description=Stylic AI FastAPI application
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/var/www/stylicai
EnvironmentFile=/var/www/stylicai/.env
ExecStart=/var/www/stylicai/venv/bin/gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --timeout 120 \
    --access-logfile /var/log/stylicai/access.log \
    --error-logfile /var/log/stylicai/error.log
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo mkdir -p /var/log/stylicai
sudo chown ubuntu:ubuntu /var/log/stylicai

sudo systemctl daemon-reload
sudo systemctl enable stylicai
sudo systemctl start stylicai
sudo systemctl status stylicai
```

---

## 5. Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/stylicai
```

Paste (replace `your-domain.com` with your actual domain or EC2 public IP):

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

Enable and test:

```bash
sudo ln -s /etc/nginx/sites-available/stylicai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 6. SSL Certificate (HTTPS)

```bash
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Certbot auto-renews. Verify:

```bash
sudo certbot renew --dry-run
```

---

## 7. GitHub Actions CI/CD Pipeline

Create the workflow file in your repo:

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to EC2

on:
  push:
    branches:
      - main

jobs:
  deploy:
    name: SSH Deploy
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Deploy to EC2 via SSH
        uses: appleboy/ssh-action@v1.0.3
        with:
          host:        ${{ secrets.EC2_HOST }}
          username:    ${{ secrets.EC2_USER }}
          key:         ${{ secrets.EC2_SSH_KEY }}
          port:        22
          script: |
            set -e

            cd /var/www/stylicai

            echo "── Pulling latest code ──"
            git pull origin main

            echo "── Activating virtualenv ──"
            source venv/bin/activate

            echo "── Installing dependencies ──"
            pip install --upgrade pip
            pip install -r requirements.txt

            echo "── Restarting Gunicorn ──"
            sudo systemctl restart stylicai

            echo "── Reloading Nginx ──"
            sudo systemctl reload nginx

            echo "✅ Deploy complete"
```

> **Note:** The `ubuntu` user needs passwordless sudo for `systemctl`. Add this on the server:
>
> ```bash
> echo "ubuntu ALL=(ALL) NOPASSWD: /bin/systemctl restart stylicai, /bin/systemctl reload nginx" 
sudo tee /etc/sudoers.d/stylicai-deploy
> ```

---

## 8. GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name   | Value                                              |
|---------------|----------------------------------------------------|
| `EC2_HOST`    | Your EC2 public IP or domain (`1.2.3.4`)           |
| `EC2_USER`    | `ubuntu`                                           |
| `EC2_SSH_KEY` | Contents of your `.pem` file (the private key)     |

---

## 9. First Deploy Checklist

```
[ ] EC2 instance running with correct security group rules
[ ] Python 3.11, Nginx, Git installed on server
[ ] Repo cloned to /var/www/stylicai
[ ] venv created and requirements installed
[ ] .env file created with all real values
[ ] Firebase serviceAccountKey.json uploaded to server
[ ] /var/log/stylicai/ directory created
[ ] stylicai.service created, enabled and running
[ ] Nginx config created, tested and reloaded
[ ] (Optional) SSL cert issued via Certbot
[ ] GitHub Secrets set: EC2_HOST, EC2_USER, EC2_SSH_KEY
[ ] Push to main branch → GitHub Actions deploys automatically
```

---

## 10. Useful Commands

```bash
# Check app service status
sudo systemctl status stylicai

# View live logs
sudo journalctl -u stylicai -f

# View access / error logs
tail -f /var/log/stylicai/access.log
tail -f /var/log/stylicai/error.log

# Restart app manually
sudo systemctl restart stylicai

# Reload Nginx
sudo systemctl reload nginx

# Test Nginx config
sudo nginx -t

# Check what's running on port 8000
sudo ss -tlnp | grep 8000
```
