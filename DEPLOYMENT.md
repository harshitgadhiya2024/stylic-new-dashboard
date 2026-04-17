# Stylic AI — Production deployment (Ubuntu 22.04 droplet)

This guide targets an **Ubuntu 22.04** **DigitalOcean Droplet**, with first-time access via **`ssh root@DROPLET_PUBLIC_IP`** (SSH key only; no password setup on the default `ubuntu` user is required). It then adds a **`deploy`** user for the app and CI.

The server runs:

- Python app behind **Gunicorn + Uvicorn workers**
- **Redis** (broker for Celery)
- **Celery** worker for the `photoshoots` queue
- **Flower** (optional Celery UI)
- **Nginx** reverse proxy + **Certbot** (Let’s Encrypt TLS)
- **CI/CD** example using **GitHub Actions** + SSH deploy

Adjust paths, domain names, and user names to match your environment.

---

## Table of contents

1. [Assumptions](#1-assumptions)
2. [DigitalOcean: SSH as root (key-only, no password)](#2-digitalocean-ssh-as-root-key-only-no-password)
3. [Server baseline](#3-server-baseline)
4. [Deploy user & app directory](#4-deploy-user--app-directory)
5. [System packages](#5-system-packages)
6. [Redis](#6-redis)
7. [Clone repository & Python venv](#7-clone-repository--python-venv)
8. [Environment file](#8-environment-file)
9. [Gunicorn (systemd)](#9-gunicorn-systemd)
10. [Celery worker (systemd)](#10-celery-worker-systemd)
11. [Flower (systemd)](#11-flower-systemd)
12. [Nginx reverse proxy](#12-nginx-reverse-proxy)
13. [Certbot (SSL)](#13-certbot-ssl)
14. [Firewall](#14-firewall)
15. [CI/CD — GitHub Actions](#15-cicd--github-actions)
16. [Operations cheat sheet](#16-operations-cheat-sheet)

---

## 1. Assumptions

| Item | Example |
|------|---------|
| Provider | **DigitalOcean** Droplet (Ubuntu 22.04) |
| First SSH | **`ssh root@DROPLET_PUBLIC_IP`** (SSH key from DO; see §2) |
| OS | Ubuntu 22.04 LTS |
| App user | `deploy` (non-root; created after bootstrap) |
| App root | `/opt/stylicai` (repo clone) |
| Public domain | `api.yourdomain.com` → droplet **A** record |
| Python | 3.11 (recommended) or 3.10+ |
| Process manager | `systemd` |
| Repo | Git over HTTPS/SSH (private OK) |

You need **MongoDB** (Atlas or self-hosted), **Redis**, **Cloudflare R2** (or compatible), and API keys as described in `.env.example`.

---

## 2. DigitalOcean: SSH as root (key-only, no password)

On a new **DigitalOcean** droplet you typically log in with an **SSH key** only. The default `ubuntu` user often has **no password set**, so any command that runs **`sudo`** will ask for a password you never created—this is expected, not a bug.

**Recommended path for this guide:** use **`root` over SSH** for the initial server setup (packages, `deploy` user, systemd, nginx, certbot). Your **SSH key** is the same one you added in the DO create-droplet flow; DigitalOcean allows **`ssh root@YOUR_DROPLET_PUBLIC_IP`** when that key is present (unless you have changed SSH config to disable root).

```bash
ssh root@YOUR_DROPLET_PUBLIC_IP
```

From here on, commands in **§3–§6** are written for a **root shell** (`#` prompt): use **`apt`** and **`systemctl`** **without** `sudo`. If you prefer a non-root admin user with `sudo`, prefix those commands with `sudo` and ensure that user has a password or passwordless-sudo configured.

**If `ssh root@…` is refused:** use the DigitalOcean control panel **Droplet → Access → Launch Droplet Console** (web-based terminal), log in as root from there, or reset access using DO’s **Recovery** / **Reset root password** docs, then return to this guide.

**Security (after bootstrap):** you may disable password login for root, rely on keys only, and optionally disable direct `root` SSH once `deploy` + `sudo` are configured the way you want. That is outside the scope of this minimal deploy walkthrough.

---

## 3. Server baseline

On the droplet, as **root**:

```bash
apt update && apt upgrade -y
timedatectl set-timezone UTC   # optional
```

---

## 4. Deploy user & app directory

Still as **root**:

```bash
adduser deploy --disabled-password --gecos ""
usermod -aG sudo deploy          # optional: allows deploy to run sudo later (e.g. CI restarts)
mkdir -p /opt/stylicai
chown deploy:deploy /opt/stylicai
```

Give **`deploy`** the same **public key** you use for root so you can SSH in as `deploy` (no password; key auth only):

```bash
install -d -m 700 -o deploy -g deploy /home/deploy/.ssh
nano /home/deploy/.ssh/authorized_keys   # paste one line: your laptop’s id_ed25519.pub / id_rsa.pub
chmod 600 /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh/authorized_keys
```

From your **laptop**, open a **new** terminal and verify:

```bash
ssh deploy@YOUR_DROPLET_PUBLIC_IP
cd /opt/stylicai
```

Use this **`deploy`** session for **§7** (clone/venv) and editing app files. Use **`root`** (or `deploy` + `sudo` if configured) for **§9–§14** (systemd unit files, nginx, certbot, ufw).

---

## 5. System packages

As **root**:

```bash
apt install -y \
  git curl build-essential \
  nginx certbot python3-certbot-nginx \
  redis-server \
  python3.11 python3.11-venv python3.11-dev
```

If `python3.11` is not in default repos, use [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa):

```bash
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.11 python3.11-venv python3.11-dev
```

---

## 6. Redis

As **root**:

```bash
systemctl enable redis-server
systemctl start redis-server
redis-cli ping   # expect PONG
```

In production `.env`, set for example:

```env
REDIS_URL=redis://127.0.0.1:6379/0
```

If Redis listens only on localhost (default), do not expose port `6379` in the firewall.

---

## 7. Clone repository & Python venv

As **`deploy`** (SSH session from §4):

```bash
cd /opt/stylicai
git clone https://github.com/YOUR_ORG/Newstylicai.git .
# or: git clone git@github.com:YOUR_ORG/Newstylicai.git .

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install flower   # Celery monitoring UI (not listed in requirements.txt by default)
```

**Note:** For heavy image work, workers may need extra RAM; start with a **2 GB+** droplet for API + Celery on the same host, or split Celery onto another droplet using the same `REDIS_URL`.

---

## 8. Environment file

As **`deploy`**:

```bash
cd /opt/stylicai
cp .env.example .env
nano .env   # fill MONGO_URI, JWT, R2_*, REDIS_URL, API keys, etc.
chmod 600 .env
```

Production recommendations:

- `DEBUG=False`
- `HOST=127.0.0.1` and `PORT=8000` (Gunicorn binds locally; Nginx talks to it)
- Strong `JWT_SECRET_KEY`
- Use **MongoDB Atlas** or a secured Mongo instance

---

## 9. Gunicorn (systemd)

Gunicorn loads **`main:app`** (FastAPI) using **Uvicorn workers** from the project root (`/opt/stylicai`).

As **root** (create unit files under `/etc/systemd/system/`). On a root shell, omit `sudo`; otherwise use `sudo` before each command.

Create `/etc/systemd/system/stylicai-api.service`:

```ini
[Unit]
Description=Stylic AI API (Gunicorn + Uvicorn)
After=network.target redis-server.service

[Service]
Type=notify
User=deploy
Group=deploy
WorkingDirectory=/opt/stylicai
Environment="PATH=/opt/stylicai/.venv/bin"
ExecStart=/opt/stylicai/.venv/bin/gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  -b 127.0.0.1:8000 \
  -w 4 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start (as **root**):

```bash
systemctl daemon-reload
systemctl enable stylicai-api
systemctl start stylicai-api
systemctl status stylicai-api
curl -sS http://127.0.0.1:8000/health
```

Tune `-w` (worker count) to CPU cores (often `2 * cores + 1` is a starting point; I/O-heavy APIs may use fewer).

---

## 10. Celery worker (systemd)

The project uses **`app.worker:celery_app`** and queue **`photoshoots`** (see `app/worker.py`).

Create `/etc/systemd/system/stylicai-celery.service`:

```ini
[Unit]
Description=Stylic AI Celery worker (photoshoots queue)
After=network.target redis-server.service

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/opt/stylicai
Environment="PATH=/opt/stylicai/.venv/bin"
ExecStart=/opt/stylicai/.venv/bin/celery -A app.worker worker \
  -Q photoshoots \
  --concurrency=1 \
  --loglevel=info
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Why `--concurrency=1`?** The photoshoot task pipeline uses `asyncio.run()` per job; the codebase recommends a single worker process per machine for predictable GPU/API usage. For more parallelism, run **multiple droplets** each with `concurrency=1`, or use `--autoscale=3,1` only if you understand resource limits (see comments in `app/worker.py`).

As **root**:

```bash
systemctl daemon-reload
systemctl enable stylicai-celery
systemctl start stylicai-celery
systemctl status stylicai-celery
```

---

## 11. Flower (systemd)

Flower should **not** be public without authentication. Options:

- **A)** Bind to `127.0.0.1` and use SSH port forwarding for admins.
- **B)** Bind to `127.0.0.1` and proxy via Nginx with HTTP basic auth (below).

Create `/etc/systemd/system/stylicai-flower.service`:

```ini
[Unit]
Description=Stylic AI Celery Flower
After=network.target redis-server.service stylicai-celery.service

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/opt/stylicai
Environment="PATH=/opt/stylicai/.venv/bin"
ExecStart=/opt/stylicai/.venv/bin/celery -A app.worker flower \
  --address=127.0.0.1 \
  --port=5555 \
  --basic_auth=CHANGE_ME_USER:CHANGE_ME_STRONG_PASSWORD
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

As **root**:

```bash
systemctl daemon-reload
systemctl enable stylicai-flower
systemctl start stylicai-flower
```

**Security:** Replace basic auth credentials; prefer SSH tunnel or VPN for production if possible.

---

## 12. Nginx reverse proxy

As **root**, create `/etc/nginx/sites-available/stylicai`:

```nginx
# Upstream FastAPI (Gunicorn)
upstream stylicai_api {
    server 127.0.0.1:8000 fail_timeout=0;
}

# HTTP — Certbot will add HTTPS + redirects
server {
    listen 80;
    listen [::]:80;
    server_name api.yourdomain.com;

    # Let’s Encrypt HTTP-01
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        proxy_pass http://stylicai_api;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        client_max_body_size 50M;
    }
}
```

Enable site and test (as **root**):

```bash
ln -sf /etc/nginx/sites-available/stylicai /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
```

Optional **Flower** behind Nginx (same server, path prefix — Flower may need `--url_prefix`; check `celery flower --help` for your Celery version). Simpler approach: **SSH tunnel**:

```bash
ssh -L 5555:127.0.0.1:5555 deploy@YOUR_DROPLET_PUBLIC_IP
# open http://127.0.0.1:5555 on your laptop
```

---

## 13. Certbot (SSL)

Ensure DNS **A** record for `api.yourdomain.com` points to the droplet, then as **root**:

```bash
certbot --nginx -d api.yourdomain.com
```

Follow prompts (email, agree to terms). Certbot will install a snippet under `/etc/letsencrypt/` and adjust the Nginx server block for **HTTPS** and redirect HTTP → HTTPS.

Auto-renewal is installed as a timer; verify:

```bash
certbot renew --dry-run
```

After Certbot edits Nginx, confirm:

```bash
nginx -t && systemctl reload nginx
curl -sS https://api.yourdomain.com/health
```

---

## 14. Firewall

With **UFW**, as **root**:

```bash
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw enable
ufw status
```

Do **not** open `8000` or `5555` publicly if they bind to `127.0.0.1` only.

---

## 15. CI/CD — GitHub Actions

Goal: on push to `main`, SSH into the droplet, `git pull`, restart **systemd** units.

### 15.1 On the droplet (once)

1. Create SSH key **used only for deploy** (on your laptop or as a GitHub deploy key with read-only repo access):

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/github_deploy_stylicai -N ""
   ```

2. Add **public** key to `deploy` user on droplet: `~/.ssh/authorized_keys`.

3. On the droplet, as **root**, allow `deploy` to restart services without a password (narrow sudoers):

   ```bash
   visudo -f /etc/sudoers.d/stylicai-deploy
   ```

   Add (single line):

   ```
   deploy ALL=(root) NOPASSWD: /bin/systemctl restart stylicai-api, /bin/systemctl restart stylicai-celery, /bin/systemctl restart stylicai-flower
   ```

### 15.2 GitHub repository secrets

In **GitHub → Settings → Secrets data and variables → Actions**, add:

| Secret | Example |
|--------|---------|
| `DEPLOY_HOST` | `203.0.113.50` or `api.yourdomain.com` |
| `DEPLOY_USER` | `deploy` |
| `DEPLOY_SSH_KEY` | **Private** key PEM (full multiline) |
| `DEPLOY_PATH` | `/opt/stylicai` |

### 15.3 Workflow file

The repo includes `.github/workflows/deploy-droplet.yml`. Adjust the branch name if yours is not `main`. Example inline:

```yaml
name: Deploy to Ubuntu droplet

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy over SSH
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.DEPLOY_HOST }}
          username: ${{ secrets.DEPLOY_USER }}
          key: ${{ secrets.DEPLOY_SSH_KEY }}
          script_stop: true
          script: |
            set -euo pipefail
            cd ${{ secrets.DEPLOY_PATH }}
            git fetch origin main
            git reset --hard origin/main
            source .venv/bin/activate
            pip install -r requirements.txt
            pip install -q flower || true
            sudo systemctl restart stylicai-api
            sudo systemctl restart stylicai-celery
            sudo systemctl restart stylicai-flower
```

**Notes:**

- `git reset --hard origin/main` matches a **single-branch** deploy server; adjust if you use tags/releases.
- Add `pip install -r requirements.txt` only when dependencies change to speed deploys, or keep as-is for simplicity.
- For **zero-downtime** rolling deploys, use two releases directories + symlink swap (blue/green); this workflow is intentionally simple.

---

## 16. Operations cheat sheet

As **root** (omit `sudo`); as **`deploy`** with sudo rights, prefix with `sudo`:

```bash
# Logs
journalctl -u stylicai-api -f
journalctl -u stylicai-celery -f
journalctl -u stylicai-flower -f

# Restart after code or .env change
systemctl restart stylicai-api stylicai-celery stylicai-flower

# Nginx
nginx -t && systemctl reload nginx
```

**Environment reload:** Gunicorn/Celery do not hot-reload `.env`; always `systemctl restart` after changing secrets.

**Modal / GPU:** If you use Modal for upscaling, ensure `modal` CLI token or secrets on the **Celery** host are configured for the `deploy` user as documented by Modal.

---

## Checklist

- [ ] `ssh root@DROPLET_PUBLIC_IP` works (DO SSH key)
- [ ] `ssh deploy@DROPLET_PUBLIC_IP` works (`authorized_keys` for `deploy`)
- [ ] DNS A record → droplet
- [ ] `.env` complete and `DEBUG=False`
- [ ] Redis running, `REDIS_URL` correct
- [ ] `stylicai-api` healthy (`/health` via Nginx HTTPS)
- [ ] `stylicai-celery` running, photoshoot jobs complete
- [ ] Flower secured or tunneled only
- [ ] UFW enabled, SSH + 80/443 only
- [ ] Certbot dry-run OK
- [ ] GitHub Actions secrets set, workflow push tested

---

*Document version: April 2026 · Ubuntu 22.04 · DigitalOcean · root SSH bootstrap · Stylic AI backend (`main.py`, `app.worker` Celery app).*
