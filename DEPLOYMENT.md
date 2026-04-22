# Stylic AI — Production deployment (Ubuntu 22.04 on Vultr)

This guide targets an **Ubuntu 22.04 LTS** **Vultr Cloud Compute** instance (VPS). The architecture is tuned for the photoshoot pipeline:

* Python 3.11 FastAPI app behind **Gunicorn + Uvicorn workers**
* **Redis** (Celery broker + result backend + cross-worker **KIE rate limiter**)
* **Celery** worker for the `photoshoots` queue (fixed concurrency + per-child memory cap — OOM-safe)
* **Flower** on a dedicated sub-domain `flower.stylic.ai`
* **Nginx** reverse proxy + **Certbot** (Let's Encrypt TLS) for both `api.stylic.ai` and `flower.stylic.ai`
* **Swap + OOM-guard** so a runaway worker never takes down the box
* CI/CD example using **GitHub Actions** + SSH deploy (see `.github/workflows/deploy-vultr.yml`)

**Vultr docs (reference):** [Deploy a new server](https://www.vultr.com/docs/deploy-a-new-server/) · [SSH keys](https://www.vultr.com/docs/how-to-use-ssh-keys/) · [Firewall](https://www.vultr.com/docs/vultr-firewall/)

---

## Table of contents

1. [Architecture & sizing](#1-architecture--sizing)
2. [Vultr: create the instance & SSH](#2-vultr-create-the-instance--ssh)
3. [Server baseline (swap, OOM guard, sysctls)](#3-server-baseline-swap-oom-guard-sysctls)
4. [Deploy user & app directory](#4-deploy-user--app-directory)
5. [System packages](#5-system-packages)
6. [Redis (broker + rate limiter backend)](#6-redis-broker--rate-limiter-backend)
7. [Clone repository & Python venv](#7-clone-repository--python-venv)
8. [Environment file (.env)](#8-environment-file-env)
9. [Gunicorn (systemd)](#9-gunicorn-systemd)
10. [Celery worker (systemd, OOM-safe)](#10-celery-worker-systemd-oom-safe)
11. [Flower on flower.stylic.ai (systemd)](#11-flower-on-flowerstylicai-systemd)
12. [Nginx — api.stylic.ai & flower.stylic.ai](#12-nginx--apistylicai--flowerstylicai)
13. [Certbot (SSL) — both sub-domains](#13-certbot-ssl--both-sub-domains)
14. [Firewall (Vultr portal + UFW)](#14-firewall-vultr-portal--ufw)
15. [Throughput tuning — KIE rate limiter](#15-throughput-tuning--kie-rate-limiter)
16. [CI/CD — GitHub Actions](#16-cicd--github-actions)
17. [Operations cheat sheet](#17-operations-cheat-sheet)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Architecture & sizing

### 1.1 The flow — one photoshoot job

```
 FastAPI  ──Redis (broker)──►  Celery worker (photoshoots queue)
                                    │
                                    ▼  (fan-out 8 poses via asyncio.gather)
                        ┌──── KIE rate limiter (Redis sliding window: 10/10s) ────┐
                        │                                                          │
                KIE nano-banana-2  →  Vertex nb-2  →  Vertex nb-pro  →  Evolink   │  (per-pose
                                                                                   │   fallback chain)
                        └─► Topaz upscale (KIE) ─► PIL 8K/4K/2K/1K encode ─► R2  ◄┘
```

Everything photoshoot-related runs on **one Vultr instance**; MongoDB (Atlas) and Cloudflare R2 are external.

### 1.2 Recommended Vultr plan — **High Performance / High Frequency, 2 vCPU × 16 GB RAM, 100 GB NVMe ($80/mo)**

| Plan ID                       | vCPU   | RAM    | NVMe   | Price/mo | Notes |
|-------------------------------|--------|--------|--------|----------|-------|
| `vhf-2c-16gb` (High Frequency)| 2      | **16** | 100 GB | **$80**  | **Recommended** — best single-thread perf for PIL 8K encoding. |
| `voc-2c-16gb` (Cloud Compute) | 2      | 16     | 100 GB | $80      | Cheaper CPU but still fine; pick if VHF is sold out. |
| `vhp-2c-8gb`                  | 2      | 8      | —      | $48      | Too little RAM once autoscale + 8K encodes land together. |

> **Why not autoscale vCPU?** The expensive part (PIL encode + HTTP download) is memory-bound, not CPU-bound. Adding vCPUs without RAM makes OOM **more** likely, not less.

### 1.3 Why this setup won't OOM

| Risk | Mitigation |
|------|------------|
| Celery autoscale spawns extra children during a queue burst → 5 × 1.2 GB PIL buffers = 6 GB + base = crash | **Fixed `--concurrency=3`** (no autoscale) + **`--max-memory-per-child=1.5 GB`**. Celery recycles the child **after** the current task completes, so no job is lost. |
| A slow leak in PIL / httpx / google-genai over hours | `--max-tasks-per-child=50` hard-recycles on a schedule. |
| One hung provider hangs the child forever | `task_soft_time_limit=30 min`, `task_time_limit=35 min` in `app/worker.py`. |
| KIE 429 cascades across workers | Redis sliding-window **KIE rate limiter** (§15) — every worker shares the same token pool. |
| Kernel OOM-killer picks SSH or Redis | **Swap = 4 GB** (§3) + `oom_score_adj` pinned so Redis and sshd are always last to die. |

### 1.4 Expected throughput

With the default `KIE_RATE_LIMIT_MAX=10` / 10 s window and 8 poses per photoshoot, a sustained burst hits the KIE limiter first (not CPU, not RAM). Empirically:

* **~7 photoshoots / minute** (steady state).
* KIE nano-banana-2 handles the first ~80% of poses; Vertex/Evolink pick up the spillover for free.
* Raise `KIE_RATE_LIMIT_MAX` to 18 once monitoring shows 429-rate stays at 0 — this pushes throughput to ~12 photoshoots/min.

---

## 2. Vultr: create the instance & SSH

1. Open [Vultr → Products → Cloud Compute](https://my.vultr.com/deploy/). Pick **Cloud Compute → High Frequency** (or Cloud Compute Regular if unavailable).
2. **Location:** choose the region closest to your users (and to KIE's region if known; `Singapore`, `Frankfurt`, or `New York` are safe defaults).
3. **Image:** Ubuntu **22.04 LTS x64**.
4. **Plan:** `2 vCPU / 16 GB / 100 GB NVMe` (≈$80/mo — **recommended**).
5. **Add SSH key** — paste your `~/.ssh/id_ed25519.pub`. This avoids password SSH entirely.
6. (Optional) Enable **Backups** ($16/mo on $80 plan). Strongly recommended for prod.
7. Hostname: `stylicai-prod`. Label: same.
8. **Deploy.** Wait ~60s, note the **public IPv4** in the Vultr console.

First login:

```bash
ssh root@INSTANCE_PUBLIC_IP
```

Create DNS A records at your DNS provider (Cloudflare, Route53, or Vultr DNS):

```
api.stylic.ai     A   INSTANCE_PUBLIC_IP
flower.stylic.ai  A   INSTANCE_PUBLIC_IP
```

Wait for propagation (`dig +short api.stylic.ai` should return the IP).

---

## 3. Server baseline (swap, OOM guard, sysctls)

All commands as `root` unless noted.

### 3.1 Update + basics

```bash
apt update && apt -y upgrade
timedatectl set-timezone UTC
apt -y install curl wget git build-essential software-properties-common \
    ufw htop btop jq unzip tmux ca-certificates
```

### 3.2 4 GB swap file (OOM safety net)

```bash
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Touch swap only when RAM is really under pressure.
sysctl -w vm.swappiness=10
sysctl -w vm.vfs_cache_pressure=50
cat > /etc/sysctl.d/99-stylicai.conf <<'EOF'
vm.swappiness=10
vm.vfs_cache_pressure=50
net.core.somaxconn=1024
net.ipv4.tcp_max_syn_backlog=2048
fs.file-max=1000000
EOF
sysctl --system
```

### 3.3 Raise file descriptor limits

```bash
cat > /etc/security/limits.d/stylicai.conf <<'EOF'
* soft nofile 65535
* hard nofile 65535
root soft nofile 65535
root hard nofile 65535
EOF
```

Log out and back in for limits to apply.

### 3.4 OOM-killer pinning — keep Redis and sshd alive

We tell the kernel to **never** pick `redis-server` or `sshd` first if memory pressure hits. The Celery worker is the one that should die (and auto-restart via systemd):

```bash
# Applied via systemd unit overrides (see §6 and §10). Sanity check:
choom -p $(pidof sshd | awk '{print $1}')     # should be -1000 after §3.5
```

### 3.5 Low-priority sshd hardening

```bash
mkdir -p /etc/systemd/system/ssh.service.d
cat > /etc/systemd/system/ssh.service.d/override.conf <<'EOF'
[Service]
OOMScoreAdjust=-1000
EOF
systemctl daemon-reload
systemctl restart ssh
```

---

## 4. Deploy user & app directory

```bash
adduser --disabled-password --gecos "" deploy
usermod -aG sudo deploy

mkdir -p /home/deploy/.ssh
cp ~/.ssh/authorized_keys /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys

mkdir -p /opt/stylicai /var/log/stylicai
chown -R deploy:deploy /opt/stylicai /var/log/stylicai
```

From now on, SSH in as `deploy`:

```bash
ssh deploy@INSTANCE_PUBLIC_IP
```

---

## 5. System packages

```bash
sudo apt -y install python3.11 python3.11-venv python3.11-dev \
    python3-pip nginx redis-server certbot python3-certbot-nginx \
    libjpeg-dev zlib1g-dev libpng-dev libwebp-dev
```

Verify:

```bash
python3.11 --version    # Python 3.11.x
redis-cli ping          # PONG
nginx -v                # nginx/1.18+
```

---

## 6. Redis (broker + rate limiter backend)

Redis is mission-critical — it is the Celery broker **and** the KIE rate limiter. Lock it down:

```bash
sudo sed -i 's/^# *maxmemory .*/maxmemory 512mb/' /etc/redis/redis.conf
sudo sed -i 's/^# *maxmemory-policy .*/maxmemory-policy noeviction/' /etc/redis/redis.conf
sudo sed -i 's/^bind .*/bind 127.0.0.1 ::1/'      /etc/redis/redis.conf
sudo sed -i 's/^# *requirepass .*/requirepass CHANGE_ME_STRONG_PASSWORD/' /etc/redis/redis.conf

# Pin Redis as low-priority for OOM killer (never kill it first).
sudo mkdir -p /etc/systemd/system/redis-server.service.d
sudo tee /etc/systemd/system/redis-server.service.d/override.conf >/dev/null <<'EOF'
[Service]
OOMScoreAdjust=-900
LimitNOFILE=65535
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now redis-server
sudo systemctl restart redis-server
redis-cli -a CHANGE_ME_STRONG_PASSWORD ping    # PONG
```

> `maxmemory-policy noeviction` is **required** — Celery loses jobs if Redis evicts keys under pressure. The 512 MB cap is plenty (rate-limiter keys use <1 MB; job results <100 MB).

Update your `REDIS_URL` in `.env` (§8) to include the password:
`redis://:CHANGE_ME_STRONG_PASSWORD@127.0.0.1:6379/0`

---

## 7. Clone repository & Python venv

```bash
cd /opt/stylicai
git clone https://github.com/<org>/Newstylicai.git .
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

---

## 8. Environment file (.env)

```bash
cp .env.example .env
nano .env
```

Fill in the real values. Mandatory keys:

| Key                         | Notes |
|-----------------------------|-------|
| `MONGO_URI`, `MONGO_DB_NAME`| MongoDB Atlas. |
| `JWT_SECRET_KEY`            | 64+ random chars. |
| `R2_*`                      | Cloudflare R2 (see `cloudflare-r2-guide.md`). |
| `KIE_API_KEY`               | kie.ai key. |
| `GOOGLE_CLOUD_API_KEY`      | Vertex fallback. |
| `EVOLINK_API_KEY`           | Evolink fallback. |
| `REDIS_URL`                 | `redis://:PASSWORD@127.0.0.1:6379/0` |
| `KIE_RATE_LIMIT_MAX=10`     | Start at 10; raise to 18 once 429 rate is confirmed 0. |
| `KIE_RATE_LIMIT_WINDOW_S=10`| KIE's window. Do not change. |
| `KIE_RATE_LIMIT_429_SLEEP_S=5` | Cool-down when a 429 slips through. |
| `FLOWER_BASIC_AUTH`         | `user:strong_password` — credentials for flower.stylic.ai. |

Lock it down:

```bash
chmod 600 .env
```

---

## 9. Gunicorn (systemd)

FastAPI is served by Gunicorn with Uvicorn workers. Memory footprint is small (~150 MB/worker × 3 workers = 450 MB).

```bash
sudo tee /etc/systemd/system/stylicai-api.service >/dev/null <<'EOF'
[Unit]
Description=Stylic AI FastAPI (Gunicorn + Uvicorn)
After=network.target redis-server.service
Requires=redis-server.service

[Service]
Type=notify
User=deploy
Group=deploy
WorkingDirectory=/opt/stylicai
EnvironmentFile=/opt/stylicai/.env
ExecStart=/opt/stylicai/venv/bin/gunicorn main:app \
    --workers 3 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile /var/log/stylicai/api.access.log \
    --error-logfile  /var/log/stylicai/api.error.log

Restart=on-failure
RestartSec=5
OOMScoreAdjust=-100
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now stylicai-api
sudo systemctl status stylicai-api --no-pager
```

---

## 10. Celery worker (systemd, OOM-safe)

This is the critical service. **Fixed concurrency + per-child memory cap** beats autoscale every time for memory-heavy workloads.

```bash
sudo tee /etc/systemd/system/stylicai-celery.service >/dev/null <<'EOF'
[Unit]
Description=Stylic AI Celery worker — photoshoots queue
After=network.target redis-server.service
Requires=redis-server.service

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/opt/stylicai
EnvironmentFile=/opt/stylicai/.env
ExecStart=/opt/stylicai/venv/bin/celery -A app.worker worker \
    -Q photoshoots \
    --concurrency=3 \
    --prefetch-multiplier=1 \
    --max-tasks-per-child=50 \
    --max-memory-per-child=1500000 \
    --loglevel=info \
    --logfile=/var/log/stylicai/celery.log

# Send TERM first, then KILL after 90s — gives the in-flight task time to
# flush the current pose to R2 before the process is killed.
KillSignal=SIGTERM
TimeoutStopSec=90
Restart=on-failure
RestartSec=10

# Let the kernel pick THIS service first if the box goes OOM — better to
# lose one worker (systemd restarts it) than Redis or the API.
OOMScoreAdjust=200
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now stylicai-celery
sudo systemctl status stylicai-celery --no-pager
sudo tail -n 50 /var/log/stylicai/celery.log
```

Key settings (also documented in `app/worker.py`):

* `--concurrency=3` → 3 simultaneous photoshoots. On 16 GB: `3 × 1.5 GB = 4.5 GB`, leaves 11+ GB headroom for the OS, Redis, Nginx, and Gunicorn.
* `--max-memory-per-child=1500000` KB (1.5 GB) → Celery recycles the child **after** the current task completes. No job is lost.
* `--prefetch-multiplier=1` → no task hoarding.
* `--max-tasks-per-child=50` → periodic flush of library-level memory leaks.

---

## 11. Flower on flower.stylic.ai (systemd)

Flower has its own sub-domain; no URL-prefix, no static-asset headaches.

### 11.1 Install Flower into the venv

```bash
source /opt/stylicai/venv/bin/activate
pip install flower==2.0.1
deactivate
```

### 11.2 Set `FLOWER_BASIC_AUTH` in `.env`

```env
FLOWER_BASIC_AUTH=admin:REPLACE_WITH_STRONG_PASSWORD
```

### 11.3 systemd unit

```bash
sudo tee /etc/systemd/system/stylicai-flower.service >/dev/null <<'EOF'
[Unit]
Description=Stylic AI Celery Flower UI
After=network.target redis-server.service stylicai-celery.service
Requires=redis-server.service

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/opt/stylicai
EnvironmentFile=/opt/stylicai/.env
ExecStart=/opt/stylicai/venv/bin/celery -A app.worker flower \
    --address=127.0.0.1 \
    --port=5555 \
    --persistent=True \
    --db=/opt/stylicai/flower.db \
    --max_tasks=10000 \
    --basic_auth=${FLOWER_BASIC_AUTH}

Restart=on-failure
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now stylicai-flower
sudo systemctl status stylicai-flower --no-pager
```

Flower is now on `127.0.0.1:5555`. Nginx (§12) publishes it on `https://flower.stylic.ai`.

---

## 12. Nginx — api.stylic.ai & flower.stylic.ai

### 12.1 API — `/etc/nginx/sites-available/stylicai-api`

```nginx
upstream stylicai_api {
    server 127.0.0.1:8000;
    keepalive 64;
}

server {
    listen 80;
    server_name api.stylic.ai;

    client_max_body_size 25m;

    access_log /var/log/nginx/stylicai-api.access.log;
    error_log  /var/log/nginx/stylicai-api.error.log warn;

    location / {
        proxy_pass http://stylicai_api;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_connect_timeout 30s;
        proxy_send_timeout   300s;
        proxy_read_timeout   300s;
    }
}
```

### 12.2 Flower — `/etc/nginx/sites-available/stylicai-flower`

```nginx
server {
    listen 80;
    server_name flower.stylic.ai;

    access_log /var/log/nginx/stylicai-flower.access.log;
    error_log  /var/log/nginx/stylicai-flower.error.log warn;

    location / {
        proxy_pass http://127.0.0.1:5555;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # WebSocket for live task updates
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
    }
}
```

### 12.3 Enable + reload

```bash
sudo ln -sf /etc/nginx/sites-available/stylicai-api    /etc/nginx/sites-enabled/stylicai-api
sudo ln -sf /etc/nginx/sites-available/stylicai-flower /etc/nginx/sites-enabled/stylicai-flower
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

---

## 13. Certbot (SSL) — both sub-domains

```bash
sudo certbot --nginx \
    -d api.stylic.ai \
    -d flower.stylic.ai \
    --redirect \
    --agree-tos \
    -m ops@stylic.ai \
    --no-eff-email

# Verify renewal
sudo certbot renew --dry-run
```

Certbot adds `listen 443 ssl`, the ACME challenge blocks, and the 80 → 443 redirect to both server blocks automatically.

---

## 14. Firewall (Vultr portal + UFW)

### 14.1 UFW (host-level)

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw --force enable
sudo ufw status verbose
```

### 14.2 Vultr Firewall (network-level)

In the Vultr portal → **Products → Firewall**:

1. Create a group `stylicai-prod`.
2. Rules:

   | Protocol | Port  | Source        | Notes |
   |----------|-------|---------------|-------|
   | TCP      | 22    | `YOUR_IP/32`  | SSH (restrict to your office/VPN IP). |
   | TCP      | 80    | `0.0.0.0/0`   | HTTP → redirects to 443. |
   | TCP      | 443   | `0.0.0.0/0`   | HTTPS (api + flower). |

3. Attach the firewall group to your instance (**Instance → Settings → Firewall**).

Redis (6379) is bound to `127.0.0.1` only — never expose it to the internet.

---

## 15. Throughput tuning — KIE rate limiter

The limiter (`app/services/kie_rate_limiter.py`) uses Redis-backed sliding-window counters so **every worker on every box sharing the same Redis** enforces the same account-wide cap.

### 15.1 Start conservative

```env
KIE_RATE_LIMIT_MAX=10        # submissions per window
KIE_RATE_LIMIT_WINDOW_S=10   # KIE's window; don't touch
KIE_RATE_LIMIT_429_SLEEP_S=5 # cool-down when a 429 slips through
KIE_RATE_LIMIT_MAX_WAIT_S=60 # caller falls through to Vertex after 60s wait
```

With `MAX=10`, a short burst of 4 simultaneous photoshoots (32 poses) saturates KIE for ~30 s, then Vertex/Evolink absorb the tail — zero 429s observed in staging.

### 15.2 Scale up safely

Watch:

```bash
# Live count of submit-tokens used in the current window:
redis-cli -a PW keys 'kie:createTask:*'
redis-cli -a PW get 'kie:createTask:<bucket>'

# 429 occurrences in the last hour:
sudo journalctl -u stylicai-celery --since '1 hour ago' | grep -c '429'
```

If the 429 count is 0 for 24 hours, raise `KIE_RATE_LIMIT_MAX` to `14`, then `18`. Do **not** cross `18` — KIE's real cap is 20 and you need headroom for poll-traffic clock skew.

### 15.3 Disable for local dev

```env
KIE_RATE_LIMIT_ENABLED=false
```

---

## 16. CI/CD — GitHub Actions

Workflow: `.github/workflows/deploy-vultr.yml`.

Required GitHub repo secrets:

| Secret           | Value |
|------------------|-------|
| `VULTR_HOST`     | Instance public IPv4 |
| `VULTR_USER`     | `deploy` |
| `VULTR_SSH_KEY`  | Private key (PEM) matching the key installed in §2 |

ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/stylicai_deploy
Add ~/.ssh/stylicai_deploy.pub to server user deploy in ~/.ssh/authorized_keys
Put contents of ~/.ssh/stylicai_deploy (private key) into DEPLOY_SSH_KEY

The deploy step runs on the box:

```bash
cd /opt/stylicai
git fetch --all --prune
git reset --hard origin/main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart stylicai-api
sudo systemctl restart stylicai-celery
sudo systemctl restart stylicai-flower
sudo systemctl reload nginx
```

---

## 17. Operations cheat sheet

```bash
# ── Service status ──────────────────────────────────────────────────────────
sudo systemctl status stylicai-api stylicai-celery stylicai-flower redis-server

# ── Logs (live) ─────────────────────────────────────────────────────────────
sudo journalctl -u stylicai-api     -f
sudo journalctl -u stylicai-celery  -f
tail -f /var/log/stylicai/celery.log
tail -f /var/log/stylicai/api.error.log

# ── Restart after .env or code change ───────────────────────────────────────
sudo systemctl restart stylicai-api
sudo systemctl restart stylicai-celery
sudo systemctl restart stylicai-flower

# ── Queue inspection ────────────────────────────────────────────────────────
redis-cli -a PW llen photoshoots          # pending jobs
redis-cli -a PW keys 'kie:createTask:*'   # active rate-limiter buckets

# ── Memory / OOM ────────────────────────────────────────────────────────────
free -h
ps -eo pid,rss,cmd --sort=-rss | head -n 10
dmesg -T | grep -i 'killed process'       # OOM-killer history
sudo journalctl -u stylicai-celery | grep -i 'max memory\|recycling'

# ── Flower ──────────────────────────────────────────────────────────────────
# Browse https://flower.stylic.ai (HTTP basic auth from FLOWER_BASIC_AUTH)

# ── Force-rotate all worker children (e.g. to pick up new model weights) ────
sudo systemctl restart stylicai-celery
```

---

## 18. Troubleshooting

### 18.1 "Worker exited prematurely: signal 9 (SIGKILL)"

Celery child was OOM-killed. Check:

```bash
sudo journalctl -k | grep -i 'out of memory'
sudo journalctl -u stylicai-celery | grep -iE 'memory|killed'
```

Mitigations (usually already applied):

1. Confirm `--max-memory-per-child=1500000` is in the systemd unit.
2. Confirm swap is on: `swapon --show`.
3. Temporarily lower `--concurrency` from 3 → 2 if the box is smaller than recommended.

### 18.2 `429 Too Many Requests` from KIE in logs

The limiter is a **gate**, not a guarantee — if another client shares your KIE account you can still trip 429. Actions:

1. Lower `KIE_RATE_LIMIT_MAX` by 2 (e.g. 10 → 8).
2. Confirm Redis is reachable from both workers: `redis-cli -a PW ping`.
3. `journalctl -u stylicai-celery | grep kie-rl` — should show `token acquired after N attempts` entries, not `redis unavailable`.

### 18.3 Flower shows no tasks / stuck spinner

```bash
sudo systemctl restart stylicai-flower
```

If still blank, confirm Celery is producing events: `celery -A app.worker events` (should show a live stream).

### 18.4 "502 Bad Gateway" on api.stylic.ai

```bash
sudo systemctl status stylicai-api
tail -n 100 /var/log/stylicai/api.error.log
```

Usually a bad `.env` value or a missing dependency after deploy — fix and `sudo systemctl restart stylicai-api`.

### 18.5 DNS not resolving after setup

```bash
dig +short api.stylic.ai
dig +short flower.stylic.ai
```

If these don't return the Vultr IP, fix the A records at your DNS provider and wait for TTL (usually 60–300 s).
