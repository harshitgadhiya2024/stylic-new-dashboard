#!/bin/bash
# =============================================================================
# EC2 service setup script — run once after pulling new code
# Sets up / updates the Celery worker and Flower monitoring as systemd services
#
# Usage:
#   chmod +x deploy/setup_services.sh
#   sudo bash deploy/setup_services.sh
# =============================================================================

set -e

APP_DIR="/home/ubuntu/stylic-ai"
LOG_DIR="/var/log/celery"
RUN_DIR="/var/run/celery"

echo "=== Creating log and PID directories ==="
mkdir -p "$LOG_DIR" "$RUN_DIR"
chown ubuntu:ubuntu "$LOG_DIR" "$RUN_DIR"

echo "=== Copying service files to systemd ==="
cp "$APP_DIR/deploy/celery-photoshoot.service" /etc/systemd/system/celery-photoshoot.service
cp "$APP_DIR/deploy/flower.service"            /etc/systemd/system/flower.service

echo "=== Reloading systemd daemon ==="
systemctl daemon-reload

echo "=== Enabling services (auto-start on reboot) ==="
systemctl enable celery-photoshoot
systemctl enable flower

echo "=== Restarting services ==="
systemctl restart celery-photoshoot
systemctl restart flower

echo ""
echo "=== Status ==="
systemctl status celery-photoshoot --no-pager
echo "---"
systemctl status flower --no-pager

echo ""
echo "Done! Services are running."
echo "  Flower UI:    http://$(curl -s ifconfig.me):5555  (user: admin  pass: changeme)"
echo "  Celery logs:  tail -f $LOG_DIR/photoshoot.log"
echo "  Flower logs:  tail -f $LOG_DIR/flower.log"
