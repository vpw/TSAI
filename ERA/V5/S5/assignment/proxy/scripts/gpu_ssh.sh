#!/usr/bin/env bash
# Thin ssh/rsync wrapper for the GPU box, so the host key and identity options live in one
# place. Host is resolved from EC2 each time, because a stopped instance gets a new public
# IP every start.
#
#   gpu_ssh.sh run  '<remote command>'
#   gpu_ssh.sh push <local path> <remote path>
#   gpu_ssh.sh pull <remote path> <local path>
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IP="${GPU_IP:-$("$HERE/aws_gpu.sh" ip)}"
KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
USER="${SSH_USER:-ubuntu}"
OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
      -o ServerAliveInterval=30 -i "$KEY")

case "${1:-}" in
  run)  shift; exec ssh "${OPTS[@]}" "$USER@$IP" "$@" ;;
  push) shift; exec rsync -a -e "ssh ${OPTS[*]}" "$1" "$USER@$IP:$2" ;;
  pull) shift; exec rsync -a -e "ssh ${OPTS[*]}" "$USER@$IP:$1" "$2" ;;
  ip)   echo "$IP" ;;
  *)    echo "usage: $0 {run <cmd>|push <src> <dst>|pull <src> <dst>|ip}"; exit 2 ;;
esac
