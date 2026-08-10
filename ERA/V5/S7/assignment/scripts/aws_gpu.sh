#!/usr/bin/env bash
# Lifecycle control for the GPU box the proxy ablation runs on.
#
#   aws_gpu.sh status        what state is it in, and what does it cost
#   aws_gpu.sh start         start it and wait until SSH answers, print the public IP
#   aws_gpu.sh ip            print the current public IP
#   aws_gpu.sh stop          stop it, then VERIFY it reached "stopped"
#
# The instance bills by the second while running, so stop is verified rather than assumed.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# awscli lives in S5's venv (this session reuses S5's GPU box rather than provisioning a new
# one); fall back to a local venv or PATH so the script works if that ever moves.
for cand in "$REPO/.venv/bin/aws" "$REPO/../../S5/assignment/.venv/bin/aws" "$(command -v aws || true)"; do
  [ -x "$cand" ] && AWS="$cand" && break
done
: "${AWS:?aws CLI not found - checked S7 .venv, S5 .venv and PATH}"
export AWS_PROFILE="${AWS_PROFILE:-AWS-ESS}"
REGION="${AWS_REGION:-ap-south-1}"
INSTANCE_ID="${INSTANCE_ID:-i-025fb7b65d7e3460e}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
SSH_USER="${SSH_USER:-ubuntu}"

q() { "$AWS" ec2 "$@" --region "$REGION"; }

state() {
  q describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null
}

public_ip() {
  q describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null
}

case "${1:-status}" in
  status)
    q describe-instances --instance-ids "$INSTANCE_ID" \
      --query 'Reservations[0].Instances[0].{State:State.Name,Type:InstanceType,IP:PublicIpAddress,Launch:LaunchTime}' \
      --output table
    ;;

  start)
    s="$(state)"
    echo "state: $s"
    if [ "$s" != "running" ]; then
      q start-instances --instance-ids "$INSTANCE_ID" --output text >/dev/null || exit 1
      echo -n "waiting for running"
      for _ in $(seq 1 60); do
        sleep 5; echo -n "."
        [ "$(state)" = "running" ] && break
      done
      echo
    fi
    ip="$(public_ip)"
    echo "public ip: $ip"
    echo -n "waiting for ssh"
    for _ in $(seq 1 60); do
      if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
             -o ConnectTimeout=5 -i "$SSH_KEY" "$SSH_USER@$ip" true 2>/dev/null; then
        echo " ok"; echo "$ip"; exit 0
      fi
      sleep 5; echo -n "."
    done
    echo " TIMED OUT"; exit 1
    ;;

  ip) public_ip ;;

  stop)
    q stop-instances --instance-ids "$INSTANCE_ID" --output text >/dev/null
    echo -n "stopping"
    # A g4dn with instance-store volumes can sit in `stopping` for several minutes while the
    # ephemeral disks are scrubbed. 5 minutes was not enough on the real run and produced a
    # false alarm; allow 15 before shouting.
    for _ in $(seq 1 180); do
      s="$(state)"
      [ "$s" = "stopped" ] && { echo " -> stopped (verified)"; exit 0; }
      sleep 5; echo -n "."
    done
    echo
    echo "!! STILL '$(state)' AFTER WAITING — CHECK THE CONSOLE, IT MAY STILL BE BILLING"
    exit 1
    ;;

  *) echo "usage: $0 {status|start|ip|stop}"; exit 2 ;;
esac
