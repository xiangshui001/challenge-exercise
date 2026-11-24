#!/bin/bash
set -euo pipefail
cd /root/challenge-grounding-confirm-20261122
py=/root/anaconda3/envs/challenge-exp5/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
if [[ ! -f base.audit.json ]]; then "$py" prepare_confirm.py; fi
if [[ ! -f freeze.json ]]; then "$py" assign.py; fi
for m in clip siglip qwen; do "$py" infer.py "$m" --smoke --gpu 0; done
for m in clip siglip qwen; do "$py" infer.py "$m" --gpu 0; done
echo CONFIRMATION_COMPLETE
