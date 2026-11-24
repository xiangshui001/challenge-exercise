#!/bin/bash
set -euo pipefail
cd /root/challenge-new-controls-20261122-a
py=/root/anaconda3/envs/challenge-exp5/bin/python
export PYTHONUNBUFFERED=1
if [[ ! -f freeze.json ]]; then "$py" prepare.py ; fi
for model in clip siglip qwen; do "$py" infer.py "$model" --smoke --gpu 0; done
for model in clip siglip qwen; do "$py" infer.py "$model" --gpu 0; done
echo "ALL MODELS SEALED"
