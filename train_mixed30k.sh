#!/bin/bash
# Compatibility wrapper for mixed30k training.

set -euo pipefail

DATASET_BACKEND=mixed30k bash ./train.sh
