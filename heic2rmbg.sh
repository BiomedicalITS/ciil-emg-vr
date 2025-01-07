#!/usr/bin/env bash

set -euo

source ./.venv/bin/activate

heic2png -i "${1}.HEIC" -q 90
rembg i "${1}.png" "${1}_nobg.png" 
