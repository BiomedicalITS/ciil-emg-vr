source ./.venv/bin/activate

heic2png -i ${1}.HEIC -w -q 100
rembg i ${1}.png ${1}_nobg.png
