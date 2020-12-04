#!usr/bin/bash
cd src

# Supplement
echo
echo "#########################"
echo "# Supplemental analyses #"
echo "#########################"
echo

echo "Running dwell time regressions..."
python Supplement_run-dwell-regression.py --verbose 2 --mixed  --timebinned --seed 2 --label timebinned
# python Supplement_run-dwell-regression.py --verbose 2 --mixed  --seed 1 --label fulltrial
