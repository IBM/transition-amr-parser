#!/bin/bash

set -o errexit
set -o pipefail


rootdir=/dccstor/jzhou1/work/EXP
files=(${rootdir}/*pmask1*)    # change the pattern here

for f in "${files[@]}"
do
    echo "removing $f"
    rm -rf $f
done