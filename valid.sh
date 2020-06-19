#!/bin/bash
if [ $# == 1 ]; then
  n=$1
else
  n=1
fi
set -e
../common/diffrgb myoutput.rgb ../common/examples/output_${n}.rgb
