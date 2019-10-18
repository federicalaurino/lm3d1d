#! /bin/bash

script=`ls test_*.py`
for i in $script
do
    echo ">>>>>>>>>>>>>" $i "<<<<<<<<<<<<<<";
    python $i;
done
