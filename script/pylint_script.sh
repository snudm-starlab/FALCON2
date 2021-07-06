#!/bin/bash
#
cd ../src/imagenetutils/

filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../../src/models/
filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../../src/train_test/

filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done

cd ../../src/utils/

filename=`ls ./*.py`
for eachfile in $filename
do
    echo $eachfile
    pylint --rcfile .pylintrc $eachfile
done
