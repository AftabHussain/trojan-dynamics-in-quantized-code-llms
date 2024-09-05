#!/bin/bash


files=$(find .. -name '*DROPTABLE*.csv')
rm -frv list.txt

for file in $files
do
	realpath $file >> list.txt
done

