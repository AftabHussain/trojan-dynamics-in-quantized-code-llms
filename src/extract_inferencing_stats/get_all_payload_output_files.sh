#!/bin/bash


files=$(find .. -name '*DROP*.csv')
rm -frv list.txt

for file in $files
do
	realpath $file >> list.txt
done

