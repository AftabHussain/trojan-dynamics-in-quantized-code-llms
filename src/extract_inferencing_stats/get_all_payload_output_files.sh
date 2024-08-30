#!/bin/bash


files=$(find .. -name '*_payload-drop.csv')
rm -frv list.txt

for file in $files
do
	realpath $file >> list.txt
done

