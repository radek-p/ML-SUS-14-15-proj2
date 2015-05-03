#!/bin/bash

rm -rf data-vis;
mkdir -p data-vis/;

let "c=0";
while read -r line; do
	mkdir -p data-vis/$c/;
	for filename in $line; do
		ln -s ../../data/$filename data-vis/$c/$filename;
	done;
	let "c++";
done < ./out.txt

# Problemy:
# e <--|--> c
# b <--|--> h
# n <--|--> u
# l <--|--> ł
# ś <--|--> ś rozbicie!
