#!/usr/bin/env bash

myVar="$(cat $1 | grep -E '!([0-9]+) = !\{[^"]+"nvvm-reflect-ftz"' | cut -d ' ' -f 1)" 
awk '!/nvvm-reflect-ftz/' $1 | sed "/^\!llvm\.module\.flags = /s/$myVar, //" > $2