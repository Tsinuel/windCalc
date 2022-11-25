#!/bin/sh
for i in "$@"; do
   find "$i" -type f -exec touch -m {} \;
done
