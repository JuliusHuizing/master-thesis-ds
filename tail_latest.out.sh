#!/bin/bash

latest_file=$(ls -v logs | grep '\.out$' | tail -n 1)

if [ -z "logs/$latest_file" ]; then
  echo "No .out file found."
  exit 1
fi

tail -fn 100000000000 "$latest_file"