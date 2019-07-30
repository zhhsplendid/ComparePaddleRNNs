#!/bin/sh

python static_rnn.py

echo "Sleeping for next benchmark"
sleep 5

python dynamic_rnn.py
