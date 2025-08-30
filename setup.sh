#!/bin/bash

# setup.sh
apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

pip install --upgrade pip
pip install -r requirements.txt
