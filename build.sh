#!/bin/bash

# Make sure setuptools and wheel are available for building packages
pip install --upgrade pip setuptools wheel

# Install app dependencies
pip install -r requirements.txt


