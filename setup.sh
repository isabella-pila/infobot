#!/bin/bash

pip install -r requirements.txt

playwright install chromium

export PLAYWRIGHT_BROWSERS_PATH=0
