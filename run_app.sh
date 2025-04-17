#!/bin/bash

# Deactivate any active environments first
if [[ -n $VIRTUAL_ENV ]]; then
    deactivate
fi

# Find the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Activate the virtual environment
source "$DIR/venv/bin/activate"

# Run the app
python -m streamlit run app.py

# Deactivate when done
deactivate 