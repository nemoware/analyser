# shellcheck shell=sh

# activates local jupyter in virtual env
#this allows colab to connect 127.0.0.1

source /Users/artem/work/nemo/goil/nlp_tools/venv/bin/activate

#jupyter  --port=8888 --NotebookApp.port_retries=0

#jupyter notebook run ../trainsets/retrain_contract_uber_model.ipynb --output=../../work/new.ipynb

jupyter nbconvert --to notebook --execute trainsets/retrain_contract_uber_model.ipynb --output=../../work/new.ipynb
