# Analyser core & nlp_tools


## Misc commands
- Create wheel: ```python setup.py bdist_wheel ```
- Collect all wheels of the project: ```pip wheel -r requirements.txt --wheel-dir=allwheels```
- Install collected wheel ``` pip install --no-index --find-links=/tmp/wheelhouse SomePackage ```