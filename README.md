# Analyser core & nlp_tools


## Misc commands
- Create wheel: 
```
python setup.py bdist_wheel 
```
- Collect all wheels of the project: 
```
pip wheel -r requirements.txt --wheel-dir=tmp/wheelhouse
```
- Install collected wheels 
``` 
pip install --no-index --find-links=tmp/wheelhouse SomePackage 
```

## Assign release tag:
1. Create a tag:
    ```                     
    > git tag -a vX.X.X -m "<release comment>"
    ```
1. Push tag:
    ```                     
    > git push origin --tags
    ```

## Usage(Windows):
1. Install Python >=3.6 and pip
1. Install ```virtualenv```( [https://virtualenv.pypa.io/en/latest/installation/]() ):
    ```
    > pip install virtualenv
    ```
1. Download ``` nemoware_analyzer-X.X.X-py3-none-any.whl ``` to working dir (e.g. ```analyser_home```)     
1. Go to working dir:
    ```
    > cd analyser_home
    ```
1. Create virtual environment (with name ```venv```):
    ```
    > virtualenv venv
    ```
1. Activate:
    ``` 
    > .\venv\Scripts\activate
    ```
1. Install ```analyser``` with all deps:
    ```
    > pip install  .\nemoware_analyzer-X.X.X-py3-none-any.whl    
    ```
1. Run:

    ```
    > analyser_run
    ```



