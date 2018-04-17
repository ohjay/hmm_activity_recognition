# HMM Activity Recognition

### Data

To download the KTH and Weizmann datasets, run `./get_data.sh`.

### Usage
```
python main.py <command> <path to config>
```

Specifically,
```
python main.py extract <path to config>  # extract features
python main.py build <path to config>  # build models
python main.py classify <path to config>  # classify activity
```
