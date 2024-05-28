# Segmentation of Historical Documents using Kraken

Example using a dataset of historical documents in the PAGE XML format, with the following split:

* train.txt (90%)
* test.txt (10%) 

### Train a model

#### Activate virtual environment with kraken 5

```bash
    $ source venv_train/bin/activate
```

#### Example command to train

```bash
    $ ketos segtrain -f page -t train.txt -N 100 -q early --min-epochs 50
```

### Test the model

#### Activate virtual environment with kraken 4

This is because version 5 is not working with the trining option

```bash
    $ source venv/bin/activate
```

#### Standard testing

```bash
    $ ketos segtest  -f page -e test.txt -m mymodel.mlmodel
```

#### Upgraded testing option

Adds the following metrics:

* Percentage of correct baselines detected
* mAP@0.50
* Precision-Recall Curve

```bash
    $ python3 segmentation.py -f page -e test.txt -m mymodel.mlmodel
```




