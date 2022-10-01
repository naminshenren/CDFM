# CDFM

A PyTorch implementation of CDFM for CTR prediction problem.

## Framework
![image](https://github.com/naminshenren/TransHawkes/blob/master/transHAWKES.PNG)

## Usage

1. Download Criteo's Kaggle display advertising challenge dataset from [here][1]( if you have had it already, skip it ), and put it in *./data/raw/*

2. Generate a preprocessed dataset.

        ./utils/dataPreprocess.py

3. Train a model and predict.

        ./main.py

## Output

## Model
Pre-tained model placed in ‘model_save’.

## Acknowledge
This work was supported by the National Key R&D Program of China under Grant No. 2020AAA0103804(Sponsor: <a  href ="https://bs.ustc.edu.cn/chinese/profile-74.html">Hefu Liu</a>). This work belongs to the University of science and technology of China.

## Reference

[1]: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
