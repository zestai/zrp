# Alabama Case Study
The purpose of this notebook is to illustrate an application of the Zest Race Predictor `ZRP`, a module that generates race & ethnicity predictions, outside of model development. This case study will involve a sample of Alabama voter registration data. 

The Alabama voter registration data sample contains 229,644 records with required name and address features. 

## Table of Contents:
* [Invoke the Zest Race Predictor](#zrp-start)
* [Performance Analysis](#perform)
    * [ZRP](#perform-zrp)
    * [BISG](#perform-bisg)
    * [Comparing Methods](#perform-compare)
    * [Alternative Measurements of Performance](#alt)
        * [Age Analysis](#alt-age)
        * [Geographic level Analysis](#alt-geo)
* [Appendix](#appendix)



```python
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi=False
```


```python
from os.path import join, expanduser
from sklearn.metrics import confusion_matrix
import pandas
import pandas as pd
import sys
import os
import re
import warnings
```

#### Set source code path here


```python
warnings.filterwarnings(action='once')
home = expanduser('~')

src_path = '{}/zrp'.format(home)
sys.path.append(src_path)
```


```python
from zrp import ZRP
from zrp.modeling.performance import *
from zrp.modeling.predict import ZRP_Predict
from zrp.prepare.utils import *
```

    /home/kam/.conda/envs/zrp_q1_22/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)


#### Load data
Load in Alabama voter registration data. Some initial processing was done on the raw Alabama data, to ensure required data is available for both ZRP & BISG.  


```python
support_files_path = "/d/shared/zrp/shared_data"
key ='ZEST_KEY'
```


```python
df = load_file("/d/shared/zrp/shared_data/processed/data/state_level/voters/processed_al_2022q1.parquet")
df.shape
```




    (229644, 13)



##### Encode key
While the Alabama voter registration data available to the public, the key is hashed to maintain voter privacy. 


```python
df.index  = pd.util.hash_pandas_object(df.index).astype(str)
df.index.name = "ZEST_KEY"
```

Distribution of Race/Ethnicity


```python
df.race.value_counts(dropna=False)
```




    WHITE       166749
    BLACK        54145
    HISPANIC      5649
    AAPI          2526
    AIAN           575
    Name: race, dtype: int64



# Invoke the Zest Race Predictor<a class="anchor" id="zrp-start"></a>
On the Alabama data


```python
%%time
zest_race_predictor = ZRP(**{"n_jobs" : 94})
zest_race_predictor.fit()
zrp_output = zest_race_predictor.transform(df)
```

    Directory already exists
    Data is loaded
       [Start] Validating input data
         Number of observations: 229644
         Is key unique: True
    Directory already exists
       [Completed] Validating input data
    
    The key is already set
       Formatting P1
       Formatting P2
       reduce whitespace
    
    [Start] Preparing geo data


      0%|          | 0/1 [00:00<?, ?it/s]

    
      The following states are included in the data: ['AL']
       ... on state: AL
    
       Data is loaded
       [Start] Processing geo data
          ...address cleaning


    
      0%|          | 0/229644 [00:00<?, ?it/s][A[Parallel(n_jobs=94)]: Using backend ThreadingBackend with 94 concurrent workers.
    [Parallel(n_jobs=94)]: Done  12 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=94)]: Done 262 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=94)]: Done 612 tasks      | elapsed:    0.1s
    
      0%|          | 846/229644 [00:00<00:27, 8373.50it/s][A[Parallel(n_jobs=94)]: Done 1062 tasks      | elapsed:    0.1s
    
      1%|          | 1836/229644 [00:00<00:25, 8779.59it/s][A[Parallel(n_jobs=94)]: Done 1612 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=94)]: Done 2262 tasks      | elapsed:    0.3s
    
      1%|          | 2726/229644 [00:00<00:25, 8798.50it/s][A[Parallel(n_jobs=94)]: Done 3012 tasks      | elapsed:    0.4s
    
      2%|▏         | 3760/229644 [00:00<00:25, 8995.94it/s][A[Parallel(n_jobs=94)]: Done 3862 tasks      | elapsed:    0.4s
    
      2%|▏         | 4700/229644 [00:00<00:25, 8872.81it/s][A[Parallel(n_jobs=94)]: Done 4812 tasks      | elapsed:    0.6s
    
      2%|▏         | 5640/229644 [00:00<00:25, 8828.59it/s][A[Parallel(n_jobs=94)]: Done 5862 tasks      | elapsed:    0.7s
    
      3%|▎         | 6580/229644 [00:00<00:25, 8854.08it/s][A[Parallel(n_jobs=94)]: Done 7012 tasks      | elapsed:    0.8s
    
      3%|▎         | 7398/229644 [00:00<00:31, 7048.65it/s][A
      4%|▎         | 8272/229644 [00:01<00:30, 7309.32it/s][A[Parallel(n_jobs=94)]: Done 8262 tasks      | elapsed:    1.0s
    
      4%|▍         | 9212/229644 [00:01<00:28, 7639.34it/s][A[Parallel(n_jobs=94)]: Done 9612 tasks      | elapsed:    1.2s
    
      5%|▍         | 10528/229644 [00:01<00:25, 8587.98it/s][A[Parallel(n_jobs=94)]: Done 11062 tasks      | elapsed:    1.3s
    
      5%|▍         | 11468/229644 [00:01<00:25, 8697.97it/s][A
      5%|▌         | 12408/229644 [00:01<00:24, 8717.81it/s][A[Parallel(n_jobs=94)]: Done 12612 tasks      | elapsed:    1.5s
    
      6%|▌         | 13303/229644 [00:01<00:24, 8730.17it/s][A
      6%|▌         | 14194/229644 [00:01<00:24, 8658.05it/s][A[Parallel(n_jobs=94)]: Done 14262 tasks      | elapsed:    1.7s
    
      7%|▋         | 15134/229644 [00:01<00:24, 8617.43it/s][A
      7%|▋         | 16074/229644 [00:01<00:24, 8730.87it/s][A[Parallel(n_jobs=94)]: Done 16012 tasks      | elapsed:    1.9s
    
      7%|▋         | 17014/229644 [00:01<00:24, 8844.62it/s][A
      8%|▊         | 17954/229644 [00:02<00:23, 8967.97it/s][A[Parallel(n_jobs=94)]: Done 17862 tasks      | elapsed:    2.1s
    
      8%|▊         | 18894/229644 [00:02<00:23, 8980.14it/s][A
      9%|▊         | 19834/229644 [00:02<00:23, 9054.10it/s][A[Parallel(n_jobs=94)]: Done 19812 tasks      | elapsed:    2.3s
    
      9%|▉         | 20774/229644 [00:02<00:23, 9053.52it/s][A
      9%|▉         | 21714/229644 [00:02<00:23, 9039.32it/s][A[Parallel(n_jobs=94)]: Done 21862 tasks      | elapsed:    2.5s
    
     10%|█         | 23030/229644 [00:02<00:21, 9829.58it/s][A
     10%|█         | 24034/229644 [00:02<00:21, 9637.36it/s][A[Parallel(n_jobs=94)]: Done 24012 tasks      | elapsed:    2.7s
    
     11%|█         | 25014/229644 [00:02<00:22, 9246.42it/s][A
     11%|█▏        | 25953/229644 [00:02<00:22, 9099.41it/s][A[Parallel(n_jobs=94)]: Done 26262 tasks      | elapsed:    3.0s
    
     12%|█▏        | 26884/229644 [00:03<00:22, 9021.92it/s][A
     12%|█▏        | 27824/229644 [00:03<00:22, 8984.80it/s][A[Parallel(n_jobs=94)]: Done 28612 tasks      | elapsed:    3.2s
    
     13%|█▎        | 28858/229644 [00:03<00:21, 9273.61it/s][A
     13%|█▎        | 29892/229644 [00:03<00:21, 9353.76it/s][A
     13%|█▎        | 30832/229644 [00:03<00:21, 9168.40it/s][A[Parallel(n_jobs=94)]: Done 31062 tasks      | elapsed:    3.5s
    
     14%|█▍        | 31772/229644 [00:03<00:21, 9028.91it/s][A
     14%|█▍        | 32712/229644 [00:03<00:21, 9025.78it/s][A
     15%|█▍        | 33652/229644 [00:03<00:21, 8917.68it/s][A[Parallel(n_jobs=94)]: Done 33612 tasks      | elapsed:    3.8s
    
     15%|█▌        | 34592/229644 [00:03<00:21, 8884.58it/s][A
     15%|█▌        | 35532/229644 [00:03<00:21, 8868.34it/s][A[Parallel(n_jobs=94)]: Done 36262 tasks      | elapsed:    4.1s
    
     16%|█▌        | 36472/229644 [00:04<00:21, 8883.27it/s][A
     16%|█▋        | 37412/229644 [00:04<00:21, 8888.00it/s][A
     17%|█▋        | 38352/229644 [00:04<00:21, 8997.19it/s][A[Parallel(n_jobs=94)]: Done 39012 tasks      | elapsed:    4.4s
    
     17%|█▋        | 39480/229644 [00:04<00:19, 9528.81it/s][A
     18%|█▊        | 40442/229644 [00:04<00:19, 9476.34it/s][A
     18%|█▊        | 41396/229644 [00:04<00:20, 9349.66it/s][A[Parallel(n_jobs=94)]: Done 41862 tasks      | elapsed:    4.7s
    
     18%|█▊        | 42336/229644 [00:04<00:20, 9146.84it/s][A
     19%|█▉        | 43256/229644 [00:04<00:20, 9047.04it/s][A
     19%|█▉        | 44180/229644 [00:04<00:20, 8865.91it/s][A[Parallel(n_jobs=94)]: Done 44812 tasks      | elapsed:    5.0s
    
     20%|█▉        | 45120/229644 [00:05<00:20, 8920.88it/s][A
     20%|██        | 46060/229644 [00:05<00:20, 8942.12it/s][A
     20%|██        | 47000/229644 [00:05<00:20, 9009.31it/s][A
     21%|██        | 47940/229644 [00:05<00:20, 9013.47it/s][A[Parallel(n_jobs=94)]: Done 47862 tasks      | elapsed:    5.4s
    
     22%|██▏       | 50666/229644 [00:05<00:15, 11272.20it/s][A[Parallel(n_jobs=94)]: Done 51012 tasks      | elapsed:    5.5s
    
     23%|██▎       | 52264/229644 [00:05<00:14, 12348.71it/s][A
     23%|██▎       | 53753/229644 [00:05<00:15, 11004.75it/s][A[Parallel(n_jobs=94)]: Done 54262 tasks      | elapsed:    5.8s
    
     24%|██▍       | 55059/229644 [00:05<00:15, 11383.29it/s][A
     25%|██▍       | 56345/229644 [00:05<00:16, 10414.56it/s][A
     25%|██▌       | 57506/229644 [00:06<00:17, 9949.80it/s] [A[Parallel(n_jobs=94)]: Done 57612 tasks      | elapsed:    6.1s
    
     26%|██▌       | 58589/229644 [00:06<00:18, 9437.15it/s][A
     26%|██▌       | 59600/229644 [00:06<00:18, 9258.08it/s][A
     26%|██▋       | 60574/229644 [00:06<00:18, 9224.22it/s][A[Parallel(n_jobs=94)]: Done 61062 tasks      | elapsed:    6.5s
    
     27%|██▋       | 61852/229644 [00:06<00:16, 10039.86it/s][A
     27%|██▋       | 62901/229644 [00:06<00:17, 9647.95it/s] [A
     28%|██▊       | 63901/229644 [00:06<00:17, 9499.23it/s][A[Parallel(n_jobs=94)]: Done 64612 tasks      | elapsed:    6.9s
    
     28%|██▊       | 64876/229644 [00:06<00:17, 9231.55it/s][A
     29%|██▊       | 65819/229644 [00:06<00:18, 9054.70it/s][A
     29%|██▉       | 66740/229644 [00:07<00:18, 8917.06it/s][A
     29%|██▉       | 67680/229644 [00:07<00:18, 8804.84it/s][A[Parallel(n_jobs=94)]: Done 68262 tasks      | elapsed:    7.3s
    
     30%|██▉       | 68620/229644 [00:07<00:18, 8718.08it/s][A
     30%|███       | 69560/229644 [00:07<00:18, 8638.06it/s][A
     31%|███       | 70500/229644 [00:07<00:18, 8695.99it/s][A
     31%|███       | 71440/229644 [00:07<00:18, 8668.80it/s][A[Parallel(n_jobs=94)]: Done 72012 tasks      | elapsed:    7.7s
    
     31%|███▏      | 72309/229644 [00:07<00:21, 7406.14it/s][A
     32%|███▏      | 73132/229644 [00:07<00:20, 7552.49it/s][A
     32%|███▏      | 73978/229644 [00:08<00:19, 7799.65it/s][A
     33%|███▎      | 74918/229644 [00:08<00:19, 8072.43it/s][A
     33%|███▎      | 75858/229644 [00:08<00:18, 8220.03it/s][A[Parallel(n_jobs=94)]: Done 75862 tasks      | elapsed:    8.3s
    
     34%|███▍      | 78396/229644 [00:08<00:14, 10298.16it/s][A
     35%|███▍      | 79733/229644 [00:08<00:15, 9910.31it/s] [A[Parallel(n_jobs=94)]: Done 79812 tasks      | elapsed:    8.5s
    
     36%|███▌      | 82250/229644 [00:08<00:12, 12098.41it/s][A
     37%|███▋      | 83846/229644 [00:08<00:13, 11119.94it/s][A[Parallel(n_jobs=94)]: Done 83862 tasks      | elapsed:    8.8s
    
     37%|███▋      | 85244/229644 [00:08<00:13, 10318.30it/s][A
     38%|███▊      | 86487/229644 [00:09<00:14, 9728.08it/s] [A
     38%|███▊      | 87615/229644 [00:09<00:14, 9513.35it/s][A[Parallel(n_jobs=94)]: Done 88012 tasks      | elapsed:    9.2s
    
     39%|███▊      | 88676/229644 [00:09<00:15, 9349.82it/s][A
     39%|███▉      | 89689/229644 [00:09<00:15, 9176.11it/s][A
     39%|███▉      | 90662/229644 [00:09<00:15, 9259.10it/s][A
     40%|███▉      | 91627/229644 [00:09<00:14, 9324.11it/s][A[Parallel(n_jobs=94)]: Done 92262 tasks      | elapsed:    9.7s
    
     40%|████      | 92587/229644 [00:09<00:14, 9145.11it/s][A
     41%|████      | 93522/229644 [00:09<00:15, 9065.70it/s][A
     41%|████      | 94443/229644 [00:09<00:15, 8868.33it/s][A
     42%|████▏     | 95341/229644 [00:10<00:15, 8517.85it/s][A
     42%|████▏     | 96256/229644 [00:10<00:15, 8507.79it/s][A[Parallel(n_jobs=94)]: Done 96612 tasks      | elapsed:   10.2s
    
     43%|████▎     | 97948/229644 [00:10<00:13, 9929.63it/s][A
     43%|████▎     | 99037/229644 [00:10<00:13, 9734.78it/s][A
     44%|████▎     | 100079/229644 [00:10<00:13, 9481.53it/s][A
     44%|████▍     | 101238/229644 [00:10<00:12, 9948.97it/s][A[Parallel(n_jobs=94)]: Done 101062 tasks      | elapsed:   10.6s
    
     45%|████▍     | 102274/229644 [00:10<00:12, 9955.45it/s][A
     45%|████▍     | 103298/229644 [00:10<00:12, 9745.06it/s][A
     45%|████▌     | 104294/229644 [00:10<00:13, 9399.04it/s][A
     46%|████▌     | 105252/229644 [00:11<00:13, 9429.70it/s][A[Parallel(n_jobs=94)]: Done 105612 tasks      | elapsed:   11.1s
    
     46%|████▋     | 106690/229644 [00:11<00:11, 10495.28it/s][A
     47%|████▋     | 107787/229644 [00:11<00:11, 10299.15it/s][A
     47%|████▋     | 108851/229644 [00:11<00:12, 9863.87it/s] [A
     48%|████▊     | 109865/229644 [00:11<00:12, 9493.91it/s][A[Parallel(n_jobs=94)]: Done 110262 tasks      | elapsed:   11.6s
    
     48%|████▊     | 111202/229644 [00:11<00:11, 10325.34it/s][A
     49%|████▉     | 112272/229644 [00:11<00:11, 9855.95it/s] [A
     49%|████▉     | 113288/229644 [00:11<00:12, 9463.54it/s][A
     50%|████▉     | 114492/229644 [00:11<00:11, 10075.79it/s][A[Parallel(n_jobs=94)]: Done 115012 tasks      | elapsed:   12.0s
    
     50%|█████     | 115528/229644 [00:12<00:11, 9706.82it/s] [A
     51%|█████     | 116522/229644 [00:12<00:11, 9572.70it/s][A
     51%|█████     | 117496/229644 [00:12<00:11, 9366.58it/s][A
     52%|█████▏    | 118445/229644 [00:12<00:12, 8885.95it/s][A
     52%|█████▏    | 119380/229644 [00:12<00:12, 8858.98it/s][A[Parallel(n_jobs=94)]: Done 119862 tasks      | elapsed:   12.6s
    
     52%|█████▏    | 120320/229644 [00:12<00:12, 8818.83it/s][A
     53%|█████▎    | 121260/229644 [00:12<00:12, 8842.09it/s][A
     53%|█████▎    | 122200/229644 [00:12<00:12, 8794.37it/s][A
     54%|█████▎    | 123140/229644 [00:12<00:12, 8722.55it/s][A
     54%|█████▍    | 124080/229644 [00:13<00:12, 8729.57it/s][A[Parallel(n_jobs=94)]: Done 124812 tasks      | elapsed:   13.1s
    
     54%|█████▍    | 125020/229644 [00:13<00:11, 8751.31it/s][A
     55%|█████▍    | 125960/229644 [00:13<00:11, 8786.36it/s][A
     56%|█████▌    | 128404/229644 [00:13<00:09, 10867.10it/s][A
     57%|█████▋    | 129749/229644 [00:13<00:09, 10239.60it/s][A[Parallel(n_jobs=94)]: Done 129862 tasks      | elapsed:   13.5s
    
     57%|█████▋    | 130961/229644 [00:13<00:10, 9747.56it/s] [A
     58%|█████▊    | 132073/229644 [00:13<00:10, 9455.21it/s][A
     58%|█████▊    | 133116/229644 [00:13<00:10, 9265.56it/s][A
     58%|█████▊    | 134112/229644 [00:13<00:10, 9213.78it/s][A
     59%|█████▉    | 135082/229644 [00:14<00:10, 8934.05it/s][A[Parallel(n_jobs=94)]: Done 135012 tasks      | elapsed:   14.1s
    
     59%|█████▉    | 136018/229644 [00:14<00:10, 8888.44it/s][A
     60%|█████▉    | 136958/229644 [00:14<00:10, 8884.26it/s][A
     60%|██████    | 137898/229644 [00:14<00:10, 8885.21it/s][A
     60%|██████    | 138838/229644 [00:14<00:10, 8914.44it/s][A
     61%|██████    | 139778/229644 [00:14<00:10, 8907.39it/s][A[Parallel(n_jobs=94)]: Done 140262 tasks      | elapsed:   14.7s
    
     62%|██████▏   | 141282/229644 [00:14<00:08, 10086.42it/s][A
     62%|██████▏   | 142348/229644 [00:14<00:09, 9666.45it/s] [A
     62%|██████▏   | 143359/229644 [00:14<00:09, 9313.83it/s][A
     63%|██████▎   | 144324/229644 [00:15<00:09, 9246.49it/s][A
     63%|██████▎   | 145273/229644 [00:15<00:09, 9208.69it/s][A[Parallel(n_jobs=94)]: Done 145612 tasks      | elapsed:   15.2s
    
     64%|██████▎   | 146211/229644 [00:15<00:10, 8334.34it/s][A
     64%|██████▍   | 147110/229644 [00:15<00:09, 8390.07it/s][A
     64%|██████▍   | 148050/229644 [00:15<00:09, 8569.21it/s][A
     65%|██████▍   | 148990/229644 [00:15<00:09, 8668.40it/s][A
     65%|██████▌   | 149930/229644 [00:15<00:09, 8722.15it/s][A
     66%|██████▌   | 150870/229644 [00:15<00:08, 8781.63it/s][A[Parallel(n_jobs=94)]: Done 151062 tasks      | elapsed:   15.9s
    
     67%|██████▋   | 152938/229644 [00:15<00:07, 10597.75it/s][A
     67%|██████▋   | 154171/229644 [00:16<00:07, 9994.56it/s] [A
     68%|██████▊   | 155299/229644 [00:16<00:07, 9670.35it/s][A
     68%|██████▊   | 156359/229644 [00:16<00:07, 9420.80it/s][A[Parallel(n_jobs=94)]: Done 156612 tasks      | elapsed:   16.4s
    
     69%|██████▉   | 158296/229644 [00:16<00:06, 11106.85it/s][A
     69%|██████▉   | 159574/229644 [00:16<00:06, 10474.15it/s][A
     70%|██████▉   | 160746/229644 [00:16<00:06, 9890.42it/s] [A
     70%|███████   | 161829/229644 [00:16<00:07, 9596.64it/s][A[Parallel(n_jobs=94)]: Done 162262 tasks      | elapsed:   16.9s
    
     71%|███████   | 162857/229644 [00:16<00:07, 9367.85it/s][A
     71%|███████▏  | 163843/229644 [00:17<00:07, 9077.59it/s][A
     72%|███████▏  | 164787/229644 [00:17<00:07, 8965.85it/s][A
     72%|███████▏  | 165722/229644 [00:17<00:07, 8910.89it/s][A
     73%|███████▎  | 166662/229644 [00:17<00:07, 8833.07it/s][A
     73%|███████▎  | 167602/229644 [00:17<00:07, 8828.36it/s][A[Parallel(n_jobs=94)]: Done 168012 tasks      | elapsed:   17.5s
    
     73%|███████▎  | 168542/229644 [00:17<00:06, 8853.52it/s][A
     74%|███████▍  | 169482/229644 [00:17<00:06, 8849.41it/s][A
     74%|███████▍  | 170422/229644 [00:17<00:06, 8890.69it/s][A
     75%|███████▍  | 171362/229644 [00:17<00:06, 8799.18it/s][A
     75%|███████▌  | 172302/229644 [00:18<00:06, 8769.01it/s][A
     75%|███████▌  | 173242/229644 [00:18<00:06, 8762.94it/s][A[Parallel(n_jobs=94)]: Done 173862 tasks      | elapsed:   18.2s
    
     76%|███████▌  | 174182/229644 [00:18<00:06, 8716.57it/s][A
     76%|███████▋  | 175122/229644 [00:18<00:06, 8701.61it/s][A
     77%|███████▋  | 176062/229644 [00:18<00:06, 8767.46it/s][A
     77%|███████▋  | 177002/229644 [00:18<00:05, 8845.00it/s][A
     77%|███████▋  | 177942/229644 [00:18<00:05, 8818.28it/s][A
     78%|███████▊  | 178882/229644 [00:18<00:05, 8876.87it/s][A
     78%|███████▊  | 179822/229644 [00:18<00:05, 8826.39it/s][A[Parallel(n_jobs=94)]: Done 179812 tasks      | elapsed:   18.9s
    
     79%|███████▉  | 182266/229644 [00:18<00:04, 10905.46it/s][A
     80%|███████▉  | 183611/229644 [00:19<00:04, 10316.32it/s][A
     80%|████████  | 184827/229644 [00:19<00:04, 9716.58it/s] [A
     81%|████████  | 185935/229644 [00:19<00:04, 9305.51it/s][A[Parallel(n_jobs=94)]: Done 185862 tasks      | elapsed:   19.4s
    
     81%|████████▏ | 186965/229644 [00:19<00:04, 9060.53it/s][A
     82%|████████▏ | 187942/229644 [00:19<00:04, 8920.44it/s][A
     82%|████████▏ | 188885/229644 [00:19<00:04, 8901.95it/s][A
     83%|████████▎ | 189811/229644 [00:19<00:04, 8823.57it/s][A
     83%|████████▎ | 190726/229644 [00:19<00:04, 8759.84it/s][A
     83%|████████▎ | 191666/229644 [00:20<00:04, 8688.78it/s][A[Parallel(n_jobs=94)]: Done 192012 tasks      | elapsed:   20.1s
    
     84%|████████▍ | 193358/229644 [00:20<00:03, 10106.19it/s][A
     85%|████████▍ | 194465/229644 [00:20<00:03, 9842.26it/s] [A
     85%|████████▌ | 195518/229644 [00:20<00:03, 9582.29it/s][A
     86%|████████▌ | 196526/229644 [00:20<00:03, 9246.84it/s][A
     86%|████████▌ | 197488/229644 [00:20<00:03, 9180.42it/s][A
     86%|████████▋ | 198433/229644 [00:20<00:03, 9110.13it/s][A[Parallel(n_jobs=94)]: Done 198262 tasks      | elapsed:   20.7s
    
     87%|████████▋ | 199363/229644 [00:20<00:03, 8932.18it/s][A
     87%|████████▋ | 200270/229644 [00:20<00:03, 8790.86it/s][A
     88%|████████▊ | 201160/229644 [00:21<00:03, 8710.03it/s][A
     88%|████████▊ | 202100/229644 [00:21<00:03, 8723.63it/s][A
     88%|████████▊ | 203040/229644 [00:21<00:03, 8770.01it/s][A
     89%|████████▉ | 204262/229644 [00:21<00:02, 9536.24it/s][A[Parallel(n_jobs=94)]: Done 204612 tasks      | elapsed:   21.4s
    
     89%|████████▉ | 205238/229644 [00:21<00:02, 9451.27it/s][A
     90%|████████▉ | 206199/229644 [00:21<00:02, 9351.52it/s][A
     90%|█████████ | 207146/229644 [00:21<00:02, 9357.91it/s][A
     91%|█████████ | 208090/229644 [00:21<00:02, 9155.43it/s][A
     91%|█████████ | 209013/229644 [00:21<00:02, 9002.74it/s][A
     91%|█████████▏| 209919/229644 [00:21<00:02, 8788.28it/s][A
     92%|█████████▏| 210842/229644 [00:22<00:02, 8672.77it/s][A[Parallel(n_jobs=94)]: Done 211062 tasks      | elapsed:   22.1s
    
     93%|█████████▎| 212722/229644 [00:22<00:01, 10323.92it/s][A
     93%|█████████▎| 213889/229644 [00:22<00:01, 9925.47it/s] [A
     94%|█████████▎| 214980/229644 [00:22<00:01, 9547.31it/s][A
     94%|█████████▍| 216007/229644 [00:22<00:01, 9536.72it/s][A
     94%|█████████▍| 217011/229644 [00:22<00:01, 9259.76it/s][A[Parallel(n_jobs=94)]: Done 217612 tasks      | elapsed:   22.8s
    
     95%|█████████▍| 217974/229644 [00:22<00:01, 9364.73it/s][A
     95%|█████████▌| 218937/229644 [00:22<00:01, 9017.13it/s][A
     96%|█████████▌| 219860/229644 [00:22<00:01, 9074.95it/s][A
     96%|█████████▌| 220783/229644 [00:23<00:00, 9002.28it/s][A
     97%|█████████▋| 221694/229644 [00:23<00:00, 8868.29it/s][A
     97%|█████████▋| 222592/229644 [00:23<00:00, 8750.10it/s][A
     97%|█████████▋| 223532/229644 [00:23<00:00, 8711.55it/s][A[Parallel(n_jobs=94)]: Done 224262 tasks      | elapsed:   23.5s
    
     98%|█████████▊| 224472/229644 [00:23<00:00, 8736.37it/s][A
     98%|█████████▊| 225412/229644 [00:23<00:00, 8815.48it/s][A
     99%|█████████▉| 227480/229644 [00:23<00:00, 10627.41it/s][A
    100%|██████████| 229644/229644 [00:23<00:00, 9578.57it/s] [A
    [Parallel(n_jobs=94)]: Done 229644 out of 229644 | elapsed:   24.0s finished


          ...replicating address
             ...Base
             ...Map street suffixes...
             ...Mapped & split by street suffixes...
             ...Number processing...
    
             Address dataframe expansion is complete! (n=327310)
             ...Base
             ...Number processing...
             House number dataframe expansion is complete! (n=327310)
          ...formatting
       [Completed] Processing geo data
       [Start] Mapping geo data
          ...merge user input & lookup table
          ...mapping
       [Completed] Validating input geo data
    Directory already exists


    100%|██████████| 1/1 [09:25<00:00, 565.20s/it]

    Output saved
       [Completed] Mapping geo data


    


    
    [Completed] Preparing geo data
    
    [Start] Preparing ACS data
       [Start] Validating ACS input data
         Number of observations: 229644
         Is key unique: True
    
       [Completed] Validating ACS input data
    
       ...loading ACS lookup tables
       ... combining ACS & user input data
     ...Copy dataframes
     ...Block group
     ...Census tract
     ...Zip code
     ...No match
     ...Merge
     ...Merging complete
    [Complete] Preparing ACS data
    
       [Start] Validating pipeline input data
         Number of observations: 229644
         Is key unique: True
       [Completed] Validating pipeline input data
    


      0%|          | 0/17 [00:00<?, ?it/s][Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 60 concurrent workers.
    100%|██████████| 17/17 [00:00<00:00, 16820.75it/s]
    [Parallel(n_jobs=-1)]: Done   6 out of  17 | elapsed:    0.1s remaining:    0.2s
    [Parallel(n_jobs=-1)]: Done  17 out of  17 | elapsed:    0.1s finished
      0%|          | 0/1 [00:00<?, ?it/s][Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 60 concurrent workers.
    100%|██████████| 1/1 [00:00<00:00, 907.07it/s]
    [Parallel(n_jobs=-1)]: Done   1 out of   1 | elapsed:    0.0s finished
      0%|          | 0/7 [00:00<?, ?it/s][Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 60 concurrent workers.
    100%|██████████| 7/7 [00:00<00:00, 7115.88it/s]
    [Parallel(n_jobs=-1)]: Done   7 out of   7 | elapsed:    0.1s finished


    Directory already exists
    Output saved
    Output saved
    CPU times: user 30min 14s, sys: 4min 17s, total: 34min 31s
    Wall time: 31min 34s



```python

```

# Performance Analysis <a class="anchor" id="perform"></a>
In this section we will generate basic performance metrics for Zest Race Predictor proxies and BISG proxies. To do so we will use `ZRP_Performance` and support functions.

The ZRP output contains the original input data which will be used for future analysis in this notebook. The ground truth race & age, from the original data, will be used to illustrate the performance gains of using ZRP over BISG. 

## ZRP <a class="anchor" id="perform-zrp"></a>


```python
zrp_output = zrp_output.set_index('ZEST_KEY')
```


```python
zrp_output.index.values[0:1]
```




    array(['6633809822925307629'], dtype=object)



#### Invoke the Performance module on the Alabama sample data



```python
zrp_perf = ZRP_Performance()
zrp_perf.fit()
output_metrics = zrp_perf.transform(zrp_output, zrp_output)
```

    The key is already set
    The key is already set


By default multiple performance metrics are generated. Lets review the True Positive Rate, the metric labeled TPR. In this case the TPR conveys the percentage of correct approximated race in the true race/ethnicity class. For reference, ideally the TPR should be as close to 1 as possible 


```python
output_metrics.keys()
```




    dict_keys(['PPV', 'TPR', 'FPR', 'FNR', 'TNR', 'AUC'])



Looking at the TPR dictionary below we see that 74% of the time African Americans (labeled as 'BLACK') are correctly identified as African American, when using ZRP.


```python
output_metrics['TPR']
```




    {'AAPI': 0.665083135391924,
     'AIAN': 0.043478260869565216,
     'BLACK': 0.7388863237602733,
     'HISPANIC': 0.85519560984245,
     'WHITE': 0.9510281920731158,
     'nan': 'None'}



A majority of the time **ZRP is able to correctly identify people of the Asian American Pacific Islander, African American, Hispanic, and White race/ethnic groups** as their true reported race/ethnicity 

### BISG <a class="anchor" id="perform-bisg"></a>
BISG is a popular standard method for race proxying. BISG proxies are automatically generated by ZRP and are saved to the `/artifacts` folder. Lets load in the BISG proxies to generate performance metrics

Note: Since we are using BISG integrated into ZRP we will higher coverage than when calling `BISGWrapper`, a module for generating BISG proxies, on the original input data.


```python
bisg_output = pd.read_feather("artifacts/bisg_proxy_output.feather")
bisg_output = bisg_output.replace("None", np.nan)
bisg_output = bisg_output.set_index('ZEST_KEY')
bisg_output
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPI</th>
      <th>AIAN</th>
      <th>BLACK</th>
      <th>HISPANIC</th>
      <th>WHITE</th>
      <th>race_proxy</th>
      <th>source_bisg</th>
    </tr>
    <tr>
      <th>ZEST_KEY</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6633809822925307629</th>
      <td>0.000330</td>
      <td>0.004636</td>
      <td>0.333697</td>
      <td>0.002674</td>
      <td>0.645880</td>
      <td>WHITE</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3692884092241884245</th>
      <td>0.000313</td>
      <td>0.004092</td>
      <td>0.319857</td>
      <td>0.002841</td>
      <td>0.661163</td>
      <td>WHITE</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8193470747390094897</th>
      <td>0.000348</td>
      <td>0.004401</td>
      <td>0.279131</td>
      <td>0.002696</td>
      <td>0.701545</td>
      <td>WHITE</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1250429590175689533</th>
      <td>0.000292</td>
      <td>0.004075</td>
      <td>0.409306</td>
      <td>0.002641</td>
      <td>0.570575</td>
      <td>WHITE</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15516224258645594574</th>
      <td>0.002026</td>
      <td>0.003685</td>
      <td>0.355842</td>
      <td>0.006504</td>
      <td>0.619576</td>
      <td>WHITE</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7642004289156802863</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15595779201190192886</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4338437641988397076</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4674780794524469353</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13567699867855876794</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>229644 rows × 7 columns</p>
</div>



To ensure the indices are in the same order for performance metric calculation, lets re-index both BISG & ZRP output. 


```python
zrp_idx = list(zrp_output.sort_index().index)
bisg_output =  bisg_output.reindex(zrp_idx)
zrp_output =  zrp_output.reindex(zrp_idx)
```

#### Invoke the Performance module on the Alabama sample data



```python
bisg_perf = ZRP_Performance()
bisg_perf.fit()
bisg_output_metrics = bisg_perf.transform(bisg_output, zrp_output)
```

    The key is already set
    The key is already set



```python
bisg_output_metrics.keys()
```




    dict_keys(['PPV', 'TPR', 'FPR', 'FNR', 'TNR', 'AUC'])



Lets look at the TPR dictionary below. When using BISG we see a TPR of 57% for African Americans. This TPR is low especially when compared to the TPR of 74% we saw using ZRP.


```python
bisg_output_metrics['TPR']
```




    {'AAPI': 0.5312747426761678,
     'AIAN': 0.04,
     'BLACK': 0.5697848370117278,
     'HISPANIC': 0.5022127810231899,
     'None': 'None',
     'WHITE': 0.8468476572573149}



**BISG falls short when proxying race/ethnicity of minority groups** exhibited by low TPRs across  minority race/ethnic groups. In this case, BISG is only able to correctly proxy people who identify as White with a good level of confidence.

## Comparing Methods <a class="anchor" id="perform-compare"></a>
### Zest Race Predictor & BISG 
To compare the ZRP and BISG we will generate performance tables of metrics by race and method. These tables will utilize the performance dictionaries generated by `ZRP_Performance` and percent difference calculations. 

Across the board, with significant class sizes, we can see ZRP outperform BISG. Here are a few highlights of using ZRP over BISG.
- 25% more Asian American Pacific Islanders are correctly identified
- 30% more African Americans are correctly identified
- 39% fewer African Americans identified as non-African American
- 68% fewer White Americans identified as non-White American



```python
comparision_dict = gen_metrics_tables(output_metrics, bisg_output_metrics)
comparision_dict.keys()
```




    dict_keys(['TPR', 'FPR', 'FNR', 'TNR', 'AUC'])



##### True Positive Rate
The percentage of people correctly labeled as the race/ethnicity they identify as.


```python
comparision_dict["TPR"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TPR</td>
      <td>AAPI</td>
      <td>0.665083</td>
      <td>0.531275</td>
      <td>25.19%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TPR</td>
      <td>AIAN</td>
      <td>0.043478</td>
      <td>0.040000</td>
      <td>8.70%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TPR</td>
      <td>BLACK</td>
      <td>0.738886</td>
      <td>0.569785</td>
      <td>29.68%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TPR</td>
      <td>HISPANIC</td>
      <td>0.855196</td>
      <td>0.502213</td>
      <td>70.29%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TPR</td>
      <td>WHITE</td>
      <td>0.951028</td>
      <td>0.846848</td>
      <td>12.30%</td>
    </tr>
  </tbody>
</table>
</div>



ZRP significantly improves the ability to proxy people who identify as Hispanic. **There is a 70% increase in correct identifications of Hispanic people, when using ZRP over BISG**.

.

##### False Positive Rate
The percentage of people incorrectly labeled as a race/ethnicity they do identify as.


```python
comparision_dict["FPR"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FPR</td>
      <td>AAPI</td>
      <td>0.003271</td>
      <td>0.001202</td>
      <td>172.16%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FPR</td>
      <td>AIAN</td>
      <td>0.000978</td>
      <td>0.000284</td>
      <td>244.62%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FPR</td>
      <td>BLACK</td>
      <td>0.032091</td>
      <td>0.092605</td>
      <td>-65.35%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FPR</td>
      <td>HISPANIC</td>
      <td>0.012558</td>
      <td>0.009375</td>
      <td>33.95%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FPR</td>
      <td>WHITE</td>
      <td>0.237284</td>
      <td>0.365959</td>
      <td>-35.16%</td>
    </tr>
  </tbody>
</table>
</div>



**37% of people who are non-White are identified as White by BISG**, which is 35% higher than when using ZRP.

.

##### False Negative Rate
The percentage of people who identify as a certain race/ethnicity but as incorrectly labeled a different race/ethnicity.


```python
comparision_dict["FNR"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FNR</td>
      <td>AAPI</td>
      <td>0.334917</td>
      <td>0.468725</td>
      <td>-28.55%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FNR</td>
      <td>AIAN</td>
      <td>0.956522</td>
      <td>0.960000</td>
      <td>-0.36%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FNR</td>
      <td>BLACK</td>
      <td>0.261114</td>
      <td>0.430215</td>
      <td>-39.31%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FNR</td>
      <td>HISPANIC</td>
      <td>0.144804</td>
      <td>0.497787</td>
      <td>-70.91%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FNR</td>
      <td>WHITE</td>
      <td>0.048972</td>
      <td>0.153152</td>
      <td>-68.02%</td>
    </tr>
  </tbody>
</table>
</div>



ZRP mislabeled Hispanic people 14% of the time whereas BISG mislabled Hispanic people 50% of the time as non-Hispanic. **Notice the 71% decrease in mislabeling Hispanic people as non-Hispanic when using ZRP** over BISG

.

##### True Negative Rate
The percentage of people correctly labeled as not belonging to a race/ethnicity they do not identify as.


```python
comparision_dict["TNR"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TNR</td>
      <td>AAPI</td>
      <td>0.996729</td>
      <td>0.998798</td>
      <td>-0.21%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TNR</td>
      <td>AIAN</td>
      <td>0.999022</td>
      <td>0.999716</td>
      <td>-0.07%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TNR</td>
      <td>BLACK</td>
      <td>0.967909</td>
      <td>0.907395</td>
      <td>6.67%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TNR</td>
      <td>HISPANIC</td>
      <td>0.987442</td>
      <td>0.990625</td>
      <td>-0.32%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TNR</td>
      <td>WHITE</td>
      <td>0.762716</td>
      <td>0.634041</td>
      <td>20.29%</td>
    </tr>
  </tbody>
</table>
</div>



20% more people who are not White are not labeled as White with ZRP

.

##### AUC
A standard method applied by scholars to be a single numeric measure for evaluating the predictive ability of learning algorithms.


```python
comparision_dict["AUC"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AUC</td>
      <td>AAPI</td>
      <td>0.830906</td>
      <td>0.765036</td>
      <td>8.61%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AUC</td>
      <td>AIAN</td>
      <td>0.521250</td>
      <td>0.519858</td>
      <td>0.27%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AUC</td>
      <td>BLACK</td>
      <td>0.853397</td>
      <td>0.738590</td>
      <td>15.54%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AUC</td>
      <td>HISPANIC</td>
      <td>0.921319</td>
      <td>0.746419</td>
      <td>23.43%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AUC</td>
      <td>WHITE</td>
      <td>0.856872</td>
      <td>0.740444</td>
      <td>15.72%</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

## Alternative Measurements of Performance <a class="anchor" id="alt"></a>
In the previous section we review pretty standard performance metrics and illustrated the significant performance gains ZRP has over BISG. In this section, we want to drive home how accurate the Zest Race Predictor is without using standard measures. We will illustrate how close ZRP is to the ground truth by incorporating age into our performance analysis.

Lets first add reported race and age, from original data, to the BISG proxy output data.


```python
bisg_output = bisg_output.merge(zrp_output.filter(regex="age|sex|race$"), left_index=True, right_index=True)
bisg_output.shape
```




    (229644, 12)



### Age Analysis <a class="anchor" id="alt-age"></a>
#### Invoke the Alternative Performance module on the Alabama sample data


```python
zac = ZRP_Age_Comparision("age", "race_proxy", "race")
age_metrics = zac.transform(zrp_data = zrp_output, bisg_data = bisg_output)
```


```python

```

Here the alternative performance analysis is an age comparison. We will generate age metrics by self reported race and proxied race. The goal here is to see which method, ZRP or BISG, generates age metrics closest to the true age. There are 4 main component to the output as seen below. We will focus on the Comparison component, as it includes all of the three prior components.


```python
age_metrics.keys()
```




    dict_keys(['Self-Reported', 'ZRP', 'BISG', 'Comparison'])



When comparing true average age by race to each proxy method. We see that ZRP proxies race age averages are closest to the true race age averages, except for American Indian/Alaskan Native.


```python
age_metrics['Comparison']['Age Averages']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>True Age</th>
      <th>ZRP Age</th>
      <th>BISG Age</th>
      <th>ZRP_Pct_Diff</th>
      <th>BISG_Pct_Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AAPI</th>
      <td>40.11</td>
      <td>39.86</td>
      <td>39.44</td>
      <td>-0.62%</td>
      <td>-1.67%</td>
    </tr>
    <tr>
      <th>AIAN</th>
      <td>37.02</td>
      <td>34.28</td>
      <td>37.86</td>
      <td>-7.4%</td>
      <td>2.27%</td>
    </tr>
    <tr>
      <th>BLACK</th>
      <td>35.71</td>
      <td>34.75</td>
      <td>36.99</td>
      <td>-2.69%</td>
      <td>3.58%</td>
    </tr>
    <tr>
      <th>HISPANIC</th>
      <td>32.17</td>
      <td>33.52</td>
      <td>34.10</td>
      <td>4.2%</td>
      <td>6.0%</td>
    </tr>
    <tr>
      <th>WHITE</th>
      <td>40.52</td>
      <td>40.59</td>
      <td>39.99</td>
      <td>0.17%</td>
      <td>-1.31%</td>
    </tr>
    <tr>
      <th>NaN</th>
      <td>NaN</td>
      <td>24.16</td>
      <td>38.51</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



For the output below we bin age in each race category. Here we are comparing the distribution of age by race between the ground truth (self-reported race) and the proxy methods. The column values are percents.


```python
age_metrics['Comparison']['Age Group']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Ground Truth</th>
      <th>ZRP</th>
      <th>BISG</th>
    </tr>
    <tr>
      <th>race</th>
      <th>age group</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9" valign="top">AAPI</th>
      <th>18-20</th>
      <td>11.60%</td>
      <td>11.02%</td>
      <td>10.40%</td>
    </tr>
    <tr>
      <th>20-30</th>
      <td>21.93%</td>
      <td>23.44%</td>
      <td>23.96%</td>
    </tr>
    <tr>
      <th>30-40</th>
      <td>19.56%</td>
      <td>19.19%</td>
      <td>19.94%</td>
    </tr>
    <tr>
      <th>40-50</th>
      <td>16.71%</td>
      <td>17.25%</td>
      <td>17.77%</td>
    </tr>
    <tr>
      <th>50-60</th>
      <td>14.92%</td>
      <td>14.44%</td>
      <td>15.17%</td>
    </tr>
    <tr>
      <th>60-70</th>
      <td>10.29%</td>
      <td>9.33%</td>
      <td>8.61%</td>
    </tr>
    <tr>
      <th>70-80</th>
      <td>3.96%</td>
      <td>4.21%</td>
      <td>3.10%</td>
    </tr>
    <tr>
      <th>80-90</th>
      <td>0.91%</td>
      <td>0.99%</td>
      <td>0.99%</td>
    </tr>
    <tr>
      <th>90+</th>
      <td>0.12%</td>
      <td>0.12%</td>
      <td>0.06%</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">AIAN</th>
      <th>18-20</th>
      <td>20.00%</td>
      <td>15.66%</td>
      <td>6.82%</td>
    </tr>
    <tr>
      <th>20-30</th>
      <td>26.09%</td>
      <td>33.73%</td>
      <td>30.68%</td>
    </tr>
    <tr>
      <th>30-40</th>
      <td>12.87%</td>
      <td>18.07%</td>
      <td>19.32%</td>
    </tr>
    <tr>
      <th>40-50</th>
      <td>14.78%</td>
      <td>13.25%</td>
      <td>18.18%</td>
    </tr>
    <tr>
      <th>50-60</th>
      <td>12.70%</td>
      <td>10.04%</td>
      <td>13.64%</td>
    </tr>
    <tr>
      <th>60-70</th>
      <td>9.39%</td>
      <td>5.22%</td>
      <td>10.23%</td>
    </tr>
    <tr>
      <th>70-80</th>
      <td>3.48%</td>
      <td>4.02%</td>
      <td>1.14%</td>
    </tr>
    <tr>
      <th>80-90</th>
      <td>0.70%</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">BLACK</th>
      <th>18-20</th>
      <td>16.52%</td>
      <td>17.11%</td>
      <td>15.21%</td>
    </tr>
    <tr>
      <th>20-30</th>
      <td>29.52%</td>
      <td>31.58%</td>
      <td>29.02%</td>
    </tr>
    <tr>
      <th>30-40</th>
      <td>18.16%</td>
      <td>18.31%</td>
      <td>17.64%</td>
    </tr>
    <tr>
      <th>40-50</th>
      <td>14.34%</td>
      <td>13.98%</td>
      <td>13.52%</td>
    </tr>
    <tr>
      <th>50-60</th>
      <td>10.83%</td>
      <td>9.55%</td>
      <td>11.37%</td>
    </tr>
    <tr>
      <th>60-70</th>
      <td>7.43%</td>
      <td>6.33%</td>
      <td>8.76%</td>
    </tr>
    <tr>
      <th>70-80</th>
      <td>2.45%</td>
      <td>2.35%</td>
      <td>3.37%</td>
    </tr>
    <tr>
      <th>80-90</th>
      <td>0.63%</td>
      <td>0.68%</td>
      <td>0.97%</td>
    </tr>
    <tr>
      <th>90+</th>
      <td>0.12%</td>
      <td>0.11%</td>
      <td>0.15%</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">HISPANIC</th>
      <th>18-20</th>
      <td>19.99%</td>
      <td>17.61%</td>
      <td>15.90%</td>
    </tr>
    <tr>
      <th>20-30</th>
      <td>36.34%</td>
      <td>34.77%</td>
      <td>34.56%</td>
    </tr>
    <tr>
      <th>30-40</th>
      <td>17.45%</td>
      <td>18.38%</td>
      <td>19.00%</td>
    </tr>
    <tr>
      <th>40-50</th>
      <td>11.56%</td>
      <td>12.07%</td>
      <td>12.54%</td>
    </tr>
    <tr>
      <th>50-60</th>
      <td>7.70%</td>
      <td>9.01%</td>
      <td>9.38%</td>
    </tr>
    <tr>
      <th>60-70</th>
      <td>4.69%</td>
      <td>5.40%</td>
      <td>6.00%</td>
    </tr>
    <tr>
      <th>70-80</th>
      <td>1.63%</td>
      <td>1.95%</td>
      <td>1.96%</td>
    </tr>
    <tr>
      <th>80-90</th>
      <td>0.53%</td>
      <td>0.67%</td>
      <td>0.53%</td>
    </tr>
    <tr>
      <th>90+</th>
      <td>0.11%</td>
      <td>0.13%</td>
      <td>0.14%</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">WHITE</th>
      <th>18-20</th>
      <td>10.32%</td>
      <td>10.45%</td>
      <td>10.95%</td>
    </tr>
    <tr>
      <th>20-30</th>
      <td>25.73%</td>
      <td>25.28%</td>
      <td>25.97%</td>
    </tr>
    <tr>
      <th>30-40</th>
      <td>18.33%</td>
      <td>18.25%</td>
      <td>18.44%</td>
    </tr>
    <tr>
      <th>40-50</th>
      <td>13.89%</td>
      <td>14.02%</td>
      <td>14.17%</td>
    </tr>
    <tr>
      <th>50-60</th>
      <td>13.52%</td>
      <td>13.76%</td>
      <td>13.24%</td>
    </tr>
    <tr>
      <th>60-70</th>
      <td>11.03%</td>
      <td>11.20%</td>
      <td>10.53%</td>
    </tr>
    <tr>
      <th>70-80</th>
      <td>5.46%</td>
      <td>5.36%</td>
      <td>5.11%</td>
    </tr>
    <tr>
      <th>80-90</th>
      <td>1.53%</td>
      <td>1.48%</td>
      <td>1.40%</td>
    </tr>
    <tr>
      <th>90+</th>
      <td>0.20%</td>
      <td>0.20%</td>
      <td>0.19%</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">NaN</th>
      <th>18-20</th>
      <td>NaN</td>
      <td>57.14%</td>
      <td>13.60%</td>
    </tr>
    <tr>
      <th>20-30</th>
      <td>NaN</td>
      <td>26.92%</td>
      <td>27.34%</td>
    </tr>
    <tr>
      <th>30-40</th>
      <td>NaN</td>
      <td>4.95%</td>
      <td>17.83%</td>
    </tr>
    <tr>
      <th>40-50</th>
      <td>NaN</td>
      <td>4.95%</td>
      <td>13.09%</td>
    </tr>
    <tr>
      <th>60-70</th>
      <td>NaN</td>
      <td>2.75%</td>
      <td>9.71%</td>
    </tr>
    <tr>
      <th>50-60</th>
      <td>NaN</td>
      <td>1.65%</td>
      <td>12.67%</td>
    </tr>
    <tr>
      <th>70-80</th>
      <td>NaN</td>
      <td>1.65%</td>
      <td>4.32%</td>
    </tr>
    <tr>
      <th>80-90</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.29%</td>
    </tr>
    <tr>
      <th>90+</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.16%</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### Geographic level Analysis <a class="anchor" id="alt-geo"></a>
#### County vs City
In this section, we will compare ZRP & BISG across different geographic areas.

**County**: Cherokee County 
- A northeastern county in Alabama with estimated population of around 25,000 people

**Densely populated City**: Birmingham
- A city in north central Alabama with estimated population of around 212,000 people


#### County


```python
chk_idx = list(zrp_output[(zrp_output.zip_code.isin(["35959", "35960", "35973", "35983",
                                                     35959, 35960, 35973, 35983]))].index.values)

cherokee = zrp_output[(zrp_output.index.isin(chk_idx))]
cherokee_bisg = bisg_output[(bisg_output.index.isin(chk_idx))]
```

Check reported race distribution


```python
cherokee.race.value_counts(dropna=False, normalize=True)
```




    WHITE       0.955128
    BLACK       0.029202
    HISPANIC    0.008547
    AAPI        0.007123
    Name: race, dtype: float64



#### City


```python
birmingham_idx = list(zrp_output[(zrp_output.city.str.upper().str.contains("BIRMINGHAM"))].index.values)
birmingham = zrp_output[(zrp_output.index.isin(birmingham_idx))]
birmingham_bisg = bisg_output[(bisg_output.index.isin(birmingham_idx))]
```

Check reported race distribution


```python
birmingham.race.value_counts(dropna=False, normalize=True)
```




    WHITE       0.618126
    BLACK       0.342214
    HISPANIC    0.020520
    AAPI        0.017872
    AIAN        0.001269
    Name: race, dtype: float64




```python

```

Initialize `ZRP_Performance`


```python
area_perf = ZRP_Performance()
area_perf.fit()
```




    <zrp.modeling.performance.ZRP_Performance at 0x7efb4d21cc90>



Transform ZRP Outputsm


```python
chk_output_metrics = area_perf.transform(cherokee, cherokee)
birmingham_output_metrics = area_perf.transform(birmingham, birmingham)
```

    The key is already set
    The key is already set
    The key is already set
    The key is already set


Transform BISG Outputs


```python
chk_bisg_output_metrics = area_perf.transform(cherokee_bisg, cherokee_bisg)
bmghm_bisg_output_metrics = area_perf.transform(birmingham_bisg, birmingham_bisg)
```

    The key is already set
    The key is already set
    The key is already set
    The key is already set


Generate comparison tables


```python
chk_comparision_dict = gen_metrics_tables(chk_output_metrics, chk_bisg_output_metrics)
birmingham_comparision_dict = gen_metrics_tables(birmingham_output_metrics, bmghm_bisg_output_metrics)
```

#### Correctly identifying people with their self-reported race/ethnicity

##### County
In Cherokee County, around 96% of voters are self-reported as White. Below we can see that both ZRP & BISG do very well in correctly predicting if a person is White. 

BISG falls short in identifying minorities. 
- ZRP is able to correctly identify 1600% more African Americans than BISG
- ZRP is able to correctly identify 700% more Hispanic Americans than BISG 




```python
chk_comparision_dict['TPR']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TPR</td>
      <td>BLACK</td>
      <td>0.414634</td>
      <td>0.024390</td>
      <td>1600.00%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TPR</td>
      <td>WHITE</td>
      <td>0.989560</td>
      <td>0.972409</td>
      <td>1.76%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TPR</td>
      <td>HISPANIC</td>
      <td>0.666667</td>
      <td>0.083333</td>
      <td>700.00%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TPR</td>
      <td>AAPI</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>N/A</td>
    </tr>
  </tbody>
</table>
</div>



##### City
In the city of Birmingham, ZRP does very well correctly predicting if a person is Asian American Pacific Islander, African American, Hispanic American, and White American. 

There is much to highlight from the table below
- 17% more Asian American Pacific Islanders are correctly identified
- 18% more African Americans are correctly identified
- 76% more Hispanic Americans are correctly identified
- 18% more White Americans are correctly identified



```python
birmingham_comparision_dict['TPR']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TPR</td>
      <td>AAPI</td>
      <td>0.805556</td>
      <td>0.691358</td>
      <td>16.52%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TPR</td>
      <td>HISPANIC</td>
      <td>0.862903</td>
      <td>0.489247</td>
      <td>76.37%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TPR</td>
      <td>BLACK</td>
      <td>0.821406</td>
      <td>0.693424</td>
      <td>18.46%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TPR</td>
      <td>AIAN</td>
      <td>0.043478</td>
      <td>0.043478</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TPR</td>
      <td>WHITE</td>
      <td>0.941103</td>
      <td>0.798412</td>
      <td>17.87%</td>
    </tr>
  </tbody>
</table>
</div>



#### Avoiding mislabeling race/ethnicity


##### County
In Cherokee County, both ZRP & BISG do very well in not labeling people as Asian American Pacific Islander, African American, or Hispanic American they did not report to be Asian American Pacific Islander, African American, or Hispanic American.

ZRP Highlights
- 271% fewer White Americans identified as non-White
- BISG struggles in this geo-location to properly identify people identify as a racial minority 




```python
chk_comparision_dict['TNR']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TNR</td>
      <td>BLACK</td>
      <td>0.992663</td>
      <td>0.997799</td>
      <td>-0.51%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TNR</td>
      <td>WHITE</td>
      <td>0.412698</td>
      <td>0.111111</td>
      <td>271.43%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TNR</td>
      <td>HISPANIC</td>
      <td>0.997845</td>
      <td>1.000000</td>
      <td>-0.22%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TNR</td>
      <td>AAPI</td>
      <td>0.999283</td>
      <td>1.000000</td>
      <td>-0.07%</td>
    </tr>
  </tbody>
</table>
</div>



##### City
In the city of Birmingham, both ZRP & BISG do very well in not labeling people as a race/ethnicity they did not report.

ZRP Highlights
- 9% fewer African Americans identified as non-African American
- 12% fewer White Americans are identified as non-White 



```python
birmingham_comparision_dict['TNR']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Race</th>
      <th>ZRP</th>
      <th>BISG</th>
      <th>Percent Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TNR</td>
      <td>AAPI</td>
      <td>0.994945</td>
      <td>0.997360</td>
      <td>-0.24%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TNR</td>
      <td>HISPANIC</td>
      <td>0.987611</td>
      <td>0.992792</td>
      <td>-0.52%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TNR</td>
      <td>BLACK</td>
      <td>0.961090</td>
      <td>0.877400</td>
      <td>9.54%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TNR</td>
      <td>AIAN</td>
      <td>0.999392</td>
      <td>1.000000</td>
      <td>-0.06%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TNR</td>
      <td>WHITE</td>
      <td>0.838365</td>
      <td>0.748664</td>
      <td>11.98%</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# Appendix <a class="anchor" id="appendix"></a>
This section records full outputs that may not have been reviewed above.


```python
output_metrics
```




    {'PPV': {'AAPI': 0.6933553446141147,
      'AIAN': 0.10040160642570281,
      'BLACK': 0.8765967703060978,
      'HISPANIC': 0.6319989534275249,
      'WHITE': 0.9139861792319618,
      'nan': 0.0},
     'TPR': {'AAPI': 0.665083135391924,
      'AIAN': 0.043478260869565216,
      'BLACK': 0.7388863237602733,
      'HISPANIC': 0.85519560984245,
      'WHITE': 0.9510281920731158,
      'nan': 'None'},
     'FPR': {'AAPI': 0.0032714271876292944,
      'AIAN': 0.000977871296421573,
      'BLACK': 0.032091350947868635,
      'HISPANIC': 0.012558316033840056,
      'WHITE': 0.23728436282693377,
      'nan': 0.000792531048056988},
     'FNR': {'AAPI': 0.334916864608076,
      'AIAN': 0.9565217391304348,
      'BLACK': 0.2611136762397267,
      'HISPANIC': 0.14480439015754998,
      'WHITE': 0.04897180792688416,
      'nan': 'None'},
     'TNR': {'AAPI': 0.9967285728123707,
      'AIAN': 0.9990221287035784,
      'BLACK': 0.9679086490521314,
      'HISPANIC': 0.9874416839661599,
      'WHITE': 0.7627156371730662,
      'nan': 0.999207468951943},
     'AUC': {'AAPI': 0.8309058541021473,
      'AIAN': 0.5212501947865719,
      'BLACK': 0.8533974864062024,
      'HISPANIC': 0.921318646904305,
      'WHITE': 0.856871914623091,
      'nan': 'None'}}




```python
bisg_output_metrics
```




    {'PPV': {'AAPI': 0.8309597523219814,
      'AIAN': 0.26136363636363635,
      'BLACK': 0.6549688979470522,
      'HISPANIC': 0.5746404699210047,
      'None': 0.0,
      'WHITE': 0.8598472854811603},
     'TPR': {'AAPI': 0.5312747426761678,
      'AIAN': 0.04,
      'BLACK': 0.5697848370117278,
      'HISPANIC': 0.5022127810231899,
      'None': 'None',
      'WHITE': 0.8468476572573149},
     'FPR': {'AAPI': 0.0012020183340818447,
      'AIAN': 0.00028375729583662856,
      'BLACK': 0.09260451626504995,
      'HISPANIC': 0.009375209268064011,
      'None': 0.05083085123060038,
      'WHITE': 0.36595913824628346},
     'FNR': {'AAPI': 0.4687252573238322,
      'AIAN': 0.96,
      'BLACK': 0.4302151629882722,
      'HISPANIC': 0.4977872189768101,
      'None': 'None',
      'WHITE': 0.15315234274268508},
     'TNR': {'AAPI': 0.9987979816659182,
      'AIAN': 0.9997162427041634,
      'BLACK': 0.90739548373495,
      'HISPANIC': 0.990624790731936,
      'None': 0.9491691487693996,
      'WHITE': 0.6340408617537165},
     'AUC': {'AAPI': 0.765036362171043,
      'AIAN': 0.5198581213520816,
      'BLACK': 0.7385901603733389,
      'HISPANIC': 0.7464187858775629,
      'None': 'None',
      'WHITE': 0.7404442595055157}}




```python
comparision_dict
```




    {'TPR':   Metric      Race       ZRP      BISG Percent Difference
     0    TPR      AAPI  0.665083  0.531275             25.19%
     1    TPR      AIAN  0.043478  0.040000              8.70%
     2    TPR     BLACK  0.738886  0.569785             29.68%
     3    TPR  HISPANIC  0.855196  0.502213             70.29%
     4    TPR     WHITE  0.951028  0.846848             12.30%,
     'FPR':   Metric      Race       ZRP      BISG Percent Difference
     0    FPR      AAPI  0.003271  0.001202            172.16%
     1    FPR      AIAN  0.000978  0.000284            244.62%
     2    FPR     BLACK  0.032091  0.092605            -65.35%
     3    FPR  HISPANIC  0.012558  0.009375             33.95%
     4    FPR     WHITE  0.237284  0.365959            -35.16%,
     'FNR':   Metric      Race       ZRP      BISG Percent Difference
     0    FNR      AAPI  0.334917  0.468725            -28.55%
     1    FNR      AIAN  0.956522  0.960000             -0.36%
     2    FNR     BLACK  0.261114  0.430215            -39.31%
     3    FNR  HISPANIC  0.144804  0.497787            -70.91%
     4    FNR     WHITE  0.048972  0.153152            -68.02%,
     'TNR':   Metric      Race       ZRP      BISG Percent Difference
     0    TNR      AAPI  0.996729  0.998798             -0.21%
     1    TNR      AIAN  0.999022  0.999716             -0.07%
     2    TNR     BLACK  0.967909  0.907395              6.67%
     3    TNR  HISPANIC  0.987442  0.990625             -0.32%
     4    TNR     WHITE  0.762716  0.634041             20.29%,
     'AUC':   Metric      Race       ZRP      BISG Percent Difference
     0    AUC      AAPI  0.830906  0.765036              8.61%
     1    AUC      AIAN  0.521250  0.519858              0.27%
     2    AUC     BLACK  0.853397  0.738590             15.54%
     3    AUC  HISPANIC  0.921319  0.746419             23.43%
     4    AUC     WHITE  0.856872  0.740444             15.72%}




```python
age_metrics
```




    {'Self-Reported': {'Age Group': race      age group
      AAPI      20-30        21.93%
                30-40        19.56%
                40-50        16.71%
                50-60        14.92%
                18-20        11.60%
                60-70        10.29%
                70-80         3.96%
                80-90         0.91%
                90+           0.12%
      AIAN      20-30        26.09%
                18-20        20.00%
                40-50        14.78%
                30-40        12.87%
                50-60        12.70%
                60-70         9.39%
                70-80         3.48%
                80-90         0.70%
      BLACK     20-30        29.52%
                30-40        18.16%
                18-20        16.52%
                40-50        14.34%
                50-60        10.83%
                60-70         7.43%
                70-80         2.45%
                80-90         0.63%
                90+           0.12%
      HISPANIC  20-30        36.34%
                18-20        19.99%
                30-40        17.45%
                40-50        11.56%
                50-60         7.70%
                60-70         4.69%
                70-80         1.63%
                80-90         0.53%
                90+           0.11%
      WHITE     20-30        25.73%
                30-40        18.33%
                40-50        13.89%
                50-60        13.52%
                60-70        11.03%
                18-20        10.32%
                70-80         5.46%
                80-90         1.53%
                90+           0.20%
      dtype: object,
      'Age Averages': race
      AAPI        40.11
      AIAN        37.02
      BLACK       35.71
      HISPANIC    32.17
      WHITE       40.52
      Name: age_num, dtype: float64},
     'ZRP': {'Age Group': race      age group
      AAPI      20-30        23.44%
                30-40        19.19%
                40-50        17.25%
                50-60        14.44%
                18-20        11.02%
                60-70         9.33%
                70-80         4.21%
                80-90         0.99%
                90+           0.12%
      AIAN      20-30        33.73%
                30-40        18.07%
                18-20        15.66%
                40-50        13.25%
                50-60        10.04%
                60-70         5.22%
                70-80         4.02%
      BLACK     20-30        31.58%
                30-40        18.31%
                18-20        17.11%
                40-50        13.98%
                50-60         9.55%
                60-70         6.33%
                70-80         2.35%
                80-90         0.68%
                90+           0.11%
      HISPANIC  20-30        34.77%
                30-40        18.38%
                18-20        17.61%
                40-50        12.07%
                50-60         9.01%
                60-70         5.40%
                70-80         1.95%
                80-90         0.67%
                90+           0.13%
      WHITE     20-30        25.28%
                30-40        18.25%
                40-50        14.02%
                50-60        13.76%
                60-70        11.20%
                18-20        10.45%
                70-80         5.36%
                80-90         1.48%
                90+           0.20%
      NaN       18-20        57.14%
                20-30        26.92%
                30-40         4.95%
                40-50         4.95%
                60-70         2.75%
                50-60         1.65%
                70-80         1.65%
      dtype: object,
      'Age Averages': race_proxy
      AAPI        39.86
      AIAN        34.28
      BLACK       34.75
      HISPANIC    33.52
      WHITE       40.59
      NaN         24.16
      Name: age_num, dtype: float64},
     'BISG': {'Age Group': race      age group
      AAPI      20-30        23.96%
                30-40        19.94%
                40-50        17.77%
                50-60        15.17%
                18-20        10.40%
                60-70         8.61%
                70-80         3.10%
                80-90         0.99%
                90+           0.06%
      AIAN      20-30        30.68%
                30-40        19.32%
                40-50        18.18%
                50-60        13.64%
                60-70        10.23%
                18-20         6.82%
                70-80         1.14%
      BLACK     20-30        29.02%
                30-40        17.64%
                18-20        15.21%
                40-50        13.52%
                50-60        11.37%
                60-70         8.76%
                70-80         3.37%
                80-90         0.97%
                90+           0.15%
      HISPANIC  20-30        34.56%
                30-40        19.00%
                18-20        15.90%
                40-50        12.54%
                50-60         9.38%
                60-70         6.00%
                70-80         1.96%
                80-90         0.53%
                90+           0.14%
      WHITE     20-30        25.97%
                30-40        18.44%
                40-50        14.17%
                50-60        13.24%
                18-20        10.95%
                60-70        10.53%
                70-80         5.11%
                80-90         1.40%
                90+           0.19%
      NaN       20-30        27.34%
                30-40        17.83%
                18-20        13.60%
                40-50        13.09%
                50-60        12.67%
                60-70         9.71%
                70-80         4.32%
                80-90         1.29%
                90+           0.16%
      dtype: object,
      'Age Averages': race_proxy
      AAPI        39.44
      AIAN        37.86
      BLACK       36.99
      HISPANIC    34.10
      WHITE       39.99
      NaN         38.51
      Name: age_num, dtype: float64},
     'Comparison': {'Age Aveages':           True Age  ZRP Age  BISG Age ZRP_Pct_Diff BISG_Pct_Diff
      AAPI         40.11    39.86     39.44       -0.62%        -1.67%
      AIAN         37.02    34.28     37.86        -7.4%         2.27%
      BLACK        35.71    34.75     36.99       -2.69%         3.58%
      HISPANIC     32.17    33.52     34.10         4.2%          6.0%
      WHITE        40.52    40.59     39.99        0.17%        -1.31%
      NaN            NaN    24.16     38.51         None          None,
      'Age Group':                    Ground Truth     ZRP    BISG
      race     age group                             
      AAPI     18-20           11.60%  11.02%  10.40%
               20-30           21.93%  23.44%  23.96%
               30-40           19.56%  19.19%  19.94%
               40-50           16.71%  17.25%  17.77%
               50-60           14.92%  14.44%  15.17%
               60-70           10.29%   9.33%   8.61%
               70-80            3.96%   4.21%   3.10%
               80-90            0.91%   0.99%   0.99%
               90+              0.12%   0.12%   0.06%
      AIAN     18-20           20.00%  15.66%   6.82%
               20-30           26.09%  33.73%  30.68%
               30-40           12.87%  18.07%  19.32%
               40-50           14.78%  13.25%  18.18%
               50-60           12.70%  10.04%  13.64%
               60-70            9.39%   5.22%  10.23%
               70-80            3.48%   4.02%   1.14%
               80-90            0.70%     NaN     NaN
      BLACK    18-20           16.52%  17.11%  15.21%
               20-30           29.52%  31.58%  29.02%
               30-40           18.16%  18.31%  17.64%
               40-50           14.34%  13.98%  13.52%
               50-60           10.83%   9.55%  11.37%
               60-70            7.43%   6.33%   8.76%
               70-80            2.45%   2.35%   3.37%
               80-90            0.63%   0.68%   0.97%
               90+              0.12%   0.11%   0.15%
      HISPANIC 18-20           19.99%  17.61%  15.90%
               20-30           36.34%  34.77%  34.56%
               30-40           17.45%  18.38%  19.00%
               40-50           11.56%  12.07%  12.54%
               50-60            7.70%   9.01%   9.38%
               60-70            4.69%   5.40%   6.00%
               70-80            1.63%   1.95%   1.96%
               80-90            0.53%   0.67%   0.53%
               90+              0.11%   0.13%   0.14%
      WHITE    18-20           10.32%  10.45%  10.95%
               20-30           25.73%  25.28%  25.97%
               30-40           18.33%  18.25%  18.44%
               40-50           13.89%  14.02%  14.17%
               50-60           13.52%  13.76%  13.24%
               60-70           11.03%  11.20%  10.53%
               70-80            5.46%   5.36%   5.11%
               80-90            1.53%   1.48%   1.40%
               90+              0.20%   0.20%   0.19%
      NaN      18-20              NaN  57.14%  13.60%
               20-30              NaN  26.92%  27.34%
               30-40              NaN   4.95%  17.83%
               40-50              NaN   4.95%  13.09%
               60-70              NaN   2.75%   9.71%
               50-60              NaN   1.65%  12.67%
               70-80              NaN   1.65%   4.32%
               80-90              NaN     NaN   1.29%
               90+                NaN     NaN   0.16%}}




```python
chk_output_metrics
```




    {'PPV': {'AAPI': 0.5,
      'BLACK': 0.6296296296296297,
      'HISPANIC': 0.7272727272727273,
      'WHITE': 0.9728739002932552},
     'TPR': {'AAPI': 0.1,
      'BLACK': 0.4146341463414634,
      'HISPANIC': 0.6666666666666666,
      'WHITE': 0.9895600298284862},
     'FPR': {'AAPI': 0.0007173601147776321,
      'BLACK': 0.007336757153338258,
      'HISPANIC': 0.0021551724137931494,
      'WHITE': 0.5873015873015873},
     'FNR': {'AAPI': 0.9,
      'BLACK': 0.5853658536585367,
      'HISPANIC': 0.33333333333333337,
      'WHITE': 0.010439970171513768},
     'TNR': {'AAPI': 0.9992826398852224,
      'BLACK': 0.9926632428466617,
      'HISPANIC': 0.9978448275862069,
      'WHITE': 0.4126984126984127},
     'AUC': {'AAPI': 0.5496413199426112,
      'BLACK': 0.7036486945940625,
      'HISPANIC': 0.8322557471264367,
      'WHITE': 0.7011292212634495}}




```python
chk_bisg_output_metrics
```




    {'PPV': {'AAPI': 'None',
      'BLACK': 0.25,
      'HISPANIC': 1.0,
      'None': 0.0,
      'WHITE': 0.9588235294117647},
     'TPR': {'AAPI': 0.0,
      'BLACK': 0.024390243902439025,
      'HISPANIC': 0.08333333333333333,
      'None': 'None',
      'WHITE': 0.9724086502609992},
     'FPR': {'AAPI': 0.0,
      'BLACK': 0.002201027146001455,
      'HISPANIC': 0.0,
      'None': 0.02777777777777779,
      'WHITE': 0.8888888888888888},
     'FNR': {'AAPI': 1.0,
      'BLACK': 0.975609756097561,
      'HISPANIC': 0.9166666666666666,
      'None': 'None',
      'WHITE': 0.027591349739000792},
     'TNR': {'AAPI': 1.0,
      'BLACK': 0.9977989728539985,
      'HISPANIC': 1.0,
      'None': 0.9722222222222222,
      'WHITE': 0.1111111111111111},
     'AUC': {'AAPI': 0.5,
      'BLACK': 0.5110946083782187,
      'HISPANIC': 0.5416666666666666,
      'None': 'None',
      'WHITE': 0.5417598806860552}}




```python
chk_comparision_dict
```




    {'TPR':   Metric      Race       ZRP      BISG Percent Difference
     0    TPR     BLACK  0.414634  0.024390           1600.00%
     1    TPR     WHITE  0.989560  0.972409              1.76%
     2    TPR  HISPANIC  0.666667  0.083333            700.00%
     3    TPR      AAPI  0.100000  0.000000                N/A,
     'FPR':   Metric      Race       ZRP      BISG Percent Difference
     0    FPR     BLACK  0.007337  0.002201            233.33%
     1    FPR     WHITE  0.587302  0.888889            -33.93%
     2    FPR  HISPANIC  0.002155  0.000000                N/A
     3    FPR      AAPI  0.000717  0.000000                N/A,
     'FNR':   Metric      Race       ZRP      BISG Percent Difference
     0    FNR     BLACK  0.585366  0.975610            -40.00%
     1    FNR     WHITE  0.010440  0.027591            -62.16%
     2    FNR  HISPANIC  0.333333  0.916667            -63.64%
     3    FNR      AAPI  0.900000  1.000000            -10.00%,
     'TNR':   Metric      Race       ZRP      BISG Percent Difference
     0    TNR     BLACK  0.992663  0.997799             -0.51%
     1    TNR     WHITE  0.412698  0.111111            271.43%
     2    TNR  HISPANIC  0.997845  1.000000             -0.22%
     3    TNR      AAPI  0.999283  1.000000             -0.07%,
     'AUC':   Metric      Race       ZRP      BISG Percent Difference
     0    AUC     BLACK  0.703649  0.511095             37.67%
     1    AUC     WHITE  0.701129  0.541760             29.42%
     2    AUC  HISPANIC  0.832256  0.541667             53.65%
     3    AUC      AAPI  0.549641  0.500000              9.93%}




```python
birmingham_output_metrics
```




    {'PPV': {'AAPI': 0.7435897435897436,
      'AIAN': 0.08333333333333333,
      'BLACK': 0.9165467625899281,
      'HISPANIC': 0.5933456561922366,
      'WHITE': 0.9040720102871839},
     'TPR': {'AAPI': 0.8055555555555556,
      'AIAN': 0.043478260869565216,
      'BLACK': 0.8214055448098001,
      'HISPANIC': 0.8629032258064516,
      'WHITE': 0.941102980546136},
     'FPR': {'AAPI': 0.005054759898904804,
      'AIAN': 0.0006075334143378353,
      'BLACK': 0.038909853249475934,
      'HISPANIC': 0.012389480204989534,
      'WHITE': 0.1616351292792142},
     'FNR': {'AAPI': 0.19444444444444442,
      'AIAN': 0.9565217391304348,
      'BLACK': 0.1785944551901999,
      'HISPANIC': 0.13709677419354838,
      'WHITE': 0.05889701945386405},
     'TNR': {'AAPI': 0.9949452401010952,
      'AIAN': 0.9993924665856622,
      'BLACK': 0.9610901467505241,
      'HISPANIC': 0.9876105197950105,
      'WHITE': 0.8383648707207858},
     'AUC': {'AAPI': 0.9002503978283254,
      'AIAN': 0.5214353637276137,
      'BLACK': 0.891247845780162,
      'HISPANIC': 0.9252568728007311,
      'WHITE': 0.8897339256334609}}




```python
bmghm_bisg_output_metrics
```




    {'PPV': {'AAPI': 0.8265682656826568,
      'AIAN': 1.0,
      'BLACK': 0.7463566967383761,
      'HISPANIC': 0.5870967741935483,
      'None': 0.0,
      'WHITE': 0.8371853653972116},
     'TPR': {'AAPI': 0.691358024691358,
      'AIAN': 0.043478260869565216,
      'BLACK': 0.6934235976789168,
      'HISPANIC': 0.489247311827957,
      'None': 'None',
      'WHITE': 0.7984115652329109},
     'FPR': {'AAPI': 0.002639707947205805,
      'AIAN': 0.0,
      'BLACK': 0.12259958071278831,
      'HISPANIC': 0.007208424846539385,
      'None': 0.06045562358651879,
      'WHITE': 0.2513361259569551},
     'FNR': {'AAPI': 0.308641975308642,
      'AIAN': 0.9565217391304348,
      'BLACK': 0.3065764023210832,
      'HISPANIC': 0.510752688172043,
      'None': 'None',
      'WHITE': 0.2015884347670891},
     'TNR': {'AAPI': 0.9973602920527942,
      'AIAN': 1.0,
      'BLACK': 0.8774004192872117,
      'HISPANIC': 0.9927915751534606,
      'None': 0.9395443764134812,
      'WHITE': 0.7486638740430449},
     'AUC': {'AAPI': 0.8443591583720761,
      'AIAN': 0.5217391304347826,
      'BLACK': 0.7854120084830643,
      'HISPANIC': 0.7410194434907088,
      'None': 'None',
      'WHITE': 0.7735377196379779}}




```python
birmingham_comparision_dict
```




    {'TPR':   Metric      Race       ZRP      BISG Percent Difference
     0    TPR      AAPI  0.805556  0.691358             16.52%
     1    TPR  HISPANIC  0.862903  0.489247             76.37%
     2    TPR     BLACK  0.821406  0.693424             18.46%
     3    TPR      AIAN  0.043478  0.043478              0.00%
     4    TPR     WHITE  0.941103  0.798412             17.87%,
     'FPR':   Metric      Race       ZRP      BISG Percent Difference
     0    FPR      AAPI  0.005055  0.002640             91.49%
     1    FPR  HISPANIC  0.012389  0.007208             71.87%
     2    FPR     BLACK  0.038910  0.122600            -68.26%
     3    FPR      AIAN  0.000608  0.000000                N/A
     4    FPR     WHITE  0.161635  0.251336            -35.69%,
     'FNR':   Metric      Race       ZRP      BISG Percent Difference
     0    FNR      AAPI  0.194444  0.308642            -37.00%
     1    FNR  HISPANIC  0.137097  0.510753            -73.16%
     2    FNR     BLACK  0.178594  0.306576            -41.75%
     3    FNR      AIAN  0.956522  0.956522              0.00%
     4    FNR     WHITE  0.058897  0.201588            -70.78%,
     'TNR':   Metric      Race       ZRP      BISG Percent Difference
     0    TNR      AAPI  0.994945  0.997360             -0.24%
     1    TNR  HISPANIC  0.987611  0.992792             -0.52%
     2    TNR     BLACK  0.961090  0.877400              9.54%
     3    TNR      AIAN  0.999392  1.000000             -0.06%
     4    TNR     WHITE  0.838365  0.748664             11.98%,
     'AUC':   Metric      Race       ZRP      BISG Percent Difference
     0    AUC      AAPI  0.900250  0.844359              6.62%
     1    AUC  HISPANIC  0.925257  0.741019             24.86%
     2    AUC     BLACK  0.891248  0.785412             13.48%
     3    AUC      AIAN  0.521435  0.521739             -0.06%
     4    AUC     WHITE  0.889734  0.773538             15.02%}




```python

```


```python

```
