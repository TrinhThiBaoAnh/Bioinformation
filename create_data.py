from glob import glob
import numpy as np
import pandas as pd
txt_file  = glob("output_cancer/*.txt")
set_features = sorted(open("features.txt", "r").readlines())
num_samples = len(txt_file)
num_features = len(set_features)
print("num_samples: ", num_samples)
print("num_features: ", num_features)
sum = 0
X = np.zeros((num_samples,num_features))
sample_idx = 0
for txt in txt_file:
    fi = open(txt)
    lst_rs = fi.readlines()
    set_rs = sorted(set(lst_rs))
    for i in range(num_features):
        if set_features[i] in set_rs:
            X[sample_idx, i] = 1
    sample_idx += 1
Y = np.random.choice([1], size=(num_samples,1))
headers = []
for feature in set_features:
    headers.append(feature.replace("\n", ""))
headers.append("label")
data = np.concatenate([X,Y], axis= 1)
df = pd.DataFrame(data)
df.to_csv("data.csv", header=headers, index=False)