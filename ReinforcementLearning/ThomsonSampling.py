import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

dataSet=pd.read_csv("Ads_CTR_Optimisation.csv")


n=10000
d=10
ads_selected=[]
reward_1=[0]*d
reward_0=[0]*d

for n in range(n):
    ad=0
    max_beta=0
    for i in range(0,d):
        random_beta=random.betavariate(reward_1[i]+1, reward_0[i]+1)
        if(random_beta>max_beta):
            max_beta=random_beta
            ad=i
    ads_selected.append(ad)        
    if(dataSet.values[n,ad]==1):
        reward_1[ad]+=1
    else:
        reward_0[ad]+=1


plt.hist(ads_selected)