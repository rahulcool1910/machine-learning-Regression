import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

dataSet=pd.read_csv("Ads_CTR_Optimisation.csv")


dataSet.isna().sum()


rows=10000
cols=10
ads_selected=[]
ad_Seleceted_count=[0]*cols
sum_of_reward=[0]*cols
totalReward=0

for n in range(rows):
    ad=0
    average_reward=0
    max_upper_bound=0
    for i in range(cols):
        if(ad_Seleceted_count[i]>0):
            average_reward=sum_of_reward[i]/ad_Seleceted_count[i]        
            deltaI=math.sqrt((3/2)*math.log(n+1)/ad_Seleceted_count[i])
            upper_bound=average_reward+deltaI
        else:
            upper_bound=1e400
        if(upper_bound>max_upper_bound):
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    ad_Seleceted_count[ad]+=1
    sum_of_reward[ad]+=dataSet.values[n,ad]        


plt.hist(ads_selected)    