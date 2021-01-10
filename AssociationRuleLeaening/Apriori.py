import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataSet=pd.read_csv("Market_Basket_Optimisation.csv",header=None)

transactions=[]

for i in range(0,7501):
    transactions.append([str(dataSet.values[i,j]) for j in range(0,20)])
    


from apyori_import import apriori as ap

rules=ap(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)



results=list(rules)

for i in results:
    print(i)
    print('**********')