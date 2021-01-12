import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

dataSet=pd.read_csv("Market_Basket_Optimisation.csv",header=None)

transactions=[]

for i in range(0,7501):
    transactions.append([str(dataSet.values[i,j]) for j in range(0,20)])
    


from apyori_import import apriori as ap

rules=ap(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2,max_length=2)



results=list(rules)


def dataframes(results):
    lhs=[]
    rhs=[]
    support=[]
    confidence=[]
    lift=[]
    for i in results:
        print(i)
        lhs.append(tuple(i[2][0][0])[0])
        support.append(i[1])
        rhs.append(tuple(i[2][0][1])[0])
        confidence.append(i[2][0][2])
        lift.append(i[2][0][3])
    
    return list(zip(lhs,rhs,support,confidence,lift))

data=pd.DataFrame(dataframes(results),columns=["prod1","prod2","support","confidence","lift"])

data.nlargest(n=10, columns="lift")
