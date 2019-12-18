import pandas as pd
import numpy as np
import utils as util
import matplotlib.pyplot as plt

if util.final:
    columns, data = util.readFile('dating.csv')
else:
    columns, data = util.readFile('test_dating.csv')

for col in util.rPP:
    distinctValues = np.sort(data[col].unique())
    success_rates={}
    for val in distinctValues:
        dfWithVal = data[data[col]==val]
        dfWithValSuccess = dfWithVal[dfWithVal.decision==1]
        success_rates[val] = len(dfWithValSuccess)/len(dfWithVal)

    plt.scatter(distinctValues,
             [success_rates[val] for val in distinctValues],
                marker='o',
                s=50)
    plt.xlabel(col)
    plt.ylabel('Success Rate')
    if util.final:
        plt.savefig('outputs/2_2/success_rate_'+col+'.pdf',format='pdf')
    else:
        plt.savefig('outputs/2_2/test_success_rate_'+col+'.pdf',format='pdf')
    plt.clf()
