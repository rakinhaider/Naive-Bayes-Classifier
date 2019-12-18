import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as util
import discretize as dscrt
import split as splt
import importlib
NBC = importlib.import_module("5_1")

B = [2,5,10,50,100,200]
if util.final:
    columns, data = util.readFile('dating.csv')
else:
    columns, data = util.readFile('test_dating.csv')

training_accuracies = []
testing_accuracies = []
for b in B:
    dataBinned, _ = dscrt.continuousToBinConverter(data, columns,b)
    train, test = splt.split(dataBinned,47,0.2)
    splt.save_train_and_test_split(train, test)
    print("Bin: "+str(b))
    model = NBC.nbc(1)
    training_accuracies.append(model.get_accuracy(train))
    testing_accuracies.append(model.get_accuracy(test))
    print('Training Accuracy: {:.2f}'.format(training_accuracies[len(training_accuracies)-1]))
    print('Testing Accuracy: {:.2f}'.format(testing_accuracies[len(testing_accuracies)-1]))

plt.plot(B,training_accuracies,'g-*', label='Train')
plt.plot(B,testing_accuracies,'b-^',label='Test')
plt.xticks(B)
plt.xlabel("Bins")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('outputs/5_2/5_2.pdf',format='pdf')
#break