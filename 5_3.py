import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as util
import discretize as dscrt
import split as splt
import importlib
NBC = importlib.import_module("5_1")

F = [0.01,0.1,0.2,0.5,0.6,0.75,0.9,1]
if util.final:
    columns, data = util.readFile('dating.csv')
else:
    columns, data = util.readFile('test_dating.csv')

training_accuracies = []
testing_accuracies = []
for f in F:
    dataBinned, _ = dscrt.continuousToBinConverter(data, columns,5)
    train, test = splt.split(dataBinned,47,0.2)
    splt.save_train_and_test_split(train, test)
    print("Fraction: "+str(f))
    model = NBC.nbc(f)
    _, train = splt.split(train, random_state=47, frac=f)
    training_accuracies.append(model.get_accuracy(train))
    testing_accuracies.append(model.get_accuracy(test))
    print('Training Accuracy: {:.2f}'.format(training_accuracies[len(training_accuracies)-1]))
    print('Testing Accuracy: {:.2f}'.format(testing_accuracies[len(testing_accuracies)-1]))

plt.plot(F, training_accuracies, 'g-*', label='Train')
plt.plot(F, testing_accuracies, 'b-^',label='Test')
plt.xticks(F)
plt.xlabel("Fractions")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('outputs/5_3/5_3.pdf',format='pdf')
#break