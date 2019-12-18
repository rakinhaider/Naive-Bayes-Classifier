import pandas as pd
import numpy as np
import utils as util
import matplotlib.pyplot as plt


if util.final:
    columns, data = util.readFile('dating.csv')
else:
    columns, data = util.readFile('test_dating.csv')

# Answer to question 2.i

groups = data.groupby(by='gender')
fGroup = groups.get_group(0)
mGroup = groups.get_group(1)

fig, ax = plt.subplots()
barwidth=1
x=[x for x in np.arange(0,2.5*len(util.psParticipants),2.5)]
ax.bar([a - barwidth/2 for a in x],
        fGroup.mean()[util.psParticipants],
        width=barwidth,
        color = 'b',
        label='Female')
ax.bar([a+barwidth/2 for a in x],
        mGroup.mean()[util.psParticipants],
        width=barwidth,
        color = 'g',
        label='Male')
ax.set_xticks(x)
ax.set_xticklabels(util.psParticipants,rotation='vertical')
ax.legend()
fig.tight_layout()
if util.final:
    plt.savefig('outputs/2_1/gender_means.pdf',format='pdf')
else:
    plt.savefig('outputs/2_1/test_gender_means.pdf',format='pdf')