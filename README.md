# Naive Bayes algorithm on speed dating data

### A better format can be found in readme.pdf

In this programming assignment, you are given a dataset of experimental speed dating events, and
your task is to predict whether the participant of a date decides to give his or her partner a second
date after the speed dating event (i.e., the “decision” column in the dataset). You will implement al-
gorithms to learn and apply some naive Bayes classification (NBC) models to make such predictions.

More specifically, the datasetdating-full.csv is to be used for this assignment. This .csv file
contains information for 6744 speed dating events in the comma-separated format. The filefield-
meaning.pdfcontains the complete description for the meaning of each column of the dataset.

You are asked to implement your algorithms in Python. Note that although there are many data
mining algorithms available online, for this assignment (as well as the next few programming as-
signments) you must design and implement yourown versionsof the algorithm. DO NOT use
any publicly available code including libraries such assklearn. Your code will be checked against
public implementations. In addition, we will not provide separate testing data to you. You are
asked to design your own tests to ensure that your code runs correctly and meets the specifications
below. Note: You may use thepandas,numpy,scipylibraries for data processing purposes.
The only restriction is that you have to write yourown versionof data mining algorithms; you
can notuse any built-in functions for your algorithm. This is a general rule for this assignment
and all the upcoming ones as well.

To make it easier to refer to a few sets of columns in the dataset, we will use the following terms
(usages will be italicized):

1. preferencescoresof participant: [attractiveimportant, sincereimportant, intelligenceimportant,
    funnyimportant, ambitionimportant, sharedinterestsimportant]
2. preferencescoresof partner: [prefoattractive, prefosincere, prefointelligence, prefofunny
    prefoambitious, prefosharedinterests]
3. continuousvaluedcolumns: All columns other than [gender, race, raceo, samerace, field,
    decision].
4. ratingofpartnerfromparticipant: [attractivepartner, sincerepartner, intelligencepartner,
    funnypartner, ambitionpartner, sharedinterestspartner]

In the following, we specify a number of steps you are asked to complete for this assignment.Note
that all results in sample outputs are fictitious and for representation only for this
assignment and all upcoming assignments as well.

## 1 Preprocessing (4 pts)

Write a Python script namedpreprocess.pywhich reads the filedating-full.csvas input and
performs the following operations to output a new filedating.csv:


```
(i) The format of values in some columns of the dataset is not unified. Strip the surrounding
quotes in the values for columnsrace,raceoandfield(e.g., ‘Asian/Pacific Islander/Asian-
American’→Asian/Pacific Islander/Asian-American), count how many cells are changed
after this pre-processing step, and output this number.
```
- Expected output line:Quotes removed from [count-of-changed-cells] cells.

```
(ii) Convert all the values in the columnfieldto lowercase if they are not already in lowercases
(e.g., Law→law). Count the number of cells that are changed after this pre-processing step,
and output this number.
```
- Expected output line:Standardized [count-of-changed-cells] cells to lower case.

(iii) Use label encoding to convert the categorical values in columnsgender,race,raceoand
fieldto numeric valuesstart from 0. The process of label encoding works by mapping
each categorical value of an attribute to an integer number between 0 andnvalues−1 where
nvaluesis the number of distinct values for that attribute. Sort the values of each categorical
attributelexicographically/alphabeticallybefore you start the encoding process. You
are then asked to output the mapped numeric values for ‘male’ in thegendercolumn, for
‘European/Caucasian-American’ in theracecolumn, for ‘Latino/Hispanic American’ in the
raceocolumn, and for ‘law’ in thefieldcolumn.

- Expected output lines:
    Value assigned for male in column gender: [value-for-male].
    Value assigned for European/Caucasian-American in column race: [value-
    for-European/Caucasian-American].
    Value assigned for Latino/Hispanic American in column raceo: [value-for-
    Latino/Hispanic American].
    Value assigned for law in column field: [value-for-law].

(iv) Normalization:As the speed dating experiments are conducted in several different batches,
the instructions participants received across different batches vary slightly. For example, in
some batches of experiments participants are asked to allocate a total of 100 points among the
six attributes (i.e., attractiveness, sincerity, intelligence, fun, ambition, shared interests) to
indicate how much they value each of these attributes in their romantic partner—that is, the
values inpreferencescoresofparticipantcolumns of a row should sum up to 100 (similarly,
values inpreferencescoresofpartnercolumns of a row should also sum up to 100)—while in
some other batches of experiments, participants are not explicitly instructed to do so.
To deal with this problem, let’s conduct one more pre-process step for values inprefer-
encescoresofparticipantandpreferencescoresofpartnercolumns. For each row, let’s first
sum up all the values in the six columns that belong to the setpreferencescoresofparticipant
(denote the sum value astotal), and then transform the value for each column in the setprefer-
encescoresofparticipantin that row as follows:newvalue=oldvalue/total. We then conduct
similar transformation for values in the setpreferencescoresofpartner.
Finally, you are asked to output themean values for each columnin these two sets after
the transformation.

- Expected output lines:
    Mean of attractiveimportant: [mean-rounded-to-2-digits].


#### ...

```
Mean of sharedinterestsimportant: [mean-rounded-to-2-digits].
Mean of prefoattractive: [mean-rounded-to-2-digits].
...
Mean of prefosharedinterests: [mean-rounded-to-2-digits].
```
```
In summary, below are the sample inputs and outputs we expect to see. We expect18 lines
of outputs in total (the numbers are ficititious):
```
```
$python preprocess.py dating-full.csv dating.csv
Quotes removed from 123 cells.
Standardized 456 cells to lower case.
Value assigned for male in column gender: 0.
Value assigned for European/Caucasian-American in column race: 1.
Value assigned for Latino/Hispanic American in column raceo: 4.
Value assigned for law in column field: 2.
Mean of attractiveimportant: 0.12.
...
Mean of sharedinterestsimportant: 0.34.
Mean of prefoattractive: 0.45.
...
Mean of prefosharedinterests: 0.56.
```
## 2 Visualizing interesting trends in data (6 pts)

```
(i) First, let’s explore how males and females differ in terms of what are the attributes they value
the most in their romantic partners. Please perform the following task ondating.csvand
include your visualization code in a file named 2 1.py.
```
```
(a) Divide the dataset into two sub-datasets by the gender of participant
(b) Within each sub-dataset, compute the mean values for each column in the setprefer-
encescoresofparticipant
(c) Use asingle barplotto contrast how females and males value the six attributes in their
romantic partners differently. Please use color of the bars to indicate gender.
```
```
What do you observe from this visualization? What characteristics do males favor in their
romantic partners? How does this differ from what females prefer?
```
```
(ii) Next, let’s explore how a participant’s rating to their partner on each of the six attributes
relate to how likely he/she will decide to give the partner a second date. Please perform the
following task ondating.csvand include your visualization code in a file named 2 2.py.
```
```
(a) Given an attribute in the setratingofpartnerfromparticipant(e.g., attractivepartner),
determine the number of distinct values for this attribute.
(b) Given a particular value for the chosen attribute (e.g., a value of 10 for attribute ‘at-
tractivepartner’), compute the fraction of participants who decide to give the partner a
second date among all participants whose rating of the partner on the chosen attribute
```

```
(e.g., attractivepartner) is the given value (e.g., 10). We refer to this probability as
thesuccess ratefor the group of partners whose rating on the chosen attribute is the
specified value.
(c) Repeat the above process for all distinct values on each of the six attributes in the set
ratingofpartnerfromparticipant.
(d) For each of the six attributes in the setratingofpartnerfromparticipant, draw a scatter
plot using the information computed above. Specifically, for the scatter plot of a partic-
ular attribute (e.g., attractivepartner), use x-axis to represent different values on that
attribute and y-axis to represent the success rate. We expect6 scatter plots in total.
```
```
What do you observe from these scatter plots?
```
## 3 Convert continuous attributes to categorical attributes (3 pts)

Write a Python script nameddiscretize.pyto discretize all columns incontinuousvaluedcolumns
by splitting them into 5 bins of equal-width in the range of values for that column (checkfield-
meaning.pdffor the range of each column; for those columns that you’ve finished pre-processing
in Question 1(iv), the range should be considered as [0, 1]). If you encounter any values that lie
outside the specified range of a certain column, please treat that value as the max value spec-
ified for that column. The script readsdating.csv as input and producesdating-binned.csv
as output. As an output of your scripts, please print the number of items in each of the 5 bins.
Bins should be sorted from small value ranges to large value rangesfor each column in
continuousvaluedcolumns.

The sample inputs and outputs are as follows. We expect 47 lines of output, and the order of the
attributes in the output should be the same as the order they occur in the dataset:
$python discretize.py dating.csv dating-binned.csv
age: [3203 1188 1110 742 511]
ageo: [2151 1292 1233 1383 685]
importancesamerace: [1282 4306 1070 58 28]
...
like: [119 473 2258 2804 1090]

## 4 Training-Test Split (2 pts)

Use thesamplefunction frompandaswith the parameters initialized asrandomstate = 47,
frac = 0.2to take a random 20% sample from the entire dataset. This sample will serve as your
test dataset, and the rest will be your training dataset. (Note: The use of the randomstate will
ensure all students have the same training and test datasets; incorrect or no initialization of this
parameter will lead to non-reproducible results). Create a new script calledsplit.pythat takes
dating-binned.csvas input and outputstrainingSet.csvandtestSet.csv.


## 5 Implement a Naive Bayes Classifier (15 pts)

- Learn a NBC model using the data in the training dataset, and then apply the learned model
    to the test dataset.
- Evaluate the accuracy of your learned model and print out the model’s accuracy on both the
    training dataset and the test dataset as specified below.

### Code Specification:

Write a function namednbc(tfrac)to train your NBC which takes a parametertfrac that
represents the fraction of the training data to sample from the original training set. Use the
samplefunction frompandaswith the parameters initialized asrandomstate = 47, frac =
tfracto generate random samples of training data of different sizes.

1. Use all the attributes and all training examples intrainingSet.csvto train the NBC by
    calling yournbc(tfrac)function withtfrac= 1. After get the learned model, apply it on all
    examples in the training dataset (i.e.,trainingSet.csv) and test dataset (i.e.,testSet.csv)
    and compute the accuracy respectively. Please put your code for this question in a file called
    5 1.py.
       - Expected output lines:
          Training Accuracy: [training-accuracy-rounded-to-2-decimals]
          Testing Accuracy: [testing-accuracy-rounded-to-2-decimals]

```
The sample inputs and outputs are as follows:
$python 51.py
Training Accuracy: 0.
Testing Accuracy: 0.
```
2. Examine the effects of varying the number of bins for continuous attributes during the dis-
    cretization step. Please put your code for this question parts(ii, iii, iv) in a file called 5 2.py.

```
(i) Given the number of binsb∈B={ 2 , 5 , 10 , 50 , 100 , 200 }, perform discretization for all
columns in setcontinuousvaluedcolumnsby splitting the values in each column intob
bins of equal width within its range. For this task, you can re-use yourdiscretize.py
code to perform the binning procedure, now taking the number of bins as a parameter
and usingdating.csvas input as earlier.)
(ii) Repeat the train-test split as described in Question 4 for the obtained dataset after
discretizing each continuous attribute intobbins.
(iii) For each value ofb, train the NBC on the corresponding new training dataset by call-
ing yournbc(tfrac)function withtfrac = 1, and apply the learned model on the
corresponding new test dataset.
(iv) Draw a plot to show how the value ofbaffects the learned NBC model’s performance
on the training dataset and the test dataset, with x-axis representing the value ofband
y-axis representing the model accuracy. Comment on what you observe in the plot.
```
```
The sample inputs and outputs are as follows:
$python 52.py
Bin size: 2
```

```
Training Accuracy: 0.
Testing Accuracy: 0.
Bin size: 5
Training Accuracy: 0.
Testing Accuracy: 0.
.
.
Bin size: 200
Training Accuracy: 0.
Testing Accuracy: 0.
```
3. Plot the learning curve. Please put your code for this question in a file called 5 3.py.

```
(i) For eachfinF={ 0. 01 , 0. 1 , 0. 2 , 0. 5 , 0. 6 , 0. 75 , 0. 9 , 1 }, randomly sample a fraction of the
training data intrainingSet.csvwith our fixed seed (i.e.,randomstate=47).
(ii) Train a NBC model on the selectedffraction of the training dataset (You can call your
nbc(tfrac)function withtfrac=f). Evaluate the performance of the learned model
on all examples in the selected samples of training data as well as all examples in the
test dataset (i.e.,testSet.csv), and compute the accuracy respectively. Do so for all
f∈F.
(iii) Draw one plot of learning curves where the x-axis representing the values off and
the y-axis representing the corresponding model’s accuracy on training/test dataset.
Comment on what you observe in this plot.
```
### Submission Instructions:

Submit through Blackboard

1. Make a directory namedyourF irstN ameyourLastN ameHW2 and copy all of your files to
    this directory.
2. DO NOTput the datasets into your directory.
3. Make sure you compress your directory into azip folderwith the same name as described
    above, and then upload your zip folder to Blackboard.
    Keep in mind that old submissions are overwritten with new ones whenever you re-upload.

```
Your submission should include the following files:
```
1. The source code in python.
2. Your evaluation & analysis in .pdf format. Note that your analysis should include visualization
    plots as well as a discussion of results, as described in details in the questions above.
3. A README file containing your name, instructions to run your code and anything you would
    like us to know about your program (like errors, special conditions, etc).
