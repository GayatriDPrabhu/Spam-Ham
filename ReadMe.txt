
Spam/Ham Filter

#To compile and run the program, do the following from the command prompt:

$ python SpamHamFilter.py <k> <e> <l> <i>

Where:-

k: feature selection size
e: eta value or the learning rate for Logistic Regression
l: lambda value for Logistic Regression
i: number of iterations for Logistic Regression

eg: python SpamHamFilter.py 1000 0.025 0.1 100

#Output for (k=1000 e=0.025 l=0.1 i=500):-
Naive Bayes Accuracy with Stop Words : 94.769874477
Naive Bayes Accuracy without Stop Words : 94.3514644351
Naive Bayes Accuracy with Feature Selection : 92.050209205
Logistic Regression Accuracy with Stop Words : 94.1422594142
Logistic Regression Accuracy without Stop Words : 94.769874477
Logistic Regression Accuracy with Feature Selection : 95.1882845188