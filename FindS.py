import pandas as pd

num_attributes = 6
a =[]
df = pd.read_csv('FindSDataset.csv', header=None)
a = df.values.tolist()

print("The given Dataset:")
for row in a:
    print(row)

hypothesis = [0]* num_attributes
print("\nInitial value of hypothesis:")
print(hypothesis)

print("\nFindS Algorithm: Finding maximally specific hypothesis\n")
for j in range(0, num_attributes):
    hypothesis[j] = a[0][j]

for i in range (0, len(a)):
    if a[i][num_attributes] == 'Yes':
        for j in range(0, num_attributes):
            if hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'
        
        print("For training instance {0} the hypothesis is ".format(i), hypothesis)

print("\nThe maximally specific hypothesis for given training example is:")
print(hypothesis)