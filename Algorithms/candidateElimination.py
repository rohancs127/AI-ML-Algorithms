import pandas as pd
import numpy as np

df = pd.read_csv('Datasets/candidateEliminationDataset.csv')
df = df.drop(['Sl.no'], axis=1)
df.head()

concepts = df.values[:,:-1]
target = df.values[:, -1]

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?' for i in range(len(specific_h))] for i in range(len(specific_h))]

    for i,h in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if target[i] == "No":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x]= specific_h[x]
                else:
                    general_h[x][x] = '?'

    indices = [i for i,val in enumerate(general_h) if val==['?','?','?','?','?','?']]
    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])
    
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print(f"Final S: {s_final}")
print(f"Final G: {g_final}")