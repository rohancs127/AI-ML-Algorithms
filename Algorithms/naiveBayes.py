def probAttr(data,attr,val):
    Total=data.shape[0]
    cnt=len(data[data[attr]==val])
    return cnt,cnt/Total


def train(data,Attr,conceptVals,concept):
 conceptProbs={}
 countConcept={}
 
 for cVal in conceptVals:
    countConcept[cVal],conceptProbs[cVal]=probAttr(data,concept,cVal)
 
 AttrConcept={}
 probability_list={}
 
 for att in Attr:
    probability_list[att]={}
    AttrConcept[att]={}
 
    for val in Attr[att]:
        AttrConcept[att][val]={}
        a,probability_list[att][val]=probAttr(data,att,val)
 
        for cVal in conceptVals:
            dataTemp=data[data[att]==val]
 
    AttrConcept[att][val][cVal]=len(dataTemp[dataTemp[concept]==cVal])/countConcept[cVal]
 
    print(f"P(A):{conceptProbs}\n")
    print(f"P(X/A):{AttrConcept}\n")
    print(f"P(X):{probability_list}\n")
 
    return conceptProbs,AttrConcept,probability_list
 

def test(examples,Attr,concept_list,conceptProbs,AttrConcept,probablity_list):
    misclassification_count=0
    Total=len(examples)
    for ex in examples:
        px={}
        for a in Attr:
            for x in ex:
                for c in concept_list:
                    if x in AttrConcept[a]:
                        if c not in px:
                            px[c]=conceptProbs[c]*AttrConcept[a][x][c]/probability_list[a][x]
                        else:
                            px[c]=px[c]*AttrConcept[a][x][c]/probability_list[a][x]
        print(px)

        classification=max(px,key=px.get)
        print(f"Classification:{classification} Expected:{ex[-1]}")
    
        if(classification!=ex[-1]):
            misclassification_count+=1
            misclassification_rate=misclassification_count*100/Total
            accuracy=100-misclassification_rate
    
    print(f"Misclassification Count={misclassification_count}")
    print(f"Misclassification Rate={misclassification_rate}%")
    print(f"Accuracy={accuracy}%")


import pandas as pd
df=pd.read_csv('//Dataset')
concept=str(list(df)[-1])
concept_list=set(df[concept])
Attr={}
for a in df.columns[:-1]:
    Attr[a]=set(df[a])
    print(f"{a}:{Attr[a]}")
 
conceptProbs,AttrConcept,probability_list=train(df,Attr,concept_list,concept)
examples=pd.read_csv('//Dataset')
test(examples.values,Attr,concept_list,conceptProbs,AttrConcept,probability_list)