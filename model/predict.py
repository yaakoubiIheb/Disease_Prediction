import pickle
import pandas as pd



@staticmethod
def predict(symptoms):
    test=pd.read_csv("Testing.csv")
    B= test.iloc[:1]
    B= B.drop(["prognosis"],axis=1)
    B.iloc[0]=symptoms

    # load the model from disk
    model = pickle.load(open("disease_prediction.sav", 'rb'))
    return(model.predict(B))





'''
def test() :
    test=pd.read_csv("Testing.csv")
    B= test.iloc[0:0]
    B = B.append(test.iloc[11], ignore_index = True)
    print(B)
    P= B.drop(["prognosis"],axis=1)
    # load the model from disk
    model = pickle.load(open("disease_prediction.sav", 'rb'))
    print(model.predict(P))
'''



    

'''
test=pd.read_csv("Testing.csv")
test=test.drop(["prognosis"],axis=1)
print(test.iloc[40].values.tolist())
result=predict(test.iloc[40].values.tolist())
print(result)
'''


