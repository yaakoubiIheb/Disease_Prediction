import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from warnings import filterwarnings
filterwarnings("ignore")




print("\n \n \n")
print("Reading files ...")
train=pd.read_csv("Training.csv")
test=pd.read_csv("Testing.csv")



#dropping the unnamed column from the train dataframe
train = train.drop("Unnamed: 133", axis=1)



print("\n \n \n")
print("the diseases i can predict and their occurence in the training dataframe :")
print(train.prognosis.value_counts())



#Define X and Y - from training data and P for testing data
Y= train[["prognosis"]]
X= train.drop(["prognosis"], axis=1)
B= test
P= B.drop(["prognosis"],axis=1)



#training data spliting
xtrain,xtest,ytrain,ytest= train_test_split(X,Y,test_size=0.9, shuffle=True,random_state=42)




#random forest model
rfc= RandomForestClassifier(random_state=42)
model_rfc = rfc.fit(xtrain,ytrain)
tr_pred_rfc = model_rfc.predict(xtrain)
ts_pred_rfc = model_rfc.predict(xtest)




print("\n \n \n")
print("Random forest accuracy")
#print(ts_pred_rfc)
print("training accuracy is:",accuracy_score(ytrain,tr_pred_rfc))
print("testing accuracy is:",accuracy_score(ytest,ts_pred_rfc))



#saving the model
filename = 'disease_prediction.sav'
pickle.dump(model_rfc, open(filename, 'wb'))













