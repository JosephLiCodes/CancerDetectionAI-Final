import sklearn
import os.path
from os import path
import pickle
import pandas
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
trainF = input("Would you like to train the model first? Y/N:")
if(trainF.lower() == "y" or trainF.lower() == "yes" ):
    print("Training model...")
    cancer = datasets.load_breast_cancer()
    #have feature names and target names
    #print(cancer.feature_names)
    #print(cancer.target_names)

    x = cancer.data
    y = cancer.target

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    classes = ['malignant', 'benign']

    clf = svm.SVC(kernel="linear")
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    pickle.dump(clf, open("finishedmodel.pickle", 'wb'))
    print("Your model accuracy is:" + str(acc))
userInput = input("Would you like to predict?: (yes/no)")
if userInput.lower() == "yes" or userInput.lower() == "y":
    try:
        model = pickle.load(open("finishedmodel.pickle", 'rb'))
    except:
        print("Model not found! Please make sure it has the file name finishedmodel.pickle")
        exit()
    fileName = input("What is the name of the file for your data?")
    try:
        results = model.predict(pandas.read_csv(fileName))
        print(results)
        f = open("results.txt", "w")
        f.writelines(["%s\n" % item  for item in results])
        f.close()

    except:
        print("File not found, or there is a problem with the content in your txt file! Please try again")
        exit()