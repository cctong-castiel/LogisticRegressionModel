import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from lib.logisticregression import _LogisticRegression
from dotenv import load_dotenv
import os

load_dotenv()
seed = int(os.getenv("seed"))

def main():

    # prepare demo dataset
    # X_train = np.array([[3.393533211,2.331273381],
    #     [3.110073483,1.781539638],
    #     [1.343808831,3.368360954],
    #     [3.582294042,4.67917911],
    #     [2.280362439,2.866990263],
    #     [7.423436942,4.696522875],
    #     [5.745051997,3.533989803],
    #     [9.172168622,2.511101045],
    #     [7.792783481,3.424088941],
    #     [7.939820817,0.791637231],
    #     [3.662294042,4.66667911]])
    # y_train = np.array([0,0,0,0,0,1,1,1,2,2,2])
    
    # X_test = np.random.randint(1, 10, (5,2))

    # load X and y
    X, y = load_wine(return_X_y=True)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # create model objects
    lg_skl = LogisticRegression(random_state=seed, max_iter=100)
    lg_clf = _LogisticRegression(random_state=seed, max_iter=10)

    # fit
    print("model fitting")
    print(f"y_test: {y_test}")
    lg_skl.fit(X_train, y_train)
    lg_clf.fit(X_train, y_train)

    # predict
    print("model prediction")
    y_pred_skl = lg_skl.predict(X_test)
    y_pred_ctm = lg_clf.predict(X_test)

    # predict proba
    y_pred_proba_skl = lg_skl.predict_proba(X_test)
    y_pred_proba_ctm = lg_clf.predict_proba(X_test)

    print(f"y_pred_proba_ctm: {y_pred_proba_ctm}")
    print(f"y_pred_ctm: {y_pred_ctm}")

    # classification report
    print(f"classification report of sklearn SVC: \n{classification_report(y_test, y_pred_skl)}\n")
    print(f"classification report of custom SVC: \n{classification_report(y_test, y_pred_ctm)}\n")



if __name__ == "__main__":
    main()