import numpy as np
from lib.logisticregression import _LogisticRegression
from dotenv import load_dotenv
import os

seed = os.getenv("seed")
np.random.seed(seed)

def main():

    # prepare demo dataset
    X_train = np.array([[3.393533211,2.331273381],
        [3.110073483,1.781539638],
        [1.343808831,3.368360954],
        [3.582294042,4.67917911],
        [2.280362439,2.866990263],
        [7.423436942,4.696522875],
        [5.745051997,3.533989803],
        [9.172168622,2.511101045],
        [7.792783481,3.424088941],
        [7.939820817,0.791637231],
        [3.662294042,4.66667911]])
    y_train = np.array([0,0,0,0,0,1,1,1,2,2,2])
    
    X_test = np.random.randint(1, 10, (5,2))

    print(f"X_test: {X_test}")

    # train model
    lg_clf = _LogisticRegression(random_state=seed, max_iter=10)
    lg_clf.fit(X_train, y_train)

    # predict
    y_pred = lg_clf.predict(X_test)

    # predict proba
    y_pred_proba = lg_clf.predict_proba(X_test)

    print(f"y_pred: {y_pred} \ny_pred_proba: {y_pred_proba}")


if __name__ == "__main__":
    main()