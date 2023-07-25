# import libraries
import mlflow
import argparse
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from mlflow.models import infer_signature



from mlflow.pyfunc import PythonModel, PythonModelContext
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-mlflow-models?view=azureml-api-2&tabs=wrapper#logging-custom-models
class ModelWrapper(PythonModel):
    def __init__(self, model):
        self._model = model

    def predict(self, context: PythonModelContext, data):
        # You don't have to keep the semantic meaning of `predict`. You can use here model.recommend(), model.forecast(), etc
        return self._model.predict_proba(data)

    # You can even add extra functions if you need to. Since the model is serialized,
    # all of them will be available when you load your model back.
    def predict_batch(self, data):
        pass





def main(args):
    # read data
    df = get_data(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.n_estimators, args.max_depth, args.model_output, X_train, X_test, y_train, y_test)

    # evaluate model
    eval_model(model, X_test, y_test)

# # function that reads the data
# def get_data(path):
#     print("Reading data...")
#     df = pd.read_csv(path)
    
#     return df

# function that reads the data
def get_data(data_path):

    # Read all csv files in the data directory
    all_files = glob.glob(data_path + "/*.csv")
    print('all_files in get-data(): ', all_files)
    df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)
    
    return df

# function that splits the data
def split_data(df):
    print("Splitting data...")
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

# function that trains the model
def train_model(n_estimators, max_depth, model_output, X_train, X_test, y_train, y_test):

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)


    clf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, random_state=0)
    model = clf.fit(X_train, y_train)


    # mlflow.log_param("Regularization rate", reg_rate)
    # print("Training model...")
    # model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)

    #Logging the model artifact
    signature = infer_signature(X_test, y_probs)
    mlflow.pyfunc.log_model("RFclassifier", 
                            python_model=ModelWrapper(model),
                            signature=signature)
    

    return model

# function that evaluates the model
def eval_model(model, X_test, y_test):
    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    mlflow.log_metric("Accuracy", acc)

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    mlflow.log_metric("AUC", auc)

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png")
    mlflow.log_artifact("ROC-Curve.png")    

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--n_estimators", dest='n_estimators',
                        type=int, default=10)
    parser.add_argument("--max_depth", dest='max_depth',
                        type=int, default=3)
    parser.add_argument("--model_output", dest='model_output',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
