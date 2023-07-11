# # %% [markdown]
# # # Train diabetes classification model
# # 
# # This notebook reads a CSV file and trains a model to predict diabetes in patients. The data is already preprocessed and requires no feature engineering.
# # 
# # The evaluation methods were used during experimentation to decide whether the model was accurate enough. Moving forward, there's a preference to use the autolog feature of MLflow to more easily deploy the model later on.

# # %% [markdown]
# # ## Read data from local file
# # 
# # 

# # %%
# import pandas as pd

# print("Reading data...")
# df = pd.read_csv('diabetes.csv')
# df.head()

# # %% [markdown]
# # ## Split data

# # %%
# print("Splitting data...")
# X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

# # %%
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# # %% [markdown]
# # ## Train model

# # %%
# from sklearn.linear_model import LogisticRegression

# print("Training model...")
# model = LogisticRegression(C=1/0.1, solver="liblinear").fit(X_train, y_train)

# # %% [markdown]
# # ## Evaluate model

# # %%
# import numpy as np

# y_hat = model.predict(X_test)
# acc = np.average(y_hat == y_test)

# print('Accuracy:', acc)

# # %%
# from sklearn.metrics import roc_auc_score

# y_scores = model.predict_proba(X_test)
# auc = roc_auc_score(y_test,y_scores[:,1])

# print('AUC: ' + str(auc))

# # %%
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt

# # plot ROC curve
# fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
# fig = plt.figure(figsize=(6, 4))
# # Plot the diagonal 50% line
# plt.plot([0, 1], [0, 1], 'k--')
# # Plot the FPR and TPR achieved by our model
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')


import argparse

def main(args):
    for i, arg in enumerate(args.inputs, 1):
        print(f"Argument {i}: {arg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input strings.')
    parser.add_argument('inputs', type=str, nargs='+', help='Input strings')


    args = parser.parse_args()
    print(args)

    main(args)
