import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from textdistance import levenshtein
from IPython.display import display, HTML
from IPython.display import clear_output
import time
import math
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay

def preprocess(df):
    # Removes empty rows and replaces empty entries with NaN values
    df = df.dropna(how='all', axis=1).replace('[Leeg]', np.nan).replace('[leeg]', np.nan).replace('[…]', np.nan).replace('[...]', np.nan).dropna(how='all', axis=0)
    return preprocess_columns(df)

def preprocess_column(col):
    # Removes al non-alphabetic characters with the exception of the whitespaces in a column
    try:
        col = col.str.lower()
        col = col.str.replace('[^a-zA-Z ]', '')
        return col
    except:
        return col

def preprocess_columns(df):
    # Preprocess all columns
    for col in df.columns:
        df[col] = preprocess_column(df[col])
    return df
        
def drop_missing(df, cols):
    df2 = df.copy()
    for col in cols:
        df2 = df2[df2[col].isnull()]
    return pd.concat([df,df2]).drop_duplicates(keep=False)

def load_transactions(path):
    # Loads transaction dataset
    file = pd.read_csv(path)
    file["Entry-ID"] = file.index
    transactions = preprocess(file)
    transactions = drop_missing(transactions, ["SlaafNaamNieuw", "KoperVoornaam", "KoperAchternaam"])
    transactions = transactions.drop(columns=["Verkoop", "Inventarisnummer", "Scan", "Plaats"])
    return transactions

def load_permissions(path):
    # Loads permission datset
    file = pd.read_csv(path)
    file["Entry-ID"] = file.index
    permissions = preprocess(file)
    permissions = drop_missing(permissions, ["SlaafNaamNieuw", "BezitterVoornaam", "BezitterAchternaam"])
    return permissions

def fuzzy_search(df, query, col_name):
    col = df[col_name].to_numpy()
    res = np.zeros(len(col))
    for i, val in enumerate(col):
        if type(val) == str and type(query[0]) == str:
            r = fuzz.ratio(val, query[0])
        elif type(val) != str and type(query[0]) != str:
            r = 70
        else:
            r = 70
        res[i]= r * query[1]
    df[f"{col_name}_dist"] = res
    return df , f"{col_name}_dist"

def fuzzy_match(df, query):
    df = df.copy()
    dist_col_names = []
    df["total_dist"] = 0
    for key in query.keys():
        df, col_name = fuzzy_search(df, query[key], key)
        dist_col_names.append(col_name)
        df["total_dist"] = df["total_dist"] + df[col_name]
    return df.sort_values(by=['total_dist'], ascending=False)

def generate_query(s, translation):
    res = dict()
    for key in s.keys():
        try:
            res[translation[key][0]] = (s[key], translation[key][1])
        except:
            pass
    return res

def get_n_dict_value(d, n=0):
    d2 = dict()
    for key in d.keys():
        d2[key] = d[key][0]
    return d2

def read_or_make_csv(path, col_names=["Annotator", "transaction_indx", "permission_indx"]):
    try:
        df = pd.read_csv(path, index_col=0)
    except:
        df = pd.DataFrame(columns=col_names)
    return df

class Annotator:
    def __init__(self, df1, df2, translation, output_file, annotator_name="Bas"):
        self.df1 = df1
        self.df2 = df2
        self.translation = translation
        self.output_file = output_file
        self.output = read_or_make_csv(output_file)
        self.annotator_name = annotator_name
        
    def get_last_index(self, df, match_df, col_name):
        try:
            last_index = match_df.iloc[-1][col_name]
            return match_df[col_name].values[-1]
        except:
            return 0
        
    def annotate(self):
        starting_indx = self.get_last_index(self.df1, self.output, "permission_indx")
        print(starting_indx)
        df1 = self.df1.loc[starting_indx:]
        for i, row in df1.iterrows():
            df1_row = df1.loc[i]
            q = generate_query(df1_row, self.translation)
            m = fuzzy_match(self.df2, q)
            print("Query:")
            display(HTML(pd.DataFrame(get_n_dict_value(q), index=[0]).to_html()))
            print("Possible matches:")
            display(HTML(m[get_n_dict_value(self.translation).values()].head(n=12).to_html()))
            res = input("Index of most likely match: ").split(" ")
            res_id = []
            if res != ['']:
                for el in res:
                    res_id = self.df2.loc[int(el)]["Entry-ID"]
                    res_df = pd.DataFrame(data={"Annotator":self.annotator_name,"transaction_indx":res_id,"permission_indx":df1_row["Entry-ID"]}, index=[0])
                    self.output = pd.concat([self.output, res_df]).reset_index(drop=True)
            else:
                    res_df = pd.DataFrame(data={"Annotator":self.annotator_name,"transaction_indx":"None","permission_indx":df1_row["Entry-ID"]}, index=[0])
                    self.output = pd.concat([self.output, res_df]).reset_index(drop=True)

            self.output.to_csv(self.output_file)
            clear_output(wait=True)
        
        
def evaluate_prediction(pred, y_true, model_name="default_model", figure_folder="../Figures/", plot=False):
    if plot:
        cm = confusion_matrix(y_true, pred)
        disp = sns.heatmap(cm, annot=True, cmap='Blues', yticklabels=["False", "True"], xticklabels=["False", "True"])
        plt.xlabel('\nPredicted Values')
        plt.ylabel('Actual Values ')
        disp.plot()
        plt.tight_layout()
        plt.savefig(f"{figure_folder}_{model_name}_Confusion_matrix.png", bbox_inches="tight")
        plt.show()
    
    return {"recall score: ": recall_score(y_true, pred), "precision score: ": precision_score(y_true, pred), "f1 score: ": f1_score(y_true, pred, zero_division=0)}

def fit_and_test_classifier(clf, X_train, X_test, y_train, y_test, model_name="default_model", figure_folder="../"):
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    evaluate_prediction(pred, y_test, model_name, figure_folder)
    
    display = PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test, name=model_name
    )
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    plt.tight_layout()
    plt.savefig(f"{figure_folder}_{model_name}_PR_curve.png")
    return clf

def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend();








