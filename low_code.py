import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine, load_diabetes

 global problem_type = None

def getdata(name):
    if name.lower() == 'iris':
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        return df
    elif name.lower() == 'digits':
        digits = load_digits()
        df = pd.DataFrame(data=digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
        df['digits'] = digits.target
        return df
    elif name.lower() == 'wine':
        wine = load_wine()
        df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
        df['wine'] = wine.target
        return df
    elif name.lower() == 'diabetes':
        diabetes = load_diabetes()
        df = pd.DataFrame(data=diabetes.data, columns=[f'feature_{i}' for i in range(diabetes.data.shape[1])])
        df['diabetes'] = diabetes.target
        return df
    elif name.split(".")[1] == "csv":
        return pd.read_csv(name)
    elif name.split(".")[1] == "xlsx":
        return pd.read_excel(name)
    else:
        return None
def setup(data,target,session_id):

    num_unique_values = len(data[target].unique())
    if num_unique_values == 2:
        problem_type = "Binary"
    elif num_unique_values > 2 and num_unique_values < 10 :
        problem_type = "MultiClass"
    else:
        problem_type = "Regression"

    setup_info = {"Descrption":["session_id", "Target","Target type", "Original data shape"],
                  "Value":[session_id, target, problem_type, data.shape]}
    df = pd.DataFrame(setup_info)
    return df
