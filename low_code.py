import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine, load_diabetes
from utils import problem_typeD, categorical_columns,columns_with_null_values
from sklearn.model_selection import train_test_split


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


class Expermint():
    def __init__(self):
        self.data = None
        self.problem_type = None
        self.shape = None
        self.tshape = None
        self.X_test = None
        self.X_train = None
        self.y_train = None
        self.y_test = None

    def setup(self, data, target, session_id):
        self.data = data
        self.problem_type = problem_typeD(data, target)
        catgCol, strCol, numCol = categorical_columns(self.data)
        self.shape = self.data.shape
        if len(catgCol) != 0:
            self.data = pd.get_dummies(self.data, columns=catgCol, drop_first=True)
        print(catgCol, strCol, numCol)
        self.tshape = self.data.shape
        x = self.data.drop(target, axis=1).values
        y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)
        null_col = columns_with_null_values(self.data)
        setup_info = {
            "Description": ["session_id", "Target", "Target type", "Original data shape", "Transformed data shape"],
            "Value": [session_id, target, self.problem_type, self.shape, self.tshape]}
        df = pd.DataFrame(setup_info)

        return df
