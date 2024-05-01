import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine, load_diabetes
from utils import problem_typeD, categorical_columns,columns_with_null_values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer



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
        self.shape = self.data.shape
        x = self.data.drop(target, axis=1)
        y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,random_state=session_id)
        catgCol, strCol, numCol = categorical_columns(self.data)
        if self.problem_type == "Binary" or self.problem_type == "MultiClass":
            catgCol.remove(target)
            ordinal_encoder = OrdinalEncoder()
            data[target] = ordinal_encoder.fit_transform(self.data[[target]])
        null_obj, null_num = columns_with_null_values(self.data, catgCol, numCol)
        print(null_obj, null_num)
        imputer = SimpleImputer(strategy="mean")
        print(type(self.X_train))
        self.X_train[null_num] = imputer.fit_transform(self.X_train[null_num])
        self.X_test[null_num] = imputer.transform(self.X_test[null_num])
        imputer = SimpleImputer(strategy="most_frequent")
        self.X_train[null_obj] = imputer.fit_transform(self.X_train[null_obj])
        self.X_test[null_obj] = imputer.transform(self.X_test[null_obj])
        null_obj, null_num = columns_with_null_values(self.X_train, catgCol, numCol)
        print(null_obj, null_num)
        print(self.X_test[catgCol].isna().sum())
        if len(catgCol) != 0:
            # self.data = pd.get_dummies(self.data, columns=catgCol, drop_first=True)
            encoder = OneHotEncoder(drop="first", sparse_output=False)
            x_train_encoded = encoder.fit_transform(self.X_train[catgCol])
            x_test_encoded = encoder.transform(self.X_test[catgCol])
            x_train_encoded_df = pd.DataFrame(x_train_encoded, columns=encoder.get_feature_names_out(catgCol))
            x_test_encoded_df = pd.DataFrame(x_test_encoded, columns=encoder.get_feature_names_out(catgCol))
            self.X_train = pd.concat([self.X_train.drop(catgCol, axis=1), x_train_encoded_df], axis=1)
            self.X_test = pd.concat([self.X_test.drop(catgCol, axis=1), x_test_encoded_df], axis=1)
        print(catgCol, strCol, numCol)
        self.tshape = self.X_train.shape

        setup_info = {
            "Description": ["session_id", "Target", "Target type", "Original data shape", "Transformed data shape",
                            "Transformed train set shape", "Transformed test set shape", "Numeric features","Categorical features", "Imputation type", "Numeric imputation", "Categorical imputation"],
            "Value": [session_id, target, self.problem_type, self.shape, self.tshape, self.X_train.shape, self.X_test.shape, len(numCol), len(catgCol), "Simple", "mean", "mode"]}
        df = pd.DataFrame(setup_info)

        return df
