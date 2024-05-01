def problem_typeD(data, target):
    num_unique_values = len(data[target].unique())
    if num_unique_values == 2:
        return "Binary"
    elif 2 < num_unique_values < 10:
        return "MultiClass"
    else:
        return "Regression"


def categorical_columns(data):
    data_types = data.dtypes
    object_columns = data_types[(data_types == 'object') | (data_types == 'bool')].index.tolist()
    numeric_columns_int = data_types[(data_types == 'int32')|(data_types == 'int64')].index.tolist()
    numeric_columns_float = data_types[(data_types == 'float64')].index.tolist()
    catgCol = []
    strCol = []
    numericCol = []
    for column in object_columns:
        unique_values = data[column].nunique()
        if unique_values < 10:  # You can adjust the threshold as needed
            catgCol.append(column)
        else:
            strCol.append(column)
    for column in numeric_columns_int:
        unique_values = data[column].nunique()
        if unique_values < 15:  # You can adjust the threshold as needed
            catgCol.append(column)
        else:
            numericCol.append(column)
    for column in numeric_columns_float:
        unique_values = data[column].nunique()
        if unique_values < 15:  # You can adjust the threshold as needed
            catgCol.append(column)
        else:
            numericCol.append(column)
    return catgCol, strCol, numericCol


def columns_with_null_values(data,catg_col,num_col):

    null_columns_obj = data[catg_col].isnull().any()
    null_columns_num = data[num_col].isnull().any()

    columns_with_null_obj = null_columns_obj[null_columns_obj].index.tolist()
    columns_with_null_num = null_columns_num[null_columns_num].index.tolist()

    return columns_with_null_obj, columns_with_null_num
