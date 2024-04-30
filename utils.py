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
    numeric_columns_int = data_types[(data_types == 'int32')].index.tolist()
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
        if unique_values < 10:  # You can adjust the threshold as needed
            catgCol.append(column)
        else:
            numericCol.append(column)
    numericCol.extend(numeric_columns_float)
    return catgCol, strCol, numericCol
def columns_with_null_values(data):
    # Check for null values in each column
    null_columns = data.isnull().any()

    # Filter to get the names of columns with null values
    columns_with_null = null_columns[null_columns].index.tolist()

    return columns_with_null