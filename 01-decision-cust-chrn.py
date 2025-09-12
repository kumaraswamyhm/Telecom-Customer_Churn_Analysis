import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #spliting the data into train and test
from sklearn import tree #loading the package
from sklearn.metrics import accuracy_score,confusion_matrix #evalution matrix
from sklearn.metrics import *
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class

DATA_SET =  "customer_churn.csv"
OUTPUT_COLUMN = "Churn"
OUTPUT_VALUES = [1, 0]

def get_dataset():
    return DATA_SET

def is_data_loaded(df):
    print(f"Top rows...")
    print(df.head())
    print()

def data_undertand_structure(df):
    print(f"Shape...")
    print(df.shape)
    print()
    
    print(f"Column Names...")
    collist = df.columns.to_list()
    print(f"{collist}")
    
    print(f"Information of data...")
    print(df.info())
    print()

def eda(df):
    is_data_loaded(df)
    data_undertand_structure(df)

def get_entropy(values, values_count):
    S = 0.0
    
    for key in values:
        fvalue = values[key]/values_count
        # print(f"key :{key}, count :{values[key]}, tot :{values_count}, fvalue :{fvalue}")
        S = S - (fvalue * math.log(fvalue, 2))

    return S.round(4)

def get_gain(df, attribute, output_feature, S):
    print()
    unique_values = df[attribute].unique()
    row_count = df.shape[0]
    t = 0

    print(f"Attribute :{attribute}, Values({attribute}) :{unique_values}")
    
    for v in unique_values:
        print(f"\tS({v})", end=": ")
        filter = df[attribute] == v
        filtered_df = df[filter]

        values_count = filtered_df[output_feature].count()
        values = dict(filtered_df.groupby(output_feature)[output_feature].count())

        S1 = get_entropy(values, values_count)
        print(f"values_count :{values_count}, row_count :{row_count}, Values :{values}, E({v}) :{S1}")
        t = t + (values_count/row_count) * S1

    Gain = S - t
    Gain = Gain.round(4)
    print(f"\tGain({attribute}) :{Gain}")
    
    return Gain

def maxvalue(r):
    # print(list(r.items())[0][1])
    return list(r.items())[0][1]

def get_entropy_by_frame(df, output_feature, col_values):
    # print(df[output_feature])
    values_count = df[output_feature].count()
    values = dict(df.groupby(output_feature)[output_feature].count())
    Gains = []
    
    print(f"S('{OUTPUT_COLUMN}') :values_count :{values_count}, Values :{values}")
    
    S = get_entropy(values, values_count)
    print(f"E('{OUTPUT_COLUMN}') :{S}")

    columns = list(df.columns)
    columns.remove(OUTPUT_COLUMN)
    
    for col in columns:
        t = get_gain(df[[col, OUTPUT_COLUMN]], col, OUTPUT_COLUMN, S)
        Gains.append({col:t})
    
    sorted(Gains, key=maxvalue)
    max_gain_col = list(Gains[0].keys())[0]

    print(f"Gains :{Gains}")    
    print(f"Max Gain :{Gains[0]}, feature :{max_gain_col}")
    print("-" * 100)
   
    print()
    return max_gain_col

def analyze_data(df):
    df['age'].hist(grid=True, bins=10)
    plt.title('Age distribuition')
    plt.show()
    
    sns.distplot(df[df['sex']==0]['age'],  label='female') #data[sex]==female[age]
    sns.distplot(df[df['sex']==1]['age'], label='male')
    plt.legend()
    plt.title('Density plot of age by sex')
    plt.show()

    
    df['trestbps'].hist()
    plt.title('Resting Blood pressure distribution')
    plt.show()

    sns.distplot(df['trestbps'], bins=10)
    plt.title('Resting Blood pressure desnity plot');
    plt.show()

    fig, axes = plt.subplots(nrows = 1, ncols=2)
    sns.boxplot(x='chol', data=df, orient='v', ax=axes[0])
    sns.boxplot(x='oldpeak', data=df,  orient='v', ax=axes[1])
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17,10))
    cat_feat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

    for idx, feature in enumerate(cat_feat):
        if feature != 'target':
            ax = axes[int(idx/4), idx%4]  # this step we are fixing the axes like example sex=0==idx=0/4=(0)0 not x=0,0%4=0,,#idx=1/4=int(0.25)0,1%4=1(0,1)
            sns.countplot(x=feature, hue='target', data=df,ax=ax)

    plt.show()
    
    plt.rcParams['figure.figsize'] = (5,5)
    sns.countplot(x='target', hue='sex', data=df)
    plt.title('Count of target feature by sex')
    plt.show()
    
def show_heat_map(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()
    
def main():
    data_set = get_dataset()
    df = pd.read_csv(data_set)
    eda(df)
    
    output_var_balance = df['Churn'].value_counts()
    print(f"Output variable Balance :{output_var_balance}\n")
    
    duplis = df.duplicated().sum()
    print(f"Duplicates :{duplis}\n")
    
    # report = sv.analyze(df)
    # report.show_html('cust_churn.html')
    

    AV = AutoViz_Class()
    report = AV.AutoViz(df)
    print(f"Report :{report}")
    
    return
    # filter = df['TotalCharges'
    column_name = 'TotalCharges'
    non_float_values = df[column_name].apply(lambda x: x if not isinstance(x, float) else None)
    # print(non_float_values)
    only_spaces = df[non_float_values]
    print(only_spaces)
    
    space_values = df[column_name].apply(lambda x: x.isspace() if isinstance(x, str) else False)
    print(space_values)
    only_spaces = df[space_values]
    print(only_spaces)

    
    return

    # analyze_data(df)
    # show_heat_map(df)

    X = df.drop(columns=['target'])#independent variable
    y = df['target']#dependnet or target value
    print(X.shape)
    print(y.shape)

    x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=0, test_size=0.3)
    print(x_train.shape)
    print(x_test.shape)
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    
    print(f"y_train_pred :{y_train_pred}")
    print(f"y_test_pred :{y_test_pred}")

    print(accuracy_score(y_train_pred,y_train))
    print(accuracy_score(y_test_pred, y_test))
    
    print(classification_report(y_test_pred,y_test))
    
    c_parameter_name = 'max_depth'
    c_parameter_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    df = pd.DataFrame(columns=[c_parameter_name, 'accuracy'])

    for input_parameter in c_parameter_values:
        model = tree.DecisionTreeClassifier(max_depth=input_parameter, splitter='best')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred) * 100
        df = pd.concat([df, pd.DataFrame({c_parameter_name: [input_parameter], 'accuracy': [acc_score]})], ignore_index=True)

    print(df)

    print(classification_report(y_test_pred,y_test))    
    
    return
    
    while(True):
        feture_column = get_entropy_by_frame(df, OUTPUT_COLUMN, OUTPUT_VALUES)
        unique_values = df[feture_column].unique()
        
        for uv in unique_values:
            filter = df[feture_column] == uv
            filter_df = df[filter]
            filter_df.drop(columns=[feture_column], inplace=True)
            print(filter_df)
            
        
        break

if (__name__ == '__main__'):
    main()
