import pandas as pd

def preprocess(df):
    df_encoded = pd.get_dummies(df, columns=['Year_of_Study', 'Academic_Program'])

    def multi_select(df, col):
        dummies = df[col].str.get_dummies(sep=', ')
        dummies.columns = [f"{col}_{c}" for c in dummies.columns]
        return dummies

    subjects = multi_select(df, 'Subjects')
    days = multi_select(df, 'Available_Days')
    times = multi_select(df, 'Time_Slots')
    comms = multi_select(df, 'Communication_Methods')

    X = pd.concat([
        df_encoded.drop(columns=['Subjects','Available_Days','Time_Slots','Communication_Methods']),
        subjects, days, times, comms
    ], axis=1)

    return X
