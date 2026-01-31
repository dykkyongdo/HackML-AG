def remove_correlations(df, var_list):
    df = df.copy()
    df.drop(columns=var_list, inplace=True, errors="ignore")
    return df
    
