def save_to_csv(df, csv_path):
    df.to_csv(csv_path, index=False)