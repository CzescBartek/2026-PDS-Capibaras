import pandas as pd
import joblib

def predictions_save(model_path, X_test_path, Y_test_path):
    predictions_path = '../result/predictions/predictions.csv'

    model = joblib.load(model_path)
    X_test = joblib.load(X_test_path)
    Y_test = joblib.load(Y_test_path)

    df = pd.read_csv('../data/features.csv')
    test_ids = df.loc[Y_test.index, 'img_id']
    


    probs = model.predict_proba(X_test)[:, -1]


    print(f"Długość ID: {len(test_ids)}")
    print(f"Długość Probs: {len(probs)}")
    
    results = pd.DataFrame({
    'img_id': test_ids.values,
    'probability': probs
    })
    results.to_csv(predictions_path, index=False)

if __name__ == "__main__":
    predictions_save(
        '../result/models/RandomForest_Model.pkl', 
        '../result/models/X_test.pkl',
        '../result/models/Y_test.pkl',
        )
    
