from src.RandomForest_model import RandomForest_model
from src.shap_analysis import run_shap_analysis
from src.evaluate_model import evaluate_model
def main(features_path, prediction_results_path, model_path, load_model, X_test_path, feature_names_path):

    if load_model:
        model = joblib.load(model_path)
        model = joblib.load(feature_names_path)
        y_test = joblib.load('./result/models/Y_test.pkl')
        X_test_scaled = joblib.load('./result/models/X_test.pkl')

        
        evaluate_model(model, X_test_scaled, y_test, features_names)
        run_shap_analysis(model_path, X_test_path, feature_names_path)

    else:
        RandomForest_model(features_path)
        run_shap_analysis(model_path, X_test_path, feature_names_path)





if __name__ == "__main__":
    features_path = "./data/features.csv"
    prediction_results_path = "./result/predictions/predictions.csv"
    model_path = "./result/models/RandomForest.pkl"
    X_test_path= './result/models/X_test.pkl'
    feature_names_path = './result/models/feature_names.pkl'
    load_model = False

        
        
    main(features_path, prediction_results_path,model_path,load_model, feature_names_path, X_test_path)