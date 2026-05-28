from src.RandomForest_model import RandomForest_model
from src.shap_analysis import run_shap_analysis
from src.evaluate_model import evaluate_model

def main(features_path, prediction_results_path, model_path, feature_names_path, X_test_path, y_test_path, load_model):
    if load_model:
        evaluate_model(X_test_path, y_test_path, model_path)
        run_shap_analysis(model_path, X_test_path, feature_names_path)
    else:
        RandomForest_model(
            features_path, 
            prediction_results_path, 
            model_path, 
            feature_names_path, 
            X_test_path, 
            y_test_path
        )
        run_shap_analysis(model_path, X_test_path, feature_names_path)

if __name__ == "__main__":
    features_path = "./data/features.csv"
    prediction_results_path = "./result/predictions/predictions.csv"
    model_path = "./result/models/RandomForest_Model.pkl"
    feature_names_path = './result/models/feature_names.pkl'
    X_test_path = './result/models/X_test.pkl'
    y_test_path = './result/models/Y_test.pkl'
    load_model = False

    main(
        features_path, 
        prediction_results_path, 
        model_path, 
        feature_names_path, 
        X_test_path, 
        y_test_path, 
        load_model
    )