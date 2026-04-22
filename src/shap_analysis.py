import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_shap_analysis(model_path, X_test_path, feature_names_path):
    print("\n--- SHAP analysis ---")
    plot_dir = '../result/figures'
    
    # 1. Wczytywanie modelu i danych
    if not os.path.exists(model_path):
        print(f"No model : {model_path}")
        return
        
    model = joblib.load(model_path)
    X_test = joblib.load(X_test_path) 
    feature_names = joblib.load(feature_names_path) 


    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    print("Sampling 500 rows")
    X_sample = X_test_df.sample(500, random_state=42)

    explainer = shap.TreeExplainer(model)
    

    print("Computing shap. May take a while.")

    all_shap_values = explainer.shap_values(X_sample)
    if isinstance(all_shap_values, list):
        shap_values_to_plot = all_shap_values[1]
    else:

        shap_values_to_plot = all_shap_values[:, :, 1]

    print("Generating summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)

    shap_plot_path = os.path.join(plot_dir, 'shap_summary_without_one.png')
    plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


    print("Generating Waterfall plot...")
    
    example_index = 1
    

    exp_for_one_case = shap.Explanation(
        values=all_shap_values[example_index, :, 1] if len(all_shap_values.shape) == 3 else all_shap_values[example_index],
        base_values=explainer.expected_value[1],
        data=X_sample.iloc[example_index],
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp_for_one_case, show=False)
    plt.title(f"Why transactiona {X_sample.index[example_index]} was graded like that?")
    plt.tight_layout()


    shap_bar_path = os.path.join(plot_dir, 'shap_bar_without_one.png')
    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == "__main__":

    run_shap_analysis(
        '../result/models/RandomForest_Model.pkl', 
        '../result/models/X_test.pkl', 
        '../result/models/feature_names.pkl'
    )