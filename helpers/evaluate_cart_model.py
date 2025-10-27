import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


def evaluate_cart_model(model, X_train, y_train, X_test, y_test, task='classification', plot_depth=3, path=None):
    """
    Huấn luyện, dự đoán, đánh giá, vẽ cây và lưu mô hình cho CART hoặc tương tự.

    Parameters
    ----------
    model : sklearn estimator
        Model đã chọn (DecisionTreeClassifier hoặc DecisionTreeRegressor)
    X_train, y_train : pandas.DataFrame/Series or numpy.array
        Dữ liệu huấn luyện
    X_test, y_test : pandas.DataFrame/Series or numpy.array
        Dữ liệu test
    task : str
        'classification' hoặc 'regression'
    plot_depth : int
        Số tầng muốn hiển thị trên cây khi plot
    path : str or None
        Đường dẫn để lưu mô hình (.joblib). Nếu None, không lưu.
    """
    # Huấn luyện
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Đánh giá
    if task == 'classification':
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        n_correct = np.sum(y_pred == y_test)
        n_total = len(y_test)
        
        print(f"Classification Results")
        print(f"Số mẫu dự đoán đúng: {n_correct}/{n_total} ({acc*100:.2f}%)")
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        
        # Vẽ cây
        plt.figure(figsize=(20,10))
        plot_tree(model, filled=True, feature_names=X_train.columns,
                  class_names=[str(c) for c in np.unique(y_train)], max_depth=plot_depth, fontsize=14)
        plt.title("Decision Tree (Classification)")
        plt.show()
        
    elif task == 'regression':
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("Regression Results")
        print("R² score:", r2)
        print("MAE:", mae)
        print("RMSE:", rmse)
        
        # Vẽ cây
        plt.figure(figsize=(20,10))
        plot_tree(model, filled=True, feature_names=X_train.columns, max_depth=plot_depth, fontsize=14)
        plt.title("Decision Tree (Regression)")
        plt.show()
        
    else:
        raise ValueError("task must be either 'classification' or 'regression'")
    
    # Lưu model nếu có path
    if path is not None:
        model_path=f"{path}"
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")
    
    return model, y_pred