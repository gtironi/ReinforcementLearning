import re
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def extrair_codigo(texto):
    try:
        pattern = r'```python(.*?)```'
        python_code = re.search(pattern, texto, re.DOTALL).group(1).strip()
    except:
        python_code = texto
    return python_code

def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de classificação ou regressão e retorna um dicionário com os resultados.
    """
    try:
        if np.issubdtype(np.array(y_true).dtype, np.integer):
            report = classification_report(y_true, y_pred, output_dict=True)
            return {
                "accuracy": report.get("accuracy", 0),
                "precision": report.get("macro avg", {}).get("precision", 0),
                "recall": report.get("macro avg", {}).get("recall", 0),
                "f1_score": report.get("macro avg", {}).get("f1-score", 0),
            }
        else:
            return {
                "mae": mean_absolute_error(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "r2_score": r2_score(y_true, y_pred),
            }
    except Exception as e:
        return {"error": str(e)}
