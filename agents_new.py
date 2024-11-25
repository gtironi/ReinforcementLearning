import openai
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

import os
import re
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

class LLMAgent:
    """
    Classe base para agentes que interagem com modelos de linguagem.
    """
    def __init__(self, model):
        self.model = model
        self.chat = ChatOllama(model=model, temperature=0.8, num_predict=800)
        self.chain = self.chat | StrOutputParser()

    def generate(self, prompt):
        """
        Gera uma resposta baseada em um prompt.
        """
        return self.chain.invoke(prompt)


class Programador(LLMAgent):
    """
    Agente responsável por gerar código com base na descrição do problema.
    """
    def __init__(self, model, problem_description):
        super().__init__(model)
        self.problem_description = problem_description

    def criar_codigo(self):
        prompt = f"""
        You are a Python Machine Learning Engineer specialized in scikit-learn. Your task is to write Python code to solve data-science problems efficiently.
        Be concise and ensure your code is well-documented. Try to not use functions if it is not necessary. Don't print anything. Always try to make it as simples as possible.

        ### Instructions:
        1. Write Python code to solve the following problem: `{self.problem_description}`.
        2. Just calculate y_pred and y_test. Don't calculate any metrics.
        3. The code should follow this structure:

        ### Example:
        ```python
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import load_iris

        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        ```

        Do not calculate any metrics. Just calculate y_pred and y_test. Try to not use functions if it is not necessary. Don't print anything.
        """
        return self.generate(prompt)


class Revisor(LLMAgent):
    """
    Agente responsável por revisar e ajustar o código gerado pelo Programador.
    """
    def __init__(self, model, problem_description):
        super().__init__(model)
        self.problem_description = problem_description

    def revisar_codigo(self, codigo, erro):
        """
        Gera um novo código revisado com base no erro encontrado.
        """

        prompt = f"""
        You are a Python Machine Learning Engineer. The following Python code failed with the error: `{erro}`.
        Please fix the issue and return the full corrected code at once. If there are funcitions used just once, you can remove them.
        Always try to make it as simples as possible.

        ### Instructions:
        1. The code is made to solve the following problem: `{self.problem_description}`.

        ### Code:
        ```python
        {codigo}
        ```
        """
        return self.generate(prompt)

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

if __name__ == "__main__":
    # Descrição do problema
    PROBLEM_DESCRIPTION = "Crie um modelo de Machine Learning de classificação utilizando o dataset sklearn.datasets.load_wine(). use 300 iterations and a learning rate of 0.01."

    # Instanciar agentes
    programador = Programador("llama3.2", PROBLEM_DESCRIPTION)
    revisor = Revisor("llama3.2", PROBLEM_DESCRIPTION)

    # Loop de interação
    interacoes = 0
    codigo_valido = False
    ambiente = {"calculate_metrics": calculate_metrics}
    local_vars = {}
    metrics = {}
    errors = []

    while not codigo_valido:
        interacoes += 1

        print(f"Começando a interação {interacoes}...")

        # Programador cria código inicial ou revisado
        if interacoes == 1:
            codigo = programador.criar_codigo()
        elif interacoes < 5:
            errors.append(erro)
            codigo = revisor.revisar_codigo(codigo, erro)
        elif interacoes == 5:
            openai.api_key = os.getenv("OPENAI_API_KEY")

            problem_description = "Crie um modelo de Machine Learning de classificação utilizando o dataset sklearn.datasets.load_wine(). use 300 iterations and a learning rate of 0.01."

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": "You are a Python Machine Learning Engineer specialized in scikit-learn. Your task is to write Python code to solve data-science problems efficiently. Be concise and ensure your code is well-documented. Try to not use functions if it is not necessary. Don't print anything. Always try to make it as simples as possible."},
                    {"role": "user", "content": f"### Instructions: 1. Write Python code to solve the following problem: `{PROBLEM_DESCRIPTION}` 2. Just calculate y_pred and y_test. Don't calculate any metrics."}
                ]
            )

            codigo = response.choices[0].message.content.strip()
        else:
            print("Código não foi validado após 5 interações.")
            print("O último código gerado foi:")
            print(codigo)
            break

        codigo_extraido = extrair_codigo(codigo) + "\n\nmetrics = calculate_metrics(y_test, y_pred)"

        try:
            # Tenta executar o código
            exec(codigo_extraido, ambiente, local_vars)
            metrics = local_vars.get("metrics", {})
            codigo_valido = True  # Se executou sem erros, código é válido
        except Exception as e:
            erro = str(e)

    # Exibe resultado
    print(f"Interações necessárias: {interacoes}")
    print(f"Métricas calculadas: {metrics}")
