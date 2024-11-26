import os
import openai

from LLM import LLMAgent
from utilities import extrair_codigo, calculate_metrics
from Revisor import Revisor

openai.api_key = os.getenv("OPENAI_API_KEY")

class Programador(LLMAgent):
    """
    Agente responsável por gerar código com base na descrição do problema.
    """
    def __init__(self, model, problem_description, limit, temperature):
        super().__init__(model, limit, temperature)
        self.problem_description = problem_description

    def criar_codigo(self):
        prompt = f"""
        You are a Python Machine Learning Engineer specialized in scikit-learn. Your task is to write Python code to solve data-science problems efficiently.
        Be concise and ensure your code is well-documented. Try to not use functions if it is not necessary. Don't print anything. Always try to make it as simples as possible.

        ### Instructions:
        1. Write Python code following this instructions:

        `{self.problem_description}`

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

    def arrumar_codigo(self, codigo, erros, sugestoes):
        prompt = f"""
        You are a Python Machine Learning Engineer. The following Python code failed with the error: `{erros}`.
        Please fix the issue and return the full corrected code at once.
        Use the following suggestions to improve the code:

        {sugestoes}

        ### Code:
        ```python
        {codigo}
        ```
        """

        return self.generate(prompt)

if __name__ == "__main__":
    # Descrição do problema
    PROBLEM_DESCRIPTION = "Crie um modelo de Machine Learning de classificação utilizando o dataset sklearn.datasets.load_wine(). Não coloque nenhum hiperparametro."

    # Instanciar agentes
    programador = Programador("qwen2.5-coder:7b", PROBLEM_DESCRIPTION, 800, temperature = 0.6)
    revisor = Revisor("qwen2.5-coder:3b", PROBLEM_DESCRIPTION, 300, temperature = 0.85)

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
            print(f"Erros encontrados nas interações anteriores: {errors}")
            sugestoes = revisor.revisar_codigo(codigo, erro)
            print(f"Sugestões do revisor: {sugestoes}")
            codigo = programador.arrumar_codigo(codigo, erro, sugestoes)
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

        #print(codigo_extraido)

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
