import openai
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

import os
import re
import ast
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

def extrair_codigo(texto):
    try:
        pattern = r'```python(.*?)```'
        python_code = re.search(pattern, texto, re.DOTALL).group(1).strip()
    except:
        python_code = texto
    return python_code

class LLMAgent:
    """
    Classe base para agentes que interagem com modelos de linguagem.
    """
    def __init__(self, model, limit, temperature):
        self.model = model
        self.chat = ChatOllama(model=model, temperature=temperature, num_predict=limit)
        self.chain = self.chat | StrOutputParser()

    def generate(self, prompt):
        """
        Gera uma resposta baseada em um prompt.
        """
        return self.chain.invoke(prompt)

class Summarizer(LLMAgent):
    """
    Agente responsável por interagir com LLMs para tarefas específicas:
    - Extrair o nome de algoritmos de Machine Learning.
    - Extrair hiperparâmetros associados.
    - Resumir processos de engenharia de características.
    """
    def __init__(self, model, limit=100, temperature=0.7):
        super().__init__(model, limit, temperature)

    def extract_ml_name(self, text):
        """
        Extrai o nome de algoritmos de Machine Learning mencionados no texto.
        """
        prompt = f"""
        Your task is to identify the machine learning model to be applied mentioned in the question. Return only the model's name function in scikit-learn in no more than 5 words.

        Question: For a classification task using the `sklearn.datasets.load_wine()` dataset, a **Random Forest Classifier** would be an excellent choice.
        Answer: RandomForestClassifier

        Question: A suitable alternative to Random Forest for this classification task could be **Support Vector Machine (SVM)**, due to its effectiveness in handling small datasets and dealing with high-dimensional data.
        Answer: SVC

        Question: {text}
        Answer:
        """
        return self.generate(prompt)

    def summarize_feature_engineering(self, text):
        """
        Resume o processo de feature engineering descrito no texto.
        """
        prompt = f"""
        You should sumarize the feature engineer described in the text given.
        Your answer should be a sumarry em few words about whats was made.

        Question: I normalized the data in the 'age' column to improve consistency.
        Answer: Normalized the 'age' column.

        Question: I applied one-hot encoding to the 'gender' column for better categorical representation.
        Answer: One-hot encoded the 'gender' column.

        Question: {text}
        Answer:
        """
        return self.generate(prompt)

class EngenheiroML(LLMAgent):
    """
    Agente responsável por propor e ajustar modelos de Machine Learning.
    """
    def __init__(self, model, problem_description, limit, temperature):
        super().__init__(model, limit, temperature)
        self.problem_description = problem_description
        self.models_history = []
        self.summarizer = Summarizer("qwen2.5:0.5b", limit = 120, temperature = 0.93)
        self.hyperparameters_names = []
        self.hyperparameters_history = {}

    def propor_modelo(self):
        prompt = f"""
        You are a Machine Learning Engineer. You can't write code, just the idea. Based on the following problem:
        `{self.problem_description}`,

        I have already tried {self.models_history}, but need a better-performing model. Return just one model.
        Suggest the most suitable scikit-learn model. Keep your response short and concise.
        Return just the name of the model.
        """

        response = self.generate(prompt)

        if len(response.split()) < 5:
            self.models_history.append(response)
        else:
            model_to_be_tried = self.summarizer.extract_ml_name(response)
            if len(model_to_be_tried.split()) > 5:
                model_to_be_tried = self.summarizer.extract_ml_name(response)

            if len(model_to_be_tried.split()) < 10:
                self.models_history.append(model_to_be_tried)

        self._extract_hyperparameters_names()

        return response + "\n\nDo not set any parameters, use the defaut ones."

    def _extract_hyperparameters_names(self):
        prompt_know_hiper = f"""
        The following code stores the hyperparameters of a scikit-learn model in a variable named "param_names".
        Adapt the code to use the {self.models_history[-1]} model instead. Keep the structure, only change the model.
        Don't use print(). Return the variable "param_names" in a python code block.

        ### Example for RandomForestClassifier:
        ```python
        from sklearn.ensemble import RandomForestClassifier
        import inspect

        signature = inspect.signature(RandomForestClassifier)
        param_names = [param.name for param in signature.parameters.values() if param.name != "self"]
        ```
        """

        response = self.generate(prompt_know_hiper)

        try:
            local_var = {}
            param_names_code = extrair_codigo(response)
            exec(param_names_code, local_var)
            param_names = local_var.get("param_names", [])
        except:
            param_names = []

        self.hyperparameters_names = param_names

    def ajustar_hiperparametros(self):
        prompt = f"""
        You are a Machine Learning Engineer tasked with fine-tuning a model. The problem is: `{self.problem_description}`.

        Suggest optimized hyperparameters for the `{self.models_history[-1]}` model to enhance performance.
        According to scikit-learn, the available hyperparameters are:

        {self.hyperparameters_names}

        We already used the following hyperparameters: {self.hyperparameters_history}.
        You can suggest new values for them or propose new hyperparameters.

        Keep your answer concise and focus only on adjusting the necessary hyperparameters for better performance. Adjust 5 hyperparameters at maximum.
        Return a dictionary with the hyperparameters and their values. Return it in a python code block.
        """

        params_dict_response = self.generate(prompt)

        params_dict = ast.literal_eval(extrair_codigo(params_dict_response))

        for key, value in params_dict.items():
            if key not in self.hyperparameters_history:
                # Inicializa a chave com uma lista se ela ainda não existir
                self.hyperparameters_history[key] = []

            if isinstance(value, list):
                value = value[0]

            # Adiciona o novo valor ao histórico
            self.hyperparameters_history[key].append(value)

        return params_dict

    def realizar_feature_engineering(self, dataset):
        try:
            dataset_info = dataset.describe(percentiles=[], include = "all").loc[["mean", "std", "top", "freq"]]
        except:
            try:
                dataset_info = dataset.describe(percentiles=[], include = "all").loc[["mean", "std"]]
            except:
                dataset_info = dataset.describe(percentiles=[], include = "all")

        prompt = f"""
        You are a Machine Learning Engineer. For the dataset described below:

        `{dataset_info}`,

        suggest specific feature engineering techniques (e.g., transformations, encodings, or selection).

        Focus on improving the data quality and model performance for the problem:
        `{self.problem_description}`.
        Be concise and specific. Suggest just one feature engineering technique.
        Specify the coluns names to be used in the feature engineering.
        Don't return the code.
        """

        return self.generate(prompt)

    def validacao_cruzada(self):
        prompt = f"""
        You are a Machine Learning Engineer tasked with fine-tuning a model. The problem is: `{self.problem_description}`.

        Propose a cross-validation strategy for the `{self.models_history[-1]}` model to enhance performance.
        According to scikit-learn, the available hyperparameters are:

        {self.hyperparameters_names}

        We already used the following hyperparameters: {self.hyperparameters_history}.
        You can suggest new values for them or propose new hyperparameters.

        Keep your answer concise and focus only on adjusting the necessary hyperparameters for better performance.
        Adjust 3 hyperparameters at maximum. The combination of hyperparameters to be tested should not exceed 30.
        Return a dictionary with the hyperparameters and their values. Return it in a python code block.
        """

        params_dict_response = self.generate(prompt)

        params_dict = ast.literal_eval(extrair_codigo(params_dict_response))

        for key, value in params_dict.items():
            if key not in self.hyperparameters_history:
                # Inicializa a chave com uma lista se ela ainda não existir
                self.hyperparameters_history[key] = []

            if isinstance(value, list):
                for v in value:
                    self.hyperparameters_history[key].append(v)
            else:
                self.hyperparameters_history[key].append(value)

        return params_dict

if __name__ == "__main__":
    # Descrição do problema
    PROBLEM_DESCRIPTION = "Crie um modelo de Machine Learning para uma tarefa de classificação de comportamento canino baseado em dados de acelerometro e giroscopio."

    # Instanciar agentes
    enginner =  EngenheiroML("qwen2.5-coder:7b", PROBLEM_DESCRIPTION, 300, temperature = 0.5)


    print(enginner.propor_modelo())
    # print(enginner.ajustar_hiperparametros())
    print(enginner.validacao_cruzada())

    import pandas as pd
    import numpy as np

    df = pd.read_csv("nina_cloe_frames.csv")

    print(enginner.realizar_feature_engineering(df))
