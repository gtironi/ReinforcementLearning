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
        Your task is to identify the machine learning model to be applied mentioned in the question. Return only the model's name in no more than 5 words.

        Question: For a classification task using the `sklearn.datasets.load_wine()` dataset, a **Random Forest Classifier** would be an excellent choice.
        Answer: Random Forest

        Question: A suitable alternative to Random Forest for this classification task could be **Support Vector Machine (SVM)**, due to its effectiveness in handling small datasets and dealing with high-dimensional data.
        Answer: Support Vector Machines

        Question: {text}
        Answer:
        """
        return self.generate(prompt)

    def extract_hyperparameters(self, text):
        """
        Extrai hiperparâmetros mencionados no texto.
        """
        prompt = f"""
        Hyperparameter:
        Question: I need hyperparameters for a Random Forest model.
        Answer: {{'max_depth': 5, 'n_estimators': 100}}

        Hyperparameter:
        Question: Provide the hyperparameters for a Logistic Regression model.
        Answer: {{'penalty': 'l2', 'C': 1.0}}

        Hyperparameter:
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

        return response + "\n\nDo not set any parameters, use the defaut ones."

    def ajustar_hiperparametros(self, modelo):
        prompt = f"""
        You are a Machine Learning Engineer. The following problem requires fine-tuning:
        `{self.problem_description}`.
        Suggest optimized hyperparameters for the model `{modelo}` to improve performance.
        Be concise and week you answer short.
        """
        return self.generate(prompt)

    def realizar_feature_engineering(self, dataset_info):
        prompt = f"""
        You are a Machine Learning Engineer. For the dataset described below:
        `{dataset_info}`,
        suggest specific feature engineering techniques (e.g., transformations, encodings, or selection).
        Focus on improving the data quality and model performance for the problem:
        `{self.problem_description}`.
        Be concise and specific.
        """
        return self.generate(prompt)

    def validacao_cruzada(self, modelo):
        prompt = f"""
        You are a Machine Learning Engineer. Propose a cross-validation strategy for the model `{modelo}`
        given the problem:
        `{self.problem_description}`.
        Include the number of folds, and whether stratification is necessary.
        Be concise and explain your reasoning.
        """
        return self.generate(prompt)

if __name__ == "__main__":
    # Descrição do problema
    PROBLEM_DESCRIPTION = "Crie um modelo de Machine Learning de regressão utilizando o dataset sklearn.datasets.load_boston()."

    # Instanciar agentes
    enginner =  EngenheiroML("qwen2.5-coder:7b", PROBLEM_DESCRIPTION, 300, temperature = 0.5)


    print(enginner.propor_modelo())
