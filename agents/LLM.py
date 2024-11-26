from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

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
