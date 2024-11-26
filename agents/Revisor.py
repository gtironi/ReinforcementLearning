from LLM import LLMAgent

class Revisor(LLMAgent):
    """
    Agente responsável por revisar e ajustar o código gerado pelo Programador.
    """
    def __init__(self, model, problem_description, limit, temperature):
        super().__init__(model, limit, temperature)
        self.problem_description = problem_description

    def revisar_codigo(self, codigo, erro):
        """
        Gera sugestões para melhorar o código com base no erro encontrado.
        """

        prompt = f"""
        You are a Senior Code Reviewer. The following Python code failed with the error: `{erro}`. Your task is to help a junior developer fix it.

        Focus on simplicity and clarity. Don't use classes. Remove unnecessary functions or arguments, and suggest minimal changes to fix the error.

        Context: The code is meant to:

        `{self.problem_description}`.

        ### Code:
        ```python
        {codigo}
        ```

        Instructions:
        - Analyze the error and provide concise suggestions.
        - Keep your response under 100 words, focusing on key fixes NOT NOT suggest details.
        - Don't include the code, just the suggestions.

        Examples od suggestions:
        1. Add `import pandas as pd` at the beginning.
        2. The `fit()` method does not take `learning_rate`, so remove that argument.
        """
        return self.generate(prompt)
