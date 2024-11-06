def __init__(self, model, problem_description):
        self.base_prompt = "You are a Python developer and data-scientist. Your job is to write code to solve data-science problems. Be concise and make sure to document your code."
        self.problema_descrito = problem_description
        self.codigo = ""
        self.documentacao = ""
        self.sugestoes = []
        self.encoding = tiktoken.encoding_for_model(model)  # Escolhe o encoding para contagem de tokens

    def criar_novo_codigo(self):
        """
        Gera código inicial com base na descrição do problema e retorna o número de tokens.
        """
        prompt = f"Escreva um código Python para resolver o seguinte problema: {self.problema_descrito}"
        codigo, tokens = self._chamada_chatgpt(prompt)
        self.codigo = codigo
        return codigo, tokens

    def revisar_codigo(self):
        """
        Revisar o próprio código internamente e retorna o número de tokens usados.
        """
        prompt = f"Revise o seguinte código para melhorar e corrigir possíveis erros:\n\n{self.codigo}"
        revisao, tokens = self._chamada_chatgpt(prompt)
        self.codigo = revisao
        return revisao, tokens

    def implementar_melhorias(self, sugestoes_revisor):
        """
        Implementa melhorias no código com base nas sugestões fornecidas e retorna o número de tokens.
        """
        self.sugestoes.extend(sugestoes_revisor)
        prompt = f"Implemente as seguintes melhorias no código:\nSugestões: {', '.join(sugestoes_revisor)}\n\nCódigo:\n{self.codigo}"
        melhoria, tokens = self._chamada_chatgpt(prompt)
        self.codigo = melhoria
        return melhoria, tokens

    def documentar_codigo(self):
        """
        Gera e aprimora a documentação do código e retorna o número de tokens usados.
        """
        prompt = f"Documente o seguinte código, explicando a lógica e o funcionamento das partes principais:\n\n{self.codigo}"
        documentacao, tokens = self._chamada_chatgpt(prompt)
        self.documentacao = documentacao
        return documentacao, tokens

    def _chamada_chatgpt(self, prompt):
        """
        Faz uma chamada à API do ChatGPT com o prompt especificado e calcula o número de tokens.
        """
        messages = [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": prompt}
        ]
        # Conta o número de tokens antes da chamada
        tokens_usados = self._num_tokens_from_messages(messages)

        # Chamada à API do ChatGPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Tokens usados na resposta
        tokens_resposta = response["usage"]["completion_tokens"]
        total_tokens = tokens_usados + tokens_resposta
        resposta = response.choices[0].message["content"]

        return resposta, total_tokens

    def _num_tokens_from_messages(self, messages):
        """
        Calcula o número de tokens em uma lista de mensagens.
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Tokens padrão para o papel e os delimitadores
            num_tokens += len(self.encoding.encode(message["content"]))
        num_tokens += 2  # Tokens adicionais para o final de cada mensagem
        return num_tokens


class Revisor:
    def __init__(self, model, problem_description=""):
        self.model = model
        self.base_prompt = ("You are a Senior Python developer and data-scientist. Your role is to review code generated by other developers and propose improvements.")
        self.problem_description = problem_description
        self.feedback_history = []  # Registra feedback e progresso
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _chamada_gemini(self, prompt):
        """
        Realiza uma chamada à API Gemini, retorna a resposta e o número de tokens usados.
        """
        messages = [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": prompt}
        ]

        tokens_usados = self.num_tokens_from_messages(messages)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )

        tokens_resposta = response["usage"]["completion_tokens"]
        total_tokens = tokens_usados + tokens_resposta
        resposta = response.choices[0].message["content"]

        return resposta, total_tokens

    def num_tokens_from_messages(self, messages):
        """
        Calcula o número de tokens em uma lista de mensagens.
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Tokens padrão para o papel e delimitadores
            num_tokens += len(self.encoding.encode(message["content"]))
        num_tokens += 2  # Tokens adicionais para o final de cada mensagem
        return num_tokens

    def propor_melhorias(self, code):
        """
        Propor refatorações e otimizações, retornando feedback e tokens usados.
        """
        prompt = (f"Considerando o problema descrito: {self.problem_description}\n\n"
                  f"Analise o seguinte código e proponha melhorias de velocidade, "
                  f"uso de memória, e boas práticas de codificação:\n{code}")
        feedback, tokens = self._chamada_gemini(prompt)
        self.feedback_history.append({"action": "propor_melhorias", "feedback": feedback})
        return feedback, tokens

    def propor_melhorias_especificas(self, code):
        pass

    def aprovar_ou_rejeitar_codigo(self, codigo):
        """
        Aprova ou rejeita o código com base na adequação ao problema e registra o feedback.
        """
        aprovado = "Aprovado"
        return decisao, tokens
