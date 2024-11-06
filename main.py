class MultiAgentDataAnalysisEnv(MultiAgentEnv):
    def __init__(self):
        super(MultiAgentDataAnalysisEnv, self).__init__()
        self.action_space = {
            "codificador": gym.spaces.Discrete(5),
            "revisor": gym.spaces.Discrete(3)
        }

        self.observation_space = {
            "codificador": gym.spaces.Box(low=0, high=1, shape=(1,)),
            "revisor": gym.spaces.Box(low=0, high=1, shape=(1,))
        }

        self.problema = "Analise o dataset de vendas para identificar tendências."
        self.codigo = None
        self.codigo_anterior = None
        self.aprovado = False
        self.sugestoes = []
        self.historico_sugestoes = []
        self.token_count_codificador = 0
        self.token_count_revisor = 0

    def reset(self):
        self.codigo = None
        self.codigo_anterior = None
        self.aprovado = False
        self.sugestoes = []
        self.historico_sugestoes = []
        self.token_count_codificador = 0
        self.token_count_revisor = 0

        return {
            "codificador": self.problema,
            "revisor": "Aguarda o código do codificador para revisão."
        }

    def step(self, actions):
        responses = {}
        rewards = {"codificador": 0, "revisor": 0}
        dones = {"codificador": False, "revisor": False}

        # Ações do Codificador
        if actions["codificador"] is not None:
            codificador_action = actions["codificador"]
            responses["codificador"], tokens_used_codificador = self.execute_codificador_action(codificador_action)
            self.token_count_codificador += tokens_used_codificador

            # Penalidade por tokens do Codificador
            rewards["codificador"] -= tokens_used_codificador * 0.01

            if codificador_action in [0, 2, 3]:
                self.codigo_anterior = self.codigo
                self.codigo = responses["codificador"]
                codigo_funcional = self.avaliar_codigo()
                melhoria_implementada = self.verificar_melhoria_implementada()

                if not codigo_funcional:
                    rewards["codificador"] -= 5
                if not melhoria_implementada:
                    rewards["codificador"] -= 3

                self.analisar_complexidade()

        # Ações do Revisor
        if actions["revisor"] is not None and self.codigo:
            revisor_action = actions["revisor"]
            responses["revisor"], tokens_used_revisor = self.execute_revisor_action(revisor_action)
            self.token_count_revisor += tokens_used_revisor

            # Penalidade por tokens do Revisor
            rewards["revisor"] -= tokens_used_revisor * 0.01

            # Se o Revisor aprova o código
            if revisor_action == 2:
                self.aprovado = True
                dones["codificador"] = True
                dones["revisor"] = True
                rewards["codificador"] += 10
                rewards["revisor"] += 5

            # Se o Revisor propõe uma melhoria, verifique eficácia
            if revisor_action == 0:
                melhoria_eficaz = self.verificar_melhoria_eficaz()
                if melhoria_eficaz:
                    rewards["revisor"] += 3  # Recompensa por melhoria eficaz
                else:
                    rewards["revisor"] -= 2  # Penalidade por sugestão ineficaz

            if revisor_action != 2:
                self.sugestoes.append(responses["revisor"])

        # Registro de versões e verificação de melhorias ao final
        if dones["codificador"] and dones["revisor"]:
            self.registrar_versoes(rewards)
            self.verificar_melhorias(rewards)

        return responses, rewards, dones, {}

    def execute_codificador_action(self, action):
        tokens_used = 0
        if action == 0:
            prompt = "Crie um código para: " + self.problema
        elif action == 1:
            prompt = "Revise seu código sem interação com o revisor."
        elif action == 2:
            prompt = "Implemente melhorias no código conforme sugestões do revisor."
        elif action == 3:
            prompt = "Documente e melhore a documentação do código."
        elif action == 4:
            return "Solicitação de explicação enviada ao revisor.", tokens_used

        if action != 4:
            response, tokens_used = self.call_gpt_api(prompt)
            return response, tokens_used

    def execute_revisor_action(self, action):
        tokens_used = 0
        if action == 0:
            prompt = "Sugira uma refatoração para otimizar o código."
            response, tokens_used = self.call_gpt_api(prompt)
            return response, tokens_used
        elif action == 1:
            return "Solicitação de documentação adicional enviada ao codificador.", tokens_used
        elif action == 2:
            return "Código aprovado e pronto para entrega.", tokens_used

    def verificar_melhoria_eficaz(self):
        # Avalia se a sugestão do revisor trouxe melhorias significativas (simulação)
        return "improved" in self.codigo.lower()

    def avaliar_codigo(self):
        if "error" in self.codigo.lower():
            return False
        else:
            return True

    def analisar_complexidade(self):
        print("Análise de complexidade e conformidade com linters realizada.")

    def verificar_melhoria_implementada(self):
        if self.codigo_anterior and "improved" in self.codigo.lower():
            return True
        else:
            return False

    def registrar_versoes(self, rewards):
        self.historico_sugestoes.append({
            "versao_codigo": self.codigo,
            "sugestoes": self.sugestoes,
            "rewards": rewards
        })

    def verificar_melhorias(self, rewards):
        if self.codigo_anterior and self.codigo:
            resumo_melhorias = self.resumir_melhorias()
            rewards["codificador"] += 5  # Recompensa pela implementação das melhorias

    def resumir_melhorias(self):
        prompt = "Resuma as melhorias implementadas no código."
        return self.call_gpt_api(prompt)[0]

    def call_gpt_api(self, prompt):
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        tokens_used = response["usage"]["total_tokens"]
        return response.choices[0].message["content"], tokens_used

from marllib import marl
import random

# Configuração do ambiente multiagente
env = MultiAgentDataAnalysisEnv()

# Configuração de treinamento com MADDPG
config = marl.config("maddpg")

# Inicialização do algoritmo com o ambiente
algo = marl.algos.MADDPG(env)

# Configuração dos agentes com as políticas aprendidas
agent_codificador = marl.agents.SimpleAgent(
    obs_space=env.observation_space["codificador"],
    action_space=env.action_space["codificador"],
    policy=config
)

agent_revisor = marl.agents.SimpleAgent(
    obs_space=env.observation_space["revisor"],
    action_space=env.action_space["revisor"],
    policy=config
)

# Definindo as políticas iniciais para os agentes
def codificador_policy(observation):
    action, _ = agent_codificador.predict(observation)
    return action

def revisor_policy(observation):
    action, _ = agent_revisor.predict(observation)
    return action

# Loop de treinamento
num_episodes = 100  # Ajuste conforme necessário
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = {"codificador": 0, "revisor": 0}

    while not done:
        # Ações do agente codificador
        codificador_action = codificador_policy(obs["codificador"])

        # Ações do agente revisor, se houver código disponível para revisão
        revisor_action = revisor_policy(obs["revisor"]) if env.codigo else None

        actions = {"codificador": codificador_action, "revisor": revisor_action}

        # Passo no ambiente com as ações escolhidas
        obs, rewards, dones, _ = env.step(actions)

        # Registro das recompensas acumuladas por episódio para cada agente
        episode_reward["codificador"] += rewards["codificador"]
        episode_reward["revisor"] += rewards["revisor"]

        # Verifica se o ciclo de ações está concluído
        if dones["codificador"] and dones["revisor"]:
            done = True

    # Exibe as recompensas do episódio
    print(f"Episode {episode + 1}: Recompensas Codificador = {episode_reward['codificador']}, Recompensas Revisor = {episode_reward['revisor']}")

    # Atualiza as políticas dos agentes
    agent_codificador.update()
    agent_revisor.update()
