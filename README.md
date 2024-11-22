# Multi-Agent Reinforcement Learning para Análise de Dados

Este projeto envolve a criação de um sistema de **Multi-Agent Reinforcement Learning (MARL)** composto por dois agentes especializados em melhorar modelos de machine learning para análise de dados. A ideia central é permitir que esses agentes colaborem para encontrar o melhor modelo de machine learning, começando de um modelo simples e melhorando progressivamente até alcançar uma solução eficiente e robusta.

## Arquitetura dos Agentes

### 1. Machine Learning Engineer
Responsável por implementar as mudanças solicitadas pelo Agente Data Scientist e executar o código para testar os modelos. Suas ações incluem:
- Criar e treinar novos modelos de machine learning.
- Alterar hiperparâmetros de modelos existentes.
- Realizar feature engineering conforme solicitado.
- Corrigir ou melhorar o código (quando necessário).

#### Recompensas e Penalidades
- **Recompensas**: Recebe pontos por melhorar a performance do modelo (em comparação com o modelo anterior).
- **Penalidades**:
  - Uso excessivo de tokens da API (quanto mais tokens, maior a penalização).
  - Se as ações tomadas não corresponderem corretamente às solicitações do Agente Data Scientist.
- **Objetivo**: Maximizar o desempenho do modelo com mudanças eficientes e eficazes, minimizando o uso de tokens.

### 2. Agente Data Scientist
Responsável por sugerir melhorias no modelo, com foco nas melhores práticas de análise de dados. Suas ações incluem:
- Propor novo modelo de ML
- Propor ajustes de hiperparâmetros.
- Sugerir alterações nas features (como engenharia de features para melhorar o desempenho).
- Decidir não implementar mais melhorias e entregar o modelo final.
- Fazer análise na base de dados.

#### Recompensas e Penalidades
- **Recompensas**: Recebe pontos se suas sugestões forem implementadas corretamente e resultarem em uma melhoria de desempenho significativa.
- **Penalidades**:
  - Uso excessivo de tokens da API (quanto mais tokens, maior a penalização).
  - Se as sugestões, implementadas corretamente, não resultarem em uma melhoria significativa das métricas.
- **Objetivo**: Maximizar o desempenho do modelo através de boas sugestões, colaborando com o Agente Executor de maneira eficaz.

## Ambiente de Treinamento

O ambiente gerencia o fluxo de treinamento e interage com ambos os agentes, recebendo suas ações e avaliando o desempenho do modelo. Suas funções incluem:
- Receber as sugestões do Agente Data Scientist e as implementações do Agente Executor.
- Executar o código do modelo no ambiente de treinamento e calcular as métricas relevantes (F1, Recall, Log-Loss, etc.).
- Calcular as recompensas e penalidades com base nas métricas, comparando o modelo atual com o anterior.
- Verificar se o Agente Executor implementou corretamente as sugestões do Agente Data Scientist.
- Aplicar penalizações com base no uso de tokens, incentivando a utilização eficiente dos recursos.

### Penalizações de Consumo de Recursos
Aplicadas a ambos os agentes, baseadas no número de tokens que utilizam durante a execução e comunicação com a API. O uso excessivo de tokens resulta em penalizações, incentivando uma utilização mais eficiente.

### Memória de Colaboração
Ambos os agentes compartilham uma memória para registrar as sugestões passadas, os modelos criados e as métricas de desempenho. Isso facilita a colaboração entre eles, evitando sugestões redundantes e permitindo um melhor rastreamento do progresso.

## Fluxo de Treinamento

1. O ambiente apresenta o problema inicial (CSV com dados) aos agentes, indicando as colunas que são **features** e quais são **target** (variável a ser prevista).
2. **Agente Data Scientist** sugere uma mudança no modelo, como a criação de um novo modelo ou ajustes de hiperparâmetros.
3. **Agente Executor** implementa as mudanças sugeridas, treina o modelo e retorna as métricas de desempenho (F1, Recall, Log-Loss, etc.).
4. O ambiente calcula as métricas de desempenho e compara o modelo atual com o anterior. As recompensas são atribuídas com base na melhoria das métricas.
5. O processo continua com o **Agente Data Scientist** propondo novas mudanças, enquanto o **Agente Executor** implementa essas mudanças até que o modelo alcance um desempenho satisfatório.

### Métricas de Desempenho
As métricas relevantes que os agentes devem focar podem ser passadas pelo usuário, um exemplo é:
- **F1-Score**
- **Recall**
- **Log-Loss**
- Outras métricas de classificação relevantes para o problema.

O ambiente atribui recompensas com base no **ganho** de desempenho entre o modelo anterior e o modelo atual. Por exemplo, se o modelo inicial tem um **Recall** de 80% e o novo modelo tem **Recall** de 92%, o **reward** seria sobre os 12%.

## Objetivo

O objetivo do sistema é **colaborar** para encontrar o melhor modelo de machine learning possível, começando com um **baseline simples** e evoluindo para modelos mais complexos à medida que os agentes fazem melhorias incrementais. Ambos os agentes devem trabalhar juntos de forma cooperativa, com o Agente Data Scientist propondo mudanças baseadas em suas análises e o Agente Executor implementando essas mudanças de forma eficiente e eficaz.

## Links Úteis

- [RLlib](https://applied-rl-course.netlify.app/en/module2)
- [LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions](https://arxiv.org/abs/2405.11106)
- [Building Cooperative Embodied Agents Modularly with Large Language Models](https://arxiv.org/abs/2307.02485)
