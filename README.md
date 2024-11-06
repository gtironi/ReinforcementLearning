# Multi-Agent Reinforcement Learning para Análise de Dados

Este projeto envolve a criação de um algoritmo de multi-agent reinforcement learning (MARL) composto por três agentes especializados em resolver problemas de análise de dados, cada um desempenhando um papel específico com base em uma arquitetura cooperativa.

## Arquitetura dos Agentes

### 1. Agente Codificador
Responsável pela criação e aprimoramento do código para análise de dados. Suas ações incluem:
- Criar novo código a partir do problema descrito.
- Revisar o próprio código, sem interação com o Agente Revisor.
- Implementar melhorias com base nas sugestões fornecidas pelo Agente Revisor.
- Documentar e aprimorar a documentação do código.
- Solicitar explicações ao Agente Revisor sobre uma sugestão recebida, caso necessário.

#### Recompensas e Penalidades
- **Recompensas**: Recebe pontos por código correto e por implementar sugestões do Agente Revisor.
- **Penalidades**:
  - Execuções excessivas do código (peso 1).
  - Número de sugestões necessárias até a aprovação (peso 3).
  - Pedidos de melhoria de documentação.
- **Objetivo**: Maximizar a qualidade final do código e implementar as sugestões de melhoria de forma eficaz.

### 2. Agente Revisor
Um agente sênior especializado em revisar o código e propor melhorias. Suas funções incluem:
- Propor refatorações e otimizações (melhoria de velocidade, uso de memória, ou boas práticas).
- Avaliar a implementação das melhorias sugeridas, verificando se foram aplicadas corretamente.
- Solicitar ao Agente Codificador que melhore a documentação.
- Aprovar ou rejeitar o código com base na adequação à tarefa, mantendo registros do progresso e fornecendo feedback contínuo ao Agente Codificador.

#### Recompensas e Penalidades
- **Recompensas**: Recebe pontos por melhorias e correções efetivas.
- **Penalidades**:
  - Se mais de três sugestões forem necessárias para a aprovação.
  - Se alguma sugestão precisar de explicação adicional ou não melhorar significativamente o desempenho.

## Ambiente de Treinamento
O ambiente gerencia o fluxo de treinamento e fornece o problema inicial e a avaliação das respostas dos agentes. Suas funções incluem:
- Executar o código e identificar erros, avaliando se atende aos requisitos.
- Fornecer feedback sobre a complexidade do código, com base em linters e ferramentas de análise (Mypy, Ruff, Bandit).
- Registrar as versões do código e as recompensas/penalidades atribuídas a cada agente conforme seu desempenho.
- Manter o histórico de sugestões e melhorias no código.
- Tentar quebrar o código e verificar se as sugestões foram implementadas (com LLM)

### Penalizações de Consumo de Recursos
Aplicadas a todos os agentes, baseadas nos tokens que utilizam da API, incentivando o uso eficiente dos recursos.

### Memória de Colaboração
Uma memória compartilhada entre o Agente Codificador e o Agente Revisor, que registra sugestões passadas e mudanças entre as versões do código. Isso facilita a cooperação e evita sugestões redundantes.

## Fluxo de Treinamento
1. O ambiente apresenta o problema inicial aos agentes em forma de prompt, designando o papel de cada um.
2. **Agente Codificador** escolhe uma ação para desenvolver o código, que é então executado e avaliado pelo ambiente, retornando recompensas ou penalidades.
3. **Agente Revisor** analisa o código, escolhe uma ação (como 'Propor Refatoração') e fornece feedback ao Agente Codificador.
4. O ambiente avalia as melhorias implementadas, recompensando o Revisor conforme a qualidade do feedback e da refatoração.
5. O processo continua até que o objetivo seja alcançado ou o problema se repita conforme necessário para refinar o treinamento.

## Objetivo
Colaborar para atingir uma solução eficaz, garantindo a qualidade e eficiência do código. As recompensas dadas pelo ambiente visam:
- **Agente Codificador**: Focar em produzir código correto e eficiente.
- **Agente Revisor**: Aprimorar o código com sugestões valiosas.
