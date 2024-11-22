# Propor Novo Modelo de Machine Learning
PROPOR_NOVO_MODELO = """
Proponha um novo modelo de machine learning para o problema descrito. As colunas de treino são {colunas_treino} e a coluna de teste é {coluna_teste}. O objetivo do modelo é {objetivo}, que é uma tarefa de {tipo_tarefa} ({tipo_atividade}). Dev ser um modelo disponivel no sklearn.

Retorne apenas o nome do modelo.
"""

# Propor Ajustes de Hiperparâmetros
PROPOR_AJUSTE_DE_PARAMETROS = """
O modelo atual é um {modelo_atual}. Os híperparametros atuais são {hiperparametro} As métricas de desempenho atuais são:
- F1-Score: {f1_score}
- Recall: {recall}
- Log-Loss: {log_loss}

Com base nisso, sugira uma lista de hiperparâmetros para testar, retornando uma variável Python com uma lista de dicionários de parâmetros.
"""

# Propor Mudança de Features
PROPOR_MUDANCA_DE_FEATURES = """
Proponha mudanças nas features do modelo. Você pode criar novas features, normalizar ou transformar as features existentes ou criar uma nova feature. Descreva exatamente o que deve ser feito para criar ou tirar cada feature. Proponha apenas uma modificação.
"""

# Criar e Treinar Novo Modelo
CRIAR_TREINAR_NOVO_MODELO = """
Crie e treine o seguinte modelo de machine learning: {modelo_especifico}. As métricas a serem calculadas são: {metricas}. O código deve ter uma variavel de hiperparametros que começa com os hiperaparametros padrão.

Retorne apeanas o código.
"""

# Implementar Mudanças nas Features
IMPLEMENTAR_MUDANCAS_FEATURES = """
Implemente as mudanças nas features conforme as sugestões passadas. As instruções são as seguinte:

{mudancas_de_features_sugeridas}
"""

# Corrigir Código (Erro Identificado)
CORRIGIR_CODIGO_COM_ERRO = """
O código abaixo está apresentando o erro: {erro_detectado}. Por favor, corrija e retorne o código corrigido.

{codigo_atual_com_erro}
"""

# Verificar Implementação das Sugestões
VERIFICAR_IMPLEMENTACAO_SUGESTOES = """
As melhorias sugeridas foram: {melhorias_sugeridas}. O código novo implementado é: {codigo_novo}. Por favor, confirme se as melhorias foram implementadas corretamente.

Retorne Sim ou Não apenas.
"""
