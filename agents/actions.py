import os
import openai

from LLM import LLMAgent
from utilities import extrair_codigo, calculate_metrics
from EngenheiroML import EngenheiroML
from Programador import Programador
from Revisor import Revisor

openai.api_key = os.getenv("OPENAI_API_KEY")

def loop_execucao(programador, revisor, descricao_problema):
    """
    Gerencia a interação entre programador e revisor para gerar ou refinar código.
    """
    interacoes = 0
    codigo_valido = False
    ambiente = {"calculate_metrics": calculate_metrics}
    local_vars = {}
    metrics = {}
    erros = []

    while not codigo_valido:
        interacoes += 1
        print(f"Iniciando interação {interacoes}...")

        # Geração ou revisão do código
        if interacoes == 1:
            codigo = programador.criar_codigo()
        elif interacoes <= 5:
            erros.append(erro)
            print(f"Erros encontrados: {erros}")
            sugestoes = revisor.revisar_codigo(codigo, erro)
            print(f"Sugestões do revisor: {sugestoes}")
            codigo = programador.arrumar_codigo(codigo, erro, sugestoes)
        else:
            print("Máximo de interações alcançado. Saindo...")
            break

        # Testar o código gerado
        codigo_extraido = extrair_codigo(codigo) + "\n\nmetrics = calculate_metrics(y_test, y_pred)"
        try:
            exec(codigo_extraido, ambiente, local_vars)
            metrics = local_vars.get("metrics", {})
            codigo_valido = True
        except Exception as e:
            erro = str(e)
            print(f"Erro na execução: {erro}")

    if codigo_valido:
        print("Código final gerado com sucesso!")
    else:
        print("Não foi possível gerar um código válido após 5 interações.")

    print(codigo)

    return codigo, metrics

def write_machine_learning_code(enginner, dataset_name):
    """
    Gera um modelo inicial proposto pelo Engenheiro e cria o código para ele.
    """
    sugestao = enginner.propor_modelo()
    descricao_problema = f"Create a {sugestao} using scikit-learn. Use the {dataset_name} dataset from scikit-learn."

    programador = Programador("qwen2.5-coder:7b", descricao_problema, 800, temperature=0.6)
    revisor = Revisor("qwen2.5-coder:3b", descricao_problema, 300, temperature=0.9)

    codigo, metricas = loop_execucao(programador, revisor, descricao_problema)
    return codigo, metricas


def adjust_hyperparams(enginner, codigo_atual):
    """
    Ajusta os hiperparâmetros do modelo existente com base nas sugestões do Engenheiro.
    """
    sugestao = enginner.ajustar_hiperparametros()
    descricao_problema = f"""Refine the existing code by adjusting the hyperparameters {sugestao}.

    #Current code
    {codigo_atual}
    """

    programador = Programador("qwen2.5-coder:7b", descricao_problema, 800, temperature=0.6)
    revisor = Revisor("qwen2.5-coder:3b", descricao_problema, 300, temperature=0.9)

    codigo, metricas = loop_execucao(programador, revisor, descricao_problema)
    return codigo, metricas


def perform_feature_engineering(enginner, dataset, codigo_atual):
    """
    Executa técnicas de engenharia de características no dataset, refinando o código existente.
    """
    sugestao = enginner.realizar_feature_engineering(dataset)
    descricao_problema = f"""Refine the existing code by applying the following feature engineering: {sugestao}.

    #Current code
    {codigo_atual}
    """

    programador = Programador("qwen2.5-coder:7b", descricao_problema, 800, temperature=0.6)
    revisor = Revisor("qwen2.5-coder:3b", descricao_problema, 300, temperature=0.9)

    codigo, metricas = loop_execucao(programador, revisor, descricao_problema)
    return codigo, metricas


def validate_model(enginner, codigo_atual):
    """
    Implementa uma estratégia de validação cruzada no código existente.
    """
    sugestao = enginner.validacao_cruzada()
    descricao_problema = f"""Add the following cross-validation strategy to the existing code: {sugestao}.

    #Current code
    {codigo_atual}

    Return the y_pred for the best set of hyperparameters.
    """

    programador = Programador("qwen2.5-coder:7b", descricao_problema, 800, temperature=0.6)
    revisor = Revisor("qwen2.5-coder:3b", descricao_problema, 300, temperature=0.9)

    codigo, metricas = loop_execucao(programador, revisor, descricao_problema)
    return codigo, metricas

if __name__ == "__main__":
    # Configuração inicial
    PROBLEM_DESCRIPTION = "Crie um modelo de classificação para load_wine()."

    # Instanciando Engenheiro
    enginner = EngenheiroML("qwen2.5-coder:7b", PROBLEM_DESCRIPTION, 300, temperature=0.5)

    # Passo 1: Geração do modelo inicial
    dataset_name = "load_wine()"
    codigo_modelo, metricas_modelo = write_machine_learning_code(enginner, dataset_name)
    print("Modelo e métricas geradas:", metricas_modelo)

    # Passo 2: Ajuste de hiperparâmetros
    codigo_hyperparams, metricas_hyperparams = adjust_hyperparams(enginner, codigo_modelo)
    print("Hiperparâmetros ajustados:", metricas_hyperparams)

    # Passo 3: Engenharia de características
    import pandas as pd
    dataset = pd.read_csv("nina_cloe_frames.csv")
    codigo_feature_engineering, metricas_fe = perform_feature_engineering(enginner, dataset, codigo_hyperparams)
    print("Engenharia de características realizada:", metricas_fe)

    # Passo 4: Validação cruzada
    codigo_validacao, metricas_validacao = validate_model(enginner, codigo_feature_engineering)
    print("Validação cruzada concluída:", metricas_validacao)
