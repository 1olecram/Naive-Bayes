# Classificador Naive Bayes Híbrido - Atividade Prática 2

Este projeto apresenta uma implementação manual (do zero) do algoritmo **Gaussian Naive Bayes** aplicado ao conjunto de dados de **Diabetes** do `scikit-learn` para resolver um problema de classificação binária, atendendo 100% às especificações da **Atividade Prática 2** da disciplina de Inteligência Artificial.

---

## 📂 Estrutura do Projeto

* **`naive_bayes.py`**: Classe `GaussianNaiveBayes` contendo toda a lógica matemática do classificador (médias, variâncias com fator de suavização, log-verossimilhança conjunta, predição de classes e de probabilidades) codificada manualmente usando apenas o **NumPy** para operações vetoriais.
* **`dataset.py`**: Módulo de carregamento e binarização da variável alvo contínua do dataset Diabetes com base na sua média global (Classe 1 para valores acima da média; Classe 0 para valores abaixo ou iguais à média).
* **`main.py`**: Script de execução principal que integra e executa as três etapas da atividade (Parte 1, Parte 2 e Parte 3).
* **`requirements.txt`**: Definição das dependências necessárias para a execução do projeto.

---

## 🚀 Como Executar o Projeto Passo a Passo

Para rodar o programa e visualizar todos os resultados no terminal, siga estas duas etapas simples:

### 1. Instalar as Bibliotecas Necessárias

O projeto requer o **NumPy** (para operações matemáticas vetoriais) e o **scikit-learn** (utilizado **exclusivamente** para carregar o dataset nativo Diabetes e realizar a divisão treino/teste).

Instale-os executando o seguinte comando no seu terminal:

```bash
pip install -r requirements.txt
```

### 2. Executar o Script Principal

Com as bibliotecas instaladas, execute o script principal com o comando:

```bash
python main.py
```

*(Caso seu sistema exija a especificação da versão do Python, tente usar `python3 main.py`).*

Após a execução, todos os resultados das três partes da atividade prática (incluindo distribuições de classes, acurácia média de 10 rodadas, matriz de confusão, detalhes probabilísticos de uma amostra, ranking de força preditiva das características e o modelo simplificado com 3 features) serão impressos diretamente no terminal de forma organizada.

---

## 📈 Resumo dos Resultados Obtidos

* **Acurácia Média do Modelo Completo (10 Features)**: `~72.58%` (desvio padrão: `~5.69%`).
* **Ranking de Importância Individual das Features (Parte 2)**:
  1. `bmi` (Índice 2): `~71.46%`
  2. `s5` (Índice 8): `~70.90%`
  3. `bp` (Índice 3): `~68.31%`
  4. `s4` (Índice 7): `~65.84%`
  5. `s3` (Índice 6): `~64.94%`
  6. `s6` (Índice 9): `~64.94%`
  7. `age` (Índice 0): `~61.12%`
  8. `s1` (Índice 4): `~58.88%`
  9. `s2` (Índice 5): `~56.97%`
  10. `sex` (Índice 1): `~56.18%`
* **Acurácia Média do Modelo Simplificado (3 Features: `bmi`, `s5`, `bp`)**: `~75.17%` (desvio padrão: `~5.47%`).

> [!NOTE]
> O ganho de acurácia de `~2.58%` obtido pelo modelo simplificado demonstra o impacto positivo da remoção de variáveis de entrada ruidosas e com baixa correlação para classificadores baseados na suposição de independência (Naive Bayes).
