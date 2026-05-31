"""Módulo de processamento de dados e binarização.

Este módulo é responsável por carregar o dataset de Diabetes nativo do
scikit-learn, calcular as estatísticas necessárias e realizar a binarização
da variável target de acordo com a média de progressão da doença.
"""

from typing import List, Tuple
import numpy as np
from sklearn.datasets import load_diabetes


def carregar_e_binarizar_dados() -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """Carrega o dataset Diabetes do scikit-learn e binariza a variável alvo.

    O dataset original de regressão é transformado em um problema de classificação
    binária. A binarização é feita da seguinte forma:
    - Classe 1: progressão da doença ACIMA da média global do target.
    - Classe 0: progressão da doença ABAIXO ou IGUAL à média global do target.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str], float]:
            - X (np.ndarray): Matriz contendo as 10 características (features).
            - y (np.ndarray): Vetor de rótulos binários (0 ou 1).
            - feature_names (List[str]): Lista com o nome das 10 features.
            - media_target (float): A média original calculada da variável alvo.
    """
    # Carrega o dataset nativo do scikit-learn
    diabetes = load_diabetes()
    X = diabetes.data
    target = diabetes.target
    feature_names = list(diabetes.feature_names)

    # Calcula a média da variável alvo contínua
    media_target = float(np.mean(target))

    # Binariza o target: classe 1 se for maior que a média, caso contrário classe 0
    y = np.where(target > media_target, 1, 0)

    return X, y, feature_names, media_target
