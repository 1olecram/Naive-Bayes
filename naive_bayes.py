"""Módulo de implementação do algoritmo Naive Bayes Gaussiano do zero.

Este módulo contém a classe GaussianNaiveBayes, construída sem depender de
bibliotecas de machine learning prontas para o classificador, utilizando
exclusivamente operações matemáticas via NumPy.
"""

import numpy as np


class GaussianNaiveBayes:
    """Classificador Naive Bayes Gaussiano para problemas de classificação binária ou multiclasse.

    Esta implementação calcula a probabilidade a priori das classes e estima
    a Função Densidade de Probabilidade (FDP) Gaussiana para cada característica
    (feature) com base na média e variância observadas nos dados de treinamento.

    Attributes:
        var_smoothing (float): Fator de suavização adicionado à variância para
            evitar divisão por zero e instabilidade numérica.
        classes_ (numpy.ndarray): Lista de classes únicas observadas no fit.
        theta_ (numpy.ndarray): Médias de cada característica por classe.
            Shape (n_classes, n_features).
        var_ (numpy.ndarray): Variâncias suavizadas de cada característica por classe.
            Shape (n_classes, n_features).
        class_prior_ (numpy.ndarray): Probabilidades a priori de cada classe.
            Shape (n_classes,).
        class_log_prior_ (numpy.ndarray): Logaritmo da probabilidade a priori.
            Shape (n_classes,).
        epsilon_ (float): O valor calculado de suavização da variância.
    """

    def __init__(self, var_smoothing=1e-9):
        """Inicializa o classificador Naive Bayes Gaussiano.

        Args:
            var_smoothing (float, optional): Proporção da maior variância de todas as
                features que é adicionada às variâncias para fins de estabilidade.
                Padrão é 1e-9.
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.theta_ = None
        self.var_ = None
        self.class_prior_ = None
        self.class_log_prior_ = None
        self.epsilon_ = None

    def fit(self, X, y):
        """Ajusta o modelo Naive Bayes Gaussiano aos dados de treinamento.

        Calcula as médias, variâncias e probabilidades a priori das classes
        com base nas amostras fornecidas.

        Args:
            X (numpy.ndarray ou pandas.DataFrame): Matriz de características de treinamento.
                Shape (n_amostras, n_features).
            y (numpy.ndarray ou pandas.Series): Vetor de rótulos de classe (alvo).
                Shape (n_amostras,).

        Returns:
            GaussianNaiveBayes: O próprio objeto ajustado.
        """
        # Conversão de segurança para arrays do NumPy
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Inicialização das estruturas de dados
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        self.class_log_prior_ = np.zeros(n_classes)

        # Cálculo da suavização baseada na maior variância observada em qualquer feature
        global_var = np.var(X, axis=0)
        self.epsilon_ = self.var_smoothing * np.max(global_var) if len(global_var) > 0 else 1e-9

        # Treinamento por classe
        for idx, c in enumerate(self.classes_):
            # Filtra amostras da classe atual
            X_c = X[y == c]
            
            # Cálculo da probabilidade a priori
            self.class_prior_[idx] = X_c.shape[0] / n_samples
            self.class_log_prior_[idx] = np.log(self.class_prior_[idx])
            
            # Cálculo da média aritmética e variância com a suavização
            self.theta_[idx, :] = np.mean(X_c, axis=0)
            self.var_[idx, :] = np.var(X_c, axis=0) + self.epsilon_

        return self

    def _joint_log_likelihood(self, X):
        """Calcula a verossimilhança logarítmica conjunta de cada classe para as amostras X.

        Para cada amostra x e classe c, calcula:
        log P(x, c) = log P(c) + soma( log P(x_j | c) )

        Args:
            X (numpy.ndarray): Matriz de características das amostras.
                Shape (n_amostras, n_features).

        Returns:
            numpy.ndarray: Matriz contendo a verossimilhança logarítmica conjunta.
                Shape (n_amostras, n_classes).
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        joint_log_likelihood = np.zeros((n_samples, n_classes))

        for idx, _ in enumerate(self.classes_):
            mean = self.theta_[idx]
            var = self.var_[idx]
            log_prior = self.class_log_prior_[idx]

            # Cálculo do log da FDP Gaussiana para cada feature
            # Formula: -0.5 * log(2 * pi * var) - 0.5 * ((x - mean)**2 / var)
            log_pdf = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2) / var
            
            # Somatório das características para cada amostra + log a priori
            joint_log_likelihood[:, idx] = log_prior + np.sum(log_pdf, axis=1)

        return joint_log_likelihood

    def predict(self, X):
        """Preve a classe de maior probabilidade para cada amostra em X.

        Args:
            X (numpy.ndarray ou pandas.DataFrame): Matriz de características a classificar.
                Shape (n_amostras, n_features).

        Returns:
            numpy.ndarray: Vetor contendo a classe prevista para cada amostra.
                Shape (n_amostras,).
        """
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        best_indices = np.argmax(jll, axis=1)
        return self.classes_[best_indices]

    def predict_proba(self, X):
        """Calcula as probabilidades percentuais a posteriori para cada classe.

        Utiliza o truque do Log-Sum-Exp para obter estabilidade numérica ao
        converter verossimilhanças logarítmicas de volta para probabilidades reais.

        Args:
            X (numpy.ndarray ou pandas.DataFrame): Matriz de características.
                Shape (n_amostras, n_features).

        Returns:
            numpy.ndarray: Matriz com a probabilidade a posteriori das classes.
                Shape (n_amostras, n_classes). Cada linha soma 1.0.
        """
        X = np.asarray(X, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        
        # Subtrai o valor máximo de cada linha (Log-Sum-Exp Trick) para evitar estouro numérico
        max_jll = np.max(jll, axis=1, keepdims=True)
        exp_jll = np.exp(jll - max_jll)
        
        # Normalização
        sum_exp_jll = np.sum(exp_jll, axis=1, keepdims=True)
        return exp_jll / sum_exp_jll
