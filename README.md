# 🧠 Inteligência Computacional (GBC073)

Repositório dedicado ao desenvolvimento, estudo teórico e resolução de projetos práticos da disciplina de **Inteligência Computacional (GBC073)** da Faculdade de Computação (FACOM) da **Universidade Federal de Uberlândia (UFU)**.

---

## 📌 Visão Geral da Disciplina

O curso aborda a transição e a fundamentação matemática dos principais paradigmas da Inteligência Computacional moderna:
1. **Redes Neurais Artificiais & Aprendizado Profundo:** De Perceptrons e Redes Multicamadas (MLP) a mecanismos de Atenção e Transformers.
2. **Computação Evolutiva:** Algoritmos genéticos e otimização livre de gradiente (aplicados quando derivadas não existem ou são intratáveis).
3. **Sistemas Nebulosos (Lógica Fuzzy):** Modelagem de graus de pertinência e relaxações contínuas/diferenciáveis (base conceitual de mecanismos modernos como *Softmax* e *Gating*).

> **Fio Condutor:** *"Tudo o que é diferenciável se treina por gradiente — e quase tudo o que importava foi tornado diferenciável."*

---

## 📂 Conteúdo Programático & Fundamentação Teórica

* **Aula 01 — Introdução: Redes Neurais como Aproximadores Universais**
  * Histórico do Conexionismo vs. IA Simbólica (McCulloch & Pitts, Rosenblatt, Minsky & Papert, Rumelhart).
  * O modelo do Perceptron: hiperplanos, limites de separabilidade linear e o problema do XOR.
  * O papel da não linearidade e introdução ao treinamento por gradiente descendente.
* **Aula 02 — Capacidade de Representação & Profundidade**
  * Portas de limiar como circuitos lógicos (universalidade booleana e a paridade de $n$ bits).
  * Classificadores geométricos: interseção de semiespaços gerando polítopos convexos.
  * Teorema da Aproximação Universal (Cybenko, Hornik, Leshno).
  * Vantagem da profundidade: particionamento em regiões lineares ($\text{ReLU}$) e reutilização de subexpressões intermediárias.

---

## 🚀 Desafios Práticos

### Desafio 1: Mapa de Características Não Linear para o Perceptron (`phi`)

* **Arquivo de referência:** `desafio1_aluno.py`
* **Objetivo:** Projetar uma transformação de espaço $\phi(X): \mathbb{R}^d \to \mathbb{R}^{d'}$ (com restrição $d < d' \le 64$) que mapeie conjuntos de dados não linearmente separáveis para um espaço de alta dimensão onde um **Perceptron de Rosenblatt com algoritmo Pocket** consiga separar as classes via um hiperplano.

#### ⚙️ Regras do Desafio
* **Não supervisionado:** O método `fit(X)` tem acesso apenas aos dados de entrada $X$ de treino, sem os rótulos $y$.
* **Orçamento de Dimensões:** A dimensão transformada deve satisfazer estritamente $d < d' \le 64$.
* **Desempenho:** Processamento de $10.000$ amostras em menos de $2$ segundos.
* **Robustez:** Ausência de valores `NaN`/`Inf` e garantia de função determinística (`torch.allclose`).

#### 🔬 Arquitetura da Solução Implementada
A classe `Submissao` lineariza o espaço combinando três abordagens geométricas e funcionais:
1. **Normalização dos Dados (Z-Score):** Centralização e padronização com $\epsilon = 10^{-8}$ para estabilidade numérica.
2. **Componentes Geométricas Explícitas:**
   * **Norma Quadrática ($\Vert{}X\Vert{}^2 = \sum x_i^2$):** Lineariza instantaneamente problemas com simetria esférica e anéis concêntricos (`circulos`, `esfera_10d`).
   * **Termo de Interação Cruzada ($x_1 \cdot x_2$):** Converte a alternância de quadrantes do `xor` em um corte linear trivial.
3. **Random Fourier Features (RFF) Multiescala:**
   * Projeções baseadas no Teorema de Bochner utilizando três larguras de banda: fina ($0.25\sigma$), média ($1.0\sigma$) e ampla ($2.0\sigma$).
   * Captura simultaneamente variações de alta frequência (braços da `espiral` e contornos de `duas_luas`) e formas globais suaves.
4. **Blindagem de Determinismo Numérico:**
   * Cálculo das projeções intermediárias de Fourier em precisão dupla (`float64`) para evitar variações de arredondamento em ponto flutuante antes de retornar ao `float32`.

```python
# Exemplo simplificado da transformação implementada
raio_quadrado = (Xs ** 2).sum(dim=1, keepdim=True)
cross = (Xs[:, 0] * Xs[:, 1]).unsqueeze(1)
rff = escala * torch.cos(Xs_64 @ W_64 + b_64).float()
Z = torch.cat([Xs, raio_quadrado, cross, rff], dim=1)  # d' = 64
