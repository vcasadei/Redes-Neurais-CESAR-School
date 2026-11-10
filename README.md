# Redes Neurais Artificiais e Deep Learning — CESAR School

Material das aulas práticas da disciplina de Redes Neurais Artificiais e Deep Learning da pós-graduação da CESAR School.
Turma **2026.01-A** · Prof. Vitor Casadei ([vcasadei.com](https://vcasadei.com))

Os notebooks seguem a ordem da disciplina: começamos implementando a regressão linear "na mão" com NumPy e, a cada notebook, acrescentamos uma abstração do PyTorch, até treinar redes neurais completas no MNIST.

## Como usar

**No Google Colab (recomendado):** clique no botão "Abrir no Colab" do notebook. Para salvar suas respostas, use *Arquivo › Salvar uma cópia no Drive*. GPU é opcional: *Ambiente de execução › Alterar o tipo de ambiente de execução*.

**Localmente:**

```bash
git clone -b 2026.01-A https://github.com/vcasadei/Redes-Neurais-CESAR-School.git
cd Redes-Neurais-CESAR-School
pip install torch torchvision scikit-learn pandas matplotlib ipywidgets jupyter
jupyter notebook
```

Os dados (Iris, MNIST, Salary Data e os datasets do módulo 5) são baixados pelos próprios notebooks para a pasta `./data`.

## Como os notebooks são organizados

Cada notebook tem **Objetivos**, o conteúdo (dados → modelo → treinamento → avaliação), **Exercícios** e **Aprendizados**.
Nos exercícios, as células com `# TODO: escreva sua solução aqui` são para você completar; nas perguntas, substitua *Escreva sua resposta aqui.* pela sua resposta.
Algumas células seguintes dependem do resultado de um exercício — isso está indicado no enunciado.

## Notebooks

### 2 - Regressão Linear

| Notebook | Conteúdo | Colab |
|---|---|---|
| `2.1_regressao_linear_demo.ipynb` | 2.1 — Regressão Linear: demonstração do gradiente descendente | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/2%20-%20Regress%C3%A3o%20Linear/2.1_regressao_linear_demo.ipynb) |
| `2.2_regressao_linear_numpy.ipynb` | 2.2 — Regressão Linear com NumPy | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/2%20-%20Regress%C3%A3o%20Linear/2.2_regressao_linear_numpy.ipynb) |
| `2.3_regressao_linear_tensores.ipynb` | 2.3 — Regressão Linear com tensores do PyTorch | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/2%20-%20Regress%C3%A3o%20Linear/2.3_regressao_linear_tensores.ipynb) |
| `2.4_regressao_linear_autograd.ipynb` | 2.4 — Regressão linear com autograd e `nn.Linear` | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/2%20-%20Regress%C3%A3o%20Linear/2.4_regressao_linear_autograd.ipynb) |
| `2.5_regressao_linear_otimizador.ipynb` | 2.5 — Regressão linear com `nn.MSELoss` e `optim.SGD` | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/2%20-%20Regress%C3%A3o%20Linear/2.5_regressao_linear_otimizador.ipynb) |
| `2.6_regressao_linear_salarios.ipynb` | 2.6 — Regressão linear com dados reais: salários (treino × validação) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/2%20-%20Regress%C3%A3o%20Linear/2.6_regressao_linear_salarios.ipynb) |

### 3 - Regressão Logística

| Notebook | Conteúdo | Colab |
|---|---|---|
| `3.1_regressao_logistica_iris.ipynb` | 3.1 — Regressão logística (softmax) no Iris: 3 classes | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/3%20-%20Regress%C3%A3o%20Log%C3%ADstica/3.1_regressao_logistica_iris.ipynb) |
| `3.2_regressao_softmax_mnist.ipynb` | 3.2 — Regressão softmax no MNIST | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/3%20-%20Regress%C3%A3o%20Log%C3%ADstica/3.2_regressao_softmax_mnist.ipynb) |
| `3.3_datasets_e_dataloaders.ipynb` | 3.3 — Datasets e DataLoaders no PyTorch | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/3%20-%20Regress%C3%A3o%20Log%C3%ADstica/3.3_datasets_e_dataloaders.ipynb) |
| `3.4_regressao_softmax_mnist_minibatch.ipynb` | 3.4 — Regressão softmax no MNIST com mini-batches | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/3%20-%20Regress%C3%A3o%20Log%C3%ADstica/3.4_regressao_softmax_mnist_minibatch.ipynb) |

### 4 - Redes Neurais

| Notebook | Conteúdo | Colab |
|---|---|---|
| `4.1_funcoes_de_ativacao.ipynb` | 4.1 — Funções de ativação | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/4%20-%20Redes%20Neurais/4.1_funcoes_de_ativacao.ipynb) |
| `4.2_mlp_iris_fronteira_de_decisao.ipynb` | 4.2 — Rede neural com camadas escondidas: fronteira de decisão no Iris | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/4%20-%20Redes%20Neurais/4.2_mlp_iris_fronteira_de_decisao.ipynb) |
| `4.3_underfitting_overfitting_regularizacao.ipynb` | 4.3 — Underfitting, overfitting e regularização | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/4%20-%20Redes%20Neurais/4.3_underfitting_overfitting_regularizacao.ipynb) |
| `4.4_otimizadores_regressao_linear.ipynb` | 4.4 — Otimizadores do gradiente descendente na regressão linear | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/4%20-%20Redes%20Neurais/4.4_otimizadores_regressao_linear.ipynb) |
| `4.5_otimizadores_comparacao.ipynb` | 4.5 — Comparação de otimizadores numa rede neural | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/4%20-%20Redes%20Neurais/4.5_otimizadores_comparacao.ipynb) |
| `4.6_mnist_pipeline_completo.ipynb` | 4.6 — Pipeline completo de treinamento no MNIST | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/4%20-%20Redes%20Neurais/4.6_mnist_pipeline_completo.ipynb) |
| `4.7_extra_autoencoder.ipynb` | 4.7 — Autoencoder (extra) | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/4%20-%20Redes%20Neurais/4.7_extra_autoencoder.ipynb) |

### 5 - Exemplos de Aplicação

| Notebook | Conteúdo | Colab |
|---|---|---|
| `5.1_doacao_de_sangue.ipynb` | 5.1 — Doação de sangue: classificação com rede densa | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/5%20-%20Exemplos%20de%20Aplica%C3%A7%C3%A3o/5.1_doacao_de_sangue.ipynb) |
| `5.2_aprovacao_de_estudantes.ipynb` | 5.2 — Aprovação de estudantes: classificação com rede densa | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/5%20-%20Exemplos%20de%20Aplica%C3%A7%C3%A3o/5.2_aprovacao_de_estudantes.ipynb) |
| `5.3_preco_de_imoveis.ipynb` | 5.3 — Preço de imóveis: regressão com rede densa | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/5%20-%20Exemplos%20de%20Aplica%C3%A7%C3%A3o/5.3_preco_de_imoveis.ipynb) |
| `5.4_mnist_rede_densa.ipynb` | 5.4 — MNIST com rede densa | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vcasadei/Redes-Neurais-CESAR-School/blob/2026.01-A/5%20-%20Exemplos%20de%20Aplica%C3%A7%C3%A3o/5.4_mnist_rede_densa.ipynb) |

## Slides

Os slides da disciplina estão em [`1 - Intro`](1%20-%20Intro).
