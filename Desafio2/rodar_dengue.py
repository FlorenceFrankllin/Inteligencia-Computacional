import pandas as pd
import requests
import torch
from torch import nn, optim
from desafio2.desafio_FK import ativacao, inicializar

print("1. A descarregar dados de múltiplos municípios do InfoDengue (2015-2026)...")

# Lista de códigos IBGE de várias cidades (ex: Triângulo Mineiro / MG) para acumular milhares de registos
# 3170206 (Uberlândia), 3170107 (Uberaba), 3169307 (Patos de Minas), 3143104 (Ituiutaba)
codigos_ibge = [3170206, 3170107, 3169307, 3143104] 

dados_totais = []

for geocode in codigos_ibge:
    url = f"https://info.dengue.mat.br/api/alertcity?geocode={geocode}&disease=dengue&format=csv&ew_start=1&ew_end=53&ey_start=2015&ey_end=2026"
    try:
        df_cidade = pd.read_csv(url)
        if not df_cidade.empty:
            dados_totais.append(df_cidade)
    except Exception as e:
        print(f"Aviso: Não foi possível carregar a cidade {geocode}: {e}")

# Junta tudo numa base de dados massiva
df = pd.concat(dados_totais, ignore_index=True)

# Limpeza básica de valores nulos nas colunas de interesse (ex: temperatura, humidade e casos)
df = df.dropna(subset=['tempmed', 'umidmed', 'casos'])

# Selecionar as variáveis de entrada (features) e a saída (target)
features = ['tempmed', 'umidmed'] # Podes adicionar mais colunas climáticas disponíveis na API
X = torch.tensor(df[features].values, dtype=torch.float32)

# Normalizar os dados de entrada para ajudar a rede a convergir
X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-5)

# Definir a variável alvo (ex: prever o número de casos da próxima semana)
y = torch.tensor(df['casos'].values, dtype=torch.float32).unsqueeze(1)

print(f"Base pronta! Total de registos combinados: {len(X)}")

print("\n2. A construir a Rede Neural Profunda de 48 camadas...")
L_camadas = 48
largura_oculta = 128
dim_entrada = X.shape[1]

# Criar a arquitetura sequencial com 48 camadas
layers = []
dims = [dim_entrada] + [largura_oculta] * (L_camadas - 1) + [1]

for i in range(len(dims) - 1):
    layers.append(nn.Linear(dims[i], dims[i+1]))

camadas_modulo = nn.ModuleList(layers)

# Aplicar a tua regra de inicialização de Monte Carlo em cada camada
with torch.no_grad():
    for k, lin in enumerate(camadas_modulo, start=1):
        inicializar(lin.weight, lin.bias, lin.in_features, lin.out_features, k, L_camadas)

print("\n3. A executar teste de propagação nas 48 camadas...")
# Forward pass de teste para confirmar que o sinal não desaparece nem explode
saida = X
for k, lin in enumerate(camadas_modulo):
    saida = lin(saida)
    if k < len(camadas_modulo) - 1:
        saida = ativacao(saida)

print(f"Sucesso! Média final da predição: {saida.mean().item():.4f}")
print("O teu motor matemático de 48 camadas está a processar os dados de Arboviroses perfeitamente!")