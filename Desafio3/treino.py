
import argparse
import math
import pandas as pd
import torch
from torch import nn

# =============================================================================
# 1. FUNÇÕES DO DESAFIO 2 (O MOTOR)
# =============================================================================
def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.selu(x)

_g = torch.Generator().manual_seed(42)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()

@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor, fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    if camada == n_camadas:
        desvio *= 0.5
    W.normal_(0.0, desvio)
    b.zero_()

# =============================================================================
# 2. FUNÇÃO DO DESAFIO 3 (O MUON)
# =============================================================================
def ortogonaliza(G, passos=5):
    """Ortogonalização de Newton-Schulz exigida para o otimizador Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T: 
        X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if T else X

# =============================================================================
# 3. CONFIGURAÇÃO (ARGUMENTS) PARA A VARREDURA
# =============================================================================
parser = argparse.ArgumentParser(description="Desafio 3 - O Melhor Treino")
parser.add_argument("--otimizador", type=str, default="Muon", choices=["SGD", "Adam", "AdamW", "Muon", "RMSprop"])
parser.add_argument("--agenda", type=str, default="cosseno", choices=["constante", "cosseno", "warmup_cosseno"])
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--semente", type=int, default=42)
args = parser.parse_args()

torch.manual_seed(args.semente)

# =============================================================================
# 4. DADOS (INFODENGUE)
# =============================================================================
print("1. A carregar os dados...")
codigos_ibge = [3170206, 3170107, 3169307, 3143104] 
dados_totais = []

for geocode in codigos_ibge:
    url = f"https://info.dengue.mat.br/api/alertcity?geocode={geocode}&disease=dengue&format=csv&ew_start=1&ew_end=53&ey_start=2015&ey_end=2026"
    try:
        df_cidade = pd.read_csv(url)
        if not df_cidade.empty:
            dados_totais.append(df_cidade)
    except Exception as e:
        pass # Ignora erros de rede por agora para ser rápido

df = pd.concat(dados_totais, ignore_index=True)
df = df.dropna(subset=['tempmed', 'umidmed', 'casos'])

X = torch.tensor(df[['tempmed', 'umidmed']].values, dtype=torch.float32)
X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-5)
y = torch.tensor(df['casos'].values, dtype=torch.float32).unsqueeze(1)

# Usamos um DataLoader para criar os lotes (batches)
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# =============================================================================
# 5. O MODELO (48 CAMADAS)
# =============================================================================
print("2. A construir a Rede de 48 camadas...")
L_camadas = 48
largura_oculta = 128
dim_entrada = X.shape[1]

layers = []
dims = [dim_entrada] + [largura_oculta] * (L_camadas - 1) + [1]
for i in range(len(dims) - 1):
    layers.append(nn.Linear(dims[i], dims[i+1]))

modelo = nn.ModuleList(layers)

with torch.no_grad():
    for k, lin in enumerate(modelo, start=1):
        inicializar(lin.weight, lin.bias, lin.in_features, lin.out_features, k, L_camadas)

# =============================================================================
# 6. SETUP DOS OTIMIZADORES (REGRA DO MUON) E AGENDAS
# =============================================================================
opts = []

if args.otimizador == "Muon":
    # Regra: Muon só para matrizes 2D das camadas ocultas. AdamW para o resto.
    muon_p = [p for n, p in modelo.named_parameters() if p.ndim == 2 and "47" not in n] # 47 é a última
    ids = {id(p) for p in muon_p}
    adamw_p = [p for p in modelo.parameters() if id(p) not in ids]
    
    opts.append(torch.optim.SGD(muon_p, lr=args.lr, momentum=0.95)) # Muon Base
    opts.append(torch.optim.AdamW(adamw_p, lr=3e-4, weight_decay=0.01))
elif args.otimizador == "AdamW":
    opts.append(torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01))
elif args.otimizador == "Adam": 
    opts.append(torch.optim.Adam(modelo.parameters(), lr=args.lr))
elif args.otimizador == "SGD":
    opts.append(torch.optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9))

scheds = []
if args.agenda == "cosseno":
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=len(dataloader)*10) for o in opts]
# Podes adicionar mais agendas aqui (Constante, OneCycle, etc.)

# =============================================================================
# 7. O TREINO COM REGISTO DE SINAIS (INSTRUMENTAÇÃO)
# =============================================================================
print(f"3. A iniciar o treino ({args.otimizador} | {args.agenda} | lr={args.lr})...")
criterio = nn.MSELoss()
CAMADAS_ALVO = ["0.weight", "24.weight", "47.weight"] # 1ª, Meio e Última (indices 0, 24, 47 na ModuleList)
logs = []
passo_global = 0
k = 20 # A cada 20 passos recolhemos os logs

for epoca in range(10): # 10 épocas para teste
    for batch_X, batch_y in dataloader:
        passo_global += 1
        
        # 1. Forward e Loss
        saida = batch_X
        for i, lin in enumerate(modelo):
            saida = lin(saida)
            if i < len(modelo) - 1:
                saida = ativacao(saida)
        
        loss = criterio(saida, batch_y)
        
        # 2. Zera os gradientes e faz Backward
        for opt in opts: opt.zero_grad()
        loss.backward()
        
        # 3. RECOLHA DE SINAIS ANTES DO STEP E DO CLIPPING (a cada k passos)
        if passo_global % k == 0:
            # Norma do Gradiente (g)
            g = torch.sqrt(sum((p.grad**2).sum() for p in modelo.parameters() if p.grad is not None))
            # Norma dos Parâmetros (theta)
            th = torch.sqrt(sum((p**2).sum() for p in modelo.parameters()))
            # Estado dos pesos antes do passo (para calcular a Razão de Atualização)
            antes = {n: p.detach().clone() for n, p in modelo.named_parameters() if n in CAMADAS_ALVO}
        
        # 4. Aplica o Muon (Ortogonalização) se for o otimizador escolhido
        if args.otimizador == "Muon":
            for p in muon_p:
                if p.grad is not None:
                    p.grad.copy_(ortogonaliza(p.grad))
                    
        # 5. Step nos otimizadores e agendas
        for opt in opts: opt.step()
        if scheds:
            for s in scheds: s.step()
            
        # 6. RECOLHA DA RAZÃO DE ATUALIZAÇÃO E GRAVAÇÃO DO LOG
        if passo_global % k == 0:
            # Razão de Atualização por camada (rho)
            rho = {n: ((p - antes[n]).norm() / antes[n].norm()).item() for n, p in modelo.named_parameters() if n in CAMADAS_ALVO}
            
            # Taxa efetiva atual
            lr_atual = opts[0].param_groups[0]["lr"]
            
            # Grava no CSV virtual
            logs.append({
                "passo": passo_global,
                "perda": loss.item(),
                "grad_norm": g.item(),
                "theta_norm": th.item(),
                "lr": lr_atual,
                **{f"rho_{n}": v for n, v in rho.items()}
            })

print("\n4. Treino concluído! A guardar logs...")
df_logs = pd.DataFrame(logs)
nome_arquivo = f"logs_{args.otimizador}_{args.agenda}_lr{args.lr}_seed{args.semente}.csv"
df_logs.to_csv(nome_arquivo, index=False)
print(f"Sinais guardados em '{nome_arquivo}'. Podes agora gerar os 4 gráficos!")