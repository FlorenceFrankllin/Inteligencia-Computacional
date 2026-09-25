import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# -------------------------------------------------------------
# 1. Definições da sua solução (autocontida)
# -------------------------------------------------------------
def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.selu(x)

# Calibração via Monte Carlo
_g = torch.Generator().manual_seed(42)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()

@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    if camada == n_camadas:
        desvio *= 0.5
    W.normal_(0.0, desvio)
    b.zero_()

# -------------------------------------------------------------
# 2. Modelo MLP que rastreia a variância em 48 camadas
# -------------------------------------------------------------
class MLPTeste(nn.Module):
    def __init__(self, L=48, largura=256, modo="aluno"):
        super().__init__()
        dims = [largura] * (L + 1)
        self.camadas = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(L)])
        self.modo = modo

        with torch.no_grad():
            for k, lin in enumerate(self.camadas, start=1):
                if modo == "aluno":
                    inicializar(lin.weight, lin.bias, lin.in_features, lin.out_features, k, L)
                elif modo == "baseline":  # tanh + U(-0.05, 0.05)
                    lin.weight.uniform_(-0.05, 0.05)
                    lin.bias.zero_()
                elif modo == "explode":   # ReLU + N(0, 0.2^2)
                    lin.weight.normal_(0.0, 0.2)
                    lin.bias.zero_()

    def forward(self, x):
        variancias = []
        for k, lin in enumerate(self.camadas):
            x = lin(x)
            variancias.append(x.detach().float().var().item())
            if k < len(self.camadas) - 1:
                if self.modo == "aluno":
                    x = ativacao(x)
                elif self.modo == "baseline":
                    x = torch.tanh(x)
                elif self.modo == "explode":
                    x = torch.relu(x)
        return variancias

# -------------------------------------------------------------
# 3. Execução dos testes
# -------------------------------------------------------------
torch.manual_seed(42)
L = 48
largura = 256
lote = 128

x_in = torch.randn(lote, largura)

vars_aluno = MLPTeste(L=L, largura=largura, modo="aluno")(x_in)
vars_base = MLPTeste(L=L, largura=largura, modo="baseline")(x_in)
vars_exp = MLPTeste(L=L, largura=largura, modo="explode")(x_in)

camadas = np.arange(1, L + 1)

# -------------------------------------------------------------
# 4. Gráfico Misto: BARRAS + LINHAS
# -------------------------------------------------------------
plt.figure(figsize=(12, 6.2), dpi=300)

# Barras: Sua Solução estável
plt.bar(camadas, vars_aluno, color="#1E8E5A", alpha=0.75, width=0.65, 
        edgecolor="#13633e", label="Sua Solução (SELU Calibrada) — Barras")

# Linha sobre as barras para reforçar a estabilidade da sua solução
plt.plot(camadas, vars_aluno, color="#13633e", linewidth=1.5, marker="o", markersize=3)

# Linhas: Regimes extremos para comparação
plt.plot(camadas, vars_base, color="#B3121B", linestyle="--", linewidth=2.4, 
         label="Baseline: tanh + U(-0.05, 0.05) [Gradiente some]")
plt.plot(camadas, vars_exp, color="#D12E6E", linestyle=":", linewidth=2.4, 
         label="Descalibrado: ReLU + N(0, 0.2²) [Perda explode / NaN]")

# Linha dourada de referência ideal (variância = 1)
plt.axhline(y=1.0, color="#E9A100", linestyle="-.", linewidth=2, 
            label="Alvo Teórico: Variância = 1.0 (Sinal Preservado)")

# Escala logarítmica (essencial para cobrir de 10^-12 a 10^12)
plt.yscale("log")
plt.ylim(1e-13, 1e13)
plt.xlim(0, L + 1)

# Configuração dos eixos e títulos
plt.xticks(ticks=[1, 12, 24, 36, 48], labels=["Camada 1", "Camada 12", "Camada 24", "Camada 36", "Camada 48"], fontsize=11)
plt.yticks(ticks=[1e12, 1e6, 1, 1e-6, 1e-12], 
           labels=[r"$10^{12}$", r"$10^{6}$", r"$10^{0} (1.0)$", r"$10^{-6}$", r"$10^{-12}$"], fontsize=11)

plt.title("Propagação de Variância Pré-Ativação (L = 48 Camadas)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Profundidade da Rede", fontsize=12)
plt.ylabel("Variância da Pré-ativação (Escala Log)", fontsize=12)

plt.grid(True, which="major", linestyle="--", alpha=0.5)
plt.legend(loc="upper right", fontsize=10, framealpha=0.95)
plt.tight_layout()

# Salva a figura
nome_arquivo = "variancia_barras_linhas.png"
plt.savefig(nome_arquivo)
print(f"Gráfico em barras e linhas salvo com sucesso como '{nome_arquivo}'!")