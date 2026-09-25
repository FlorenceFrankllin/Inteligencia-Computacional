import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons

# =============================================================================
# 1. Funções de Activação Disponíveis
# =============================================================================
ATIVACOES = {
    "SELU": torch.nn.functional.selu,
    "GELU": torch.nn.functional.gelu,
    "ReLU": torch.relu,
    "LeakyReLU": lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01),
    "SiLU (Swish)": torch.nn.functional.silu,
    "Tanh": torch.tanh,
    "Sigmoid": torch.sigmoid,
    "Identidade": lambda x: x
}

def estimar_ganho(ativ_fn):
    """Estima E[f(z)^2] via Monte Carlo para calibrar o ganho."""
    g = torch.Generator().manual_seed(42)
    z = torch.randn(500_000, generator=g)
    return ativ_fn(z).pow(2).mean().item()

# =============================================================================
# 2. Modelo MLP para Teste de Variância
# =============================================================================
class MLPInterativo(nn.Module):
    def __init__(self, ativ_fn, modo_init="Calibrado (Monte Carlo)", L=48, largura=256, amortecer_ultima=True):
        super().__init__()
        dims = [largura] * (L + 1)
        self.camadas = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(L)])
        self.ativ = ativ_fn

        E_f2 = estimar_ganho(ativ_fn)

        with torch.no_grad():
            for k, lin in enumerate(self.camadas, start=1):
                fan_in = lin.in_features
                
                if modo_init == "Calibrado (Monte Carlo)":
                    desvio = math.sqrt(1.0 / max(fan_in * E_f2, 1e-7))
                    if amortecer_ultima and k == L:
                        desvio *= 0.5
                    lin.weight.normal_(0.0, desvio)
                    lin.bias.zero_()
                    
                elif modo_init == "He Normal (Kaiming)":
                    desvio = math.sqrt(2.0 / fan_in)
                    lin.weight.normal_(0.0, desvio)
                    lin.bias.zero_()
                    
                elif modo_init == "Xavier / Glorot Normal":
                    desvio = math.sqrt(2.0 / (fan_in + lin.out_features))
                    lin.weight.normal_(0.0, desvio)
                    lin.bias.zero_()
                    
                elif modo_init == "Uniforme Pequena U(-0.05, 0.05)":
                    lin.weight.uniform_(-0.05, 0.05)
                    lin.bias.zero_()
                    
                elif modo_init == "Descalibrado N(0, 0.2²)":
                    lin.weight.normal_(0.0, 0.2)
                    lin.bias.zero_()

    def forward(self, x):
        variancias = []
        for k, lin in enumerate(self.camadas):
            x = lin(x)
            variancias.append(x.detach().float().var().item())
            if k < len(self.camadas) - 1:
                x = self.ativ(x)
        return variancias

# =============================================================================
# 3. Construção da Interface com Matplotlib
# =============================================================================
torch.manual_seed(42)
L = 48
largura = 256
lote = 128
x_entrada = torch.randn(lote, largura)
camadas_eixo = np.arange(1, L + 1)

# Estado inicial
estado = {
    "ativ_nome": "SELU",
    "init_nome": "Calibrado (Monte Carlo)",
    "amortecer": True
}

fig, ax = plt.subplots(figsize=(13, 7))
plt.subplots_adjust(left=0.32, bottom=0.15)  # Espaço para o painel de controlo à esquerda

def calcular_variancias():
    modelo = MLPInterativo(
        ativ_fn=ATIVACOES[estado["ativ_nome"]],
        modo_init=estado["init_nome"],
        L=L,
        largura=largura,
        amortecer_ultima=estado["amortecer"]
    )
    return modelo(x_entrada)

# Elementos visuais persistentes
ax.axhline(y=1.0, color="#E9A100", linestyle="-.", linewidth=2, label="Alvo Ideal = 1.0 (Preservado)")
ax.set_yscale("log")
ax.set_ylim(1e-14, 1e14)
ax.set_xlim(0, L + 1)
ax.set_xticks([1, 12, 24, 36, 48])
ax.set_xticklabels(["C1", "C12", "C24", "C36", "C48"])
ax.set_ylabel("Variância da Pré-ativação (Escala Log)")
ax.set_xlabel("Camada Linear")
ax.grid(True, which="major", linestyle="--", alpha=0.5)

# Traçados dinâmicos
barras = ax.bar(camadas_eixo, [1.0]*L, color="#1E8E5A", alpha=0.6, width=0.6)
linha, = ax.plot(camadas_eixo, [1.0]*L, color="#13633e", marker="o", markersize=3, linewidth=1.5)
texto_aviso = ax.text(0.5, 0.92, "", transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold")

def actualizar(_=None):
    vars_novas = calcular_variancias()
    
    # Determina a cor com base no comportamento do sinal
    final_var = vars_novas[-2] if len(vars_novas) > 1 else vars_novas[-1]
    if final_var > 1e4:
        cor = "#B3121B"  # Vermelho: Explodiu
        texto_aviso.set_text("AVISO: O sinal explodiu! Perda geraria NaN no 1º lote.")
        texto_aviso.set_color("#B3121B")
    elif final_var < 1e-4:
        cor = "#B3121B"  # Vermelho: Sumiu
        texto_aviso.set_text("AVISO: O gradiente desapareceu! Variância < 10⁻⁴.")
        texto_aviso.set_color("#B3121B")
    else:
        cor = "#1E8E5A"  # Verde: Estável
        texto_aviso.set_text("Sinal Estável: Propagação saudável ao longo das 48 camadas.")
        texto_aviso.set_color("#1E8E5A")

    for rect, h in zip(barras, vars_novas):
        rect.set_height(h)
        rect.set_color(cor)
        rect.set_alpha(0.65)
        
    linha.set_ydata(vars_novas)
    linha.set_color(cor)
    
    ax.set_title(f"Propagação: {estado['ativ_nome']} + {estado['init_nome']}", fontsize=12, fontweight="bold")
    fig.canvas.draw_idle()

# =============================================================================
# 4. Painéis de Controlo Interativo (Widgets)
# =============================================================================
# Painel da Activação
ax_radio_ativ = plt.axes([0.03, 0.50, 0.23, 0.42], facecolor="#F5F7FB")
radio_ativ = RadioButtons(ax_radio_ativ, list(ATIVACOES.keys()), active=0)

def muda_ativ(rotulo):
    estado["ativ_nome"] = rotulo
    actualizar()
radio_ativ.on_clicked(muda_ativ)
ax_radio_ativ.set_title("1. Função de Activação", fontsize=10, fontweight="bold")

# Painel da Inicialização
inits_lista = [
    "Calibrado (Monte Carlo)",
    "He Normal (Kaiming)",
    "Xavier / Glorot Normal",
    "Uniforme Pequena U(-0.05, 0.05)",
    "Descalibrado N(0, 0.2²)"
]
ax_radio_init = plt.axes([0.03, 0.18, 0.23, 0.28], facecolor="#F5F7FB")
radio_init = RadioButtons(ax_radio_init, inits_lista, active=0)

def muda_init(rotulo):
    estado["init_nome"] = rotulo
    actualizar()
radio_init.on_clicked(muda_init)
ax_radio_init.set_title("2. Inicialização dos Pesos", fontsize=10, fontweight="bold")

# Opção Extra: Amortecer última camada
ax_check = plt.axes([0.03, 0.05, 0.23, 0.08], facecolor="#F5F7FB")
check = CheckButtons(ax_check, ["Amortecer última camada"], [True])

def muda_check(rotulo):
    estado["amortecer"] = check.get_status()[0]
    actualizar()
check.on_clicked(muda_check)

# Arranque inicial
actualizar()
plt.show()