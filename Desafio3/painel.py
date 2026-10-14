import pandas as pd
import matplotlib.pyplot as plt
import os

# As 4 configurações exigidas pelo professor
arquivos = {
    "Baseline (AdamW Const)": "logs/logs_AdamW_constante_lr0.001_seed42.csv",
    "Vencedor & Muon (Muon Cosseno)": "logs/logs_Muon_cosseno_lr0.01_seed42.csv",
    "Alternativa (AdamW Cosseno)": "logs/logs_AdamW_cosseno_lr0.001_seed42.csv",
    "Instável (Muon lr=0.05)": "logs/logs_Muon_cosseno_lr0.05_seed42.csv"
}

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
fig.suptitle("Desafio 3 - Comparação de Otimizadores (InfoDengue)", fontsize=16, fontweight='bold')

cores = ['tab:gray', 'tab:blue', 'tab:green', 'tab:red']

for (rotulo, caminho), cor in zip(arquivos.items(), cores):
    if not os.path.exists(caminho):
        print(f"Aviso: Ficheiro não encontrado - {caminho}")
        continue
        
    df = pd.read_csv(caminho)
    
    # 1. Perda
    ax1.plot(df['passo'], df['perda'], label=rotulo, color=cor, linewidth=2)
    
    # 2. Norma do Gradiente
    ax2.plot(df['passo'], df['grad_norm'], label=rotulo, color=cor, linewidth=2)
    
    # 3. Norma dos Parâmetros
    ax3.plot(df['passo'], df['theta_norm'], label=rotulo, color=cor, linewidth=2)
    
    # 4. Razão de Atualização (usamos apenas a camada do meio para o gráfico não virar esparguete)
    ax4.plot(df['passo'], df['rho_24.weight'], label=rotulo, color=cor, linewidth=2)

# Formatação dos eixos
ax1.set_ylabel('Perda (Loss)')
ax1.set_yscale('log') # Log ajuda a ver melhor a diferença com a instável
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

ax2.set_ylabel('Norma Gradiente $||g||$')
ax2.set_yscale('log')
ax2.grid(True, linestyle='--', alpha=0.6)

ax3.set_ylabel('Norma Parâmetros $||\\theta||$')
ax3.grid(True, linestyle='--', alpha=0.6)

ax4.set_ylabel('Razão Atualização $\\rho$ (Meio)')
ax4.set_xlabel('Passos')
ax4.set_yscale('log')
ax4.axhline(1e-3, color='black', linestyle='--', linewidth=1, label='Ref $10^{-3}$')
ax4.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.subplots_adjust(top=0.94)

nome_saida = "painel_final.pdf"
plt.savefig(nome_saida, format='pdf')
print(f"Sucesso! Gráfico final guardado em '{nome_saida}'.")
plt.show()