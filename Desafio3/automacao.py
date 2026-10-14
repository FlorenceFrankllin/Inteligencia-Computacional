import os
import time
import pandas as pd

# 1. Cria a pasta 'logs' exigida pelo professor na entrega final
os.makedirs("logs", exist_ok=True)

# 2. Cria o ficheiro resumo exigido pelo professor
arquivo_resumo = "experimentos.csv"
with open(arquivo_resumo, "w") as f:
    f.write("fase,otimizador,agenda,lr,semente,perda_final,divergiu,tempo_segundos\n")

# 3. Lista de todas as configurações a testar (Protocolo do Desafio)
comandos = []

# A. BASELINE (3 sementes) - Ponto de partida
for seed in [42, 43, 44]:
    comandos.append({"fase": "baseline", "opt": "AdamW", "agenda": "constante", "lr": 0.001, "seed": seed})

# B. VARREDURA DE TAXA (LR) - 4 Otimizadores x 5 LRs (grade logarítmica)
otimizadores = ["SGD", "Adam", "AdamW", "Muon"]
taxas = [0.1, 0.05, 0.01, 0.005, 0.001]
for opt in otimizadores:
    for lr in taxas:
        comandos.append({"fase": "varredura_lr", "opt": opt, "agenda": "cosseno", "lr": lr, "seed": 42})

# C. VARREDURA DE AGENDAS (Foco nos melhores: Muon e AdamW)
agendas = ["constante", "cosseno", "warmup_cosseno"] 
for opt in ["Muon", "AdamW"]:
    for agenda in agendas:
        # Usa um lr menor para AdamW e maior para Muon
        lr_escolhido = 0.01 if opt == "Muon" else 0.001
        comandos.append({"fase": "varredura_agenda", "opt": opt, "agenda": agenda, "lr": lr_escolhido, "seed": 42})

# Função para extrair a perda e ver se "divergiu"
def avaliar_resultado(nome_arquivo):
    try:
        df = pd.read_csv(nome_arquivo)
        perda = df['perda'].iloc[-1] # Pega no último valor da perda
        # Se a perda for NaN ou gigante, significa que o sinal explodiu (divergiu)
        divergiu = pd.isna(perda) or perda > 1e5 
        return perda, divergiu
    except Exception:
        return "Erro", True

print(f"Total de execuções planeadas: {len(comandos)} testes.")
print("Podes ir beber um café! ☕\n")

# 4. Executa o loop de testes
for c in comandos:
    print(f"A correr: {c['opt']} | {c['agenda']} | lr={c['lr']} | seed={c['seed']}...")
    
    inicio = time.time()
    # Chama o teu ficheiro de treino
    os.system(f"python treino.py --otimizador {c['opt']} --agenda {c['agenda']} --lr {c['lr']} --semente {c['seed']}")
    fim = time.time()
    
    tempo_gasto = round(fim - inicio, 2)
    nome_log = f"logs_{c['opt']}_{c['agenda']}_lr{c['lr']}_seed{c['seed']}.csv"
    
    perda_final, divergiu = avaliar_resultado(nome_log)
    
    # Escreve o resultado no ficheiro de experimentos.csv
    with open(arquivo_resumo, "a") as f:
        f.write(f"{c['fase']},{c['opt']},{c['agenda']},{c['lr']},{c['seed']},{perda_final},{divergiu},{tempo_gasto}\n")
        
    # Move o ficheiro para a pasta logs/ para ficar tudo organizado para a entrega
    if os.path.exists(nome_log):
        # Substitui se já lá existir
        destino = os.path.join("logs", nome_log)
        if os.path.exists(destino): os.remove(destino)
        os.rename(nome_log, destino)

print("\nVarredura concluída! Verifica o teu 'experimentos.csv' e a pasta 'logs/'.")