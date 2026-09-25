import math
import torch

def ativacao(x: torch.Tensor) -> torch.Tensor:
    # Utilizamos a SiLU (Swish), uma ativação moderna e suave
    return torch.nn.functional.selu(x)

# E[f(z)^2] com z ~ N(0,1), estimado uma vez por amostragem (Monte Carlo)
_g = torch.Generator().manual_seed(42)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()

@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    
    # Aplicamos a fórmula do ganho: s^2 = 1 / (fan_in * E[f(z)^2])
    desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    
    # Reduzimos a escala da última camada (logits) para ajudar o SGD
    if camada == n_camadas:
        desvio *= 0.5
        
    # Preenche in-place, conforme a regra do desafio
    W.normal_(0.0, desvio)
    b.zero_()