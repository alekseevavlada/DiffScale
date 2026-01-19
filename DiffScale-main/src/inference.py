import os

import matplotlib.pyplot as plt
import torch
import yaml

from .model import ConditionalDiffusionModel


def load_config(config_path="configs/diffusion_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def sample(model, config, num_samples=1, start_offset=0, scale_type=1, distortion_type=0, device='cpu'):
    """Генерация сэмплов с простым обратным процессом (DDPM)"""
    model.eval()

    # Параметры диффузии из конфига 
    T = config["diffusion"]["timesteps"]
    beta_start = config["diffusion"]["beta_start"]
    beta_end = config["diffusion"]["beta_end"]
    
    # Бета-расписание
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    # Коэффициенты для обратного процесса
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # Подготовка условий
    x = torch.randn(num_samples, 1, config["data"]["image_height"], config["data"]["image_width"]).to(device)
    scale_cond = torch.full((num_samples,), scale_type, dtype=torch.long).to(device)
    distortion_cond = torch.full((num_samples,), distortion_type, dtype=torch.long).to(device)

    with torch.no_grad():
        for t in reversed(range(T)):
            t_batch = torch.full((num_samples,), t, dtype=torch.long).to(device)
            offset_cond = torch.full((num_samples,), start_offset % config["data"]["image_width"], dtype=torch.long).to(device)
            noise_pred = model(x, t_batch, scale_cond, distortion_cond, offset_cond)
            
            # Коэффициенты на шаге t
            alpha_t = alphas[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alphas_cumprod[t]
            
            # Предсказание x0
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_t
            
            # Среднее для обратного шага
            if t == 0:
                x = x0_pred
            else:
                # Стандартная формула DDPM
                mean = (
                    sqrt_recip_alphas[t] * (x - (1 - alpha_t) / sqrt_one_minus_alpha_bar_t * noise_pred)
                )
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(posterior_variance[t])
                x = mean + sigma_t * noise
                
    return x

def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Загрузка модели
    model = ConditionalDiffusionModel(config)

    # Загружаем только веса
    state_dict = torch.load("outputs/model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Папка для сэмплов
    os.makedirs("outputs/samples", exist_ok=True)

    # Генерация для разных условий
    sample_configs = [
        # (scale_type, distortion_type, prefix)
        (1, 0, "incremental_blur"),
        (1, 1, "incremental_noise"),
        (1, 2, "incremental_scratch"),
        (1, 6, "incremental_spot")
    ]

    for scale_type, distortion_type, prefix in sample_configs:
        samples = sample(
            model,
            config,
            num_samples=5,
            scale_type=scale_type,
            distortion_type=distortion_type,
            device=device
        )
        samples = samples.cpu().numpy()
        for i in range(samples.shape[0]):
            plt.imsave(f"outputs/samples/{prefix}_{i}.png", samples[i, 0], cmap="gray")

if __name__ == "__main__":
    main()
