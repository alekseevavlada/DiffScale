import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
import numpy as np
from tqdm import tqdm

from .model import ConditionalDiffusionModel
from .dataset import CodeScalesDataset

class DiffusionTrainer:
    def __init__(self, config_path: str = "configs/diffusion_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConditionalDiffusionModel(self.config).to(self.device)
        
        # Диффузионные параметры
        self.timesteps = self.config["diffusion"]["timesteps"]
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Оптимизатор
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(self.config["training"]["learning_rate"])
        )
        
        # Датасет
        self.dataset = CodeScalesDataset("data/metadata.csv")
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.config["training"]["batch_size"], 
            shuffle=True
        )
        
        # TensorBoard
        self.writer = SummaryWriter("logs/diffusion")
        
        os.makedirs("checkpoints", exist_ok=True)
    
    def _get_beta_schedule(self):
        schedule = self.config["diffusion"]["beta_schedule"]
        beta_start = self.config["diffusion"]["beta_start"]
        beta_end = self.config["diffusion"]["beta_end"]
        
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, self.timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """Прямой процесс диффузии"""
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt, noise
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch["image"].to(self.device)
            scale_cond = batch["scale_condition"].to(self.device)
            distortion_cond = batch["distortion_condition"].to(self.device)
            offset_cond = batch["offset"].to(self.device)
            
            # Случайные временные шаги
            t = torch.randint(0, self.timesteps, (images.size(0),), device=self.device)
            
            # Прямой процесс
            xt, noise = self.forward_diffusion(images, t)
            
            # Предсказание
            noise_pred = self.model(xt, t, scale_cond, distortion_cond, offset_cond)
            
            # Loss
            loss = nn.MSELoss()(noise_pred, noise)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
        
        avg_loss = total_loss / len(self.dataloader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        
        return avg_loss
    
    def train(self):
        for epoch in range(self.config["training"]["epochs"]):
            loss = self.train_epoch(epoch)
            
            if epoch % self.config["training"]["save_interval"] == 0:
                self.save_checkpoint(epoch, loss)
                
    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config
        }
        
        # Для дообучения
        torch.save(checkpoint, f"checkpoints/diffusion_epoch_{epoch}.pt")

        # Для инференса (только веса)
        torch.save(self.model.state_dict(), "outputs/model.pth")

if __name__ == "__main__":
    trainer = DiffusionTrainer()
    trainer.train()
