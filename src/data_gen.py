import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import List, Tuple
import yaml

class CodeScaleGenerator:
    def __init__(self, config_path: str = "configs/diffusion_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config["data"]["image_width"]
        self.height = self.config["data"]["image_height"]

        self.line_width = self.config["generation"]["line_width"]
        self.gap_width = self.config["generation"]["gap_width"]
        self.mark_interval = self.line_width + self.gap_width
        
    # def generate_prbs_scale(self, sequence: List[int], start_pos: int = 0) -> np.ndarray:
    #     """Генерация псевдослучайной шкалы"""
    #     scale = np.zeros((self.height, self.width), dtype=np.float32)
    #     line_width = self.config["generation"]["line_width"]
    #     gap_width = self.config["generation"]["gap_width"]
        
    #     x_pos = start_pos
    #     for bit in sequence:
    #         if bit == 1:
    #             end_x = min(x_pos + line_width, self.width)
    #             if x_pos < self.width:
    #                 scale[:, x_pos:end_x] = 1.0
    #         x_pos += (line_width if bit == 1 else gap_width)
    #         if x_pos >= self.width:
    #             break
                
    #     return scale
    
    def generate_incremental_scale(self, start_value: int = 0) -> np.ndarray:
        """Генерация инкрементальной шкалы"""
        scale = np.zeros((self.height, self.width), dtype=np.float32)

        # Вычисляем смещение, на сколько пикселей сдвинута вся шкала
        offset = start_value % self.mark_interval
        
        # Генерируем метки по всей ширине, но со смещением
        for i in range(0, (self.width + self.mark_interval) // self.mark_interval):
            x_start = i * self.mark_interval - offset
            
            # Проверяем, что метка попадает в видимую область
            if x_start < self.width and x_start + self.line_width > 0:
                actual_start = max(0, x_start)
                actual_end = min(x_start + self.line_width, self.width)
                
                if actual_start < actual_end:
                    scale[:, actual_start:actual_end] = 1.0
                
        return scale
    
    def generate_continuous_incremental_sequence(self, start_value: int = 0, num_images: int = 5) -> List[np.ndarray]:
        """Генерация непрерывной последовательности инкрементных шкал"""
        sequences = []
    
        for i in range(num_images):
            # Плавное смещение для каждого следующего изображения
            current_offset = start_value + i 
            scale = self.generate_incremental_scale(current_offset)
            sequences.append(scale)
        
        return sequences

    def apply_distortions(self, image: np.ndarray, distortion_type: str) -> np.ndarray:
        """Применение искажений к изображению"""
        distorted = image.copy()
        
        if distortion_type == "blur":
            # Motion blur
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            for i in range(self.height):
                distorted[i] = np.convolve(distorted[i], kernel, mode="same")
                
        elif distortion_type == "noise":
            # Gaussian noise
            noise = np.random.normal(0, 0.1, distorted.shape)
            distorted = distorted + noise
            
        elif distortion_type == "scratch":
            # Симуляция царапины
            scratch_pos = np.random.randint(0, self.height)
            scratch_width = np.random.randint(1, 3)
            scratch_intensity = np.random.uniform(0.3, 0.7)
            distorted[scratch_pos:scratch_pos+scratch_width, :] *= scratch_intensity
            
        elif distortion_type == "lighting":
            # Неравномерное освещение
            lighting_profile = np.linspace(0.7, 1.3, self.width)
            distorted *= lighting_profile.reshape(1, -1)

        elif distortion_type == "spot":
            # Генерация одного или нескольких случайных пятен
            num_spots = np.random.randint(1, 4)  
            for _ in range(num_spots):
                # Случайная позиция (y, x)
                y = np.random.randint(0, self.height)
                x = np.random.randint(0, self.width)
                # Размер пятна: 1–3 пикселя по ширине, высота от 1 до 4 
                spot_h = np.random.randint(1, min(3, self.height - y + 1))
                spot_w = np.random.randint(1, min(4, self.width - x + 1))
                # Интенсивность: может быть светлой (ближе к 1) или тёмной (ближе к 0)
                intensity = np.random.choice([np.random.uniform(0.0, 0.3), np.random.uniform(0.7, 1.0)])
                distorted[y:y+spot_h, x:x+spot_w] = intensity

        elif distortion_type == "noise_scratch_blur":
            # Применяем все три искажения последовательно
            distorted = self.apply_distortions(image, "noise")
            distorted = self.apply_distortions(distorted, "scratch")
            distorted = self.apply_distortions(distorted, "blur")
                        
        elif distortion_type == "combined":
            distortions = ["blur", "noise", "scratch", "lighting", "spot"]
            selected = np.random.choice(distortions, size=np.random.randint(3, 4), replace=False)
            for dist_type in selected:
                distorted = self.apply_distortions(distorted, dist_type)
        
        return np.clip(distorted, 0, 1)
    
    def generate_dataset(self, num_samples: int = 1000):
        """Генерация полного датасета"""
        os.makedirs("data/clean", exist_ok=True)
        os.makedirs("data/distorted", exist_ok=True)
        
        metadata = []
        # prbs_sequences = self.config["generation"]["prbs_sequences"]
        distortion_types = self.config["generation"]["distortion_types"]

        seq_length = 5
        for i in range(num_samples):
            # Выбор типа шкалы
            # scale_type = "prbs" if np.random.random() > 0.5 else "incremental"
            
            # if scale_type == "prbs":
            #     sequence = prbs_sequences[np.random.randint(0, len(prbs_sequences))]
            #     start_pos = np.random.randint(0, 20)
            #     image = self.generate_prbs_scale(sequence, start_pos)
            # else:
            #     start_value = np.random.randint(0, 5)
            #     image = self.generate_incremental_scale(start_value)

            base_offset = np.random.randint(0, 20)
            sequence = self.generate_continuous_incremental_sequence(base_offset, seq_length)
            
            for j in range(seq_length):
                image = sequence[j]
                ideal_path = f"data/clean/{i:04d}_{j}.png"
                plt.imsave(ideal_path, image, cmap="gray", vmin=0, vmax=1)
                
                for dist_type in distortion_types:
                    distorted_image = self.apply_distortions(image, dist_type)
                    distorted_path = f"data/distorted/{i:04d}_{j}_{dist_type}.png"
                    plt.imsave(distorted_path, distorted_image, cmap="gray", vmin=0, vmax=1)
                
                metadata.append({
                    "image_id": f"scale_{i:04d}_{j}",
                    "scale_type": "incremental",
                    "distortion_type": dist_type,
                    "offset": int(base_offset + j),
                    "ideal_path": ideal_path,
                    "distorted_path": distorted_path
                })
        
        # Сохранение метаданных
        df = pd.DataFrame(metadata)
        df.to_csv("data/metadata.csv", index=False)
        print(f"Сгенерировано {len(metadata)} изображений")

if __name__ == "__main__":
    generator = CodeScaleGenerator()
    generator.generate_dataset(50)  