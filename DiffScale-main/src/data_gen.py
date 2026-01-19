import os
from typing import List, Tuple

from scipy.ndimage import shift as nd_shift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


class CodeScaleGenerator:
    def __init__(self, config_path: str = "DiffScale-main/configs/diffusion_config.yaml"):
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

    def generate_incremental_scale(self, position: float) -> np.ndarray:
        """Генерация инкрементальной шкалы с субпиксельным сдвигом"""
        # Создаём базовую шкалу без сдвига
        base_scale = np.zeros((self.height, self.width), dtype=np.float32)
        
        x_pos = 0
        while x_pos < self.width:
            end_x = min(x_pos + self.line_width, self.width)
            if x_pos < self.width:
                base_scale[:, int(x_pos):int(end_x)] = 1.0
            x_pos += self.mark_interval
        
        # Сдвигаем шкалу на position пикселей с помощью интерполяции
        shifted_scale = nd_shift(base_scale, position, mode='wrap', order=3)
        
        return shifted_scale
    
    def generate_coherent_sequence(
        self,
        base_offset: int = 0,
        seq_length: int = 30,
        distortion_type: str = "scratch",
        seed: int = None
    ) -> List[np.ndarray]:
        """Генерация физически согласованной последовательности с первыми 10 статичными кадрами"""
        if seed is not None:
            np.random.seed(seed)
        
        # Генерируем идеальную последовательность
        ideal_frames = []
        
        for t in range(seq_length):
            if t < 10:
                # Первые 10 кадров статичные (одинаковое смещение)
                absolute_pos = float(base_offset)
            else:
                # Остальные кадры двигаются с шагом 0.5 px
                absolute_pos = base_offset + (t - 10) * 0.5
            
            frame = self.generate_incremental_scale(absolute_pos)
            ideal_frames.append(frame)

        # Применяем одно и то же искажение ко всем кадрам
        if distortion_type in ["scratch", "spot", "lighting"]:
            # Создаём параметры дефекта один раз
            defect_params = self._sample_defect_params(distortion_type)
            distorted_frames = [
                self._apply_defect_with_params(frame, distortion_type, defect_params)
                for frame in ideal_frames
            ]
        else:
            # Для шума/blur можно применять независимо
            distorted_frames = [
                self.apply_distortions(frame, distortion_type)
                for frame in ideal_frames
            ]

        return distorted_frames

    def _sample_defect_params(self, distortion_type: str):
        """Сохраняет параметры дефекта для воспроизводимости"""
        if distortion_type == "scratch":
            return {
                "pos": np.random.randint(0, self.height),
                "width": np.random.randint(1, 3),
                "intensity": np.random.uniform(0.3, 0.7)
            }
        elif distortion_type == "spot":
            return {
                "y": np.random.randint(0, self.height),
                "x": np.random.randint(0, self.width),
                "h": np.random.randint(1, min(3, self.height)),
                "w": np.random.randint(1, min(4, self.width)),
                "intensity": np.random.choice([
                    np.random.uniform(0.0, 0.3),
                    np.random.uniform(0.7, 1.0)
                ])
            }
        elif distortion_type == "lighting":
            return {
                "profile": np.linspace(0.7, 1.3, self.width)
            }
        return {}
            
    def _apply_defect_with_params(self, image: np.ndarray, distortion_type: str, params: dict):
        """Применяет дефект с заданными параметрами"""
        distorted = image.copy()
        if distortion_type == "scratch":
            y = params["pos"]
            distorted[y:y+params["width"], :] *= params["intensity"]
        elif distortion_type == "spot":
            distorted[
                params["y"]:params["y"]+params["h"],
                params["x"]:params["x"]+params["w"]
            ] = params["intensity"]
        elif distortion_type == "lighting":
            distorted *= params["profile"].reshape(1, -1)
        return np.clip(distorted, 0, 1)
    
    def apply_distortions(self, image: np.ndarray, distortion_type: str) -> np.ndarray:
        """Применение искажений к изображению"""
        distorted = image.copy()
        
        if distortion_type == "blur":
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            for i in range(self.height):
                distorted[i] = np.convolve(distorted[i], kernel, mode="same")
                
        elif distortion_type == "noise":
            noise = np.random.normal(0, 0.1, distorted.shape)
            distorted = distorted + noise
            
        elif distortion_type == "scratch":
            scratch_pos = np.random.randint(0, self.height)
            scratch_width = np.random.randint(1, 3)
            scratch_intensity = np.random.uniform(0.3, 0.7)
            distorted[scratch_pos:scratch_pos+scratch_width, :] *= scratch_intensity
            
        elif distortion_type == "lighting":
            lighting_profile = np.linspace(0.7, 1.3, self.width)
            distorted *= lighting_profile.reshape(1, -1)

        elif distortion_type == "spot":
            num_spots = np.random.randint(1, 4)  
            for _ in range(num_spots):
                y = np.random.randint(0, self.height)
                x = np.random.randint(0, self.width)
                spot_h = np.random.randint(1, min(3, self.height - y + 1))
                spot_w = np.random.randint(1, min(4, self.width - x + 1))
                intensity = np.random.choice([np.random.uniform(0.0, 0.3), np.random.uniform(0.7, 1.0)])
                distorted[y:y+spot_h, x:x+spot_w] = intensity

        elif distortion_type == "noise_scratch_blur":
            distorted = self.apply_distortions(image, "noise")
            distorted = self.apply_distortions(distorted, "scratch")
            distorted = self.apply_distortions(distorted, "blur")
                        
        elif distortion_type == "combined":
            distortions = ["blur", "noise", "scratch", "lighting", "spot"]
            selected = np.random.choice(distortions, size=np.random.randint(3, 4), replace=False)
            for dist_type in selected:
                distorted = self.apply_distortions(distorted, dist_type)
        
        return np.clip(distorted, 0, 1)
    
    def generate_dataset(self, num_sequences: int = 1, seq_length: int = 30):
        """Генерация полного датасета с первыми 10 статичными кадрами"""
        os.makedirs("DiffScale-main/data/clean", exist_ok=True)
        os.makedirs("DiffScale-main/data/distorted", exist_ok=True)
        
        metadata = []
        distortion_types = self.config["generation"]["distortion_types"]

        for seq_id in range(num_sequences):
            base_offset = np.random.randint(0, 20)
            
            # Генерируем идеальную последовательность с помощью нового метода
            clean_sequence = self.generate_coherent_sequence(
                base_offset=base_offset,
                seq_length=seq_length,
                distortion_type="none",  # Нет искажений для чистых кадров
                seed=seq_id
            )
            
            # Для каждого типа искажения создаём свою искажённую последовательность
            for dist_type in distortion_types:
                if dist_type in ["scratch", "spot", "lighting"]:
                    # Фиксированные дефекты: генерируем параметры один раз
                    defect_params = self._sample_defect_params(dist_type)
                    distorted_sequence = [
                        self._apply_defect_with_params(frame, dist_type, defect_params)
                        for frame in clean_sequence
                    ]
                else:
                    # Независимые искажения 
                    distorted_sequence = [
                        self.apply_distortions(frame, dist_type)
                        for frame in clean_sequence
                    ]
                
                # Сохраняем пары
                for t in range(seq_length):
                    clean_path = f"DiffScale-main/data/clean/{seq_id:04d}_frame_{t:02d}.png"
                    distorted_path = f"DiffScale-main/data/distorted/{seq_id:04d}_{dist_type}_frame_{t:02d}.png"

                    # Вычисляем истинное смещение в пикселях
                    if t < 10:
                        current_offset = base_offset  # Статичные кадры
                    else:
                        current_offset = base_offset + (t - 10) * 0.5  # Движущиеся кадры
                    
                    true_dx = current_offset % self.mark_interval
                    
                    # Сохраняем чистый кадр
                    plt.imsave(clean_path, clean_sequence[t], cmap="gray", vmin=0, vmax=1)
                    # Сохраняем искаженный кадр
                    plt.imsave(distorted_path, distorted_sequence[t], cmap="gray", vmin=0, vmax=1)

                    metadata.append({
                        "sequence_id": seq_id,
                        "frame_id": t,
                        "scale_type": "incremental",
                        "distortion_type": dist_type,
                        "offset": current_offset,
                        "true_dx": true_dx,
                        "clean_path": clean_path,
                        "distorted_path": distorted_path
                    })
        
        # Сохранение метаданных
        df = pd.DataFrame(metadata)
        df.to_csv("DiffScale-main/data/metadata.csv", index=False)
        print(f"Сгенерировано {len(metadata)} кадров.")

if __name__ == "__main__":
    generator = CodeScaleGenerator()
    generator.generate_dataset()  