import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import os

class CustomTransforms:
    """Кастомные трансформы заменяющие torchvision.transforms"""
    
    @staticmethod
    def to_tensor(pic):
        """Замена transforms.ToTensor()"""
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic)
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
            img = img.permute((2, 0, 1)).contiguous()
        
        return img.float().div(255)
    
    @staticmethod
    def normalize(tensor, mean=0.5, std=0.5):
        """Замена transforms.Normalize()"""
        tensor = tensor.clone()
        tensor.sub_(mean).div_(std)
        return tensor
    
    @staticmethod
    def resize(img, size, interpolation=Image.BILINEAR):
        """Замена transforms.Resize()"""
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)
    
    @staticmethod
    def random_horizontal_flip(img, p=0.5):
        """Случайное отражение по горизонтали"""
        if np.random.random() < p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class CodeScalesDataset(Dataset):
    def __init__(self, metadata_path: str, transform=None, is_training: bool = True):
        self.metadata = pd.read_csv(metadata_path)
        self.is_training = is_training
        
        # Кастомные трансформы по умолчанию
        self.transform = transform or self.default_transform()
        
        # Маппинг условий
        self.scale_type_map = {"prbs": 0, "incremental": 1}
        self.distortion_map = {
            "blur": 0,
            "noise": 1,
            "scratch": 2,
            "lighting": 3,
            "combined": 4,
            "noise_scratch_blur": 5,
            "spot": 6  
        }
    
    def default_transform(self):
        """Трансформы по умолчанию"""
        def transform_func(image):
            # Конвертируем в tensor
            if isinstance(image, Image.Image):
                img_array = np.array(image, dtype=np.float32)
            else:
                img_array = image.astype(np.float32)
            
            # Нормализация если нужно
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            
            # Добавляем channel dimension
            if len(img_array.shape) == 2:
                img_array = img_array[np.newaxis, :, :]
            
            return torch.tensor(img_array)
        
        return transform_func
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Загрузка изображения
        image_path = row["distorted_path"] if self.is_training else row["ideal_path"]
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        # Условия
        scale_type = torch.tensor(self.scale_type_map[row["scale_type"]], dtype=torch.long)
        distortion_type = torch.tensor(self.distortion_map[row["distortion_type"]], dtype=torch.long)
        offset_cond = torch.tensor(row["offset"], dtype=torch.long)
        
        return {
            "image": image,
            "scale_condition": scale_type,
            "distortion_condition": distortion_type,
            "offset": offset_cond
        }
    