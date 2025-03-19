# simplenn/utils/data_utils.py
"""
Модуль с утилитами для предобработки и манипуляции данными.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional

class DataUtils:
    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        """Нормализация данных (значения от 0 до 1)"""
        return X / 255.0 if X.max() > 1 else X

    @staticmethod
    def standardize(X: np.ndarray) -> np.ndarray:
        """Стандартизация данных (среднее = 0, стандартное отклонение = 1)"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-7)  # Добавляем эпсилон для избежания деления на 0

    @staticmethod
    def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
        """Преобразование меток классов в one-hot формат"""
        return np.eye(num_classes)[y.astype(int)]

    @staticmethod
    def train_test_split(
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2, 
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разделение данных на обучающую и тестовую выборки"""
        if random_state is not None:
            np.random.seed(random_state)
        
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        test_count = int(num_samples * test_size)
        
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test

    @staticmethod
    def batch_generator(
        X: np.ndarray, 
        y: np.ndarray, 
        batch_size: int, 
        shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Генератор батчей для обучения"""
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield X[batch_indices], y[batch_indices]

    @staticmethod
    def k_fold_split(
        X: np.ndarray, 
        y: np.ndarray, 
        n_folds: int, 
        random_state: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Разделение данных на k разбиений для кросс-валидации"""
        if random_state is not None:
            np.random.seed(random_state)
        
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        
        fold_size = num_samples // n_folds
        folds = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else num_samples
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            folds.append((X_train, X_val, y_train, y_val))
        
        return folds

    @staticmethod
    def data_augmentation(
        X: np.ndarray, 
        y: np.ndarray, 
        augmentation_functions: List[Callable[[np.ndarray], np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Аугментация данных с помощью заданных функций преобразования"""
        X_augmented = [X]
        y_augmented = [y]
        
        for aug_func in augmentation_functions:
            X_aug = np.array([aug_func(x) for x in X])
            X_augmented.append(X_aug)
            y_augmented.append(y)
        
        return np.vstack(X_augmented), np.concatenate(y_augmented)

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Поворот изображения на заданный угол"""
        from scipy.ndimage import rotate
        return rotate(image, angle, reshape=False)

    @staticmethod
    def flip_image_horizontal(image: np.ndarray) -> np.ndarray:
        """Отражение изображения по горизонтали"""
        return np.fliplr(image)

    @staticmethod
    def add_noise(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Добавление шума к изображению"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1) if image.max() <= 1 else np.clip(noisy_image, 0, 255)