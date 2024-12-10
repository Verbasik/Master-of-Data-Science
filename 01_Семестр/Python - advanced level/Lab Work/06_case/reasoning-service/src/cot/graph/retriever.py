#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Path: cot/graph/retriever.py

# Стандартные импорты Python
import logging
import os
import pickle
from typing import Dict, Tuple, List, Optional, Union

# Импорты для работы с данными
import numpy as np

# Импорты для векторного поиска
import faiss
from sentence_transformers import SentenceTransformer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Description:
        Компонент для семантического поиска в графе с использованием
        векторных эмбеддингов. Поддерживает кэширование и GPU-ускорение.

    Args:
        embedder_name: Название модели sentence-transformers для эмбеддингов
        cache_dir: Директория для хранения кэша эмбеддингов
        use_gpu: Флаг использования GPU для ускорения поиска

    Raises:
        ValueError: При некорректном названии модели или пути к кэшу
        RuntimeError: При ошибке инициализации GPU

    Examples:
        >>> retriever = Retriever('all-MiniLM-L6-v2')
        >>> retriever.init_index(['документ 1', 'документ 2'], 'test_cache')
        >>> results = retriever.search('поисковый запрос', k=1)
    """

    def __init__(
        self,
        embedder_name: str,
        cache_dir: str = ".cache",
        use_gpu: bool = False
    ) -> None:
        self.model = SentenceTransformer(embedder_name)
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        self.index: Optional[Union[faiss.IndexFlatIP, faiss.GpuIndexFlatIP]] = None
        self.doc_lookup: List[str] = []

        # Создаем директорию для кэша, если она не существует
        os.makedirs(cache_dir, exist_ok=True)

    def init_index(self, docs: List[str], cache_key: str) -> None:
        """
        Description:
            Инициализирует поисковый индекс для списка документов с поддержкой
            кэширования вычисленных эмбеддингов.

        Args:
            docs: Список документов для индексации
            cache_key: Ключ для кэширования эмбеддингов

        Raises:
            ValueError: Если список документов пуст
            OSError: При ошибках работы с кэш-файлом

        Examples:
            >>> retriever = Retriever('all-MiniLM-L6-v2')
            >>> retriever.init_index(['текст 1', 'текст 2'], 'my_cache')
        """
        if not docs:
            raise ValueError("Список документов не может быть пустым")

        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        # Пытаемся загрузить эмбеддинги из кэша
        if os.path.exists(cache_path):
            logger.info("Loading embeddings from cache...")
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            # Вычисляем эмбеддинги и сохраняем в кэш
            logger.info("Computing embeddings...")
            embeddings = self.model.encode(docs, show_progress_bar=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)

        self._build_index(embeddings)
        self.doc_lookup = docs

    def _build_index(self, embeddings: np.ndarray) -> None:
        """
        Description:
            Создает FAISS индекс из матрицы эмбеддингов с опциональным
            использованием GPU.

        Args:
            embeddings: Матрица эмбеддингов документов

        Raises:
            RuntimeError: При ошибке инициализации GPU
            ValueError: При некорректной размерности эмбеддингов

        Examples:
            >>> embeddings = np.random.rand(10, 384)  # 10 документов, 384 размерность
            >>> retriever = Retriever('all-MiniLM-L6-v2')
            >>> retriever._build_index(embeddings)
        """
        # Получаем размерность эмбеддингов из входной матрицы
        dim = embeddings.shape[1]
        
        # Создаем базовый CPU индекс
        self.index = faiss.IndexFlatIP(dim)

        # Переносим индекс на GPU если требуется
        if self.use_gpu:
            if faiss.get_num_gpus() > 0:
                self.index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(),
                    0,
                    self.index
                )
            else:
                logger.warning("GPU not available, using CPU index instead")

        # Добавляем векторы в индекс
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Description:
            Выполняет семантический поиск ближайших документов к запросу.

        Args:
            query: Поисковый запрос
            k: Количество ближайших документов для поиска

        Returns:
            Список пар (документ, score), отсортированный по убыванию score

        Raises:
            ValueError: Если индекс не инициализирован или k некорректен
            RuntimeError: При ошибке поиска

        Examples:
            >>> retriever = Retriever('all-MiniLM-L6-v2')
            >>> retriever.init_index(['документ о кошках', 'документ о собаках'], 'test')
            >>> results = retriever.search('кошка', k=1)
            >>> print(results[0][0])  # Первый найденный документ
            'документ о кошках'
        """
        if self.index is None:
            raise ValueError("Индекс не инициализирован")
        if k <= 0:
            raise ValueError("k должно быть положительным числом")

        # Вычисляем эмбеддинг запроса
        query_vector = self.model.encode([query])[0].reshape(1, -1)
        
        # Выполняем поиск в индексе
        distances, indices = self.index.search(query_vector, k)

        # Формируем результаты
        results = [
            (self.doc_lookup[idx], float(score))
            for idx, score in zip(indices[0], distances[0])
        ]

        return results