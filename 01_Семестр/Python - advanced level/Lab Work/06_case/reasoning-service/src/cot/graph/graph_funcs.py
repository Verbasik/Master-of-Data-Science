#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Path: cot/graph/graph_funcs.py

# Стандартные импорты Python
from typing import Dict, List, Tuple, Optional, Any, Union

class GraphFunctions:
    """
    Description:
        Класс, предоставляющий функции для работы с графом. Реализует
        базовые операции доступа к узлам и их свойствам.

    Args:
        graph: Словарь, представляющий структуру графа. Ключи - типы узлов,
              значения - словари узлов с их данными.

    Raises:
        KeyError: Если запрашиваемый узел не найден в графе
        ValueError: Если структура графа некорректна

    Examples:
        >>> graph_data = {
        ...     'user': {
        ...         'u1': {'features': {'name': 'John'}, 'neighbors': {'follows': ['u2']}}
        ...     }
        ... }
        >>> graph = GraphFunctions(graph_data)
        >>> graph.get_node_feature('u1', 'name')
        'John'
    """

    def __init__(self, graph: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        self.graph = graph
        # Создаем индекс при инициализации
        self._build_index()

    def _build_index(self) -> None:
        """
        Description:
            Создает индекс для быстрого доступа к узлам графа. Индекс хранит
            кортежи (тип_узла, данные_узла) для каждого идентификатора узла.

        Raises:
            ValueError: Если структура графа не соответствует ожидаемому формату

        Examples:
            >>> graph = GraphFunctions({'user': {'u1': {'features': {}, 'neighbors': {}}}})
            >>> 'u1' in graph.node_index
            True
        """
        self.node_index: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        
        # Проходим по всем типам узлов и их данным
        for node_type in self.graph:
            for node_id, node_data in self.graph[node_type].items():
                self.node_index[node_id] = (node_type, node_data)

    def get_node_feature(
        self,
        node_id: str,
        feature: str
    ) -> Optional[str]:
        """
        Description:
            Получает значение указанного атрибута узла.

        Args:
            node_id: Идентификатор узла
            feature: Название запрашиваемого атрибута

        Returns:
            Значение атрибута узла или None, если атрибут не найден

        Raises:
            KeyError: Если узел с указанным идентификатором не существует

        Examples:
            >>> graph = GraphFunctions({
            ...     'user': {
            ...         'u1': {'features': {'name': 'John'}, 'neighbors': {}}
            ...     }
            ... })
            >>> graph.get_node_feature('u1', 'name')
            'John'
        """
        # Получаем информацию об узле из индекса
        node_type, node_data = self.node_index.get(node_id, (None, None))
        
        if node_data is None:
            raise KeyError(f"Node {node_id} not found")
            
        # Возвращаем значение атрибута
        return node_data['features'].get(feature)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """
        Description:
            Получает список соседних узлов для указанного узла.

        Args:
            node_id: Идентификатор узла
            edge_type: Тип связи для фильтрации соседей (опционально)

        Returns:
            Список идентификаторов соседних узлов или словарь соседей по типам связей,
            если тип связи не указан

        Raises:
            KeyError: Если узел с указанным идентификатором не существует

        Examples:
            >>> graph = GraphFunctions({
            ...     'user': {
            ...         'u1': {'features': {}, 'neighbors': {'follows': ['u2']}}
            ...     }
            ... })
            >>> graph.get_neighbors('u1', 'follows')
            ['u2']
        """
        # Получаем информацию об узле из индекса
        node_type, node_data = self.node_index.get(node_id, (None, None))
        
        if node_data is None:
            raise KeyError(f"Node {node_id} not found")

        # Возвращаем соседей в зависимости от указанного типа связи
        if edge_type:
            return node_data['neighbors'].get(edge_type, [])
        return node_data['neighbors']

    def get_node_degree(self, node_id: str, edge_type: str) -> int:
        """
        Description:
            Вычисляет степень узла по указанному типу связи.

        Args:
            node_id: Идентификатор узла
            edge_type: Тип связи для подсчета степени

        Returns:
            Количество связей указанного типа для данного узла

        Raises:
            KeyError: Если узел с указанным идентификатором не существует

        Examples:
            >>> graph = GraphFunctions({
            ...     'user': {
            ...         'u1': {'features': {}, 'neighbors': {'follows': ['u2', 'u3']}}
            ...     }
            ... })
            >>> graph.get_node_degree('u1', 'follows')
            2
        """
        # Получаем список соседей и возвращаем их количество
        neighbors = self.get_neighbors(node_id, edge_type)
        return len(neighbors)