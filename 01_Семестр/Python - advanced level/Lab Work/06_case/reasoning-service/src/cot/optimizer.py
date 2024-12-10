#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Path: reasoning-service/src/cot/optimizer.py
from typing import List, Dict, Tuple, Set, Any
from dataclasses import dataclass
from .processor import ThoughtStep, ActionType
import numpy as np
import time


@dataclass
class OptimizationMetrics:
    """
    Description:
        Метрики оптимизации цепочки рассуждений.

    Attributes:
        coherence: Связность рассуждений.
        relevance: Релевантность рассуждений контексту.
        depth: Глубина рассуждений.
        breadth: Широта рассуждений.
        overall_score: Общая оценка качества цепочки.
    """
    coherence: float
    relevance: float
    depth: float
    breadth: float
    overall_score: float


class ChainOptimizer:
    """
    Description:
        Оптимизатор цепочек рассуждений.
        Улучшает качество и связность рассуждений.

    Attributes:
        weights: Веса для расчета общей оценки.
    """

    def __init__(self,
                 coherence_weight: float = 0.3,
                 relevance_weight: float = 0.3,
                 depth_weight: float = 0.2,
                 breadth_weight: float = 0.2):
        """
        Description:
            Инициализация оптимизатора.

        Args:
            coherence_weight: Вес связности.
            relevance_weight: Вес релевантности.
            depth_weight: Вес глубины.
            breadth_weight: Вес широты.
        """
        self.weights = {
            'coherence': coherence_weight,
            'relevance': relevance_weight,
            'depth': depth_weight,
            'breadth': breadth_weight
        }

    def optimize_chain(self,
                       chain: List[ThoughtStep],
                       context: Dict[str, Any]) -> Tuple[List[ThoughtStep], OptimizationMetrics]:
        """
        Description:
            Оптимизация цепочки рассуждений.

        Args:
            chain: Исходная цепочка рассуждений.
            context: Контекст для оценки релевантности.

        Returns:
            Улучшенная цепочка и метрики оптимизации.
        """
        # Оценка текущего состояния
        current_metrics = self._evaluate_chain(chain, context)

        # Попытка улучшения цепочки
        improved_chain = self._improve_chain(chain, current_metrics, context)

        # Оценка улучшенной цепочки
        final_metrics = self._evaluate_chain(improved_chain, context)

        return improved_chain, final_metrics

    def _evaluate_chain(self, chain: List[ThoughtStep], context: Dict[str, Any]) -> OptimizationMetrics:
        """
        Description:
            Оценка качества цепочки рассуждений.

        Args:
            chain: Цепочка рассуждений.
            context: Контекст для оценки релевантности.

        Returns:
            Метрики оптимизации.
        """
        print("\n📊 Оценка цепочки рассуждений...")
        print("=" * 80)

        coherence = self._evaluate_coherence(chain)
        print(f"📈 Связность: {coherence:.2f}")

        relevance = self._evaluate_relevance(chain, context)
        print(f"🎯 Релевантность: {relevance:.2f}")

        depth = self._evaluate_depth(chain)
        print(f"🔍 Глубина: {depth:.2f}")

        breadth = self._evaluate_breadth(chain)
        print(f"🌐 Широта: {breadth:.2f}")

        overall_score = (
            self.weights['coherence'] * coherence +
            self.weights['relevance'] * relevance +
            self.weights['depth'] * depth +
            self.weights['breadth'] * breadth
        )

        print(f"⭐ Общая оценка: {overall_score:.2f}")

        return OptimizationMetrics(
            coherence=coherence,
            relevance=relevance,
            depth=depth,
            breadth=breadth,
            overall_score=overall_score
        )

    def _improve_chain(self,
                       chain: List[ThoughtStep],
                       current_metrics: OptimizationMetrics,
                       context: Dict[str, Any]) -> List[ThoughtStep]:
        """
        Description:
            Улучшение цепочки рассуждений.

        Args:
            chain: Исходная цепочка рассуждений.
            current_metrics: Текущие метрики.
            context: Контекст для оценки релевантности.

        Returns:
            Улучшенная цепочка.
        """
        print("\n🔄 Начало процесса улучшения цепочки...")
        improved_chain = chain.copy()

        # Улучшение связности
        if current_metrics.coherence < 0.7:
            print("\n📈 Требуется улучшение связности...")
            improved_chain = self._improve_coherence(improved_chain)

        # Улучшение релевантности
        if current_metrics.relevance < 0.7:
            print("\n🎯 Требуется улучшение релевантности...")
            improved_chain = self._improve_relevance(improved_chain, context)

        # Улучшение глубины
        if current_metrics.depth < 0.5:
            print("\n🔍 Требуется улучшение глубины...")
            improved_chain = self._improve_depth(improved_chain, context)

        # Улучшение широты
        if current_metrics.breadth < 0.5:
            print("\n🌐 Требуется улучшение широты...")
            improved_chain = self._improve_breadth(improved_chain, context)

        print("\n✨ Процесс улучшения цепочки завершен")
        return improved_chain

    def _evaluate_coherence(self, chain: List[ThoughtStep]) -> float:
        """
        Description:
            Оценка связности рассуждений.

        Args:
            chain: Цепочка рассуждений.

        Returns:
            Оценка связности.
        """
        if len(chain) <= 1:
            return 1.0

        coherence_scores = []
        for i in range(1, len(chain)):
            prev_step = chain[i - 1]
            curr_step = chain[i]
            # TODO: Имплементация оценки связности между шагами
            score = 0.8  # Placeholder
            coherence_scores.append(score)

        return np.mean(coherence_scores)

    def _evaluate_relevance(self, chain: List[ThoughtStep], context: Dict[str, Any]) -> float:
        """
        Description:
            Оценка релевантности рассуждений контексту.

        Args:
            chain: Цепочка рассуждений.
            context: Контекст для оценки релевантности.

        Returns:
            Оценка релевантности.
        """
        relevance_scores = []
        for step in chain:
            # TODO: Имплементация оценки релевантности
            score = 0.7  # Placeholder
            relevance_scores.append(score)

        return np.mean(relevance_scores)

    def _evaluate_depth(self, chain: List[ThoughtStep]) -> float:
        """
        Description:
            Оценка глубины рассуждений.

        Args:
            chain: Цепочка рассуждений.

        Returns:
            Оценка глубины.
        """
        if not chain:
            return 0.0

        depth_score = 0.0

        # Оценка по количеству шагов (максимум 10 шагов)
        steps_score = min(1.0, len(chain) / 10)

        # Оценка по развитию идей
        idea_development = 0.0
        for i in range(1, len(chain)):
            prev_thought = chain[i - 1].content
            curr_thought = chain[i].content
            novelty = self._evaluate_novelty(prev_thought, curr_thought)
            idea_development += novelty

        if len(chain) > 1:
            idea_development /= (len(chain) - 1)

        # Оценка по использованию различных типов действий
        action_types = set(step.action_type for step in chain)
        action_diversity = len(action_types) / len(ActionType)

        # Взвешенная сумма всех факторов
        depth_score = (
            0.4 * steps_score +
            0.4 * idea_development +
            0.2 * action_diversity
        )

        return min(1.0, depth_score)

    def _evaluate_novelty(self, prev_thought: str, curr_thought: str) -> float:
        """
        Description:
            Оценка новизны мысли.

        Args:
            prev_thought: Предыдущая мысль.
            curr_thought: Текущая мысль.

        Returns:
            Оценка новизны.
        """
        prompt = f"""
        Оцените, насколько вторая мысль добавляет новую информацию к первой.

        Первая мысль: {prev_thought}
        Вторая мысль: {curr_thought}

        Оцените по шкале от 0 до 1, где:
        0 - полное повторение
        1 - полностью новая информация

        Верните только число.
        """

        try:
            response = float(self.thought_evaluator(prompt).strip())
            return max(0.0, min(1.0, response))
        except:
            return 0.5

    def _evaluate_breadth(self, chain: List[ThoughtStep]) -> float:
        """
        Description:
            Оценка широты рассуждений.

        Args:
            chain: Цепочка рассуждений.

        Returns:
            Оценка широты.
        """
        if not chain:
            return 0.0

        breadth_score = 0.0

        # Извлекаем ключевые концепции из всех мыслей
        all_concepts = set()
        for step in chain:
            concepts = self._extract_concepts(step.content)
            all_concepts.update(concepts)

        # Оценка по количеству уникальных концепций
        concept_score = min(1.0, len(all_concepts) / 10)

        # Оценка по разнообразию подходов
        perspectives = self._analyze_perspectives(chain)
        perspective_score = min(1.0, len(perspectives) / 5)

        # Оценка по использованию графа
        graph_usage = sum(1 for step in chain if step.action_type != ActionType.THINK)
        graph_score = min(1.0, graph_usage / len(chain))

        # Взвешенная сумма всех факторов
        breadth_score = (
            0.4 * concept_score +
            0.4 * perspective_score +
            0.2 * graph_score
        )

        return breadth_score

    def _extract_concepts(self, thought: str) -> Set[str]:
        """
        Description:
            Извлечение ключевых концепций из мысли.

        Args:
            thought: Текущая мысль.

        Returns:
            Множество ключевых концепций.
        """
        prompt = f"""
        Извлеките ключевые концепции из следующей мысли:
        {thought}

        Верните концепции одной строкой, разделенные запятыми.
        """

        try:
            response = self.thought_evaluator(prompt)
            concepts = {c.strip() for c in response.split(',')}
            return concepts
        except:
            return set()

    def _analyze_perspectives(self, chain: List[ThoughtStep]) -> Set[str]:
        """
        Description:
            Анализ различных подходов в рассуждении.

        Args:
            chain: Цепочка рассуждений.

        Returns:
            Множество подходов.
        """
        prompt = f"""
        Определите различные подходы/перспективы в следующем рассуждении:
        {' '.join(step.content for step in chain)}

        Верните подходы одной строкой, разделенные запятыми.
        """

        try:
            response = self.thought_evaluator(prompt)
            perspectives = {p.strip() for p in response.split(',')}
            return perspectives
        except:
            return set()

    def _improve_coherence(self, chain: List[ThoughtStep]) -> List[ThoughtStep]:
        """
        Description:
            Улучшение связности цепочки рассуждений.

        Args:
            chain: Исходная цепочка рассуждений.

        Returns:
            Улучшенная цепочка.
        """
        print("\n🔄 Улучшение связности рассуждений...")
        improved_chain = chain.copy()

        # Проходим по цепочке и улучшаем связность между последовательными мыслями
        for i in range(1, len(improved_chain)):
            prev_step = improved_chain[i - 1]
            curr_step = improved_chain[i]

            # Обновляем метаданные для отслеживания связности
            curr_step.metadata.update({
                "improved_coherence": True,
                "parent_step": prev_step.content[:100],  # Сохраняем ссылку на родительский шаг
                "coherence_timestamp": time.time()
            })

            # Добавляем ссылку на предыдущий шаг
            if prev_step.content not in curr_step.references:
                curr_step.references.append(prev_step.content)

        print(f"✅ Улучшена связность для {len(improved_chain)} шагов")
        return improved_chain

    def _improve_relevance(self, chain: List[ThoughtStep], context: Dict[str, Any]) -> List[ThoughtStep]:
        """
        Description:
            Улучшение релевантности рассуждений контексту.

        Args:
            chain: Исходная цепочка рассуждений.
            context: Контекст для оценки релевантности.

        Returns:
            Улучшенная цепочка.
        """
        print("\n🎯 Улучшение релевантности к контексту...")
        improved_chain = chain.copy()

        # Добавляем контекстную информацию в метаданные каждого шага
        for step in improved_chain:
            step.metadata.update({
                "context_background": context.get('background', ''),
                "improved_relevance": True,
                "relevance_timestamp": time.time()
            })

            # Обновляем ссылки с учетом контекста
            if context.get('background') and context['background'] not in step.references:
                step.references.append(context['background'])

        print(f"✅ Улучшена релевантность для {len(improved_chain)} шагов")
        return improved_chain

    def _improve_depth(self, chain: List[ThoughtStep], context: Dict[str, Any]) -> List[ThoughtStep]:
        """
        Description:
            Улучшение глубины рассуждений.

        Args:
            chain: Исходная цепочка рассуждений.
            context: Контекст для оценки релевантности.

        Returns:
            Улучшенная цепочка.
        """
        print("\n🔍 Улучшение глубины рассуждений...")
        improved_chain = chain.copy()

        for step in improved_chain:
            # Добавляем метаданные о глубине рассуждения
            step.metadata.update({
                "depth_level": len(step.references),  # Уровень глубины на основе количества ссылок
                "improved_depth": True,
                "depth_timestamp": time.time()
            })

            # Расширяем ссылки для увеличения глубины
            if "depth_references" not in step.metadata:
                step.metadata["depth_references"] = []

            step.metadata["depth_references"].extend([
                ref for ref in chain[-3:]
                if ref.content not in step.metadata["depth_references"]
            ])

        print(f"✅ Улучшена глубина для {len(improved_chain)} шагов")
        return improved_chain

    def _improve_breadth(self, chain: List[ThoughtStep], context: Dict[str, Any]) -> List[ThoughtStep]:
        """
        Description:
            Улучшение широты рассуждений.

        Args:
            chain: Исходная цепочка рассуждений.
            context: Контекст для оценки релевантности.

        Returns:
            Улучшенная цепочка.
        """
        print("\n🌐 Улучшение широты рассуждений...")
        improved_chain = chain.copy()

        # Собираем все уникальные концепции из цепочки
        all_concepts = set()
        for step in chain:
            all_concepts.update(step.references)

        # Обогащаем каждый шаг дополнительными ссылками
        for step in improved_chain:
            step.metadata.update({
                "breadth_concepts": list(all_concepts),
                "improved_breadth": True,
                "breadth_timestamp": time.time()
            })

            # Добавляем связи с другими концепциями
            for concept in all_concepts:
                if concept not in step.references and concept != step.content:
                    step.references.append(concept)

        print(f"✅ Улучшена широта для {len(improved_chain)} шагов")
        return improved_chain