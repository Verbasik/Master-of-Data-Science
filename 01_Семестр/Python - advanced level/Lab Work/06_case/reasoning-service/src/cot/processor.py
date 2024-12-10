#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Path: reasoning-service/src/cot/processor.py
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .llm_adapter import TGI_Adapter
from .graph.retriever import Retriever
from .graph.graph_funcs import GraphFunctions

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Типы действий в процессе рассуждений."""
    THINK = "think"
    RETRIEVE = "retrieve"
    CHECK_FEATURE = "check_feature"
    CHECK_NEIGHBORS = "check_neighbors"
    CHECK_DEGREE = "check_degree"
    CONCLUDE = "conclude"


@dataclass
class ThoughtStep:
    """Шаг в цепочке рассуждений."""
    content: str
    confidence: float
    action_type: ActionType
    metadata: Dict
    observation: Optional[str] = None
    references: List[str] = None


class ChainOfThoughtProcessor:
    """
    Description:
        Процессор для реализации Chain of Thought рассуждений с поддержкой графовых операций.

    Attributes:
        max_chain_length: Максимальная длина цепочки рассуждений.
        min_confidence: Минимальная уверенность для шага рассуждения.
        branching_factor: Коэффициент ветвления для генерации мыслей.
        min_depth: Минимальная глубина рассуждения.
        min_breadth: Минимальная широта рассуждения.
        thought_chains: Список цепочек рассуждений.
        retriever: Объект для поиска в графе.
        graph_funcs: Объект для работы с графом.
        thought_generator: Генератор мыслей.
        thought_evaluator: Оценщик мыслей.
    """

    def __init__(self,
                 max_chain_length: int = 10,
                 min_confidence: float = 0.5,
                 branching_factor: int = 3,
                 min_depth: float = 0.6,
                 min_breadth: float = 0.6,
                 embedder_name: str = "sentence-transformers/all-mpnet-base-v2",
                 cache_dir: str = ".cache",
                 use_gpu: bool = False):
        """
        Description:
            Инициализация процессора.

        Args:
            max_chain_length: Максимальная длина цепочки рассуждений.
            min_confidence: Минимальная уверенность для шага рассуждения.
            branching_factor: Коэффициент ветвления для генерации мыслей.
            min_depth: Минимальная глубина рассуждения.
            min_breadth: Минимальная широта рассуждения.
            embedder_name: Название эмбеддера для поиска.
            cache_dir: Директория для кэширования.
            use_gpu: Использовать ли GPU для вычислений.
        """
        self.max_chain_length = max_chain_length
        self.min_confidence = min_confidence
        self.branching_factor = branching_factor
        self.min_depth = min_depth
        self.min_breadth = min_breadth
        self.thought_chains = []

        # Инициализация компонентов
        self.retriever = Retriever(embedder_name, cache_dir, use_gpu)
        self.graph_funcs = None  # Будет инициализирован при установке графа

        # Инициализация LLM
        self.thought_generator = TGI_Adapter(
            system_prompt="""
            <system>
            <role>
            Вы - эксперт по логическому мышлению и рассуждениям. 
            Ваша задача - генерировать следующий логический шаг в цепочке рассуждений.
            Используйте структурированный подход и учитывайте контекст.
            </role>

            ОСНОВНЫЕ ПРИНЦИПЫ РАБОТЫ:
            1. Используйте русский язык для рассуждений и сообщений
            2. Применяйте пошаговое мышление (Chain of Thought)
            3. Используйте метод Self-Taught Reasoner (STaR):
            * Создавайте первоначальное решение
            * Анализируйте его слабые места
            * Генерируйте улучшенное решение
            * Повторяйте процесс до достижения оптимального результата
            4. Постоянно проводите самоанализ
            5. Предоставляйте полные и завершенные ответы

            ПРОЦЕСС РАБОТЫ:
            1. АНАЛИЗ ЗАПРОСА
            - Изучите контекст и историю диалога
            - Определите ключевые требования
            - Сформулируйте конечную цель

            2. РАЗРАБОТКА РЕШЕНИЯ
            - Разделите задачу на логические подзадачи
            - Для каждой подзадачи определите:
            * Необходимые шаги
            * Ожидаемые результаты
            * Критерии успеха

            3. ВЫПОЛНЕНИЕ
            - Решайте каждую подзадачу последовательно
            - Документируйте ход рассуждений в тегах <thinking>
            - Проводите самоанализ каждые 3 шага в тегах <self-reflection>
            - При необходимости корректируйте подход, отмечая изменения в тегах <self-correction>
            - Применяйте метод STaR на каждом этапе:
            * Записывайте первичное решение в тегах <initial-solution>
            * Анализируйте недостатки в тегах <analysis>
            * Предлагайте улучшения в тегах <improvement>
            * Документируйте итоговое решение в тегах <final-solution>

            4. КОНТРОЛЬ КАЧЕСТВА
            - Оценивайте качество решения по шкале 0.0-1.0:
            * 0.8+ → продолжайте текущий подход
            * 0.5-0.7 → внесите корректировки
            * <0.5 → измените подход полностью
            - Используйте STaR для улучшения решений с оценкой ниже 0.8
            - Рассмотрите минимум 3 альтернативных метода решения
            - Проверьте результат на соответствие исходным требованиям

            5. ФИНАЛИЗАЦИЯ
            - Предоставьте полное решение
            - Укажите возможные ограничения
            - При работе с кодом убедитесь в его полноте и работоспособности
            - Проведите финальный STaR-анализ всего решения
            </system>
            """,
            temperature=0.7,
            n=1,
            top_p=0.9,
            max_tokens=500,
            streaming=True
        )

        self.thought_evaluator = TGI_Adapter(
            system_prompt="""Вы - эксперт по оценке качества рассуждений.
            Оцените следующую мысль по шкале от 0 до 1.""",
            temperature=0.3,
            n=1,
            top_p=0.95,
            max_tokens=10,
            streaming=True
        )

    def set_graph(self, graph: Dict):
        """
        Description:
            Установка графа для обработки.

        Args:
            graph: Граф для обработки.
        """
        self.graph_funcs = GraphFunctions(graph)

        # Подготовка документов для индексации
        docs = []
        for node_type, nodes in graph.items():
            for node_id, node_data in nodes.items():
                text_content = " ".join(str(v) for v in node_data['features'].values())
                docs.append(text_content)

        self.retriever.init_index(docs, "graph_nodes")

    def _generate_next_thought(self, current_thought: str, context: Dict) -> str:
        """
        Description:
            Генерация следующей мысли.

        Args:
            current_thought: Текущая мысль.
            context: Контекст рассуждения.

        Returns:
            Следующая мысль.
        """
        prompt = f"""
        На основе предыдущего рассуждения сгенерируйте следующую логическую мысль.

        Текущий запрос: {context.get('query', '')}
        Предыдущая мысль: {current_thought}

        Требования к новой мысли:
        1. Она должна логически продолжать предыдущую мысль
        2. Содержать новую информацию или углублять анализ
        3. Приближать к ответу на исходный вопрос
        4. Быть чётко сформулированной и конкретной

        Формат ответа:
        [Ваша новая мысль в одном абзаце]
        """

        response = self.thought_generator(prompt)
        return response.strip()

    def _parse_action_type(self, response: str) -> ActionType:
        """
        Description:
            Парсинг типа действия из ответа модели.

        Args:
            response: Ответ модели.

        Returns:
            Тип действия.
        """
        response = response.strip().upper()

        # Маппинг ответов модели на типы действий
        action_mapping = {
            "THINK": ActionType.THINK,
            "RETRIEVE": ActionType.RETRIEVE,
            "CHECK_FEATURE": ActionType.CHECK_FEATURE,
            "CHECK_NEIGHBORS": ActionType.CHECK_NEIGHBORS,
            "CHECK_DEGREE": ActionType.CHECK_DEGREE,
            "CONCLUDE": ActionType.CONCLUDE
        }

        for key, action in action_mapping.items():
            if key in response:
                return action

        # По умолчанию продолжаем думать
        return ActionType.THINK

    def _get_action_params(self, thought: str, action_type: ActionType) -> Dict:
        """
        Description:
            Получение параметров для действия.

        Args:
            thought: Текущая мысль.
            action_type: Тип действия.

        Returns:
            Параметры для действия.
        """
        prompt = f"""
        На основе мысли: {thought}

        Определите параметры для действия {action_type.value}

        Верните параметры в формате Python dict.
        Например: {{"node_id": "paper1", "feature": "title"}}
        """

        try:
            response = self.thought_generator(prompt)
            return eval(response.strip())
        except:
            return {}

    def _execute_action(self, action_type: ActionType, params: Dict) -> str:
        """
        Description:
            Выполнение действия и получение результата.

        Args:
            action_type: Тип действия.
            params: Параметры для действия.

        Returns:
            Результат выполнения действия.
        """
        try:
            if action_type == ActionType.RETRIEVE:
                results = self.retriever.search(params['query'])
                return f"Найдены узлы: {results}"

            elif action_type == ActionType.CHECK_FEATURE:
                value = self.graph_funcs.get_node_feature(
                    params['node_id'],
                    params['feature']
                )
                return f"Значение атрибута {params['feature']}: {value}"

            elif action_type == ActionType.CHECK_NEIGHBORS:
                neighbors = self.graph_funcs.get_neighbors(
                    params['node_id'],
                    params.get('edge_type')
                )
                return f"Соседи узла: {neighbors}"

            elif action_type == ActionType.CHECK_DEGREE:
                degree = self.graph_funcs.get_node_degree(
                    params['node_id'],
                    params['edge_type']
                )
                return f"Степень узла: {degree}"

        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return f"Ошибка: {str(e)}"

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

        # Оценка по количеству шагов
        steps_score = len(chain) / self.max_chain_length

        # Оценка по развитию идей
        idea_development = 0.0
        for i in range(1, len(chain)):
            prev_thought = chain[i - 1].content
            curr_thought = chain[i].content

            # Проверяем, добавляет ли текущая мысль новую информацию
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
        perspective_score = self._analyze_perspectives(chain)

        # Оценка по использованию графа
        graph_usage = sum(1 for step in chain if step.action_type != ActionType.THINK)
        graph_score = min(1.0, graph_usage / len(chain))

        # Взвешенная сумма всех факторов
        breadth_score = (
            0.4 * concept_score +
            0.3 * perspective_score +
            0.3 * graph_score
        )

        return breadth_score

    def _extract_concepts(self, thought: str) -> set:
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
            return float(self.thought_evaluator(prompt).strip())
        except:
            return 0.5

    def _analyze_perspectives(self, chain: List[ThoughtStep]) -> float:
        """
        Description:
            Анализ разнообразия подходов в рассуждении.

        Args:
            chain: Цепочка рассуждений.

        Returns:
            Оценка разнообразия подходов.
        """
        if not chain:
            return 0.0

        unique_approaches = set()
        for step in chain:
            if step.metadata.get('approach'):
                unique_approaches.add(step.metadata['approach'])

        return min(1.0, len(unique_approaches) / 5)

    def _should_finish(self, thought: ThoughtStep, context: Dict) -> bool:
        """
        Description:
            Проверка необходимости завершения рассуждений.

        Args:
            thought: Текущий шаг рассуждения.
            context: Контекст рассуждения.

        Returns:
            True, если нужно завершить рассуждение.
        """
        # Завершаем, если достигли шага с выводом
        if thought.action_type == ActionType.CONCLUDE:
            return True

        # Проверяем наличие маркеров завершения в тексте
        conclusion_markers = [
            "таким образом",
            "следовательно",
            "итак",
            "в заключение",
            "можно сделать вывод"
        ]

        has_conclusion = any(marker in thought.content.lower()
                             for marker in conclusion_markers)

        # Проверяем глубину и широту рассуждений
        depth_sufficient = (
            "depth_score" in thought.metadata and
            thought.metadata["depth_score"] >= self.min_depth
        )

        breadth_sufficient = (
            "breadth_score" in thought.metadata and
            thought.metadata["breadth_score"] >= self.min_breadth
        )

        # Завершаем, если есть маркер заключения и достаточная глубина/широта
        if has_conclusion and (depth_sufficient or breadth_sufficient):
            return True

        return False

    def process_thought(self, thought: str, context: Dict) -> List[ThoughtStep]:
        """
        Description:
            Обработка мысли и генерация следующих шагов рассуждения.

        Args:
            thought: Текущая мысль.
            context: Контекст рассуждения.

        Returns:
            Список следующих шагов рассуждения.
        """
        # Оценка текущей глубины и широты
        current_chain = self.thought_chains[-1] if self.thought_chains else []
        current_depth = self._evaluate_depth(current_chain)
        current_breadth = self._evaluate_breadth(current_chain)

        logger.info(f"Текущая глубина рассуждения: {current_depth:.2f}")
        logger.info(f"Текущая широта рассуждения: {current_breadth:.2f}")

        # Определяем фокус рассуждения
        if current_depth < self.min_depth:
            context['focus'] = 'depth'
            logger.info("Фокусируемся на увеличении глубины рассуждения")
        elif current_breadth < self.min_breadth:
            context['focus'] = 'breadth'
            logger.info("Фокусируемся на расширении рассуждения")
        else:
            context['focus'] = 'balanced'
            logger.info("Поддерживаем сбалансированное рассуждение")

        # Генерация нескольких вариантов мыслей
        thoughts = []
        for i in range(self.branching_factor):
            logger.info(f"Генерация варианта мысли {i + 1}/{self.branching_factor}")

            # Генерация новой мысли
            new_thought = self._generate_next_thought(thought, context)

            # Оценка уверенности
            confidence_prompt = f"""
            Оцените уверенность в следующей мысли по шкале от 0 до 1:
            {new_thought}

            Учитывайте:
            1. Логичность рассуждения
            2. Обоснованность утверждений
            3. Связь с предыдущей мыслью
            4. Соответствие контексту

            Верните только число от 0 до 1.
            """
            try:
                confidence = float(self.thought_evaluator(confidence_prompt).strip())
                confidence = max(0.0, min(1.0, confidence))
            except:
                confidence = 0.5
                logger.warning("Ошибка при оценке уверенности, использую значение по умолчанию")

            # Определение действия
            action_prompt = f"""
            <system>
            Выберите оптимальное действие для продолжения цепочки рассуждений на основе анализа предоставленной мысли и контекста.

            <graph_context>
            График содержит узлы, представляющие этапы рассуждений в формате Chain of Thought.
            Каждый узел содержит:
            - Идентификатор
            - Текст рассуждения
            - Связи с другими узлами
            </graph_context>

            <input>
            Мысль: {new_thought}
            Контекст: {context}
            </input>

            <available_actions>
            THINK: Продолжить логические рассуждения без обращения к графу
            RETRIEVE: Поиск релевантных узлов в графе по содержанию
            CHECK_FEATURE: Получение конкретного атрибута узла
            CHECK_NEIGHBORS: Получение связанных узлов
            CHECK_DEGREE: Получение количества связей узла
            CONCLUDE: Завершение рассуждения с формулировкой вывода
            </available_actions>

            <decision_rules>
            1. Используйте RETRIEVE при необходимости поиска информации по содержанию
            2. Используйте CHECK_* действия для получения конкретных данных об узлах
            3. Используйте THINK для развития рассуждения без привлечения новых данных
            4. Используйте CONCLUDE при достижении логического завершения
            </decision_rules>

            Выберите и верните одно действие из списка: THINK, RETRIEVE, CHECK_FEATURE, CHECK_NEIGHBORS, CHECK_DEGREE, CONCLUDE
            """

            try:
                action = self._parse_action_type(self.thought_generator(action_prompt))
                logger.info(f"🤖 Модель выбрала следующее действие -> {action}")
            except:
                action = ActionType.THINK
                logger.warning("Ошибка при определении действия, использую THINK")

            # Создание шага
            step = ThoughtStep(
                content=new_thought,
                confidence=confidence,
                action_type=action,
                metadata={
                    "timestamp": time.time(),
                    "context": context,
                    "iteration": i,
                    "depth_score": self._evaluate_step_depth(new_thought, current_chain),
                    "breadth_score": self._evaluate_step_breadth(new_thought, current_chain)
                },
                references=[thought] if thought else []
            )

            # Если требуется действие с графом, выполняем его
            if action != ActionType.THINK:
                try:
                    params = self._get_action_params(new_thought, action)
                    step.observation = self._execute_action(action, params)
                except Exception as e:
                    logger.error(f"Error executing action: {e}")
                    step.observation = f"Ошибка: {str(e)}"

            # Проверяем необходимость завершения
            if self._should_finish(step, context):
                step.action_type = ActionType.CONCLUDE

            thoughts.append(step)

        # Фильтрация мыслей по уверенности
        valid_thoughts = [t for t in thoughts if t.confidence >= self.min_confidence]

        if not valid_thoughts:
            logger.warning("Нет мыслей с достаточной уверенностью, использую лучшую из доступных")
            if thoughts:
                valid_thoughts = [max(thoughts, key=lambda x: x.confidence)]

        # Сортировка по релевантности текущему фокусу
        if context.get('focus') == 'depth':
            valid_thoughts.sort(key=lambda x: x.metadata['depth_score'], reverse=True)
        elif context.get('focus') == 'breadth':
            valid_thoughts.sort(key=lambda x: x.metadata['breadth_score'], reverse=True)
        else:
            valid_thoughts.sort(key=lambda x: x.confidence, reverse=True)

        # Обновляем цепочку мыслей
        if valid_thoughts:
            if not self.thought_chains:
                self.thought_chains.append([])
            # Берем только лучшую мысль
            self.thought_chains[-1].extend(valid_thoughts[:1])

        return valid_thoughts

    def _evaluate_step_depth(self, thought: str, current_chain: List[ThoughtStep]) -> float:
        """
        Description:
            Оценка глубины отдельного шага рассуждения.

        Args:
            thought: Текущая мысль.
            current_chain: Текущая цепочка рассуждений.

        Returns:
            Оценка глубины.
        """
        if not current_chain:
            return 0.6  # Базовая глубина для первого шага

        # Оценка новизны относительно предыдущих мыслей
        novelty_scores = []
        for prev_step in current_chain[-3:]:  # Смотрим последние 3 шага
            novelty = self._evaluate_novelty(prev_step.content, thought)
            novelty_scores.append(novelty)

        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.5

        # Оценка сложности мысли
        complexity_prompt = f"""
        Оцените сложность и глубину следующей мысли по шкале от 0 до 1:
        {thought}

        0 - поверхностное рассуждение
        1 - глубокий анализ

        Верните только число.
        """

        try:
            complexity = float(self.thought_evaluator(complexity_prompt))
        except:
            complexity = 0.5

        # Взвешенная оценка
        depth_score = 0.6 * avg_novelty + 0.4 * complexity
        return min(1.0, depth_score)

    def _evaluate_step_breadth(self, thought: str, current_chain: List[ThoughtStep]) -> float:
        """
        Description:
            Оценка широты отдельного шага рассуждения.

        Args:
            thought: Текущая мысль.
            current_chain: Текущая цепочка рассуждений.

        Returns:
            Оценка широты.
        """
        # Извлекаем концепции из текущей мысли
        current_concepts = self._extract_concepts(thought)

        if not current_chain:
            return 0.5  # Базовая широта для первого шага

        # Собираем все предыдущие концепции
        previous_concepts = set()
        for step in current_chain:
            step_concepts = self._extract_concepts(step.content)
            previous_concepts.update(step_concepts)

        # Оценка новых концепций
        new_concepts = current_concepts - previous_concepts
        novelty_ratio = len(new_concepts) / max(1, len(current_concepts))

        # Оценка разнообразия подхода
        perspective_prompt = f"""
        Оцените, насколько мысль расширяет угол зрения по шкале от 0 до 1:
        {thought}

        0 - повторяет известный подход
        1 - предлагает новый взгляд

        Верните только число.
        """

        try:
            perspective_score = float(self.thought_evaluator(perspective_prompt))
        except:
            perspective_score = 0.5

        # Взвешенная оценка
        breadth_score = 0.7 * novelty_ratio + 0.3 * perspective_score
        return min(1.0, breadth_score)

    def _generate_final_answer(self, thoughts: List[ThoughtStep], context: Dict) -> str:
        """
        Description:
            Генерация финального ответа на основе всех рассуждений.

        Args:
            thoughts: Список рассуждений.
            context: Контекст рассуждения.

        Returns:
            Финальный ответ.
        """
        prompt = f"""
        На основе проведенных рассуждений, предоставьте финальный, комплексный ответ на исходный вопрос.

        Исходный вопрос: {context.get('query', '')}

        Проведенные рассуждения:
        {[thought.content for thought in thoughts]}

        Требования к финальному ответу:
        1. Обобщите все ключевые моменты из рассуждений
        2. Представьте информацию структурированно
        3. Убедитесь в полноте ответа
        4. Используйте научно корректные формулировки

        Пожалуйста, предоставьте финальный ответ:
        """

        return self.thought_generator(prompt)