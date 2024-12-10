import os
import json
import logging
from typing import List, Dict, Optional, Any

from cot.processor import ChainOfThoughtProcessor, ThoughtStep, ActionType
from cot.optimizer import ChainOptimizer
from cot.llm_adapter import TGI_Adapter
from cot.graph.generator import ensure_test_graph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReasoningService:
    """
    Description:
        Основной сервис для обработки запросов с использованием Chain of Thought.

    Attributes:
        thought_generator: Генератор мыслей.
        processor: Процессор Chain of Thought.
        optimizer: Оптимизатор цепочки рассуждений.
    """

    def __init__(self,
                 graph_path: str = None,
                 max_chain_length: int = 5,
                 min_confidence: float = 0.6,
                 cache_dir: str = ".cache",
                 use_gpu: bool = False):
        """
        Description:
            Инициализация сервиса.

        Args:
            graph_path: Путь к файлу с графом.
            max_chain_length: Максимальная длина цепочки рассуждений.
            min_confidence: Минимальная уверенность для шага рассуждения.
            cache_dir: Директория для кэширования.
            use_gpu: Использовать ли GPU для вычислений.
        """
        # Инициализация LLM для генерации мыслей
        self.thought_generator = TGI_Adapter(
            system_prompt="""Вы - эксперт по логическому мышлению и рассуждениям. 
            Ваша задача - генерировать следующий логический шаг в цепочке рассуждений.
            Используйте структурированный подход и учитывайте контекст.""",
            temperature=0.7,
            n=1,
            top_p=0.9,
            max_tokens=500,
            streaming=False
        )

        self.processor = ChainOfThoughtProcessor(
            max_chain_length=max_chain_length,
            min_confidence=min_confidence,
            cache_dir=cache_dir,
            use_gpu=use_gpu
        )
        self.optimizer = ChainOptimizer()

        # Загрузка графа, если указан путь
        if graph_path:
            self.load_graph(graph_path)

    def load_graph(self, graph_path: str):
        """
        Description:
            Загрузка графа из файла.

        Args:
            graph_path: Путь к файлу с графом.
        """
        logger.info(f"Loading graph from {graph_path}")
        try:
            with open(graph_path, 'r') as f:
                graph = json.load(f)
            self.processor.set_graph(graph)
            logger.info("Graph loaded successfully")
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            raise

    def _display_progress(self, message: str, progress: float = None):
        """
        Description:
            Отображение прогресса с визуальным индикатором.

        Args:
            message: Сообщение для отображения.
            progress: Прогресс в диапазоне от 0 до 1.
        """
        if progress is not None:
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            logger.info(f"[{bar}] {progress * 100:.1f}% | {message}")
        else:
            logger.info(message)

    def _display_thought(self, step_n: int, thought_step: ThoughtStep):
        """
        Description:
            Отображение шага рассуждения.

        Args:
            step_n: Номер шага.
            thought_step: Шаг рассуждения.
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Шаг {step_n}: {thought_step.action_type.value.capitalize()}")
        logger.info(f"{'=' * 80}")

        # Вывод мысли
        logger.info(f"Мысль: {thought_step.content}")

        # Вывод результата действия
        if thought_step.observation:
            logger.info(f"Результат: {thought_step.observation}")

        # Отображение уверенности
        conf_bar = '■' * int(thought_step.confidence * 20)
        logger.info(f"Уверенность: [{conf_bar:<20}] {thought_step.confidence:.2f}")

    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Description:
            Обработка запроса с пошаговым рассуждением и генерацией финального ответа.

        Args:
            query: Исходный вопрос.
            context: Дополнительный контекст для рассуждения.

        Returns:
            Результаты рассуждения, включая пошаговые мысли, метрики и финальный ответ.
        """

        def _generate_final_answer(thoughts: List[ThoughtStep]) -> str:
            """
            Description:
                Генерация финального ответа на основе всех рассуждений.

            Args:
                thoughts: Список рассуждений.

            Returns:
                Финальный ответ.
            """
            prompt = f"""
            На основе проведенных рассуждений составьте финальный, комплексный ответ на вопрос.

            Исходный вопрос: {query}

            Проведенные рассуждения:
            {' '.join([f"Шаг {i + 1}: {t.content}" for i, t in enumerate(thoughts)])}

            Требования к финальному ответу:
            1. Обобщите все ключевые моменты из рассуждений
            2. Структурируйте информацию логически
            3. Убедитесь в полноте ответа
            4. Используйте научно корректные формулировки
            5. Сделайте ответ понятным и хорошо организованным

            Финальный ответ должен включать:
            - Краткое введение
            - Основные механизмы и принципы
            - Важные факторы и зависимости
            - Заключение

            Пожалуйста, предоставьте структурированный финальный ответ:
            """

            return self.thought_generator(prompt).strip()

        # Инициализация
        self._display_progress("Инициализация процесса рассуждений...", 0.0)

        if context is None:
            context = {
                "background": "General reasoning task",
                "requirements": ["logical", "structured", "detailed"]
            }

        try:
            # Начало рассуждения
            logger.info("\n📝 Начальный запрос: %s", query)

            # Генерация мыслей
            thoughts = []
            current_step = 1

            while current_step <= self.processor.max_chain_length:
                logger.info(f"\n🤔 Генерация шага {current_step}...")

                # Получаем текущую цепочку мыслей
                current_chain = [t.content for t in thoughts]

                # Генерируем следующие мысли
                new_thoughts = self.processor.process_thought(
                    current_chain[-1] if current_chain else query,
                    context
                )

                progress = current_step / self.processor.max_chain_length
                self._display_progress(
                    f"Шаг {current_step}: Сгенерировано {len(new_thoughts)} мыслей",
                    progress
                )

                if new_thoughts:
                    # Берем лучшую мысль по уверенности
                    best_thought = max(new_thoughts, key=lambda x: x.confidence)
                    thoughts.append(best_thought)
                    self._display_thought(current_step, best_thought)

                    # Проверяем завершение
                    if best_thought.action_type == ActionType.CONCLUDE:
                        break

                current_step += 1

            # Оптимизация
            logger.info("\n🔄 Оптимизация цепочки рассуждений...")
            optimized_chain, metrics = self.optimizer.optimize_chain(thoughts, context)
            self._display_progress("Оптимизация завершена", 1.0)

            # Вывод метрик
            logger.info("\n📊 Метрики качества:")
            logger.info("=" * 40)
            for metric, value in metrics.__dict__.items():
                bar = '■' * int(value * 20)
                logger.info(f"{metric:15}: [{bar:<20}] {value:.2f}")

            # Генерация финального ответа
            logger.info("\n🎯 Генерация финального ответа...")
            final_answer = _generate_final_answer(optimized_chain)

            # Вывод финального ответа
            logger.info("\n📝 Финальный ответ:")
            logger.info("=" * 80)
            logger.info(final_answer)
            logger.info("=" * 80)

            return {
                "original_query": query,
                "reasoning_steps": [
                    {
                        "content": t.content,
                        "confidence": t.confidence,
                        "action_type": t.action_type.value,
                        "observation": t.observation,
                        "metadata": t.metadata
                    }
                    for t in optimized_chain
                ],
                "metrics": metrics.__dict__,
                "final_answer": final_answer,
                "success": True
            }

        except Exception as e:
            logger.error(f"❌ Ошибка: {str(e)}")
            return {
                "error": "Failed to process query",
                "message": str(e),
                "success": False
            }


def main():
    """Основная функция для демонстрации работы сервиса."""
    # Определяем пути для файлов
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data", "test_graph.json")
    cache_dir = os.path.join(base_dir, ".cache")

    # Убедимся, что граф существует
    graph_path = ensure_test_graph(graph_path)

    # Создаем сервис
    service = ReasoningService(
        graph_path=graph_path,
        cache_dir=cache_dir,
        use_gpu=True
    )

    logger.info("\n" + "=" * 80)
    logger.info("🤖 Chain of Thought Reasoning with Graph Support".center(80))
    logger.info("=" * 80 + "\n")

    try:
        # Пример запроса
        query = """
        Задача: Разработать алгоритм оптимизации расписаний и маршрутов для флота грузовых дронов, 
        минимизирующий совокупные издержки и время доставки при динамически меняющихся тарифах, и показать, 
        что классическая задача коммивояжёра (TSP) является лишь её частным случаем.
        """

        logger.info("\n🔄 Начинаем процесс рассуждений...\n")
        result = service.process_query(query)

        if "error" not in result:
            logger.info("\nУспешно завершено!")
        else:
            logger.error(f"\nПроизошла ошибка: {result['message']}")

    except KeyboardInterrupt:
        logger.info("\n\n👋 Программа завершена пользователем.")
    except Exception as e:
        logger.error(f"\n❌ Произошла ошибка: {str(e)}")
        logger.info("Попробуйте еще раз.")


if __name__ == "__main__":
    main()