# -*- coding: utf-8 -*-
"""
Модуль веб-сервера Flask для игры "Крестики-нолики" с различными типами ИИ
"""
# Стандартные библиотеки
import os
import sys
from typing import Dict, Tuple, Any

# Сторонние библиотеки
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

# Добавление корневой директории проекта в путь импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорт классов агентов из проекта
from agents.q_learning_agent import QLearningAgent
from environment.opponent import MinimaxOpponent, RandomOpponent, RuleBasedOpponent

# Инициализация Flask приложения
app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)

# Словарь для хранения загруженных Q-learning агентов
Q_AGENTS: Dict[str, QLearningAgent] = {}


def load_q_agents() -> None:
    """
    Description:
    ---------------
        Загружает обученные модели Q-learning агента для разных уровней сложности.
        Модели ищутся в директории 'models' в корне проекта.
        
    Args:
    ---------------
        Нет аргументов
        
    Returns:
    ---------------
        None: Функция не возвращает значений, но заполняет глобальный словарь Q_AGENTS
        
    Raises:
    ---------------
        Exception: При ошибке загрузки модели
        
    Examples:
    ---------------
        >>> load_q_agents()
        Загружено моделей: 1
    """
    # Определяем путь к директории с моделями
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models'
    )
    
    # Если директория с моделями не существует, создаем её
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Директория моделей создана: {models_dir}")
        print("Внимание: Модели не найдены. Сначала обучите модели.")
        return
    
    # Ищем файлы моделей
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("Внимание: Модели не найдены. Сначала обучите модели.")
        return
    
    # Загружаем модели по именам файлов
    try:
        # Проверяем наличие основной модели Q-learning агента
        q_agent_final = os.path.join(models_dir, 'q_agent_final.pkl')
        if os.path.exists(q_agent_final):
            # Создаем экземпляр агента с пространством действий 9 (3x3 поле)
            agent = QLearningAgent(action_space=9)
            # Загружаем обученную модель
            if agent.load(q_agent_final):
                Q_AGENTS['q_learning'] = agent
                print(f"Загружена модель: q_learning")
            
        # Можно добавить загрузку других моделей, если они есть
        
        print(f"Загружено моделей: {len(Q_AGENTS)}")
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {e}")


# Маршруты Flask
@app.route('/')
def index() -> str:
    """
    Description:
    ---------------
        Отображение главной страницы приложения.
        
    Args:
    ---------------
        Нет аргументов
        
    Returns:
    ---------------
        str: HTML-страница index.html
        
    Examples:
    ---------------
        >>> app.test_client().get('/')
        200 OK
    """
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path: str) -> Any:
    """
    Description:
    ---------------
        Обслуживание статических файлов (CSS, JavaScript, изображения).
        
    Args:
    ---------------
        path: Путь к запрашиваемому статическому файлу
        
    Returns:
    ---------------
        Any: Запрошенный статический файл
        
    Examples:
    ---------------
        >>> app.test_client().get('/static/js/main.js')
        200 OK
    """
    return send_from_directory('static', path)


@app.route('/api/get_ai_move', methods=['POST'])
def get_ai_move() -> Tuple[Any, int]:
    """
    Description:
    ---------------
        API-эндпоинт для получения хода ИИ. Принимает текущее состояние 
        игрового поля, тип ИИ и метку ИИ (X или O), и возвращает 
        выбранный ИИ ход.
        
    Args:
    ---------------
        Нет прямых аргументов, данные извлекаются из JSON-запроса:
            board: Текущее состояние игрового поля (массив 3x3)
            ai_type: Тип ИИ ('random', 'rule_based', 'minimax', 'q_learning')
            ai_mark: Метка ИИ (1 для X, -1 для O)
        
    Returns:
    ---------------
        Tuple[Any, int]: JSON-ответ с выбранным ходом и код состояния HTTP
        
    Raises:
    ---------------
        Exception: При ошибке обработки запроса или выборе хода ИИ
        
    Examples:
    ---------------
        >>> app.test_client().post('/api/get_ai_move', json={
        ...     'board': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ...     'ai_type': 'random',
        ...     'ai_mark': 1
        ... })
        200 OK {'move': 4}
    """
    try:
        # Получаем данные из JSON-запроса
        data = request.json
        
        # Преобразуем плоский массив в матрицу 3x3
        board = np.array(data['board']).reshape(3, 3)
        
        # Получаем тип ИИ и его метку
        ai_type = data['ai_type']
        ai_mark = data['ai_mark']  # 1 для X, -1 для O
        
        # Получение доступных ходов (клетки со значением 0)
        flat_board = board.flatten()
        valid_moves = [i for i, val in enumerate(flat_board) if val == 0]
        
        # Если нет доступных ходов, возвращаем ошибку
        if not valid_moves:
            return jsonify({'error': 'No valid moves available'}), 400
        
        # Выбор хода в зависимости от типа ИИ
        move = None
        
        if ai_type == 'random':
            # Случайный ход
            move = np.random.choice(valid_moves)
        
        elif ai_type == 'rule_based':
            # Ход на основе эвристических правил
            opponent = RuleBasedOpponent()
            move = opponent.choose_action(board, ai_mark, -ai_mark)
        
        elif ai_type == 'minimax':
            # Ход с использованием алгоритма минимакс
            opponent = MinimaxOpponent()
            move = opponent.choose_action(board, ai_mark, -ai_mark)
        
        elif ai_type == 'q_learning':
            # Ход с использованием обученного Q-learning агента
            if 'q_learning' in Q_AGENTS:
                agent = Q_AGENTS['q_learning']
                # При выборе хода в боевом режиме training=False
                move = agent.choose_action(board, valid_moves, training=False)
            else:
                # Если модель не загружена, используем MinimaxOpponent как резервный вариант
                opponent = MinimaxOpponent()
                move = opponent.choose_action(board, ai_mark, -ai_mark)
        
        else:
            # Если указан неизвестный тип ИИ
            return jsonify({'error': 'Unknown AI type'}), 400
        
        # Возвращаем выбранный ход
        return jsonify({'move': int(move)}), 200
    
    except Exception as e:
        # Логируем ошибку и возвращаем информативный ответ
        print(f"Error in get_ai_move: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai_types')
def ai_types() -> Any:
    """
    Description:
    ---------------
        Возвращает список доступных типов ИИ с их описаниями.
        
    Args:
    ---------------
        Нет аргументов
        
    Returns:
    ---------------
        Any: JSON-ответ с доступными типами ИИ
        
    Examples:
    ---------------
        >>> app.test_client().get('/api/ai_types')
        200 OK {'random': 'Easy (Random)', 'rule_based': 'Medium (Rule-Based)', ...}
    """
    # Базовые типы ИИ, доступные всегда
    types = {
        'random': 'Easy (Random)',
        'rule_based': 'Medium (Rule-Based)',
        'minimax': 'Hard (Minimax)',
    }
    
    # Добавляем Q-learning, только если модель загружена
    if 'q_learning' in Q_AGENTS:
        types['q_learning'] = 'Reinforcement Learning (Trained Agent)'
    
    return jsonify(types)


if __name__ == '__main__':
    # Загрузка моделей при запуске
    load_q_agents()
    
    # Запуск сервера Flask
    # debug=True для удобства разработки
    app.run(debug=True)