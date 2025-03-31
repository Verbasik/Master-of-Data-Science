/**
 * JavaScript для игры "Крестики-нолики" с различными типами ИИ
 * Обрабатывает логику игры, взаимодействие с сервером и UI-компоненты
 */

document.addEventListener('DOMContentLoaded', function() {
    /**
     * Description:
     * ---------------
     *     Основное состояние игры, содержащее текущее поле, текущего игрока,
     *     статистику и настройки.
     */
    const gameState = {
        // Игровое поле: 0 = пусто, 1 = X (ИИ по умолчанию), -1 = O (Человек по умолчанию)
        board: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        // Текущий игрок: 1 это X, -1 это O
        currentPlayer: 1,
        // Флаг окончания игры
        gameOver: false,
        // Статистика игр
        wins: { 
            human: 0, 
            ai: 0, 
            draws: 0 
        },
        // Текущая сложность ИИ
        difficulty: 'random',
        // Маркеры для игроков (могут меняться в зависимости от того, кто ходит первым)
        aiMark: 1,
        humanMark: -1
    };

    // DOM-элементы
    const DOM = {
        // Игровое поле и клетки
        boardElement: document.getElementById('game-board'),
        cells: document.querySelectorAll('.cell'),
        
        // Модальное окно результатов
        resultModal: document.getElementById('result-modal'),
        modalContent: document.getElementById('modal-content'),
        resultIcon: document.getElementById('result-icon'),
        resultText: document.getElementById('result-text'),
        resultMessage: document.getElementById('result-message'),
        playAgainBtn: document.getElementById('play-again-btn'),
        
        // Элементы интерфейса
        thinkingOverlay: document.getElementById('thinking-overlay'),
        difficultySelect: document.getElementById('difficulty'),
        firstPlayerSelect: document.getElementById('first-player'),
        difficultyDescription: document.getElementById('difficulty-description'),
        aiMoveInfo: document.getElementById('ai-move-info'),
        
        // Элементы статистики
        humanStats: document.getElementById('human-stats'),
        aiStats: document.getElementById('ai-stats'),
        playerHuman: document.getElementById('player-human'),
        playerAi: document.getElementById('player-ai')
    };

    /**
     * Description:
     * ---------------
     *     Описания сложности ИИ для отображения пользователю.
     */
    const difficultyDescriptions = {
        'random': 'Легкий: ИИ делает случайные ходы',
        'rule_based': 'Средний: ИИ следует базовым правилам',
        'minimax': 'Сложный: ИИ использует продвинутый алгоритм (сложно победить)',
        'q_learning': 'Обучен с помощью обучения с подкреплением'
    };

    /**
     * Description:
     * ---------------
     *     Названия позиций на игровом поле для информативных сообщений.
     */
    const positions = [
        "верхний левый", "верхний центр", "верхний правый",
        "средний левый", "центр", "средний правый",
        "нижний левый", "нижний центр", "нижний правый"
    ];

    // Инициализация игры
    loadAITypes();
    initGame();

    // Назначение обработчиков событий
    DOM.playAgainBtn.addEventListener('click', resetGame);
    DOM.difficultySelect.addEventListener('change', updateDifficulty);
    DOM.firstPlayerSelect.addEventListener('change', initGame);

    /**
     * Description:
     * ---------------
     *     Загружает доступные типы ИИ с сервера.
     */
    async function loadAITypes() {
        try {
            const response = await fetch('/api/ai_types');
            if (!response.ok) {
                throw new Error(`HTTP ошибка: ${response.status}`);
            }
            
            const aiTypes = await response.json();
            
            // Очистка текущих опций
            DOM.difficultySelect.innerHTML = '';
            
            // Добавление новых опций
            Object.entries(aiTypes).forEach(([value, text]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = text;
                DOM.difficultySelect.appendChild(option);
            });
            
            // Обновление описания сложности
            updateDifficulty();
        } catch (error) {
            console.error('Не удалось загрузить типы ИИ:', error);
        }
    }

    /**
     * Description:
     * ---------------
     *     Инициализирует новую игру.
     */
    function initGame() {
        // Сброс игрового поля
        resetBoard();
        
        // Установка параметров игры
        gameState.difficulty = DOM.difficultySelect.value;
        gameState.wins = { human: 0, ai: 0, draws: 0 };
        updateStats();
        
        // Определение первого игрока
        setFirstPlayer();

        // Обновление индикаторов игроков
        updatePlayerIndicators();

        // Если ИИ ходит первым, инициируем его ход
        if (gameState.currentPlayer === gameState.aiMark) {
            setTimeout(makeAiMove, 500);
        }
    }

    /**
     * Description:
     * ---------------
     *     Устанавливает, кто будет ходить первым на основе выбора пользователя.
     */
    function setFirstPlayer() {
        const firstPlayer = DOM.firstPlayerSelect.value;
        
        // Сначала устанавливаем метки игрокам
        if (firstPlayer === 'human') {
            // Если человек ходит первым - он играет крестиками (1)
            gameState.humanMark = 1;
            gameState.aiMark = -1;
        } else {
            // В противном случае (ИИ первый или случайный выбор) - 
            // ИИ играет крестиками (1)
            gameState.humanMark = -1;
            gameState.aiMark = 1;
        }
        
        // Теперь определяем, кто ходит первым
        if (firstPlayer === 'random') {
            gameState.currentPlayer = Math.random() < 0.5 ? 1 : -1;
        } else if (firstPlayer === 'ai') {
            gameState.currentPlayer = 1; // Всегда крестики (X)
        } else { // 'human'
            gameState.currentPlayer = 1; // Всегда крестики (X)
        }
    }

    /**
     * Description:
     * ---------------
     *     Сбрасывает игру после завершения.
     */
    function resetGame() {
        // Сброс игрового поля
        resetBoard();
        
        // Скрытие модального окна результата
        DOM.resultModal.classList.remove('show');

        // Установка первого игрока для новой игры
        setFirstPlayer();
        
        // Обновление индикаторов игроков
        updatePlayerIndicators();

        // Если ИИ ходит первым, инициируем его ход
        if (gameState.currentPlayer === gameState.aiMark) {
            setTimeout(makeAiMove, 500);
        }
    }

    /**
     * Description:
     * ---------------
     *     Сбрасывает игровое поле до начального состояния.
     */
    function resetBoard() {
        // Сброс состояния игры
        gameState.board = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        gameState.gameOver = false;
        
        // Очистка UI игрового поля
        DOM.cells.forEach(cell => {
            cell.innerHTML = '';
            cell.className = 'cell';
            cell.style.pointerEvents = 'auto';
        });

        // Включение всех клеток
        enableBoard();
    }

    /**
     * Description:
     * ---------------
     *     Обновляет сложность ИИ на основе выбора пользователя.
     */
    function updateDifficulty() {
        gameState.difficulty = DOM.difficultySelect.value;
        // Обновляем описание сложности
        DOM.difficultyDescription.textContent = 
            difficultyDescriptions[gameState.difficulty] || '';
    }

    /**
     * Description:
     * ---------------
     *     Обновляет индикаторы игроков (активный игрок, иконки X/O).
     */
    function updatePlayerIndicators() {
        // Сброс активных состояний
        DOM.playerHuman.classList.remove('active');
        DOM.playerAi.classList.remove('active');

        // Установка активного игрока
        if (gameState.currentPlayer === gameState.humanMark) {
            DOM.playerHuman.classList.add('active');
        } else {
            DOM.playerAi.classList.add('active');
        }

        // Обновление иконок в зависимости от того, кто X, а кто O
        const humanIcon = DOM.playerHuman.querySelector('.player-icon');
        const aiIcon = DOM.playerAi.querySelector('.player-icon');
        
        if (gameState.humanMark === -1) { // Человек играет O
            humanIcon.className = 'player-icon fas fa-circle';
            aiIcon.className = 'player-icon fas fa-times';
        } else { // Человек играет X
            humanIcon.className = 'player-icon fas fa-times';
            aiIcon.className = 'player-icon fas fa-circle';
        }
    }

    /**
     * Description:
     * ---------------
     *     Обновляет статистику побед и ничьих.
     */
    function updateStats() {
        DOM.humanStats.textContent = `Побед: ${gameState.wins.human}`;
        DOM.aiStats.textContent = `Побед: ${gameState.wins.ai}`;
    }

    /**
     * Description:
     * ---------------
     *     Включает взаимодействие с игровым полем.
     */
    function enableBoard() {
        DOM.cells.forEach(cell => {
            cell.addEventListener('click', handleCellClick);
        });
    }

    /**
     * Description:
     * ---------------
     *     Отключает взаимодействие с игровым полем.
     */
    function disableBoard() {
        DOM.cells.forEach(cell => {
            cell.removeEventListener('click', handleCellClick);
        });
    }

    /**
     * Description:
     * ---------------
     *     Обрабатывает клик по клетке игрового поля.
     */
    function handleCellClick(e) {
        // Если игра окончена, выходим
        if (gameState.gameOver) return;

        const cellIndex = parseInt(e.target.getAttribute('data-index'));
        
        // Проверяем, что клетка пуста
        if (gameState.board[cellIndex] !== 0) return;

        // Делаем ход человека
        makeMove(cellIndex, gameState.humanMark);
        
        // Проверяем, не завершилась ли игра после хода человека
        if (checkGameOver()) {
            return;
        }
        
        // Делаем ход ИИ с небольшой задержкой
        setTimeout(makeAiMove, 500);
    }

    /**
     * Description:
     * ---------------
     *     Выполняет ход, обновляя состояние игры и UI.
     */
    function makeMove(cellIndex, player) {
        // Обновляем состояние игры
        gameState.board[cellIndex] = player;
        const cell = DOM.cells[cellIndex];
        
        // Обновляем UI
        cell.style.pointerEvents = 'none';
        
        if (player === 1) { // X
            cell.innerHTML = '<i class="fas fa-times"></i>';
            cell.classList.add('x');
        } else { // O
            cell.innerHTML = '<i class="fas fa-circle"></i>';
            cell.classList.add('o');
        }

        // Переключаем игрока
        gameState.currentPlayer = -gameState.currentPlayer;
        updatePlayerIndicators();
    }

    /**
     * Description:
     * ---------------
     *     Выполняет ход ИИ, запрашивая решение с сервера.
     */
    async function makeAiMove() {
        // Если игра окончена, выходим
        if (gameState.gameOver) return;

        // Показываем оверлей "ИИ думает"
        DOM.thinkingOverlay.classList.add('active');
        disableBoard();

        try {
            // Преобразуем 1D доску в 2D для бэкенда
            const board2D = [
                [gameState.board[0], gameState.board[1], gameState.board[2]],
                [gameState.board[3], gameState.board[4], gameState.board[5]],
                [gameState.board[6], gameState.board[7], gameState.board[8]]
            ];

            // Подготавливаем данные запроса
            const requestData = {
                board: board2D,
                ai_type: gameState.difficulty,
                ai_mark: gameState.aiMark
            };

            // Запрашиваем ход ИИ с сервера
            const response = await fetch('/api/get_ai_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`Ошибка сервера: ${response.status}`);
            }

            // Получаем выбранный ход
            const data = await response.json();
            const move = data.move;
            
            // Показываем информацию о ходе ИИ
            DOM.aiMoveInfo.textContent = `ИИ поставил ${gameState.aiMark === 1 ? 'X' : 'O'} на ${positions[move]}`;
            DOM.aiMoveInfo.classList.add('show');
            
            // Скрываем информацию через некоторое время
            setTimeout(() => {
                DOM.aiMoveInfo.classList.remove('show');
            }, 1500);

            // Выполняем ход ИИ
            makeMove(move, gameState.aiMark);
        } catch (error) {
            console.error('Ошибка при получении хода ИИ:', error);
            
            // Резервный вариант: случайный ход в случае ошибки
            const availableMoves = gameState.board
                .map((val, idx) => val === 0 ? idx : null)
                .filter(val => val !== null);

            if (availableMoves.length > 0) {
                const move = availableMoves[Math.floor(Math.random() * availableMoves.length)];
                makeMove(move, gameState.aiMark);
            }
        } finally {
            // Скрываем оверлей и проверяем окончание игры
            DOM.thinkingOverlay.classList.remove('active');
            enableBoard();
            checkGameOver();
        }
    }

    /**
     * Description:
     * ---------------
     *     Проверяет, выиграл ли игрок на текущем поле.
     */
    function checkWin(board, player) {
        // Проверка строк
        for (let i = 0; i < 9; i += 3) {
            if (board[i] === player && board[i+1] === player && board[i+2] === player) {
                return true;
            }
        }
        
        // Проверка столбцов
        for (let i = 0; i < 3; i++) {
            if (board[i] === player && board[i+3] === player && board[i+6] === player) {
                return true;
            }
        }
        
        // Проверка диагоналей
        if (board[0] === player && board[4] === player && board[8] === player) {
            return true;
        }
        if (board[2] === player && board[4] === player && board[6] === player) {
            return true;
        }
        
        return false;
    }

    /**
     * Description:
     * ---------------
     *     Проверяет, является ли текущее состояние ничьей.
     */
    function checkDraw(board) {
        return !board.includes(0) && 
               !checkWin(board, 1) && 
               !checkWin(board, -1);
    }

    /**
     * Description:
     * ---------------
     *     Проверяет, закончилась ли игра, и обрабатывает результат.
     */
    function checkGameOver() {
        let winner = null;
        
        // Проверяем выигрыш X или O
        if (checkWin(gameState.board, 1)) { // X wins
            winner = 1;
        } else if (checkWin(gameState.board, -1)) { // O wins
            winner = -1;
        } else if (checkDraw(gameState.board)) {
            // Ничья - оставляем winner = null
        } else {
            // Игра не окончена
            return false;
        }
        
        // Отмечаем, что игра окончена
        gameState.gameOver = true;
        disableBoard();
        
        // Подсвечиваем выигрышные клетки, если есть победитель
        if (winner !== null) {
            const winPattern = findWinPattern(gameState.board, winner);
            winPattern.forEach(index => {
                DOM.cells[index].classList.add('winning-cell');
            });
        }
        
        // Обновляем статистику и показываем модальное окно результата
        updateResultStats(winner);
        
        // Показываем модальное окно результата после небольшой задержки
        setTimeout(() => {
            DOM.resultModal.classList.add('show');
        }, 1000);
        
        return true;
    }

    /**
     * Description:
     * ---------------
     *     Обновляет статистику и настраивает модальное окно результата.
     */
    function updateResultStats(winner) {
        if (winner === gameState.humanMark) {
            // Человек выиграл
            gameState.wins.human++;
            
            // Настраиваем модальное окно для победы
            DOM.modalContent.className = 'modal-content win';
            DOM.resultIcon.className = 'result-icon fas fa-trophy';
            DOM.resultText.textContent = 'Вы выиграли!';
            DOM.resultMessage.textContent = 'Поздравляем с победой над ИИ!';
        } else if (winner === gameState.aiMark) {
            // ИИ выиграл
            gameState.wins.ai++;
            
            // Настраиваем модальное окно для поражения
            DOM.modalContent.className = 'modal-content lose';
            DOM.resultIcon.className = 'result-icon fas fa-robot';
            DOM.resultText.textContent = 'ИИ выиграл!';
            DOM.resultMessage.textContent = 'Попробуйте снова и посмотрите, сможете ли вы победить ИИ!';
        } else {
            // Ничья
            gameState.wins.draws++;
            
            // Настраиваем модальное окно для ничьей
            DOM.modalContent.className = 'modal-content draw';
            DOM.resultIcon.className = 'result-icon fas fa-handshake';
            DOM.resultText.textContent = "Ничья!";
            DOM.resultMessage.textContent = 'Игра закончилась вничью. Хороший матч!';
        }
        
        // Обновляем отображение статистики
        updateStats();
    }

    /**
     * Description:
     * ---------------
     *     Находит выигрышную комбинацию на игровом поле.
     */
    function findWinPattern(board, player) {
        // Проверка строк
        for (let i = 0; i < 9; i += 3) {
            if (board[i] === player && board[i+1] === player && board[i+2] === player) {
                return [i, i+1, i+2];
            }
        }
        
        // Проверка столбцов
        for (let i = 0; i < 3; i++) {
            if (board[i] === player && board[i+3] === player && board[i+6] === player) {
                return [i, i+3, i+6];
            }
        }
        
        // Проверка диагоналей
        if (board[0] === player && board[4] === player && board[8] === player) {
            return [0, 4, 8];
        }
        if (board[2] === player && board[4] === player && board[6] === player) {
            return [2, 4, 6];
        }
        
        return [];
    }
});