#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Adapter Text Generation Inference
# Path: reasoning-service/src/cot/llm_adapter.py

# Импорты для работы с типами
from typing import List

# Импорт библиотек LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

class TGI_Adapter:
    """
    Description:
        Класс для взаимодействия с моделью ChatOpenAI.
    """
    
    def __init__(self, system_prompt: str, temperature: float, n: int, top_p: float, max_tokens: int, streaming: bool):
        """
        Description:
            Инициализирует экземпляр TGI_Adapter.
        
        Args:
            system_prompt: Начальный системный промпт для модели.
            temperature: Температура для генерации текста.
            n: Количество генераций.
            top_p: Порог для использования токенов на основе их вероятности.
            max_tokens: Максимальное количество токенов.
            streaming: Включить потоковый режим вывода.
        """
        self.system_prompt = system_prompt
        
        self.client = ChatOpenAI(
            base_url='',
            api_key="-",
            model='/meta-llama/Meta-Llama-3.1-70B-Instruct',
            temperature=temperature,
            n=n,
            top_p=top_p,
            max_tokens=max_tokens,
            streaming=streaming
        )
    
    def __call__(self, message: str) -> str:
        """Генерация ответа от модели."""
        print("\n🤖 Запрос к языковой модели...")
        print("-" * 80)
        print("📝 Системный промпт:", self.system_prompt)
        print("💭 Сообщение:", message)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=message)
        ]
        
        try:
            print("\n⏳ Ожидание ответа...")
            response = self.client.invoke(messages)
            print("\n🤖 Ответ модели:", response)
            return response.content
        except Exception as e:
            print(f"\n❌ Ошибка при запросе к модели: {e}")
            raise