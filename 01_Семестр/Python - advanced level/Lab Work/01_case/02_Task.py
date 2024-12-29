import logging
from typing import List

# Настройка логирования
logging.basicConfig(filename='warehouse.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class Product:
    """
    Description:
        Класс, представляющий товар.

    Attributes:
        name (str): Название товара.
        quantity (int): Количество товара.
        price (float): Цена товара.
    """
    def __init__(self, name: str, quantity: int, price: float):
        """
        Description:
            Инициализация объекта Product.

        Args:
            name (str): Название товара.
            quantity (int): Количество товара.
            price (float): Цена товара.
        """
        self.name = name
        self.quantity = quantity
        self.price = price

    def increase_quantity(self, amount: int) -> None:
        """
        Description:
            Увеличивает количество товара.

        Args:
            amount (int): Количество, на которое нужно увеличить.
        """
        self.quantity += amount
        logging.info(f"Увеличено количество товара '{self.name}' на {amount}. Новое количество: {self.quantity}")

    def decrease_quantity(self, amount: int) -> None:
        """
        Description:
            Уменьшает количество товара.

        Args:
            amount (int): Количество, на которое нужно уменьшить.
        """
        if self.quantity >= amount:
            self.quantity -= amount
            logging.info(f"Уменьшено количество товара '{self.name}' на {amount}. Новое количество: {self.quantity}")
        else:
            logging.warning(f"Недостаточно товара '{self.name}' для уменьшения на {amount}. Текущее количество: {self.quantity}")

    def calculate_cost(self) -> float:
        """
        Description:
            Рассчитывает стоимость товара.

        Returns:
            float: Стоимость товара.
        """
        return self.quantity * self.price

class Warehouse:
    """
    Description:
        Класс, представляющий склад.

    Attributes:
        products (List[Product]): Список товаров на складе.
    """
    def __init__(self):
        """
        Description:
            Инициализация объекта Warehouse.
        """
        self.products = []

    def add_product(self, product: Product) -> None:
        """
        Description:
            Добавляет товар на склад.

        Args:
            product (Product): Товар для добавления.
        """
        self.products.append(product)
        logging.info(f"Добавлен товар '{product.name}' на склад.")

    def remove_product(self, product_name: str) -> None:
        """
        Description:
            Удаляет товар со склада.

        Args:
            product_name (str): Название товара для удаления.
        """
        for product in self.products:
            if product.name == product_name:
                self.products.remove(product)
                logging.info(f"Удален товар '{product.name}' со склада.")
                return
        logging.warning(f"Товар '{product_name}' не найден на складе.")

    def calculate_total_cost(self) -> float:
        """
        Description:
            Рассчитывает общую стоимость всех товаров на складе.

        Returns:
            float: Общая стоимость товаров.
        """
        return sum(product.calculate_cost() for product in self.products)

class Seller:
    """
    Description:
        Класс, представляющий продавца.

    Attributes:
        name (str): Имя продавца.
        sales_history (List[dict]): История продаж.
    """
    def __init__(self, name: str):
        """
        Description:
            Инициализация объекта Seller.

        Args:
            name (str): Имя продавца.
        """
        self.name = name
        self.sales_history = []

    def sell_product(self, warehouse: Warehouse, product_name: str, quantity: int) -> None:
        """
        Description:
            Продает товар со склада.

        Args:
            warehouse (Warehouse): Склад, с которого продается товар.
            product_name (str): Название товара.
            quantity (int): Количество товара для продажи.
        """
        for product in warehouse.products:
            if product.name == product_name:
                if product.quantity >= quantity:
                    product.decrease_quantity(quantity)
                    revenue = quantity * product.price
                    self.sales_history.append({
                        'product_name': product_name,
                        'quantity': quantity,
                        'revenue': revenue
                    })
                    logging.info(f"Продавец '{self.name}' продал {quantity} единиц товара '{product_name}'. Выручка: {revenue}")
                    return
                else:
                    logging.warning(f"Недостаточно товара '{product_name}' для продажи {quantity} единиц.")
                    return
        logging.warning(f"Товар '{product_name}' не найден на складе.")

    def sales_report(self) -> List[dict]:
        """
        Description:
            Формирует отчёт о продажах.

        Returns:
            List[dict]: Список проданных товаров с указанием количества и стоимости.
        """
        return self.sales_history

# Пример использования
warehouse = Warehouse()
seller = Seller("Иван")

product1 = Product("Телефон", 10, 500)
product2 = Product("Ноутбук", 5, 1000)

warehouse.add_product(product1)
warehouse.add_product(product2)

seller.sell_product(warehouse, "Телефон", 2)
seller.sell_product(warehouse, "Ноутбук", 1)

print("Отчёт о продажах:")
for sale in seller.sales_report():
    print(f"Товар: {sale['product_name']}, Количество: {sale['quantity']}, Выручка: {sale['revenue']}")

print(f"Общая стоимость товаров на складе: {warehouse.calculate_total_cost()}")