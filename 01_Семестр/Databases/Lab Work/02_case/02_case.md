1. Выбрать все заказы (SALES_ORDER):
```sql
SELECT * FROM SALES_ORDER;
```

2. Выбрать все заказы, введенные после 1 января 2016 года:
```sql
SELECT * FROM SALES_ORDER
WHERE ORDER_DATE > '2016-01-01';
```

3. Выбрать все заказы, введенные после 1 января 2016 года и до 15 июля 2016 года:
```sql
SELECT * FROM SALES_ORDER
WHERE ORDER_DATE > '2016-01-01' AND ORDER_DATE < '2016-07-15';
```

4. Найти менеджеров с именем 'Henry':
```sql
SELECT * FROM MANAGER
WHERE MANAGER_FIRST_NAME = 'Henry';
```

5. Выбрать все заказы менеджеров с именем Henry:
```sql
SELECT SO.* 
FROM SALES_ORDER SO
JOIN MANAGER M ON SO.MANAGER_ID = M.MANAGER_ID
WHERE M.MANAGER_FIRST_NAME = 'Henry';
```

6. Выбрать все уникальные страны из таблицы CITY:
```sql
SELECT DISTINCT COUNTRY FROM CITY;
```

7. Выбрать все уникальные комбинации страны и региона из таблицы CITY:
```sql
SELECT DISTINCT COUNTRY, REGION FROM CITY;
```

8. Выбрать все страны из таблицы CITY с количеством городов в них:
```sql
SELECT COUNTRY, COUNT(*) AS CITY_COUNT
FROM CITY
GROUP BY COUNTRY;
```

9. Выбрать количество товаров (QTY), проданное с 1 по 30 января 2016 года:
```sql
SELECT SUM(SOL.PRODUCT_QTY) AS TOTAL_QTY
FROM SALES_ORDER_LINE SOL
JOIN SALES_ORDER SO ON SOL.SALES_ORDER_ID = SO.SALES_ORDER_ID
WHERE SO.ORDER_DATE BETWEEN '2016-01-01' AND '2016-01-30';
```

10. Выбрать все уникальные названия городов, регионов и стран в одной колонке одним запросом:
```sql
SELECT CITY_NAME AS LOCATION FROM CITY
UNION
SELECT REGION FROM CITY
UNION
SELECT COUNTRY FROM CITY;
```

11. Вывести имена и фамилии менеджер(ов), продавшего товаров в январе 2016 года на наибольшую сумму:
```sql
WITH MANAGER_SALES AS (
    SELECT 
        M.MANAGER_ID,
        M.MANAGER_FIRST_NAME,
        M.MANAGER_LAST_NAME,
        SUM(SOL.PRODUCT_QTY * SOL.PRODUCT_PRICE) AS TOTAL_SALES
    FROM MANAGER M
    JOIN SALES_ORDER SO ON M.MANAGER_ID = SO.MANAGER_ID
    JOIN SALES_ORDER_LINE SOL ON SO.SALES_ORDER_ID = SOL.SALES_ORDER_ID
    WHERE SO.ORDER_DATE BETWEEN '2016-01-01' AND '2016-01-31'
    GROUP BY M.MANAGER_ID, M.MANAGER_FIRST_NAME, M.MANAGER_LAST_NAME
)
SELECT MANAGER_FIRST_NAME, MANAGER_LAST_NAME
FROM MANAGER_SALES
WHERE TOTAL_SALES = (SELECT MAX(TOTAL_SALES) FROM MANAGER_SALES);
```