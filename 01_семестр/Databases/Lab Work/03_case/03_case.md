1. Каждый месяц компания выдает премию в размере 5% от суммы продаж менеджеру, который за предыдущие 3 месяца продал товаров на самую большую сумму
Выведите месяц, manager_id, manager_first_name, manager_last_name, премию за период с января по декабрь 2014 года
```sql
WITH monthly_sales AS (
    SELECT
        manager_id,
        manager_first_name,
        manager_last_name,
        DATE_TRUNC('month', sale_date) AS month,
        SUM(sale_amount) AS total_sales
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY manager_id, manager_first_name, manager_last_name, DATE_TRUNC('month', sale_date)
),
previous_3_months_sales AS (
    SELECT
        manager_id,
        manager_first_name,
        manager_last_name,
        month,
        SUM(total_sales) OVER (
            PARTITION BY manager_id
            ORDER BY month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS total_sales_3_months
    FROM monthly_sales
),
top_manager AS (
    SELECT
        month,
        manager_id,
        manager_first_name,
        manager_last_name,
        total_sales_3_months,
        ROW_NUMBER() OVER (PARTITION BY month ORDER BY total_sales_3_months DESC) AS rn
    FROM previous_3_months_sales
)
SELECT
    month,
    manager_id,
    manager_first_name,
    manager_last_name,
    total_sales_3_months * 0.05 AS bonus
FROM top_manager
WHERE rn = 1
ORDER BY month;
```

2. Компания хочет оптимизировать количество офисов, проанализировав относительные объемы продаж по офисам в течение периода с 2013-2014 гг.
Выведите год, office_id, city_name, country, относительный объем продаж за текущий год
Офисы, которые демонстрируют наименьший относительной объем в течение двух лет скорее всего будут закрыты.
```sql
WITH yearly_sales AS (
    SELECT
        office_id,
        city_name,
        country,
        EXTRACT(YEAR FROM sale_date) AS year,
        SUM(sale_amount) AS total_sales
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2013-01-01' AND '2014-12-31'
    GROUP BY office_id, city_name, country, EXTRACT(YEAR FROM sale_date)
),
total_sales_per_year AS (
    SELECT
        year,
        SUM(total_sales) AS total_sales_year
    FROM yearly_sales
    GROUP BY year
)
SELECT
    ys.year,
    ys.office_id,
    ys.city_name,
    ys.country,
    ys.total_sales / tsy.total_sales_year AS relative_sales
FROM yearly_sales ys
JOIN total_sales_per_year tsy ON ys.year = tsy.year
ORDER BY ys.year, ys.office_id;
```

3. Для планирования закупок, компанию оценивает динамику роста продаж по товарам.
Динамика оценивается как отношение объема продаж в текущем месяце к предыдущему.
Выведите товары, которые демонстрировали наиболее высокие темпы роста продаж в течение первого полугодия 2014 года.
```sql
WITH monthly_sales AS (
    SELECT
        product_id,
        product_name,
        DATE_TRUNC('month', sale_date) AS month,
        SUM(sale_amount) AS total_sales
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-06-30'
    GROUP BY product_id, product_name, DATE_TRUNC('month', sale_date)
),
sales_growth AS (
    SELECT
        product_id,
        product_name,
        month,
        total_sales,
        LAG(total_sales) OVER (PARTITION BY product_id ORDER BY month) AS prev_total_sales
    FROM monthly_sales
)
SELECT
    product_id,
    product_name,
    month,
    total_sales / prev_total_sales AS growth_rate
FROM sales_growth
WHERE prev_total_sales IS NOT NULL
ORDER BY growth_rate DESC;
```

4. Напишите запрос, который выводит отчет о прибыли компании за 2014 год: помесячно и поквартально.
Отчет включает сумму прибыли за период и накопительную сумму прибыли с начала года по текущий период.
```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        SUM(sale_amount) AS total_sales
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY DATE_TRUNC('month', sale_date)
),
quarterly_sales AS (
    SELECT
        DATE_TRUNC('quarter', sale_date) AS quarter,
        SUM(sale_amount) AS total_sales
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY DATE_TRUNC('quarter', sale_date)
),
cumulative_sales AS (
    SELECT
        month,
        total_sales,
        SUM(total_sales) OVER (ORDER BY month) AS cumulative_sales
    FROM monthly_sales
)
SELECT
    month,
    total_sales,
    cumulative_sales
FROM cumulative_sales
UNION ALL
SELECT
    quarter AS month,
    total_sales,
    SUM(total_sales) OVER (ORDER BY quarter) AS cumulative_sales
FROM quarterly_sales
ORDER BY month;
```

5. Найдите вклад в общую прибыль за 2014 год 10% наиболее дорогих товаров и 10% наиболее дешевых товаров.
Выведите product_id, product_name, total_sale_amount, percent
```sql
WITH total_sales AS (
    SELECT
        product_id,
        product_name,
        SUM(sale_amount) AS total_sale_amount
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY product_id, product_name
),
ranked_products AS (
    SELECT
        product_id,
        product_name,
        total_sale_amount,
        NTILE(10) OVER (ORDER BY total_sale_amount DESC) AS decile
    FROM total_sales
),
top_10_percent AS (
    SELECT
        product_id,
        product_name,
        total_sale_amount,
        'Top 10%' AS category
    FROM ranked_products
    WHERE decile = 1
),
bottom_10_percent AS (
    SELECT
        product_id,
        product_name,
        total_sale_amount,
        'Bottom 10%' AS category
    FROM ranked_products
    WHERE decile = 10
),
combined_products AS (
    SELECT * FROM top_10_percent
    UNION ALL
    SELECT * FROM bottom_10_percent
)
SELECT
    product_id,
    product_name,
    total_sale_amount,
    (total_sale_amount / SUM(total_sale_amount) OVER ()) * 100 AS percent
FROM combined_products
ORDER BY category, total_sale_amount DESC;
```

6. Компания хочет премировать трех наиболее продуктивных (по объему продаж, конечно) менеджеров в каждой стране в 2014 году.
Выведите country, <список manager_last_name manager_first_name, разделенный запятыми> которым будет выплачена премия
```sql
WITH manager_sales AS (
    SELECT
        manager_id,
        manager_first_name,
        manager_last_name,
        country,
        SUM(sale_amount) AS total_sales
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY manager_id, manager_first_name, manager_last_name, country
),
top_managers AS (
    SELECT
        country,
        STRING_AGG(manager_last_name || ' ' || manager_first_name, ', ') AS top_managers
    FROM (
        SELECT
            country,
            manager_last_name,
            manager_first_name,
            ROW_NUMBER() OVER (PARTITION BY country ORDER BY total_sales DESC) AS rn
        FROM manager_sales
    ) sub
    WHERE rn <= 3
    GROUP BY country
)
SELECT
    country,
    top_managers
FROM top_managers
ORDER BY country;
```

7. Выведите самый дешевый и самый дорогой товар, проданный за каждый месяц в течение 2014 года.
cheapest_product_id, cheapest_product_name, expensive_product_id, expensive_product_name, month, cheapest_price, expensive_price
```sql
WITH monthly_prices AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        MIN(sale_price) AS cheapest_price,
        MAX(sale_price) AS expensive_price
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY DATE_TRUNC('month', sale_date)
),
cheapest_products AS (
    SELECT
        month,
        product_id AS cheapest_product_id,
        product_name AS cheapest_product_name,
        sale_price AS cheapest_price
    FROM public.v_fact_sale
    JOIN monthly_prices ON DATE_TRUNC('month', sale_date) = month AND sale_price = cheapest_price
),
expensive_products AS (
    SELECT
        month,
        product_id AS expensive_product_id,
        product_name AS expensive_product_name,
        sale_price AS expensive_price
    FROM public.v_fact_sale
    JOIN monthly_prices ON DATE_TRUNC('month', sale_date) = month AND sale_price = expensive_price
)
SELECT
    c.month,
    c.cheapest_product_id,
    c.cheapest_product_name,
    e.expensive_product_id,
    e.expensive_product_name,
    c.cheapest_price,
    e.expensive_price
FROM cheapest_products c
JOIN expensive_products e ON c.month = e.month
ORDER BY c.month;
```

8. Менеджер получает оклад в 30 000 + 5% от суммы своих продаж в месяц. Средняя наценка стоимости товара - 10%
Посчитайте прибыль предприятия за 2014 год по месяцам (сумма продаж - (исходная стоимость товаров + зарплата))
month, sales_amount, salary_amount, profit_amount
```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        SUM(sale_amount) AS total_sales
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY DATE_TRUNC('month', sale_date)
),
manager_salaries AS (
    SELECT
        manager_id,
        DATE_TRUNC('month', sale_date) AS month,
        SUM(sale_amount) * 0.05 AS bonus,
        30000 AS base_salary
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY manager_id, DATE_TRUNC('month', sale_date)
),
total_salaries AS (
    SELECT
        month,
        SUM(bonus + base_salary) AS total_salary
    FROM manager_salaries
    GROUP BY month
),
product_costs AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        SUM(sale_qty * sale_price * 0.9) AS total_cost
    FROM public.v_fact_sale
    WHERE sale_date BETWEEN '2014-01-01' AND '2014-12-31'
    GROUP BY DATE_TRUNC('month', sale_date)
)
SELECT
    ms.month,
    ms.total_sales,
    ts.total_salary,
    ms.total_sales - (pc.total_cost + ts.total_salary) AS profit_amount
FROM monthly_sales ms
JOIN total_salaries ts ON ms.month = ts.month
JOIN product_costs pc ON ms.month = pc.month
ORDER BY ms.month;
```