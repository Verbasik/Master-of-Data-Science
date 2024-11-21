import requests 
from bs4 import BeautifulSoup

def fetch_data(url: str) -> str:
    """
    Description:
        Получает HTML-код веб-страницы по указанному URL.

    Copy
    Args:
        url: URL веб-страницы.

    Returns:
        HTML-код страницы в виде строки.

    Raises:
        requests.exceptions.RequestException: Если запрос не удался.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе данных: {e}")
        return ""

def parse_data(html: str) -> list:
    """
    Description:
        Парсит HTML-код и извлекает заголовки статей.

    Copy
    Args:
        html: HTML-код страницы.

    Returns:
        Список заголовков статей.
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        headers = soup.find_all('h2')
        return [header.text for header in headers]
    except Exception as e:
        print(f"Ошибка при парсинге данных: {e}")
        return []
    
if __name__ == "__main__":
    url = "https://example.com"
    html_content = fetch_data(url)
    headers = parse_data(html_content)

    for header in headers:
        print(header)