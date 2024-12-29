#!/usr/bin/env python
"""
Description:
    Django's command-line utility for administrative tasks.
"""

import os
import sys


def main() -> None:
    """
    Description:
        Запускает административные задачи Django.
        Устанавливает переменную окружения DJANGO_SETTINGS_MODULE
        и выполняет команду из командной строки.

    Raises:
        ImportError: Если Django не удалось импортировать.
    """
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blog_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()