# django-blog-main/blog/templatetags/markdown_extras.py
from django import template
from django.utils.safestring import mark_safe
import markdown
from markdown.extensions import fenced_code, tables, toc
import re

register = template.Library()

def protect_math(text):
    # Защищаем блоки математических формул
    math_blocks = []
    
    def math_replace(match):
        math_blocks.append(match.group(0))
        return f'MATHBLOCK{len(math_blocks)-1}ENDMATH'
    
    # Сохраняем блоки формул
    text = re.sub(r'\$\$(.*?)\$\$', math_replace, text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', math_replace, text)
    
    return text, math_blocks

def restore_math(text, math_blocks):
    # Восстанавливаем блоки математических формул
    for i, block in enumerate(math_blocks):
        text = text.replace(f'MATHBLOCK{i}ENDMATH', block)
    return text

@register.filter(name='markdownify')
def markdownify(text):
    # Защищаем математические выражения
    text, math_blocks = protect_math(text)
    
    # Преобразуем markdown
    html = markdown.markdown(text, extensions=[
        'markdown.extensions.fenced_code',
        'markdown.extensions.tables',
        'markdown.extensions.toc',
        'markdown.extensions.extra',
        'markdown.extensions.codehilite',
        'markdown.extensions.nl2br',
        'markdown.extensions.sane_lists'
    ])
    
    # Восстанавливаем математические выражения
    html = restore_math(html, math_blocks)
    
    return mark_safe(html)