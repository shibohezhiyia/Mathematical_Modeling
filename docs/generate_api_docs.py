"""
API 文档自动生成脚本

从 core/ 模块的 docstring 中提取文档，生成 Markdown 格式的 API 文档。
"""
import os
import sys
import inspect
import importlib
import pkgutil
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_object_signature(obj) -> str:
    """获取对象的签名字符串"""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "(...)"


def format_docstring(doc: str) -> str:
    """格式化 docstring 为 Markdown"""
    if not doc:
        return "*暂无文档*"
    lines = doc.strip().split('\n')
    result = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('>>>'):
            if not in_code:
                result.append('```python')
                in_code = True
            result.append(stripped.replace('>>> ', '').replace('>>>', ''))
        elif stripped.startswith('...'):
            result.append(stripped.replace('... ', '').replace('...', ''))
        elif in_code and not stripped:
            result.append('```')
            in_code = False
            result.append('')
        else:
            if in_code and not stripped.startswith('>>>'):
                result.append('```')
                in_code = False
            result.append(line)
    if in_code:
        result.append('```')
    return '\n'.join(result)


def extract_module_docs(module_name: str) -> Dict[str, Any]:
    """提取单个模块的文档信息"""
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        return {'error': str(e)}

    classes = []
    functions = []

    for name in dir(module):
        if name.startswith('_'):
            continue
        obj = getattr(module, name)
        if inspect.isclass(obj):
            methods = []
            for mname in dir(obj):
                if mname.startswith('_') and mname != '__init__':
                    continue
                meth = getattr(obj, mname)
                if callable(meth):
                    methods.append({
                        'name': mname,
                        'signature': get_object_signature(meth),
                        'doc': format_docstring(meth.__doc__ or ''),
                    })
            classes.append({
                'name': name,
                'signature': get_object_signature(obj),
                'doc': format_docstring(obj.__doc__ or ''),
                'methods': methods,
            })
        elif inspect.isfunction(obj):
            functions.append({
                'name': name,
                'signature': get_object_signature(obj),
                'doc': format_docstring(obj.__doc__ or ''),
            })

    return {
        'module': module_name,
        'doc': format_docstring(module.__doc__ or ''),
        'classes': classes,
        'functions': functions,
    }


def generate_markdown(docs: List[Dict[str, Any]]) -> str:
    """生成 Markdown 文档"""
    lines = []
    lines.append('# Mathematical Modeling API 文档')
    lines.append('')
    lines.append('> 自动生成于 API 文档生成脚本')
    lines.append('')
    lines.append('---')
    lines.append('')

    for mod_doc in docs:
        if 'error' in mod_doc:
            continue
        lines.append(f"## `{mod_doc['module']}`")
        lines.append('')
        if mod_doc['doc'] and mod_doc['doc'] != '*暂无文档*':
            lines.append(mod_doc['doc'])
            lines.append('')

        # Functions
        if mod_doc['functions']:
            lines.append('### 函数')
            lines.append('')
            for fn in mod_doc['functions']:
                lines.append(f"#### `{fn['name']}{fn['signature']}`")
                lines.append('')
                lines.append(fn['doc'])
                lines.append('')

        # Classes
        for cls in mod_doc['classes']:
            lines.append(f"### `{cls['name']}{cls['signature']}`")
            lines.append('')
            lines.append(cls['doc'])
            lines.append('')

            if cls['methods']:
                lines.append('**方法：**')
                lines.append('')
                for meth in cls['methods']:
                    lines.append(f"- `{meth['name']}{meth['signature']}`")
                    if meth['doc'] and meth['doc'] != '*暂无文档*':
                        lines.append(f"  - {meth['doc'].replace(chr(10), ' ')}")
                lines.append('')

        lines.append('---')
        lines.append('')

    return '\n'.join(lines)


def main():
    core_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core')
    modules = []
    for _, name, _ in pkgutil.iter_modules([core_dir]):
        modules.append(f'core.{name}')

    docs = []
    for mod_name in sorted(modules):
        print(f'Processing {mod_name}...')
        doc = extract_module_docs(mod_name)
        docs.append(doc)

    md = generate_markdown(docs)

    output_path = os.path.join(os.path.dirname(__file__), 'API.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f'API documentation generated: {output_path}')


if __name__ == '__main__':
    main()
