"""
Filesystem functions
"""
import chardet
import dotenv
import os
dotenv.load_dotenv()

# Writeable directory for app working files
storage_path = os.getenv('MC_STORAGE_PATH', 'storage')
default_encoding = os.getenv('MC_DEFAULT_ENCODING', 'utf-8')


def read(name: str, encoding: str = None):
    if '.' in name:
        parts = name.split('.')
        name = '.'.join(parts[:-1])
        ext = parts[-1]
    else:
        ext = 'txt'
    full_path = f'{storage_path}/{name}.{ext}'
    if encoding is None:
        with open(full_path, 'rb') as f:
            rawdata = f.read()
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        return rawdata.decode(encoding)
    else:
        with open(full_path, 'r', encoding=encoding) as f:
            return f.read()


def write(name: str, content: str = None, rewrite_existing: bool = False, encoding: str = default_encoding):

    if content is None:
        content = name
        name = 'out.txt'

    if '.' in name:
        parts = name.split('.')
        name = '.'.join(parts[:-1])
        ext = parts[-1]
    else:
        ext = 'txt'

    counter = 0
    while True:
        file_name = f'{name}{"_%d" % counter if counter else ""}.{ext}'
        full_path = f'{storage_path}/{file_name}'
        if not os.path.isfile(full_path) or rewrite_existing:
            break
        counter += 1

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w', encoding=encoding) as f:
        f.write(content)
    return file_name
