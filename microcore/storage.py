"""
File storage functions
"""
import shutil
import chardet
import os

from .env import env


class _Storage:
    @property
    def storage_path(self):
        return env().config.STORAGE_PATH

    @property
    def default_encoding(self):
        return env().config.DEFAULT_ENCODING

    def read(self, name: str, encoding: str = None):
        encoding = encoding or self.default_encoding
        if not os.path.isabs(name) and not name.startswith("./"):
            if "." in name:
                parts = name.split(".")
                name = ".".join(parts[:-1])
                ext = parts[-1]
            else:
                ext = "txt"
            name = f"{self.storage_path}/{name}.{ext}"

        if encoding is None:
            with open(name, "rb") as f:
                rawdata = f.read()
            result = chardet.detect(rawdata)
            encoding = result["encoding"]
            return rawdata.decode(encoding)
        else:
            with open(name, "r", encoding=encoding) as f:
                return f.read()

    def write(
        self,
        name: str,
        content: str = None,
        rewrite_existing: bool = False,
        encoding: str = None,
    ):
        encoding = encoding or self.default_encoding
        """
        :return: str File name for further usage
        """
        if content is None:
            content = name
            name = "out.txt"

        if "." in name:
            parts = name.split(".")
            name = ".".join(parts[:-1])
            ext = parts[-1]
        else:
            ext = "txt"

        counter = 0
        while True:
            file_name = f'{name}{"_%d" % counter if counter else ""}.{ext}'
            full_path = f"{self.storage_path}/{file_name}"
            if not os.path.isfile(full_path) or rewrite_existing:
                break
            counter += 1

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding=encoding) as f:
            f.write(content)
        return file_name

    def clean(self, path: str):
        full_path = f"{self.storage_path}/{path}"
        os.path.exists(full_path) and shutil.rmtree(full_path)


storage = _Storage()
