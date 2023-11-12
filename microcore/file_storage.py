"""
File storage functions
"""
import json
import os
import shutil
from pathlib import Path
import chardet

from ._env import env, config


class Storage:
    @property
    def storage_path(self) -> Path:
        return Path(config().STORAGE_PATH)

    @property
    def default_ext(self) -> str | None:
        ext = config().STORAGE_DEFAULT_FILE_EXT
        if ext and not ext.startswith("."):
            ext = "." + ext
        return ext

    @property
    def default_encoding(self) -> str:
        return config().DEFAULT_ENCODING

    def read(self, name: str, encoding: str = None):
        encoding = encoding or self.default_encoding
        if not os.path.isabs(name) and not name.startswith("./"):
            if "." in name:
                parts = name.split(".")
                name = ".".join(parts[:-1])
                ext = "." + parts[-1]
            else:
                ext = self.default_ext
                if not os.path.exists(f"{self.storage_path}/{name}{ext}"):
                    ext = ""
            name = f"{self.storage_path}/{name}{ext}"
        if encoding is None:
            with open(name, "rb") as f:
                rawdata = f.read()
            result = chardet.detect(rawdata)
            encoding = result["encoding"]
            return rawdata.decode(encoding)

        with open(name, "r", encoding=encoding) as f:
            return f.read()

    def write_json(self, name, data, rewrite_existing: bool = False):
        return self.write(name, json.dumps(data, indent=4), rewrite_existing)

    def read_json(self, name):
        return json.loads(self.read(name))

    def write(
        self,
        name: str,
        content: str = None,
        rewrite_existing: bool = False,
        encoding: str = None,
    ) -> str | os.PathLike:
        """
        :return: str File name for further usage
        """
        encoding = encoding or self.default_encoding
        if content is None:
            content = name
            name = f"out{self.default_ext}"

        base_name = Path(name).with_suffix("")
        ext = Path(name).suffix or self.default_ext

        counter = 0
        while True:
            file_name = f"{base_name}{'_%d' % counter if counter else ''}{ext}"  # noqa
            full_path = self.storage_path / file_name
            if not full_path.is_file() or rewrite_existing:
                break
            counter += 1

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding=encoding)
        return file_name

    def clean(self, path: str):
        """
        Removes the directory specified by `path` within the `storage_path`.
        :raises ValueError: If the path is outside the storage area.
        """
        full_path = (self.storage_path / path).resolve()

        # Verify that the path is inside the storage_path
        if self.storage_path.resolve() not in full_path.parents:
            raise ValueError("Cannot delete directories outside the storage path.")

        if full_path.exists() and full_path.is_dir():
            shutil.rmtree(full_path)


storage = Storage()
"""
File system operations within the storage folder.

See `Storage` for details.

Related configuration options:

    - `microcore.config.Config.STORAGE_PATH`
    - `microcore.config.Config.DEFAULT_ENCODING`
"""
