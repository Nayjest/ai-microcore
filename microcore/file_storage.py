"""
File storage functions
"""
import json
import os
import shutil
from pathlib import Path
import chardet

from ._env import config
from .utils import file_link


class Storage:
    @property
    def path(self) -> Path:
        return Path(config().STORAGE_PATH)

    @property
    def default_ext(self) -> str | None:
        ext = config().STORAGE_DEFAULT_FILE_EXT
        if ext and not ext.startswith("."):
            ext = "." + ext
        return ext

    def file_link(self, file_name: str | Path) -> str:
        """Returns file name in format displayed in PyCharm console as a link."""
        return file_link(self.path / file_name)

    @property
    def default_encoding(self) -> str:
        return config().DEFAULT_ENCODING

    def exists(self, name: str | Path) -> bool:
        return (self.path / name).exists()

    def abs_path(self, name: str | Path) -> Path:
        if os.path.isabs(name):
            return Path(name)
        return self.path / name

    def read(self, name: str | Path, encoding: str = None):
        name = str(name)
        encoding = encoding or self.default_encoding
        if not os.path.isabs(name) and not name.startswith("./"):
            if "." in name:
                parts = name.split(".")
                name = ".".join(parts[:-1])
                ext = "." + parts[-1]
            else:
                ext = self.default_ext
                if not self.exists(f"{name}{ext}"):
                    ext = ""
            name = f"{self.path}/{name}{ext}"
        if encoding is None:
            with open(name, "rb") as f:
                rawdata = f.read()
            result = chardet.detect(rawdata)
            encoding = result["encoding"]
            return rawdata.decode(encoding)

        with open(name, "r", encoding=encoding) as f:
            return f.read()

    def write_json(
            self,
            name: str | Path,
            data,
            rewrite_existing: bool = True,
            backup_existing: bool = True,
    ):
        return self.write(
            name, json.dumps(data, indent=4), rewrite_existing, backup_existing
        )

    def read_json(self, name: str | Path):
        return json.loads(self.read(name))

    def write(
            self,
            name: str | Path,
            content: str = None,
            rewrite_existing: bool = True,
            backup_existing: bool = True,
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

        file_name = f"{base_name}{ext}"
        if (self.path / file_name).is_file() and (
                backup_existing or not rewrite_existing
        ):
            counter = 1
            while True:
                file_name1 = f"{base_name}_{counter}{ext}"  # noqa
                if not (self.path / file_name1).is_file():
                    break
                counter += 1
            if not rewrite_existing:
                file_name = file_name1
            elif backup_existing:
                os.rename(self.path / file_name, self.path / file_name1)
        (self.path / file_name).parent.mkdir(parents=True, exist_ok=True)
        (self.path / file_name).write_text(content, encoding=encoding)
        return file_name

    def clean(self, path: str | Path):
        """
        Removes the directory specified by `path` within the `storage_path`.
        :raises ValueError: If the path is outside the storage area.
        """
        full_path = (self.path / path).resolve()

        # Verify that the path is inside the storage_path
        if self.path.resolve() not in full_path.parents:
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
