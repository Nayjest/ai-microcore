"""
File storage functionality.


Provides a Storage class for file operations within a configured storage directory.
Supports automatic file numbering, backups, JSON serialization, encoding detection,
file copying, directory listing, etc.

Usage:
    from microcore import storage
    storage.write("data.txt", "content")
    content = storage.read("data.txt")
"""

import fnmatch
import json
import logging
import os
from dataclasses import dataclass, field
import shutil
from pathlib import Path
import chardet

from ._env import config
from .utils import file_link, list_files

_missing = object()
"""Sentinel value to distinguish between None and 'not provided'."""


@dataclass
class Storage:
    custom_path: str = field(default="")
    file_number_placeholder: str | None = field(default="<n>")
    """
    When writing files, if the file name contains this placeholder,
    it will be replaced with an incrementing number to avoid overwriting existing files,
    starting from 1, or considered in naming of backup copies when rewriting same files.
    For example, if the file name is "output_<n>.txt", the first file will be
    "output_1.txt", the second "output_2.txt", and so on.
    If set to None or an empty string, this feature is disabled.
    """

    def __call__(
        self,
        custom_path: str,
        file_number_placeholder: str | None = "<n>"
    ) -> "Storage":
        """
        Creates a new Storage instance with a customized path and other attributes.
        """
        return Storage(
            custom_path=custom_path,
            file_number_placeholder=file_number_placeholder
        )

    @property
    def path(self) -> Path:
        """Returns the file storage path as a Path object."""
        return Path(str(self.custom_path) or config().STORAGE_PATH)

    @property
    def default_ext(self) -> str | None:
        """Returns the default file extension according to configuration."""
        ext = config().STORAGE_DEFAULT_FILE_EXT
        if ext and not ext.startswith("."):
            ext = "." + ext
        return ext

    def file_link(self, file_name: str | Path) -> str:
        """Returns file name in format displayed in PyCharm console as a link."""
        return file_link(self.path / file_name)

    @property
    def default_encoding(self) -> str:
        """Returns the default file encoding according to configuration."""
        return config().DEFAULT_ENCODING

    def exists(self, name: str | Path) -> bool:
        """Checks if a file or directory exists within the storage path."""
        if isinstance(name, Path):
            name = name.as_posix()
        return (self.path / name).exists()

    def abs_path(self, name: str | Path) -> Path:
        """
        Returns the absolute path of the file or directory within the storage path.
        """
        if os.path.isabs(name):
            return Path(name)
        return self.path / name

    def relative_path(self, name: str | Path) -> Path:
        """
        Returns the relative path of the file or directory within the storage path.
        """
        return Path(name).relative_to(self.path)

    def read(
        self,
        name: str | Path,
        encoding: str = None,
        default=_missing,
        binary: bool = False,
    ) -> str | bytes:
        """
        Read file from the storage and return its content.
        Args:
            name (str | Path): File name within the storage.
            encoding (str, optional): File encoding. Defaults to config().DEFAULT_ENCODING (utf-8).
            default (any, optional): Default value to return if file not found.
                If not provided, FileNotFoundError is raised.
            binary (bool, optional): Whether to read the file in binary mode. Defaults to False.
        Returns:
            str | bytes: File content as string or bytes.
        """
        name = str(name)
        if binary:
            if encoding is not None:
                logging.warning("Encoding is ignored when reading binary files")
        else:
            if not encoding:
                encoding = self.default_encoding
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
        try:
            if binary or encoding is None:
                with open(name, "rb") as f:
                    rawdata = f.read()
                    if binary:
                        return rawdata
                result = chardet.detect(rawdata)
                encoding = result["encoding"]
                return rawdata.decode(encoding)

            with open(name, "r", encoding=encoding) as f:
                return f.read()
        except FileNotFoundError as e:
            if default is not _missing:
                return default
            raise e

    def write_json(
        self,
        name: str | Path,
        data,
        rewrite_existing: bool = True,
        backup_existing: bool = True,
        ensure_ascii: bool = False,
    ):
        """
        Writes JSON data to a file in the storage.
        """
        serialized_data = json.dumps(data, indent=4, ensure_ascii=ensure_ascii)
        return self.write(name, serialized_data, rewrite_existing, backup_existing)

    def read_json(self, name: str | Path, default=_missing):
        """
        Reads JSON data from a file in the storage.
        """
        try:
            return json.loads(self.read(name))
        except FileNotFoundError as e:
            if default is not _missing:
                return default
            raise e

    def delete(self, target: str | Path | list[str | Path]) -> bool:
        """
        Remove the file or directory specified by target path
        related to the storage root (if exists).
        Args:
            target (str | Path | list[str | Path]):
                The target file / directory path or list of paths.
        Returns:
            bool:
                True if any file(s) or director(ies) was deleted,
                False if no target paths existed.
        """
        if isinstance(target, list):
            any_deletions_performed = False
            for t in target:
                if self.delete(t):
                    any_deletions_performed = True
            return any_deletions_performed
        path = (self.path / target).resolve()
        if not path.exists():
            return False
        if path.is_dir():
            shutil.rmtree(path)
        else:
            os.remove(path)
        return True

    def write(
        self,
        name: str | Path,
        content: str | bytes = _missing,
        rewrite_existing: bool = None,
        backup_existing: bool = None,
        encoding: str = None,
        append: bool = False,
    ) -> str | os.PathLike:
        """
        Writes file to the storage.
        Args:
            name (str | Path): File name within the storage.
            content (str | bytes): Content to write.
                If not provided, uses `name` as content and defaults the file name.
            rewrite_existing (bool, optional): Whether to overwrite existing files.
                Defaults to True
            backup_existing (bool, optional): Whether to back up existing files
                in case of overwrite.
                Defaults to True if not appending, else False.
            encoding (str, optional): Defaults to config().DEFAULT_ENCODING (utf-8).
            append (bool, optional): Whether to append to the file if it exists.
                Defaults to False.
        Returns:
            str | os.PathLike: The actual file name used for writing.
                (may differ from `name` argument
                if rewrite_existing: False is used)
        """
        if content == _missing:
            content = name
            name = f"out{self.default_ext}"

        if isinstance(content, bytes):
            if encoding is not None:
                logging.warning("Encoding is ignored when writing bytes content")

        if rewrite_existing is None:
            rewrite_existing = True
        elif append and rewrite_existing is False:
            raise ValueError("Cannot both append and prevent rewriting existing files")
        if backup_existing is None:
            backup_existing = not append
        encoding = encoding or self.default_encoding

        base_name = Path(name).with_suffix("")
        ext = Path(name).suffix or self.default_ext

        file_name = f"{base_name}{ext}"
        use_file_num_pattern = (
            self.file_number_placeholder
            and self.file_number_placeholder in file_name
        )
        if use_file_num_pattern and append:
            raise ValueError(
                f"Cannot use file number pattern '{self.file_number_placeholder}' "
                "when appending or preventing rewriting existing files"
            )
        if (self.path / file_name).is_file() and (
            backup_existing or not rewrite_existing
        ) or use_file_num_pattern:
            counter = 1
            while True:
                if use_file_num_pattern:
                    fn_incremented = file_name.replace(self.file_number_placeholder, str(counter))
                else:
                    fn_incremented = f"{base_name}_{counter}{ext}"  # noqa
                if not (self.path / fn_incremented).is_file():
                    break
                counter += 1
            if not rewrite_existing:
                file_name = fn_incremented
            elif backup_existing:
                if use_file_num_pattern:
                    file_name = file_name.replace(self.file_number_placeholder, "1")
                if file_name != fn_incremented:
                    os.rename(self.path / file_name, self.path / fn_incremented)
        (self.path / file_name).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            with (self.path / file_name).open(mode="ab" if append else "wb") as f:
                f.write(content)
        else:
            with (self.path / file_name).open(mode="a" if append else "w", encoding=encoding) as f:
                f.write(content)
        return file_name

    def clean(self, path: str | Path):
        """
        Removes the directory specified by `path` within the `storage_path`.
        :raises ValueError: If the path is outside the storage area.
        @deprecated use storage.delete() instead
        """
        full_path = (self.path / path).resolve()

        # Verify that the path is inside the storage_path
        if self.path.resolve() not in full_path.parents:
            raise ValueError("Cannot delete directories outside the storage path.")

        if full_path.exists() and full_path.is_dir():
            shutil.rmtree(full_path)

    def list_files(
        self,
        target_dir: str | Path = "",
        exclude: list[str | Path] = None,
        relative_to: str | Path = None,
        absolute: bool = False,
        posix: bool = False,
    ) -> list[Path | str]:
        """
        List files in a target directory, excluding those that match given patterns.

        Args:
            target_dir (str | Path): The directory to search in.
            exclude (list[str | Path], optional): Patterns of files to exclude.
            relative_to (str | Path, optional): Base directory for relative paths.
                If None, paths are relative to `target_dir`. Defaults to None.
            absolute (bool, optional): If True, returns absolute paths. Defaults to False.
            posix (bool, optional): If True, returns posix paths. Defaults to False.
        """
        target_dir = self.path / target_dir
        return list_files(
            target_dir=target_dir,
            exclude=exclude,
            relative_to=relative_to,
            absolute=absolute,
            posix=posix,
        )

    def copy(self, src: str | Path, dest: str | Path, exclude=None):
        """
        Copy a file or folder from src to dest, overwriting content,
        but skipping paths in exceptions.
        Supports Unix shell-style wildcards in exceptions. Accepts Path objects.

        Args:
            src (Path): Source file or directory Path object.
            dest (Path): Destination file or directory Path object.
            exclude (list of str, optional):
                List of Unix shell-style wildcard patterns relative to src.
                These paths will be excluded from the copy. Defaults to None.
        """
        src = self.path / Path(src)
        dest = self.path / Path(dest)
        exclude = exclude or []
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            files = list_files(src, exclude)
            for f in files:
                if (src / f).is_dir():
                    (dest / f).mkdir(parents=True, exist_ok=True)
                elif (src / f).is_file():
                    (dest / f).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src / f, dest / f)
                else:
                    raise ValueError(f"{src / f} is not a file or directory")
        elif src.is_file():
            if not any(fnmatch.fnmatch(src.name, pattern) for pattern in exclude):
                if dest.is_dir():
                    dest = dest / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        else:
            raise ValueError(f"{src} is not a file or directory")


storage = Storage()
"""
File system operations within the storage folder.

See `Storage` for details.

Related configuration options:

    - `microcore.config.Config.STORAGE_PATH`
    - `microcore.config.Config.DEFAULT_ENCODING`
"""
