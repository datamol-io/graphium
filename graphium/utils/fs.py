"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""

from typing import Union
from typing import Optional

import os
import io
import platformdirs
import pathlib

from tqdm.auto import tqdm
import fsspec


def get_cache_dir(suffix: str = None, create: bool = True) -> pathlib.Path:
    """Get a local cache directory. You can append a suffix folder to it and optionnaly create
    the folder if it doesn't exist.
    """

    cache_dir = pathlib.Path(platformdirs.user_cache_dir(appname="graphium"))

    if suffix is not None:
        cache_dir /= suffix

    if create:
        cache_dir.mkdir(exist_ok=True, parents=True)

    return cache_dir


def get_mapper(path: Union[str, os.PathLike]):
    """Get the fsspec mapper.
    Args:
        path: a path supported by `fsspec` such as local, s3, gcs, etc.
    """
    return fsspec.get_mapper(str(path))


def get_basename(path: Union[str, os.PathLike]):
    """Get the basename of a file or a folder.
    Args:
        path: a path supported by `fsspec` such as local, s3, gcs, etc.
    """
    path = str(path)
    mapper = get_mapper(path)
    clean_path = path.rstrip(mapper.fs.sep)
    return str(clean_path).split(mapper.fs.sep)[-1]


def get_extension(path: Union[str, os.PathLike]):
    """Get the extension of a file.
    Args:
        path: a path supported by `fsspec` such as local, s3, gcs, etc.
    """
    basename = get_basename(path)
    return basename.split(".")[-1]


def exists(path: Union[str, os.PathLike, fsspec.core.OpenFile, io.IOBase]):
    """Check whether a file exists.
    Args:
        path: a path supported by `fsspec` such as local, s3, gcs, etc.
    """

    if isinstance(path, fsspec.core.OpenFile):
        return path.fs.exists(path.path)

    elif isinstance(path, (str, pathlib.Path)):
        mapper = get_mapper(str(path))
        return mapper.fs.exists(path)

    else:
        # NOTE(hadim): file-like objects always exist right?
        return True


def exists_and_not_empty(path: Union[str, os.PathLike]):
    """Check whether a directory exists and is not empty."""

    if not exists(path):
        return False

    fs = get_mapper(path).fs

    return len(fs.ls(path)) > 0


def mkdir(path: Union[str, os.PathLike], exist_ok: bool = True):
    """Create directory including potential parents."""
    fs = get_mapper(path).fs
    fs.mkdirs(path, exist_ok=exist_ok)


def rm(path: Union[str, os.PathLike], recursive=False, maxdepth=None):
    """Delete a file or a directory with all nested files."""
    fs = get_mapper(path).fs
    fs.rm(path, recursive=recursive, maxdepth=maxdepth)


def join(*paths):
    """Join paths together. The first element determine the
    filesystem to use (and so the separator.
    Args:
        paths: a list of paths supported by `fsspec` such as local, s3, gcs, etc.
    """
    paths = [str(path) for path in paths]
    source_path = paths[0]
    fs = get_mapper(source_path).fs
    full_path = fs.sep.join(paths)
    return full_path


def get_size(file: Union[str, os.PathLike, io.IOBase, fsspec.core.OpenFile]) -> Optional[int]:
    """Get the size of a file given its path. Return None if the
    size can't be retrieved.
    """

    if isinstance(file, io.IOBase) and hasattr(file, "name"):
        fs_local = fsspec.filesystem("file")
        file_size = fs_local.size(getattr(file, "name"))

    elif isinstance(file, (str, pathlib.Path)):
        fs = get_mapper(str(file)).fs
        file_size = fs.size(str(file))

    elif isinstance(file, fsspec.core.OpenFile):
        file_size = file.fs.size(file.path)

    else:
        file_size = None

    return file_size


def copy(
    source: Union[str, os.PathLike, io.IOBase, fsspec.core.OpenFile],
    destination: Union[str, os.PathLike, io.IOBase, fsspec.core.OpenFile],
    chunk_size: int = None,
    force: bool = False,
    progress: bool = False,
    leave_progress: bool = True,
):
    """Copy one file to another location across different filesystem (local, S3, GCS, etc).

    Args:
        source: path or file-like object to copy from.
        destination: path or file-like object to copy to.
        chunk_size: the chunk size to use. If progress is enabled the chunk
            size is `None`, it is set to 2048.
        force: whether to overwrite the destination file it it exists.
        progress: whether to display a progress bar.
        leave_progress: whether to hide the progress bar once the copy is done.
    """

    if progress and chunk_size is None:
        chunk_size = 2048

    if isinstance(source, (str, os.PathLike)):
        source_file = fsspec.open(str(source), "rb")
    else:
        source_file = source

    if isinstance(destination, (str, os.PathLike)):
        # adapt the file mode of the destination depending on the source file.
        destination_mode = "wb"
        if hasattr(source_file, "mode"):
            destination_mode = "wb" if "b" in getattr(source_file, "mode") else "w"
        elif isinstance(source_file, io.BytesIO):
            destination_mode = "wb"
        elif isinstance(source_file, io.StringIO):
            destination_mode = "w"

        destination_file = fsspec.open(str(destination), destination_mode)
    else:
        destination_file = destination

    if not exists(source_file):
        raise ValueError(f"The file being copied does not exist: {source}")

    if not force and exists(destination_file):
        raise ValueError(f"The destination file to copy already exists: {destination}")

    with source_file as source_stream:
        with destination_file as destination_stream:
            if chunk_size is None:
                # copy without chunks
                destination_stream.write(source_stream.read())

            else:
                # copy with chunks

                # determine the size of the source file
                source_size = None
                if progress:
                    source_size = get_size(source)

                # init progress bar
                pbar = tqdm(
                    total=source_size,
                    leave=leave_progress,
                    disable=not progress,
                    unit="B",
                    unit_divisor=1024,
                    unit_scale=True,
                )

                # start the loop
                while True:
                    data = source_stream.read(chunk_size)
                    if not data:
                        break
                    destination_stream.write(data)
                    pbar.update(chunk_size)

                pbar.close()
