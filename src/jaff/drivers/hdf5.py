"""
HDF5 file driver.

This module provides :class:`HDF5`, a high-level wrapper around :mod:`h5py`
that converts between JAFF's in-memory :class:`~jaff.types.HDF5Dict`
representation and on-disk HDF5 files, and can also export HDF5 data to CSV.

The :class:`~jaff.types.HDF5Dict` schema
-----------------------------------------
Each leaf in the nested dictionary corresponds to an HDF5 dataset and must
contain at least the following keys:

``_kind``
    ``"linear"`` for a plain 1-D/N-D array; ``"compound"`` for a NumPy
    structured (record) array.
``_data``
    The actual data (NumPy array or compatible).
``_dtype``
    For ``"linear"``: a JAFF dtype string (e.g. ``"f64"``).
    For ``"compound"``: a ``{field_name: dtype_string}`` mapping.
``_attrs``
    Optional ``{attr_name: attr_value}`` dictionary written as HDF5
    dataset attributes.
``_name``
    Optional human-readable column name stored as the ``_name`` HDF5
    attribute.

Non-leaf nodes in the dictionary produce HDF5 groups.  The special key
``_attrs`` at any level writes HDF5 attributes on the parent group/file root.
"""

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd

from ..types import HDF5Dict


class HDF5:
    """
    High-level HDF5 read/write driver backed by :mod:`h5py`.

    Provides methods to convert between :class:`~jaff.types.HDF5Dict` and
    HDF5 files, and to export HDF5 data to CSV.

    Parameters
    ----------
    compression : str or None, optional
        HDF5 compression filter to apply when creating datasets (e.g.
        ``"gzip"``).  ``None`` disables compression.  Defaults to ``None``.
    """

    def __init__(self, compression: str | None = None):
        """Initialise the HDF5 driver with an optional compression filter.

        Parameters
        ----------
        compression : str or None, optional
            HDF5 compression filter (e.g. ``"gzip"``).  ``None`` disables
            compression.  Defaults to ``None``.
        """
        self.compression = compression

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_dict(
        self,
        h5file: h5py.File | h5py.Group | Path | str,
        *,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> HDF5Dict:
        """
        Load an HDF5 file (or sub-group) into a nested :class:`~jaff.types.HDF5Dict`.

        Parameters
        ----------
        h5file : h5py.File, h5py.Group, Path, or str
            An open :class:`h5py.File`, an open :class:`h5py.Group` (parsed
            from that group onward), or a path to an HDF5 file on disk.  A
            ``"file.h5::/internal/group"`` string (``::`` delimiter) parses
            from the named internal group onward.
        include : str, list of str, or None, optional
            Keep only datasets whose own name — or the name of any parent group
            — matches one of these bare-name patterns (:func:`fnmatch.fnmatch`,
            so exact names and ``*``/``?``/``[seq]`` wildcards both work).
            Datasets filtered out are never read into memory.  ``None``
            (default) keeps everything.
        exclude : str, list of str, or None, optional
            Drop any dataset or group whose name, or any ancestor group name,
            matches one of these bare-name patterns.  Excluding a group drops
            its whole subtree.  Takes precedence over *include*.  ``None``
            (default) drops nothing.

        Returns
        -------
        HDF5Dict
            Nested dictionary representation of the HDF5 contents.
        """
        return HDF5Dict(h5file, include=include, exclude=exclude)

    def from_dict(
        self, h5file: str | Path, h5dict: dict | HDF5Dict, *, mode: str = "a"
    ) -> None:
        """
        Write an :class:`~jaff.types.HDF5Dict` to an HDF5 file.

        Opens (or creates) *h5file* and recursively resolves all datasets and
        groups from *h5dict*.

        Parameters
        ----------
        h5file : str or Path
            Path to the target HDF5 file.  Created if it does not exist;
            existing files are updated in-place.
        h5dict : dict or HDF5Dict
            Data to write.  Plain ``dict`` objects are automatically converted
            to :class:`~jaff.types.HDF5Dict`.
        mode : str, optional
            File open mode passed to :class:`h5py.File`.  Defaults to ``"a"``
            (append, keeping existing contents).  Use ``"w"`` to truncate the
            file first so stale contents are not retained.

        Returns
        -------
        None
        """
        if not isinstance(h5dict, HDF5Dict):
            h5dict = HDF5Dict(h5dict)
        with h5py.File(h5file, mode) as f:
            self.__resolve_to_h5(f, h5dict)

    _LINEAR_TABLE_STEM = "_group"

    def to_csv(self, h5file: str | Path, outdir: str | Path, sep: str = " ") -> None:
        """
        Export all datasets from an HDF5 file to CSV files under *outdir*.

        The HDF5 group hierarchy is mirrored as a directory tree under
        *outdir*, so datasets are never overwritten by same-named siblings in
        other groups.  Within each group directory, all ``"linear"`` datasets
        are combined into a single reserved table file
        (``_group.csv``), and each ``"compound"`` dataset produces its own file
        named after its HDF5 key.  The ``_name`` attribute is used only as a
        column label, never as a filesystem path.

        Parameters
        ----------
        h5file : str or Path
            Path to the source HDF5 file.
        outdir : str or Path
            Directory in which the CSV tree is written.  Created (including
            parents) if it does not exist.
        sep : str, optional
            Column separator character.  Defaults to a single space.

        Raises
        ------
        ValueError
            If two datasets would be written to the same output file (e.g. a
            ``"compound"`` dataset keyed like the reserved linear-table stem).

        Returns
        -------
        None
        """
        outdir = Path(outdir)
        if not outdir.exists():
            outdir.mkdir(parents=True)

        h5dict = HDF5Dict(h5file)
        self.__generate_csv(h5dict, outdir, "", sep, set())

    # ------------------------------------------------------------------
    # HDF5 write helpers
    # ------------------------------------------------------------------

    def __resolve_to_h5(self, h5file: h5py.File, h5dict: dict, path: str = "") -> None:
        """
        Recursively write *h5dict* into an open HDF5 file.

        Handles three kinds of entries:

        * ``_attrs`` key — write as HDF5 attributes on the current group/root.
        * Dict with ``_kind`` — delegate to :meth:`__create_dataset`.
        * Plain dict — create an HDF5 group and recurse.

        Parameters
        ----------
        h5file : h5py.File
            Open HDF5 file handle in write or append mode.
        h5dict : dict
            Current level of the nested dictionary to process.
        path : str, optional
            Current HDF5 group path being processed.  Empty string means root.

        Returns
        -------
        None
        """
        for key, val in h5dict.items():
            # "_attrs" is a metadata key, not a dataset — write its contents
            # as HDF5 attributes on the enclosing group or root object.
            if key == "_attrs":
                target = h5file[path] if path and path != "/" else h5file
                for a_key, a_val in val.items():
                    target.attrs[a_key] = a_val
                continue

            if isinstance(val, dict) and "_kind" in val:
                # Leaf node — resolve the dataset path and delegate.
                dataset_path = f"{path}/{key}".replace("//", "/")
                self.__create_dataset(h5file, dataset_path, val)
                continue

            if isinstance(val, dict):
                # Intermediate node — ensure the group exists and recurse.
                sub_path = f"{path}/{key}".replace("//", "/")
                h5file.require_group(sub_path)
                self.__resolve_to_h5(h5file, val, sub_path)

    def __create_dataset(self, file: h5py.File, path: str, props: dict[str, Any]) -> None:
        """
        Create or replace a single HDF5 dataset at *path*.

        Deletes any pre-existing dataset at the same path before creating the
        new one.  Applies the instance's compression filter.  Handles both
        ``"linear"`` (plain array) and ``"compound"`` (structured array) kinds.

        Parameters
        ----------
        file : h5py.File
            Open HDF5 file handle.
        path : str
            Absolute HDF5 path at which the dataset is created.
        props : dict[str, Any]
            Dataset descriptor dictionary with required key ``_kind`` and
            optional keys ``_data``, ``_dtype``, ``_attrs``, ``_name``.

        Raises
        ------
        ValueError
            If ``props["_kind"]`` is not ``"linear"`` or ``"compound"``.

        Returns
        -------
        None
        """
        # Remove any pre-existing dataset at this path to avoid conflicts.
        if path in file:
            del file[path]

        kwargs = {"compression": self.compression}
        kind = props.get("_kind")
        data = props.get("_data")
        dtype_spec = props.get("_dtype")

        if kind == "linear":
            # Map the JAFF dtype string to a NumPy dtype, if provided.
            dtype = (
                HDF5Dict._to_np().get(cast(str, dtype_spec))
                if isinstance(dtype_spec, str)
                else None
            )
            ds = file.create_dataset(path, data=data, dtype=dtype, **kwargs)

        elif kind == "compound":
            if isinstance(dtype_spec, dict):
                # Build a NumPy structured dtype from the field mapping.
                dtype = np.dtype(
                    [(k, HDF5Dict._to_np()[v]) for k, v in dtype_spec.items()]
                )
                ds = file.create_dataset(path, data=data, dtype=dtype, **kwargs)
            else:
                # No explicit dtype spec — let h5py infer from the data.
                ds = file.create_dataset(path, data=data, **kwargs)
        else:
            raise ValueError(f"Unknown _kind '{kind}' at {path}")

        # Write per-dataset attributes.
        if "_attrs" in props and props["_attrs"]:
            for a_key, a_val in props["_attrs"].items():
                ds.attrs[a_key] = a_val

        # Store the human-readable column name as a special HDF5 attribute.
        if props.get("_name") is not None:
            ds.attrs["_name"] = props["_name"]

    # ------------------------------------------------------------------
    # CSV export helper
    # ------------------------------------------------------------------

    def __generate_csv(
        self,
        data_dict: dict,
        outdir: Path,
        current_path: str,
        sep: str,
        written: set[Path],
    ) -> None:
        """
        Recursively traverse *data_dict* and write CSV files.

        The current group maps to the directory ``outdir / current_path``.
        All ``"linear"`` datasets at this level are combined into a single
        reserved table file inside that directory; each ``"compound"`` dataset
        produces its own file keyed by its HDF5 name.  Sub-groups descend into
        their own sub-directories, so same-named siblings never collide.

        Parameters
        ----------
        data_dict : dict
            Current level of the :class:`~jaff.types.HDF5Dict` tree.
        outdir : Path
            Root output directory.
        current_path : str
            Slash-separated path of the current group, relative to the root.
        sep : str
            Column separator character for the CSV files.
        written : set[Path]
            Output paths already written this run, used to detect collisions.

        Raises
        ------
        ValueError
            If a computed output path was already written this run.

        Returns
        -------
        None
        """
        # This group's directory mirrors its HDF5 path.
        group_dir = outdir.joinpath(*current_path.split("/")) if current_path else outdir

        def _write(df: pd.DataFrame, path: Path) -> None:
            if path in written:
                raise ValueError(f"Output path collision: {path}")
            written.add(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, sep=sep)

        linear_dfs: list[pd.DataFrame] = []
        for key, val in data_dict.items():
            # Skip metadata keys.
            if key == "_attrs":
                continue

            if isinstance(val, dict):
                if "_kind" in val:
                    if val["_kind"] == "linear":
                        # Accumulate linear columns for a combined CSV later;
                        # _name is only a column label, never a path.
                        col_name = val.get("_name", key)
                        linear_dfs.append(pd.DataFrame({col_name: val["_data"]}))
                    elif val["_kind"] == "compound":
                        # Each compound dataset becomes its own CSV file, keyed
                        # by its HDF5 name inside this group's directory.
                        df = pd.DataFrame(val["_data"])
                        _write(df, group_dir / f"{key}.csv")
                else:
                    # Sub-group — recurse into a nested directory.
                    self.__generate_csv(
                        val,
                        outdir,
                        f"{current_path}/{key}" if current_path else key,
                        sep,
                        written,
                    )

        # Write all accumulated linear columns as one reserved table per group.
        if linear_dfs:
            combined_df = pd.concat(linear_dfs, axis=1)
            _write(combined_df, group_dir / f"{self._LINEAR_TABLE_STEM}.csv")
