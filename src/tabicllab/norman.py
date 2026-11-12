"""Read-only access to the processed Norman et al. (2019) Perturb-seq cache.

The cache is the one the sibling project ssl-lab builds and grades with
(``tokens_meta.npz``, ``splits.json``, ``de_genes.json`` and ``manifest.json``
under ``data/norman2019``, a link into the shared data lake). This lab reads it
and never writes to it, so both projects score the same cells, the same split and
the same genes.

Memory is the constraint on the development laptop, so the expression matrix is
never loaded whole. ``hvg_X`` is 109,737 cells by 5,000 genes of float32 (2.2 GB)
and is stored uncompressed inside the ``.npz`` archive, which means its bytes sit
contiguously in the file. :func:`mmap_npz_member` maps them read-only in place,
and every mean below reads only the rows of the cells it averages. The largest
group, the 11,726 control cells, is about 235 MB.

Conventions, chosen to match ssl-lab's additive reference and Round 8 scoring
exactly (the parity test in ``tests/test_norman_parity.py`` checks it):

- expression is the cache's per-gene normalized ``hvg_X`` (log1p of counts per
  10,000), over the 5,000 highly variable genes;
- the control mean is over *all* control cells;
- a single gene's effect uses that single's cells on the *training* side of the
  combination split; a pair's measured profile uses the pair's own cells, which
  under this split all sit in one split (training, validation or test);
- each mean is computed in float32, as the cache stores it, then cast to float64.
"""

from __future__ import annotations

import json
import struct
import zipfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

CONTROL = "control"
SEP = "+"
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_HEADER_SIGNATURE = 0x04034B50


def mmap_npz_member(npz_path: str | Path, member: str) -> np.memmap:
    """Map one array of an *uncompressed* ``.npz`` archive read-only, without copying it.

    ``np.savez`` stores each array as a ``.npy`` file inside a zip archive. When the
    archive is not compressed, the array's bytes lie contiguously in the archive, so
    they can be memory-mapped at their offset: the zip local-file header gives where
    the member starts, and the ``.npy`` header gives the dtype, shape and where the
    data starts. Raises if the member is compressed, since its bytes are then not
    the array's.
    """
    npz_path = Path(npz_path)
    name = member if member.endswith(".npy") else f"{member}.npy"
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{name} in {npz_path} is compressed; it cannot be memory-mapped")
    with open(npz_path, "rb") as f:
        f.seek(info.header_offset)
        fields = _LOCAL_HEADER.unpack(f.read(_LOCAL_HEADER.size))
        if fields[0] != _LOCAL_HEADER_SIGNATURE:
            raise ValueError(f"bad zip local header for {name} in {npz_path}")
        name_len, extra_len = fields[9], fields[10]
        f.seek(info.header_offset + _LOCAL_HEADER.size + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    return np.memmap(npz_path, dtype=dtype, mode="r", shape=shape,
                     order="F" if fortran else "C", offset=offset)


def pair_genes(name: str) -> tuple[str, str]:
    """``"AHR+KLF1"`` -> ``("AHR", "KLF1")``; raises for anything but a two-gene pair."""
    genes = [g for g in str(name).split(SEP) if g and g != CONTROL]
    if len(genes) != 2:
        raise ValueError(f"{name!r} is not a two-gene combination")
    return genes[0], genes[1]


@dataclass
class NormanCache:
    """The processed Norman cache, with the expression matrix memory-mapped.

    Small per-cell arrays (perturbation index, control flag, split code) are held in
    memory; ``hvg_X`` stays on disk and is read row group by row group.
    """

    cache_dir: Path
    hvg_X: np.memmap
    pert_id: np.ndarray
    is_control: np.ndarray
    split_combo: np.ndarray
    pert_names: np.ndarray
    gene_ids: np.ndarray
    splits: dict
    de_genes: dict
    manifest: dict
    _index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {str(n): i for i, n in enumerate(self.pert_names)}

    @property
    def n_genes(self) -> int:
        return int(self.hvg_X.shape[1])

    # ------------------------------------------------------------------ cells
    def cells(self, name: str, split: str | None = None) -> np.ndarray:
        """Row indices of a perturbation's cells, optionally on one side of the combination split."""
        if name not in self._index:
            raise KeyError(f"no perturbation named {name!r}")
        mask = self.pert_id == self._index[name]
        if split is not None:
            mask &= self.split_combo == SPLIT_CODE[split]
        return np.flatnonzero(mask)

    def control_cells(self) -> np.ndarray:
        return np.flatnonzero(self.is_control.astype(bool))

    def pair_split(self, name: str) -> str:
        """Which side of the combination split a pair is on (its cells are all on one side)."""
        combo = self.splits["combo"]
        for split, key in (("test", "test_perts"), ("val", "val_perts"), ("train", "train_perts")):
            if name in combo[key]:
                return split
        raise KeyError(f"{name!r} is not in the combination split")

    def pairs(self, split: str) -> list[str]:
        key = {"train": "train_perts", "val": "val_perts", "test": "test_perts"}[split]
        return [p for p in self.splits["combo"][key] if SEP in p]

    # ------------------------------------------------------------------ means
    def rows(self, rows: np.ndarray) -> np.ndarray:
        """The expression rows of these cells, read from disk (float32, in memory)."""
        return np.asarray(self.hvg_X[np.sort(np.asarray(rows))])

    def mean(self, rows: np.ndarray) -> np.ndarray:
        """Mean profile (G,) of these cells, float32 arithmetic cast to float64 as ssl-lab does."""
        rows = np.asarray(rows)
        if rows.size == 0:
            raise ValueError("no cells to average")
        return self.rows(rows).mean(0).astype(np.float64)

    @cached_property
    def control_mean(self) -> np.ndarray:
        return self.mean(self.control_cells())

    def single_delta(self, gene: str) -> np.ndarray:
        """A single gene's measured effect (G,): its training cells' mean minus the control mean."""
        rows = self.cells(gene, "train")
        if rows.size == 0:
            raise ValueError(f"single {gene!r} has no training cells")
        return self.mean(rows) - self.control_mean

    def pair_mean(self, name: str) -> np.ndarray:
        """A pair's measured mean profile (G,) from all its cells (they sit on one side of the split)."""
        return self.mean(self.cells(name, self.pair_split(name)))

    def scored_genes(self, name: str, top_k: int = 20) -> np.ndarray:
        """The pair's top-``k`` differentially expressed genes, the set ssl-lab scores on."""
        top = self.de_genes["per_pert"][name]["top_idx"][:top_k]
        return np.asarray([i for i in top if i < self.n_genes], dtype=int)


def load_norman(cache_dir: str | Path = "data/norman2019") -> NormanCache:
    """Open the cache read-only; the expression matrix is memory-mapped, not loaded."""
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / "tokens_meta.npz"
    small = np.load(npz_path, allow_pickle=True)
    return NormanCache(
        cache_dir=cache_dir,
        hvg_X=mmap_npz_member(npz_path, "hvg_X"),
        pert_id=small["pert_id"],
        is_control=small["is_control"],
        split_combo=small["split_combo"],
        pert_names=small["pert_names"],
        gene_ids=small["gene_ids"],
        splits=json.loads((cache_dir / "splits.json").read_text()),
        de_genes=json.loads((cache_dir / "de_genes.json").read_text()),
        manifest=json.loads((cache_dir / "manifest.json").read_text()),
    )
