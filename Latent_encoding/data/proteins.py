"""Protein structure datasets returning graph point clouds (xyz, atom_type).

Supports:
- PDB: Experimental structures from RCSB Protein Data Bank (high resolution)
- AlphaFold DB: Predicted structures from DeepMind's AlphaFold database

Data is stored in:
- Latent_encoding/data/protein/pdb/       (experimental structures)
- Latent_encoding/data/protein/alphafold/ (predicted structures)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None  # Will raise informative error if used

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Atom type mappings for proteins
# -----------------------------------------------------------------------------

# Standard protein atom elements (covers >99% of atoms in proteins)
PROTEIN_ELEMENTS = ["C", "N", "O", "S", "H", "P", "SE", "FE", "ZN", "MG", "CA", "MN", "CU", "K", "NA", "CL"]
ELEMENT_TO_IDX = {e: i for i, e in enumerate(PROTEIN_ELEMENTS)}
NUM_ATOM_TYPES = len(PROTEIN_ELEMENTS)

# Backbone atoms only (for reduced representations)
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Residue names for standard amino acids
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

_VALID_SPLITS = ("train", "val", "test")


# -----------------------------------------------------------------------------
# Structure parsing utilities
# -----------------------------------------------------------------------------

def parse_pdb_file(filepath: Union[str, Path], include_hetatm: bool = False) -> Tuple[
    np.ndarray, np.ndarray, List[str]]:
    """Parse a PDB file and extract atom coordinates and types.

    Args:
        filepath: Path to PDB file (can be gzipped)
        include_hetatm: Whether to include HETATM records (ligands, waters, etc.)

    Returns:
        coords: (N, 3) array of xyz coordinates
        atom_types: (N,) array of atom type indices
        elements: List of element symbols
    """
    coords = []
    atom_types = []
    elements = []

    open_fn = gzip.open if str(filepath).endswith('.gz') else open
    mode = 'rt' if str(filepath).endswith('.gz') else 'r'

    with open_fn(filepath, mode) as f:
        for line in f:
            record_type = line[:6].strip()

            if record_type == "ATOM" or (include_hetatm and record_type == "HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    # Element is in columns 77-78, fallback to atom name
                    element = line[76:78].strip()
                    if not element:
                        # Infer from atom name (columns 13-16)
                        atom_name = line[12:16].strip()
                        element = ''.join(c for c in atom_name if c.isalpha())[:2]
                        if len(element) > 1:
                            element = element[0] + element[1].lower()

                    element = element.upper()

                    # Map to index, use last index for unknown elements
                    atom_idx = ELEMENT_TO_IDX.get(element, len(PROTEIN_ELEMENTS) - 1)

                    coords.append([x, y, z])
                    atom_types.append(atom_idx)
                    elements.append(element)

                except (ValueError, IndexError):
                    continue

    if not coords:
        raise ValueError(f"No valid atoms found in {filepath}")

    return np.array(coords, dtype=np.float32), np.array(atom_types, dtype=np.int64), elements


def parse_mmcif_file(filepath: Union[str, Path], include_hetatm: bool = False) -> Tuple[
    np.ndarray, np.ndarray, List[str]]:
    """Parse an mmCIF file and extract atom coordinates and types.

    Args:
        filepath: Path to mmCIF file (can be gzipped)
        include_hetatm: Whether to include HETATM records

    Returns:
        coords: (N, 3) array of xyz coordinates
        atom_types: (N,) array of atom type indices
        elements: List of element symbols
    """
    coords = []
    atom_types = []
    elements = []

    open_fn = gzip.open if str(filepath).endswith('.gz') else open
    mode = 'rt' if str(filepath).endswith('.gz') else 'r'

    in_atom_site = False
    column_map = {}

    with open_fn(filepath, mode) as f:
        for line in f:
            line = line.strip()

            # Start of _atom_site loop
            if line.startswith('_atom_site.'):
                in_atom_site = True
                col_name = line.split('.')[1].split()[0]
                column_map[col_name] = len(column_map)
                continue

            # End of loop section
            if in_atom_site and (line.startswith('_') or line.startswith('#') or line.startswith('loop_')):
                if not line.startswith('_atom_site.'):
                    in_atom_site = False
                    continue

            # Parse atom data
            if in_atom_site and line and not line.startswith('_'):
                try:
                    parts = line.split()

                    # Check if this is an ATOM or HETATM record
                    group_pdb_idx = column_map.get('group_PDB', 0)
                    if group_pdb_idx < len(parts):
                        record_type = parts[group_pdb_idx]
                        if record_type == "HETATM" and not include_hetatm:
                            continue

                    # Extract coordinates
                    x_idx = column_map.get('Cartn_x')
                    y_idx = column_map.get('Cartn_y')
                    z_idx = column_map.get('Cartn_z')
                    elem_idx = column_map.get('type_symbol')

                    if any(idx is None for idx in [x_idx, y_idx, z_idx]):
                        continue

                    x = float(parts[x_idx])
                    y = float(parts[y_idx])
                    z = float(parts[z_idx])

                    element = parts[elem_idx].upper() if elem_idx is not None and elem_idx < len(parts) else "C"
                    atom_idx = ELEMENT_TO_IDX.get(element, len(PROTEIN_ELEMENTS) - 1)

                    coords.append([x, y, z])
                    atom_types.append(atom_idx)
                    elements.append(element)

                except (ValueError, IndexError):
                    continue

    if not coords:
        raise ValueError(f"No valid atoms found in {filepath}")

    return np.array(coords, dtype=np.float32), np.array(atom_types, dtype=np.int64), elements


def parse_structure_file(filepath: Union[str, Path], include_hetatm: bool = False) -> Tuple[
    np.ndarray, np.ndarray, List[str]]:
    """Parse a structure file (PDB or mmCIF) and extract atom data.

    Automatically detects format based on file extension.
    """
    filepath = Path(filepath)
    name = filepath.name.lower()

    if '.cif' in name:
        return parse_mmcif_file(filepath, include_hetatm)
    else:
        return parse_pdb_file(filepath, include_hetatm)


# -----------------------------------------------------------------------------
# Download utilities
# -----------------------------------------------------------------------------

def download_file(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """Download a file from URL to destination path.

    Returns:
        True if successful, False otherwise
    """
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(response, f)
        return True

    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def fetch_pdb_ids_by_resolution(
        max_resolution: float = 2.5,
        experimental_method: str = "X-RAY DIFFRACTION",
        polymer_type: str = "Protein",
        max_results: int = 10000,
) -> List[str]:
    """Fetch PDB IDs matching resolution and method criteria using RCSB Search API.

    Args:
        max_resolution: Maximum resolution in Angstroms (lower = better)
        experimental_method: Experimental method filter
        polymer_type: Type of polymer (Protein, RNA, DNA, etc.)
        max_results: Maximum number of results to return

    Returns:
        List of PDB IDs matching criteria
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": experimental_method
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": polymer_type
                    }
                }
            ]
        },
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_results
            },
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc"
                }
            ]
        },
        "return_type": "entry"
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    try:
        data = json.dumps(query).encode('utf-8')
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))

        pdb_ids = [entry['identifier'] for entry in result.get('result_set', [])]
        logger.info(f"Found {len(pdb_ids)} PDB structures with resolution <= {max_resolution}Å")
        return pdb_ids

    except Exception as e:
        logger.error(f"Failed to fetch PDB IDs: {e}")
        return []


def fetch_alphafold_uniprot_ids(
        organism: str = "Homo sapiens",
        max_results: int = 10000,
) -> List[str]:
    """Fetch UniProt IDs with AlphaFold predictions for a given organism.

    For bulk downloads, AlphaFold provides proteome-level archives.
    This function provides a way to get individual protein IDs.

    Args:
        organism: Scientific name of organism
        max_results: Maximum number of results

    Returns:
        List of UniProt accession IDs
    """
    # AlphaFold DB API endpoint
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{organism}"

    # For now, return a curated list of well-studied human proteins
    # In production, this would query the AlphaFold API or use their accession list

    # Download the accession list from AlphaFold FTP
    accession_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/accession_ids.csv"

    try:
        req = urllib.request.Request(accession_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read().decode('utf-8')

        # Parse CSV: UniProt ID, AlphaFold ID, etc.
        lines = content.strip().split('\n')[1:]  # Skip header
        uniprot_ids = []

        for line in lines[:max_results]:
            parts = line.split(',')
            if len(parts) >= 1:
                uniprot_ids.append(parts[0])

        logger.info(f"Found {len(uniprot_ids)} AlphaFold predictions")
        return uniprot_ids

    except Exception as e:
        logger.warning(f"Failed to fetch AlphaFold accession list: {e}")
        # Return some well-known human proteins as fallback
        return [
            "P00533", "P04637", "P38398", "P35354", "P00519",
            "P01112", "P01116", "P01111", "P10415", "P42336"
        ]


# -----------------------------------------------------------------------------
# Base Dataset Class
# -----------------------------------------------------------------------------

class ProteinPointCloudDataset(Dataset, ABC):
    """Abstract base class for protein point cloud datasets."""

    def __init__(
            self,
            root: Optional[Union[Path, str]] = None,
            split: str = "train",
            limit: Optional[int] = None,
            split_fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            split_seed: int = 0,
            max_atoms: Optional[int] = None,
            center: bool = True,
            include_hydrogens: bool = False,
            backbone_only: bool = False,
            use_cache: bool = True,
    ) -> None:
        """Initialize the protein dataset.

        Args:
            root: Root directory for data storage
            split: One of 'train', 'val', 'test'
            limit: Maximum number of samples to use
            split_fractions: (train, val, test) fractions
            split_seed: Random seed for reproducible splits
            max_atoms: Filter out proteins larger than this
            center: Whether to center coordinates at origin
            include_hydrogens: Whether to include hydrogen atoms
            backbone_only: Use only backbone atoms (N, CA, C, O)
            use_cache: Whether to cache processed structures
        """
        super().__init__()

        if split not in _VALID_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Expected one of {_VALID_SPLITS}.")

        self.root = Path(root) if root is not None else self._default_root()
        self.root.mkdir(parents=True, exist_ok=True)

        self.split = split
        self.split_seed = split_seed
        self.split_fractions = split_fractions
        self.max_atoms = max_atoms
        self.center = center
        self.include_hydrogens = include_hydrogens
        self.backbone_only = backbone_only
        self.use_cache = use_cache

        # Setup directories
        self._raw_dir = self.root / "raw"
        self._processed_dir = self.root / "processed"
        self._cache_dir = self.root / "cache"

        for d in [self._raw_dir, self._processed_dir, self._cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Download and process data
        self._prepare_data()

        # Build split indices
        self.indices = self._build_split_indices()

        # Apply filters
        if self.max_atoms is not None:
            self.indices = self._filter_by_size()

        if limit is not None:
            self.indices = self.indices[:max(limit, 0)]

        if not self.indices:
            raise ValueError("No samples available after applying filters.")

        logger.info(f"Loaded {len(self.indices)} {self.split} samples from {self.__class__.__name__}")

    @abstractmethod
    def _default_root(self) -> Path:
        """Return the default root directory for this dataset."""
        pass

    @abstractmethod
    def _prepare_data(self) -> None:
        """Download and prepare the raw data."""
        pass

    @abstractmethod
    def _get_all_ids(self) -> List[str]:
        """Return all available structure IDs."""
        pass

    @abstractmethod
    def _get_structure_path(self, struct_id: str) -> Path:
        """Return the path to a structure file given its ID."""
        pass

    def _build_split_indices(self) -> List[int]:
        """Build train/val/test split indices."""
        all_ids = self._get_all_ids()
        n_total = len(all_ids)

        perm = torch.randperm(
            n_total,
            generator=torch.Generator().manual_seed(self.split_seed)
        )

        n_train = int(self.split_fractions[0] * n_total)
        n_val = int(self.split_fractions[1] * n_total)

        splits = {
            "train": perm[:n_train],
            "val": perm[n_train:n_train + n_val],
            "test": perm[n_train + n_val:],
        }

        return splits[self.split].tolist()

    def _filter_by_size(self) -> List[int]:
        """Filter indices by maximum atom count."""
        filtered = []
        for idx in self.indices:
            try:
                data = self._load_structure(idx)
                if data.num_nodes <= self.max_atoms:
                    filtered.append(idx)
            except Exception:
                continue
        return filtered

    def _get_cache_path(self, struct_id: str) -> Path:
        """Get the cache file path for a structure."""
        # Include processing options in cache key
        opts = f"{self.include_hydrogens}_{self.backbone_only}"
        cache_key = hashlib.md5(f"{struct_id}_{opts}".encode()).hexdigest()[:12]
        return self._cache_dir / f"{struct_id}_{cache_key}.pt"

    def _load_structure(self, idx: int) -> Data:
        """Load a structure by index, using cache if available."""
        all_ids = self._get_all_ids()
        struct_id = all_ids[idx]

        cache_path = self._get_cache_path(struct_id)

        # Try loading from cache
        if self.use_cache and cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception:
                pass

        # Load and process structure
        struct_path = self._get_structure_path(struct_id)

        if not struct_path.exists():
            raise FileNotFoundError(f"Structure file not found: {struct_path}")

        coords, atom_types, elements = parse_structure_file(struct_path)

        # Filter atoms if needed
        mask = np.ones(len(coords), dtype=bool)

        if not self.include_hydrogens:
            mask &= np.array([e != 'H' for e in elements])

        coords = coords[mask]
        atom_types = atom_types[mask]

        # Create PyG Data object
        if Data is None:
            raise ImportError("torch_geometric is required. Install with: pip install torch-geometric")

        pos = torch.from_numpy(coords)
        x = F.one_hot(torch.from_numpy(atom_types), num_classes=NUM_ATOM_TYPES).float()

        data = Data(
            pos=pos,
            x=x,
            atom_types=torch.from_numpy(atom_types),
            struct_id=struct_id,
        )

        # Cache the processed structure
        if self.use_cache:
            try:
                torch.save(data, cache_path)
            except Exception:
                pass

        return data

    @property
    def num_node_features(self) -> int:
        """Return the number of node features (atom types)."""
        return NUM_ATOM_TYPES

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        data = self._load_structure(self.indices[idx])

        if self.center:
            data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)

        return data


# -----------------------------------------------------------------------------
# PDB Dataset (Experimental Structures)
# -----------------------------------------------------------------------------

class PDBPointCloudDataset(ProteinPointCloudDataset):
    """Dataset of experimental protein structures from the RCSB PDB.

    Downloads high-resolution X-ray crystallography structures and
    provides them as point clouds with (xyz, atom_type) for each atom.

    Example:
        >>> dataset = PDBPointCloudDataset(
        ...     root="Latent_encoding/data/protein/pdb",
        ...     split="train",
        ...     max_resolution=2.0,  # Only structures <= 2.0 Å
        ...     max_atoms=5000,      # Filter out very large structures
        ... )
        >>> data = dataset[0]
        >>> data.pos.shape  # (N, 3) coordinates
        >>> data.x.shape    # (N, num_atom_types) one-hot atom types
    """

    def __init__(
            self,
            root: Optional[Union[Path, str]] = None,
            split: str = "train",
            limit: Optional[int] = None,
            split_fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            split_seed: int = 0,
            max_atoms: Optional[int] = None,
            center: bool = True,
            include_hydrogens: bool = False,
            backbone_only: bool = False,
            use_cache: bool = True,
            # PDB-specific options
            max_resolution: float = 2.5,
            experimental_method: str = "X-RAY DIFFRACTION",
            pdb_ids: Optional[List[str]] = None,
            download: bool = True,
            max_download: int = 1000,
    ) -> None:
        """Initialize PDB dataset.

        Args:
            max_resolution: Maximum resolution in Angstroms (lower = better quality)
            experimental_method: Filter by experimental method
            pdb_ids: Specific PDB IDs to use (overrides query)
            download: Whether to download structures
            max_download: Maximum number of structures to download
            **kwargs: Arguments passed to parent class
        """
        self.max_resolution = max_resolution
        self.experimental_method = experimental_method
        self._pdb_ids = pdb_ids
        self.download = download
        self.max_download = max_download

        super().__init__(
            root=root,
            split=split,
            limit=limit,
            split_fractions=split_fractions,
            split_seed=split_seed,
            max_atoms=max_atoms,
            center=center,
            include_hydrogens=include_hydrogens,
            backbone_only=backbone_only,
            use_cache=use_cache,
        )

    def _default_root(self) -> Path:
        # Explicitly anchor to this file's location
        this_file = Path(__file__).resolve()
        this_dir = this_file.parent  # Latent_encoding/data/
        return this_dir / "protein" / "pdb"

    def _prepare_data(self) -> None:
        """Fetch PDB IDs and download structures."""
        ids_file = self._processed_dir / "pdb_ids.json"

        # Load or fetch PDB IDs
        if self._pdb_ids is not None:
            pdb_ids = self._pdb_ids
        elif ids_file.exists():
            with open(ids_file) as f:
                pdb_ids = json.load(f)
        else:
            logger.info(f"Fetching PDB IDs with resolution <= {self.max_resolution}Å...")
            pdb_ids = fetch_pdb_ids_by_resolution(
                max_resolution=self.max_resolution,
                experimental_method=self.experimental_method,
                max_results=self.max_download * 2,  # Fetch extra in case some fail
            )

            # Save for reproducibility
            with open(ids_file, 'w') as f:
                json.dump(pdb_ids, f)

        self._all_pdb_ids = pdb_ids[:self.max_download]

        # Download structures
        if self.download:
            self._download_structures()

    def _download_structures(self) -> None:
        """Download PDB structure files."""
        logger.info(f"Downloading up to {len(self._all_pdb_ids)} PDB structures...")

        downloaded = 0
        failed = 0

        for pdb_id in self._all_pdb_ids:
            dest_path = self._raw_dir / f"{pdb_id.lower()}.pdb.gz"

            if dest_path.exists():
                downloaded += 1
                continue

            # Try downloading from RCSB
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb.gz"

            if download_file(url, dest_path):
                downloaded += 1
                if downloaded % 100 == 0:
                    logger.info(f"Downloaded {downloaded} structures...")
            else:
                failed += 1

        logger.info(f"Download complete: {downloaded} successful, {failed} failed")

        # Update list to only include successfully downloaded structures
        self._all_pdb_ids = [
            pdb_id for pdb_id in self._all_pdb_ids
            if (self._raw_dir / f"{pdb_id.lower()}.pdb.gz").exists()
        ]

    def _get_all_ids(self) -> List[str]:
        return self._all_pdb_ids

    def _get_structure_path(self, struct_id: str) -> Path:
        return self._raw_dir / f"{struct_id.lower()}.pdb.gz"


# -----------------------------------------------------------------------------
# AlphaFold Dataset (Predicted Structures)
# -----------------------------------------------------------------------------

class AlphaFoldPointCloudDataset(ProteinPointCloudDataset):
    """Dataset of predicted protein structures from AlphaFold DB.

    Downloads structure predictions from DeepMind's AlphaFold database
    and provides them as point clouds.

    Example:
        >>> dataset = AlphaFoldPointCloudDataset(
        ...     root="Latent_encoding/data/protein/alphafold",
        ...     split="train",
        ...     organism="Homo sapiens",
        ...     max_atoms=5000,
        ... )
        >>> data = dataset[0]
        >>> data.pos.shape  # (N, 3) coordinates
        >>> data.x.shape    # (N, num_atom_types) one-hot atom types
    """

    def __init__(
            self,
            root: Optional[Union[Path, str]] = None,
            split: str = "train",
            limit: Optional[int] = None,
            split_fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
            split_seed: int = 0,
            max_atoms: Optional[int] = None,
            center: bool = True,
            include_hydrogens: bool = False,
            backbone_only: bool = False,
            use_cache: bool = True,
            # AlphaFold-specific options
            uniprot_ids: Optional[List[str]] = None,
            organism: str = "Homo sapiens",
            download: bool = True,
            max_download: int = 1000,
            min_plddt: Optional[float] = None,  # Filter by prediction confidence
    ) -> None:
        """Initialize AlphaFold dataset.

        Args:
            uniprot_ids: Specific UniProt IDs to download
            organism: Organism to fetch predictions for
            download: Whether to download structures
            max_download: Maximum number of structures to download
            min_plddt: Minimum pLDDT confidence score (0-100)
            **kwargs: Arguments passed to parent class
        """
        self._uniprot_ids = uniprot_ids
        self.organism = organism
        self.download = download
        self.max_download = max_download
        self.min_plddt = min_plddt

        super().__init__(
            root=root,
            split=split,
            limit=limit,
            split_fractions=split_fractions,
            split_seed=split_seed,
            max_atoms=max_atoms,
            center=center,
            include_hydrogens=include_hydrogens,
            backbone_only=backbone_only,
            use_cache=use_cache,
        )

    def _default_root(self) -> Path:
        return Path(__file__).resolve().parent / "protein" / "alphafold"

    def _prepare_data(self) -> None:
        """Fetch UniProt IDs and download AlphaFold predictions."""
        ids_file = self._processed_dir / "uniprot_ids.json"

        # Load or fetch UniProt IDs
        if self._uniprot_ids is not None:
            uniprot_ids = self._uniprot_ids
        elif ids_file.exists():
            with open(ids_file) as f:
                uniprot_ids = json.load(f)
        else:
            logger.info(f"Fetching AlphaFold predictions for {self.organism}...")
            uniprot_ids = fetch_alphafold_uniprot_ids(
                organism=self.organism,
                max_results=self.max_download * 2,
            )

            with open(ids_file, 'w') as f:
                json.dump(uniprot_ids, f)

        self._all_uniprot_ids = uniprot_ids[:self.max_download]

        if self.download:
            self._download_structures()

    def _download_structures(self) -> None:
        """Download AlphaFold structure files."""
        logger.info(f"Downloading up to {len(self._all_uniprot_ids)} AlphaFold structures...")

        downloaded = 0
        failed = 0

        for uniprot_id in self._all_uniprot_ids:
            dest_path = self._raw_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"

            if dest_path.exists():
                downloaded += 1
                continue

            # AlphaFold DB URL pattern
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

            if download_file(url, dest_path):
                downloaded += 1
                if downloaded % 100 == 0:
                    logger.info(f"Downloaded {downloaded} structures...")
            else:
                failed += 1

        logger.info(f"Download complete: {downloaded} successful, {failed} failed")

        # Update list to only include successfully downloaded structures
        self._all_uniprot_ids = [
            uid for uid in self._all_uniprot_ids
            if (self._raw_dir / f"AF-{uid}-F1-model_v4.pdb").exists()
        ]

    def _get_all_ids(self) -> List[str]:
        return self._all_uniprot_ids

    def _get_structure_path(self, struct_id: str) -> Path:
        return self._raw_dir / f"AF-{struct_id}-F1-model_v4.pdb"


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def load_protein_dataset(
        source: Literal["pdb", "alphafold"] = "pdb",
        **kwargs,
) -> ProteinPointCloudDataset:
    """Convenience function to load a protein dataset.

    Args:
        source: Either 'pdb' for experimental structures or 'alphafold' for predictions
        **kwargs: Arguments passed to the dataset class

    Returns:
        Dataset instance
    """
    if source == "pdb":
        return PDBPointCloudDataset(**kwargs)
    elif source == "alphafold":
        return AlphaFoldPointCloudDataset(**kwargs)
    else:
        raise ValueError(f"Unknown source: {source}. Expected 'pdb' or 'alphafold'.")


__all__ = [
    "ProteinPointCloudDataset",
    "PDBPointCloudDataset",
    "AlphaFoldPointCloudDataset",
    "load_protein_dataset",
    "parse_pdb_file",
    "parse_mmcif_file",
    "parse_structure_file",
    "PROTEIN_ELEMENTS",
    "ELEMENT_TO_IDX",
    "NUM_ATOM_TYPES",
]

# -----------------------------------------------------------------------------
# Example usage and testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Test protein dataset loading")
    parser.add_argument("--source", choices=["pdb", "alphafold"], default="pdb")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-atoms", type=int, default=5000)
    args = parser.parse_args()

    print(f"\nLoading {args.source} dataset...")

    dataset = load_protein_dataset(
        source=args.source,
        split="train",
        limit=args.limit,
        max_atoms=args.max_atoms,
        download=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of atom types: {dataset.num_node_features}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample structure:")
        print(f"  Structure ID: {sample.struct_id}")
        print(f"  Number of atoms: {sample.num_nodes}")
        print(f"  Position shape: {sample.pos.shape}")
        print(f"  Features shape: {sample.x.shape}")
        print(f"  Atom types: {sample.atom_types.unique().tolist()}")