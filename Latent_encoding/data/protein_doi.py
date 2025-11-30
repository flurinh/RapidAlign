"""PDB to DOI mapper.

Fetches PDB IDs (high resolution X-ray/cryo-EM) and their primary publication DOIs.
Outputs a CSV: pdb_id,doi

Usage:
    python Latent_encoding/data/protein_doi.py --max-download 1000
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        total = kwargs.get('total', None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if total and i % 50 == 0:
                print(f"{desc}: {i}/{total}")
            yield item


def search_high_res_structures(
    max_resolution: float = 2.5,
    methods: tuple[str, ...] = ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"),
) -> tuple[list[str], int]:
    """
    Use RCSB Search API to get ALL PDB IDs matching resolution and method criteria.
    Paginates through results (API limit is ~10k per request).
    """

    # Build method nodes for OR group
    method_nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": method
            }
        }
        for method in methods
    ]

    base_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                # Resolution filter (use range operator)
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "range",
                        "value": {"from": 0, "to": max_resolution}
                    }
                },
                # Method filter (OR group)
                {
                    "type": "group",
                    "logical_operator": "or",
                    "nodes": method_nodes
                }
            ]
        },
        "return_type": "entry"
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    all_pdb_ids = []
    page_size = 10000  # Safe limit per request
    start = 0
    total_count = None

    while True:
        query = base_query.copy()
        query["request_options"] = {
            "paginate": {"start": start, "rows": page_size}
        }

        try:
            data = json.dumps(query).encode('utf-8')
            req = urllib.request.Request(
                url, data=data,
                headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))

            if total_count is None:
                total_count = result.get('total_count', 0)
                print(f"  Total matching structures: {total_count}")

            result_set = result.get('result_set', [])
            if not result_set:
                break

            pdb_ids = [entry['identifier'] for entry in result_set]
            all_pdb_ids.extend(pdb_ids)

            print(f"  Fetched {len(all_pdb_ids)}/{total_count} IDs...")

            if len(result_set) < page_size:
                break  # Last page

            start += page_size
            time.sleep(0.1)  # Be nice to the API

        except HTTPError as e:
            print(f"Search API HTTP Error: {e.code}")
            try:
                error_body = e.read().decode('utf-8')
                print(f"Error details: {error_body[:300]}")
            except:
                pass
            break
        except Exception as e:
            print(f"Search API failed: {e}")
            break

    return all_pdb_ids, total_count or 0


def get_doi_from_page(pdb_id: str) -> Optional[str]:
    """Scrape RCSB structure page for Primary Publication DOI."""
    url = f"https://www.rcsb.org/structure/{pdb_id}"

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')

        # Find all DOI links
        doi_pattern = r'href="(https://doi\.org/10\.[^"]+)"'
        all_dois = re.findall(doi_pattern, html)

        # Filter out the PDB DOI - we want primary publication DOI
        for doi in all_dois:
            # Skip PDB DOIs like https://doi.org/10.2210/pdb4hhb/pdb
            if '/pdb' in doi.lower() or '10.2210' in doi:
                continue
            return doi

        return None

    except (HTTPError, URLError):
        return None


def build_pdb_doi_csv(
    output_path: Path,
    max_download: int = 1000,
    max_resolution: float = 2.5,
    delay: float = 0.1,
) -> int:
    """Build CSV of PDB IDs and primary publication DOIs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing entries
    existing = {}
    if output_path.exists():
        with open(output_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    existing[row[0]] = row[1]
        print(f"Found {len(existing)} existing entries in {output_path}")

    # Step 1: Search for ALL matching structures (paginated)
    print(f"\nSearching for structures with resolution <= {max_resolution}Å (X-ray or cryo-EM)...")
    pdb_ids, total_count = search_high_res_structures(max_resolution=max_resolution)

    print(f"\nRetrieved {len(pdb_ids)} PDB IDs\n")

    if not pdb_ids:
        print("No structures found!")
        return len(existing)

    # Calculate how many new entries we need
    needed = max_download - len(existing)
    if needed <= 0:
        print(f"Already have {len(existing)} entries, nothing to fetch")
        return len(existing)

    # Filter out already processed
    pdb_ids_to_process = [pid for pid in pdb_ids if pid not in existing]
    print(f"Need {needed} new entries, {len(pdb_ids_to_process)} candidates available\n")

    # Step 2: Scrape DOIs (slow - one request per structure)
    print(f"Scraping DOIs (with {delay}s delay between requests)...")
    new_entries = []
    no_doi_count = 0

    for pdb_id in tqdm(pdb_ids_to_process, desc="Scraping DOIs", total=min(len(pdb_ids_to_process), needed * 2)):
        if len(new_entries) >= needed:
            break

        doi = get_doi_from_page(pdb_id)
        if doi:
            new_entries.append((pdb_id, doi))
        else:
            no_doi_count += 1

        time.sleep(delay)

    print(f"\nGot {len(new_entries)} DOIs ({no_doi_count} structures had no primary publication DOI)")

    # Step 3: Write CSV
    print(f"Writing {len(existing) + len(new_entries)} entries to {output_path}")

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pdb_id', 'doi'])

        # Write existing
        for pdb_id, doi in existing.items():
            writer.writerow([pdb_id, doi])

        # Write new
        for pdb_id, doi in new_entries:
            writer.writerow([pdb_id, doi])

    total = len(existing) + len(new_entries)
    print(f"Done! Total entries: {total}")
    return total


def get_default_output_path() -> Path:
    return Path(__file__).resolve().parent / "protein_text" / "pdb_doi.csv"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PDB to DOI CSV")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--max-download", type=int, default=1000, help="Max entries to collect")
    parser.add_argument("--max-resolution", type=float, default=2.5, help="Max resolution in Angstroms")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between DOI requests (seconds)")
    args = parser.parse_args()

    output = args.output or get_default_output_path()

    build_pdb_doi_csv(
        output_path=output,
        max_download=args.max_download,
        max_resolution=args.max_resolution,
        delay=args.delay,
    )