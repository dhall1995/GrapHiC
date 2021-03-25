from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import cooler
import numpy as np


def parse_cooler(
    cooler_file: Path, regions: Dict[str, np.ndarray]
) -> Tuple[Cooler, List[np.ndarray]]:
    # Load cooler
    c = cooler.Cooler(cooler_file)

    # Fetch relevant bin_ids from the cooler file
    b_ids = fetch_bins_from_cooler(cooler, regions)
    # Identify unique bin_ids and isolate disjoint regions
    slices = get_unique_bins(b_ids)

    return c, slices


def fetch_bins_from_cooler(
    cooler: Cooler, 
    regions: Dict[str, np.ndarray],
    names: Optional[dict] = {}
) -> List[List[np.int64]]:
    # Fetch relevant bin_ids from the cooler file
    b_ids = {}
    n_ids = {}
    for chrom in regions:
        b_ids[chrom] = []
        n_ids[chrom] = []
        for idx, row in enumerate(regions[chrom]):
            b_add = list(
                    cooler.bins()
                    .fetch("{}:{}-{}".format(chrom, row[0], row[1]))
                    .index.values
            )
            try:
                n_ids[chrom].append(names[chrom][idx])
            except:
                n_ids[chrom].append("{}:{}-{}".format(chrom, row[0], row[1]))
                
            b_ids[chrom].append(
                b_add
            )
    return b_ids, n_ids


def get_unique_bins(b_ids: List[List[np.int64]]) -> List[np.ndarray]:
    # Identify unique bin_ids and isolate disjoint regions
    b_ids = np.sort(list(set(b_ids[0]).union(*[item for item in b_ids[1:]])))
    gaps = np.append([0], np.where(abs(np.diff(b_ids)) > 1)[0] + 1)
    gaps = np.append(gaps, len(b_ids))
    slices = []
    for idx, gap in enumerate(gaps[:-1]):
        s_add = b_ids[gaps[idx] : gaps[idx + 1]]
        slices.append(
                s_add
            )

    return slices

def make_slices(
    clr: Cooler,
    regions: Dict[str, np.ndarray],
    names:Optional[dict] = {},
    force_disjoint:Optional[bool] = False
) --> List[np.ndarray,np.ndarray]:
    # Fetch relevant bin_ids from the cooler file
    b_ids, n_ids = fetch_bins_from_cooler(cooler=clr, 
                                          regions=regions, 
                                          names=names)
    if force_disjoint:
        # Identify unique bin_ids and isolate disjoint regions
        slices = {chrom: get_unique_bins(b_ids=b_ids[chrom]) for chrom in b_ids}
        n_ids = {}
        for chrom in slices:
            n_ids[chrom] = []
            for sl in slices[chrom]:
                # start, end, bins and node names for region
                stl = clr.bins()[sl[0]]["start"].values[0]
                el = clr.bins()[sl[-1] + 1]["end"].values[0]

                sl_id = f"{chrom}:{stl}-{el}"
                n_ids[chrom].append(sl_id)
    else:
        slices = {chrom:[np.array(item) for item in b_ids[chrom]] for chrom in b_ids}
        
    return slices, n_ids


if __name__ == "__main__":

    parse_cooler_file(
        "Dixon2012-H1hESC-HindIII-allreps-filtered.1000kb.cool", {}
    )
