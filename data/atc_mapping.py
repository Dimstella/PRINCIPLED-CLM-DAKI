"""
atc_mapping.py  —  Algorithm 2 from Dimitsaki et al. (2026)
------------------------------------------------------------
Unify free-text drug names and map them to ATC codes using
fuzzy string matching at a configurable similarity threshold.

Public API
----------
unify_drug_names(drug_list)          -> set[str]
build_atc_lookup(atc_csv_path)       -> dict[str, str]   name -> ATC
map_drugs_to_atc(med_df, atc_lookup, threshold) -> pd.DataFrame
load_manual_atc_overrides()          -> dict[str, str]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from fuzzywuzzy import process

from config import ATC_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithm 1 – drug-name unification
# ---------------------------------------------------------------------------

def unify_drug_names(drug_list: list[str]) -> set[str]:
    """
    Collapse near-duplicate drug names by removing entries that are a
    whole-word substring of any longer name already in the set.

    Implements Algorithm 1 (Dimitsaki et al., 2026):
      1. Remove names of length <= 3.
      2. Sort ascending by length.
      3. Keep a name only if it does NOT appear as a whole word inside
         any name already kept.

    Parameters
    ----------
    drug_list : list[str]
        Raw drug name strings from MIMIC-IV prescriptions.

    Returns
    -------
    set[str]
        Deduplicated canonical drug name set.
    """
    drug_list = [d for d in drug_list if isinstance(d, str) and len(d) > 3]
    unified: set[str] = set()
    for drug in sorted(drug_list, key=len):
        pattern = r"\b" + re.escape(drug) + r"\b"
        already_covered = any(
            re.search(pattern, existing, re.IGNORECASE)
            for existing in unified
        )
        if not already_covered:
            unified.add(drug)
    return unified


def _build_name_map(drug_series: pd.Series) -> dict[str, str]:
    """
    Build a dict mapping every original name to its canonical form
    derived from unify_drug_names.
    """
    names = drug_series.dropna().tolist()
    unified = unify_drug_names(names)
    return {
        drug: next(
            (u for u in unified if re.search(r"\b" + re.escape(u) + r"\b", drug, re.IGNORECASE)),
            drug,
        )
        for drug in names
    }


def apply_name_unification(med_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``drug_new`` and ``medication_new`` columns to *med_df* with
    unified names.  The original columns are preserved.
    """
    med_df = med_df.copy()

    if "drug" in med_df.columns:
        name_map = _build_name_map(med_df["drug"])
        med_df["drug_new"] = med_df["drug"].map(name_map).fillna(med_df["drug"])
        # Fall back: if drug_new is still 'Bag', use the 'medication' column
        bag_mask = med_df["drug_new"] == "Bag"
        if "medication" in med_df.columns:
            med_df.loc[bag_mask, "drug_new"] = med_df.loc[bag_mask, "medication"]
        med_df["drug_new"] = med_df["drug_new"].fillna("Bag")

    if "medication" in med_df.columns:
        med_map = _build_name_map(med_df["medication"])
        med_df["medication_new"] = med_df["medication"].map(med_map)

    return med_df


# ---------------------------------------------------------------------------
# ATC look-up table construction
# ---------------------------------------------------------------------------

def build_atc_lookup(atc_csv_path: str | Path) -> dict[str, str]:
    """
    Build a ``{drug_name_lower: ATC_code}`` dictionary from the WHO ATC
    reference CSV (columns: ``ATC.code``, ``Name``).

    The manual GPT-assisted overrides from the notebooks are merged in
    automatically via ``load_manual_atc_overrides()``.

    Parameters
    ----------
    atc_csv_path : str or Path
        Path to ``atc_codes.csv`` (WHO reference file).

    Returns
    -------
    dict[str, str]
        Lower-cased drug name -> ATC code.
    """
    db = pd.read_csv(atc_csv_path)
    # Drop unnamed index columns if present
    db = db.loc[:, ~db.columns.str.startswith("Unnamed")]
    db = db[~db["Name"].isnull()].copy()
    db["Name"] = db["Name"].str.lower()

    # Build name -> code lookup (reversed: name is key)
    atc_lookup: dict[str, str] = (
        db.set_index("Name")["ATC.code"].to_dict()
    )

    # Merge manual overrides
    atc_lookup.update(load_manual_atc_overrides())
    return atc_lookup


# ---------------------------------------------------------------------------
# Algorithm 2 – fuzzy ATC mapping
# ---------------------------------------------------------------------------

def map_drugs_to_atc(
    med_df: pd.DataFrame,
    atc_lookup: dict[str, str],
    threshold: int = ATC_SIMILARITY_THRESHOLD,
    mapping_cache_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Map ``drug_new`` column to ATC codes using fuzzy matching.

    Implements Algorithm 2 (Dimitsaki et al., 2026):
      For each drug name:
        1. Find the closest match in *atc_lookup* with fuzzywuzzy.
        2. If similarity >= *threshold*, assign the matched ATC code.
        3. Otherwise, keep the original drug name (flagged for manual review).

    A drug-to-ATC mapping table is optionally saved to *mapping_cache_path*
    (Excel, one row per unique drug) so it can be manually validated and
    reloaded without re-running the fuzzy matching.

    Parameters
    ----------
    med_df : pd.DataFrame
        Must contain a ``drug_new`` column (output of apply_name_unification).
    atc_lookup : dict[str, str]
        Drug name -> ATC code lookup (from build_atc_lookup).
    threshold : int
        Minimum fuzzywuzzy similarity score (0-100). Default 85.
    mapping_cache_path : str or Path, optional
        If given, save the mapping table as an Excel file.

    Returns
    -------
    pd.DataFrame
        *med_df* with an added ``ATC`` column.
    """
    med_df = med_df.copy()
    unique_drugs = med_df["drug_new"].dropna().str.lower().unique()

    non_match: list[str] = []
    mapping: dict[str, str] = {}

    for drug_name in unique_drugs:
        result = process.extractOne(drug_name, atc_lookup.keys())
        if result is not None and result[1] >= threshold:
            mapping[drug_name] = atc_lookup[result[0]]
        else:
            non_match.append(drug_name)
            mapping[drug_name] = drug_name  # unchanged; flag for manual review

    if non_match:
        logger.warning(
            "%d drug names did not reach the %d%% similarity threshold: %s",
            len(non_match),
            threshold,
            non_match[:10],
        )

    # Build a mapping DataFrame for optional manual validation
    mapping_df = pd.DataFrame(
        list(mapping.items()), columns=["Drug_Name", "ATC_Code"]
    )
    if mapping_cache_path is not None:
        mapping_df.to_excel(mapping_cache_path, index=False)
        logger.info("ATC mapping table saved to %s", mapping_cache_path)

    # Apply to med_df
    med_df["ATC"] = (
        med_df["drug_new"]
        .str.lower()
        .map(mapping)
        .fillna("N/A")
    )

    return med_df


def load_mapping_from_cache(mapping_cache_path: str | Path) -> dict[str, str]:
    """
    Reload a previously saved (and optionally manually corrected)
    Drug_Name -> ATC_Code mapping from an Excel file.

    Parameters
    ----------
    mapping_cache_path : str or Path

    Returns
    -------
    dict[str, str]
    """
    df = pd.read_excel(mapping_cache_path)
    df = df[~df["ATC_Code"].isna()]
    return df.set_index("Drug_Name")["ATC_Code"].to_dict()


def apply_atc_from_cache(
    med_df: pd.DataFrame,
    mapping_cache_path: str | Path,
) -> pd.DataFrame:
    """
    Apply a saved Drug_Name->ATC mapping to *med_df*.
    Useful when the mapping has been manually corrected after a first run.
    """
    med_df = med_df.copy()
    atc_map = load_mapping_from_cache(mapping_cache_path)
    med_df["ATC"] = (
        med_df["drug_new"].str.lower().map(atc_map).fillna("N/A")
    )
    return med_df


# ---------------------------------------------------------------------------
# Manual ATC overrides
# ---------------------------------------------------------------------------
# These were accumulated interactively across the original notebooks.
# Keys are lower-cased drug names; values are ATC codes ('N/A' = no code).
# ---------------------------------------------------------------------------

def load_manual_atc_overrides() -> dict[str, str]:
    """
    Return the consolidated manual ATC overrides collected during the
    original notebook-based analysis.  This dictionary is intentionally
    kept in code (rather than a CSV) so it is version-controlled and
    reproducible without any external file dependency.
    """
    return {
        # ---- GPT batch 1 (cell 082 in AKI notebook) ----
        "aspirin ec": "B01AC06",
        "divalproex": "N03AG01",
        "sterile water": "V07AB",
        "dextrose 50%": "B05BA03",
        "glycopyrrolate": "A03AB02",
        "d5w": "B05BA03",
        "acetaminophen": "N02BE01",
        "lactated ringers": "B05BB01",
        "neutra-phos": "A12CX",
        "albuterol inhaler": "R03AC02",
        "descovy": "J05AR21",
        "nitroglycerin": "C01DA02",
        "breo ellipta": "R03AK10",
        "moviprep": "A06AD65",
        "hydroxyurea": "L01XX05",
        "latuda": "N05AE05",
        "creon 12": "A09AA02",
        "bactrim": "J01EE01",
        "cosyntropin": "H01AA02",
        "ursodiol": "A05AA02",
        "augmentin suspension": "J01CR02",
        "relpax": "N02CC06",
        "xopenex neb": "R03AC12",
        "adderall": "N06BA02",
        "glyburide": "A10BB01",
        "mucinex": "R05CA10",
        "claravis": "D10BA01",
        "niaspan": "C10AD02",
        "nexium": "A02BC05",
        "bismuth subsalicylate": "A07BB",
        "humalog": "A10AD04",
        "qvar": "R03BA08",
        "advair diskus": "R03AK06",
        "pamidronate": "M05BA03",
        "cromolyn": "R01AC01",
        "golytely": "A06AD65",
        "ambisome": "J02AA01",
        "evista": "G03XC01",
        "depakote": "N03AG01",
        "lumigan": "S01EE03",
        "systane": "S01XA20",
        "tecfidera": "N07XX09",
        "methylene blue": "V03AB27",
        "pulmozyme": "R05CB13",
        "mirapex": "N04BC05",
        "symbicort": "R03AK07",
        "myfortic": "L04AA06",
        "epipen": "C01CA24",
        "zyflo": "R03DC01",
        "kytril": "A04AA02",
        "namenda": "N06DX01",
        "celebrex": "M01AH01",
        # ---- GPT batch 2 (cell 085) ----
        "succinylcholine": "M03AB01",
        "gammagard liquid": "J06BA02",
        "psyllium": "A06AC01",
        "albuterol mdi": "R03AC02",
        "pramipexole": "N04BC05",
        "sulfazine ec": "A07EC01",
        "amphetamine-dextroamphetamine": "N06BA01",
        "gastrografin": "V08AA02",
        "stribild": "J05AR09",
        "levalbuterol neb": "R03AC13",
        "pancrelipase 5000": "A09AA02",
        "gastrocrom": "R01AC01",
        "stalevo 150": "N04BA03",
        "dibucaine": "D04AB04",
        "carbidopa": "N04BA02",
        "granisetron hcl": "A04AA02",
        "levalbuterol hcl": "R03AC13",
        "sutent": "L01XE04",
        "effexor": "N06AX16",
        "incruse ellipta": "R03BB07",
        "zofran odt": "A04AA01",
        "pulmicort": "R03BA02",
        "pred forte": "S01BA04",
        "fleet enema (saline)": "A06AG01",
        "zafirlukast": "R03DC01",
        "bactroban": "D06AX09",
        "novolin r": "A10AB01",
        "isoproterenol": "C01CA02",
        "imodium": "A07DA03",
        "bromday": "S01BC11",
        "humira pen": "L04AB04",
        "lotensin": "C09AA07",
        "aromasin": "L02BG06",
        "pletal": "B01AC23",
        "provigil": "N06BA07",
        # ---- Drug-specific non-matches (cell 093) ----
        "cyclosporine (sandimmune)": "N/A",
        "dicyclomine": "A03AA07",
        "phytonad": "N/A",
        "phosphorus": "N/A",
        "mycophen": "N/A",
        "sw": "N/A",
        "cephalexin": "J01DB01",
        "meperidine": "N02AB02",
        "granulex": "N/A",
        "amitiza": "A06AX03",
        "benztropine mesylate": "N04AC01",
        "protopic": "D11AX13",
        "mesalamine dr": "A07EC02",
        "xibrom": "S01BC11",
        "actonel": "M05BA07",
        "avodart": "G04CB02",
        "calcipotriene 0.005% cream": "D05AX02",
        "restasis": "S01XA18",
        "erleada": "L02BB04",
        "raloxifene": "G03XC01",
        "mechlorethamine": "L01AD01",
        "voltaren": "M01AB05",
        "vimpat": "N03AX18",
        "revlimid": "L04AX04",
        "lurasidone": "N05AE05",
        "proair hfa": "R03AC02",
        "reclast": "M05BA08",
        "beclomethasone dipropionate": "R03BA01",
        "lamictal": "N03AX09",
        "prograf": "L04AD02",
        "travatan z": "S01EE04",
        "avapro": "C09CA07",
        "provera": "G03DA02",
        "creon dr": "A09AA02",
        "zytiga": "L02BX03",
        "vigamox": "S01AE07",
        "tricor": "C10AB05",
        "tasigna": "L01EA03",
        "lialda": "A07EC02",
        "vesicare": "G04BD08",
        "bystolic": "C07AB12",
        "miralax": "A06AD65",
        "nucynta": "N02AX06",
        "prevnar 13 (pf)": "J07AL02",
        "azopt": "S01EC04",
        "adcirca": "G04BE08",
        "apriso": "A07EC02",
        "afrin": "R01AA07",
        "exelon": "N06DA03",
        "keppra xr": "N03AX14",
        "lyrica": "N03AX16",
        "anoro ellipta": "R03AL03",
        "welchol": "C10AX09",
        "amaryl": "A10BB12",
        "exjade": "V03AC03",
        "rocaltrol": "A11CC04",
        "xopenex hfa": "R03AC12",
        "prolensa": "S01BC11",
        "vivelle-dot": "G03CA03",
        "parcopa": "N04BA02",
        "januvia": "A10BH01",
        "salagen": "N07AX01",
        "genotropin miniquick": "H01AC01",
        "lotemax": "S01BA14",
        "risedronate": "M05BA07",
        "uceris": "A07EA07",
        "chantix": "N07BA03",
        "enbrel": "L04AB01",
        "aspirin 325 mg": "B01AC06",
        "forteo": "H05AA02",
        "enablex": "G04BD10",
        "isordil titradose": "C01DA08",
        "levemir": "A10AE05",
        "levoleucovorin": "V03AF04",
        "jakafi": "L01XE18",
        "risperdal consta": "N05AX08",
        "durezol": "S01BA12",
        "aranesp (polysorbate)": "B03XA02",
        "androgel": "G03BA03",
        "remeron soltab": "N06AX11",
        "synthroid": "H03AA01",
        "dexilant": "A02BC06",
        "lantus": "A10AE04",
        "zyrtec": "R06AE07",
        "tessalon": "R05DB01",
        "detrol la": "G04BD07",
        "valtrex": "J05AB11",
        "hespan": "B05AA06",
        "flovent diskus": "R03BA01",
        "nasonex": "R01AD09",
        "arava": "L04AA13",
        "xarelto": "B01AF01",
        "flovent hfa": "R03BA01",
        "promacta": "B03XA04",
        "bimatoprost": "S01EE03",
        "epogen": "B03XA01",
        "opsumit": "C02KX04",
        "edurant": "J05AG05",
        "renagel": "V03AE02",
        "feraheme": "B03AC07",
        "novolog": "A10AB05",
        "claritin": "R06AX13",
        "prozac": "N06AB03",
        "asacol": "A07EC02",
        "apidra": "A10AB06",
        "foradil aerolizer": "R03AC13",
        "alinia": "P01AX10",
        "estraderm": "G03CA03",
        "brovana": "R03AC12",
        "gleevec": "L01XE01",
        "rapaflo": "G04CA04",
        "entresto": "C09DX04",
        "tarceva": "L01EB02",
        "zaditor": "S01GX04",
        "atg (horse) desensitization": "N/A",
    }
