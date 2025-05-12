import json
from pygbif import species
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rockdaisy
from rockdaisy.nomenclator import Nomenclator

def gbif_backbone_lookup(nomenclator, verbose=False):
    """
    Query GBIF backbone for all names in a Nomenclator instance.

    Args:
        nomenclator (Nomenclator): Loaded Nomenclator object
        verbose (bool): Print results during execution

    Returns:
        dict: {name: gbif_match_dict}
    """
    matches = {}
    all_names = nomenclator.all_names()

    for name in tqdm(all_names, desc="Querying GBIF"):
        try:
            match = species.name_backbone(name=name)
            matches[name] = match
            if verbose:
                print(f"{name}: {match.get('scientificName', 'No match')}")
        except Exception as e:
            matches[name] = {"error": str(e)}
            if verbose:
                print(f"{name}: ERROR - {e}")

    return matches

def main():
    # === Load Nomenclator ===
    filepath = "../data/nomenclator.txt"
    nomen = Nomenclator(filepath)

    # === Query GBIF ===
    gbif_matches = gbif_backbone_lookup(nomen, verbose=False)

    # === Save results ===
    with open("gbif_matches.json", "w", encoding="utf-8") as f:
        json.dump(gbif_matches, f, indent=2, ensure_ascii=False)

    print(f"\n✅ GBIF matches saved to gbif_matches.json")

if __name__ == "__main__":
    main()
