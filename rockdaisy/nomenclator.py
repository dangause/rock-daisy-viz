import re

def parse_nomenclator(filepath):
    """
    Parse a botanical nomenclator file into a standardized dictionary.

    Returns a dict where each name (accepted or synonym) maps to:
    {
        "species": str,               # e.g., 'Perityle canescens'
        "authors": str,               # e.g., 'Everly'
        "accepted_name": str,         # accepted species name
        "accepted_authors": str,      # authorship of the accepted name
        "relationship": "accepted" | "synonym"
    }
    """
    species_dict = {}
    accepted_species = None
    accepted_authors = None

    def split_species_and_authors(name_str):
        parts = name_str.strip().split()
        species = " ".join(parts[:2])
        authors = " ".join(parts[2:]) if len(parts) > 2 else ""
        return species, authors

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            is_synonym_line = stripped.startswith(('=', '+'))

            # Remove leading =/+ markers
            clean = stripped.lstrip('=+').strip()

            # Case: synonym explicitly mapped to accepted name
            if '=' in clean and is_synonym_line:
                parts = [p.strip() for p in re.split(r'\s*=\s*', clean)]
                if len(parts) == 2:
                    syn_full, acc_full = parts
                    syn_species, syn_authors = split_species_and_authors(syn_full)
                    acc_species, acc_authors = split_species_and_authors(acc_full)

                    # Store synonym entry
                    species_dict[syn_species] = {
                        "species": syn_species,
                        "authors": syn_authors,
                        "accepted_name": acc_species,
                        "accepted_authors": acc_authors,
                        "relationship": "synonym"
                    }

                    # Ensure accepted name also recorded
                    if acc_species not in species_dict:
                        species_dict[acc_species] = {
                            "species": acc_species,
                            "authors": acc_authors,
                            "accepted_name": acc_species,
                            "accepted_authors": acc_authors,
                            "relationship": "accepted"
                        }

                continue

            # Case: synonym indented under last accepted name
            if indent > 0 and accepted_species:
                syn_species, syn_authors = split_species_and_authors(clean)
                species_dict[syn_species] = {
                    "species": syn_species,
                    "authors": syn_authors,
                    "accepted_name": accepted_species,
                    "accepted_authors": accepted_authors,
                    "relationship": "synonym"
                }
                continue

            # Case: new accepted name (no =/+ at start)
            if not is_synonym_line:
                accepted_species, accepted_authors = split_species_and_authors(clean)
                species_dict[accepted_species] = {
                    "species": accepted_species,
                    "authors": accepted_authors,
                    "accepted_name": accepted_species,
                    "accepted_authors": accepted_authors,
                    "relationship": "accepted"
                }

            # Case: synonym with no mapped accepted name (e.g., standalone "+Name Author")
            elif is_synonym_line and accepted_species:
                syn_species, syn_authors = split_species_and_authors(clean)
                species_dict[syn_species] = {
                    "species": syn_species,
                    "authors": syn_authors,
                    "accepted_name": accepted_species,
                    "accepted_authors": accepted_authors,
                    "relationship": "synonym"
                }

    return species_dict
