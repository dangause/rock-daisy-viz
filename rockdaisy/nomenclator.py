import re
import json
import pandas as pd

class Nomenclator:
    def __init__(self, filepath):
        self.species_dict = {}
        self._parse_file(filepath)

    def _split_species_and_authors(self, name_str):
        parts = name_str.strip().split()
        if len(parts) < 2:
            return None, None, None, None, None, name_str.strip()

        genus = parts[0].capitalize()
        species = parts[1].lower()

        variety = None
        variety_original_author = None
        variety_combination_author = None
        authors = ""

        if "var." in parts:
            var_index = parts.index("var.")
            variety = parts[var_index + 1].lower()
            remaining = parts[var_index + 2:]
        else:
            remaining = parts[2:]

        if remaining:
            joined = " ".join(remaining)
            var_orig_match = re.search(r"\\(([^)]+)\\)", joined)
            if var_orig_match:
                variety_original_author = var_orig_match.group(1).strip()
                variety_combination_author = re.sub(r"\\([^)]+\\)", "", joined).strip()
            else:
                variety_combination_author = joined.strip()

        if not variety:
            authors = variety_combination_author
            variety_combination_author = None

        return genus, species, variety, variety_original_author, variety_combination_author, authors

    def _format_record(self, genus, species, authors, acc_genus, acc_species, acc_authors, relationship,
                       variety=None, variety_original_author=None, variety_combination_author=None):
        # Construct scientific_name with authors included
        if variety:
            name_key = f"{genus} {species} var. {variety}"
            scientific_name = name_key
            if variety_original_author:
                scientific_name += f" ({variety_original_author})"
            if variety_combination_author:
                scientific_name += f" {variety_combination_author}"
        else:
            name_key = f"{genus} {species}"
            scientific_name = name_key
            if authors:
                scientific_name += f" {authors}"

        acc_key = f"{acc_genus} {acc_species}" + (f" var. {variety}" if variety else "")

        record = {
            "scientific_name": scientific_name,
            "genus": genus,
            "species": species,
            "authors": authors,
            "accepted_name": acc_key,
            "accepted_authors": acc_authors,
            "relationship": relationship,
            "variety": variety,
            "variety_original_author": variety_original_author,
            "variety_combination_author": variety_combination_author
        }
        return record

    def _parse_file(self, filepath):
        accepted_genus = accepted_species = accepted_authors = None
        accepted_variety = accepted_var_orig = accepted_var_comb = None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if not line.strip():
                    continue

                indent = len(line) - len(line.lstrip())
                stripped = line.strip()
                is_synonym_line = stripped.startswith(('=', '+'))
                clean = stripped.lstrip('=+').strip()

                # Handle synonym = accepted pattern
                if '=' in clean and is_synonym_line:
                    parts = [p.strip() for p in re.split(r'\s*=\s*', clean)]
                    if len(parts) == 2:
                        syn_full, acc_full = parts
                        syn_g, syn_s, syn_var, syn_var_orig, syn_var_comb, syn_auth = self._split_species_and_authors(syn_full)
                        acc_g, acc_s, acc_var, acc_var_orig, acc_var_comb, acc_auth = self._split_species_and_authors(acc_full)

                        syn_key = f"{syn_g} {syn_s}" + (f" var. {syn_var}" if syn_var else "")
                        acc_key = f"{acc_g} {acc_s}" + (f" var. {acc_var}" if acc_var else "")

                        self.species_dict[syn_key] = self._format_record(
                            syn_g, syn_s, syn_auth,
                            acc_g, acc_s, acc_auth,
                            "synonym",
                            variety=syn_var,
                            variety_original_author=syn_var_orig,
                            variety_combination_author=syn_var_comb
                        )
                        if acc_key not in self.species_dict:
                            self.species_dict[acc_key] = self._format_record(
                                acc_g, acc_s, acc_auth,
                                acc_g, acc_s, acc_auth,
                                "accepted",
                                variety=acc_var,
                                variety_original_author=acc_var_orig,
                                variety_combination_author=acc_var_comb
                            )
                    continue

                # Handle indented synonyms under an accepted block
                if indent > 0 and accepted_genus:
                    syn_g, syn_s, syn_var, syn_var_orig, syn_var_comb, syn_auth = self._split_species_and_authors(clean)
                    syn_key = f"{syn_g} {syn_s}" + (f" var. {syn_var}" if syn_var else "")
                    self.species_dict[syn_key] = self._format_record(
                        syn_g, syn_s, syn_auth,
                        accepted_genus, accepted_species, accepted_authors,
                        "synonym",
                        variety=syn_var,
                        variety_original_author=syn_var_orig,
                        variety_combination_author=syn_var_comb
                    )
                    continue

                # Start new accepted block
                if not is_synonym_line:
                    accepted_genus, accepted_species, accepted_variety, accepted_var_orig, accepted_var_comb, accepted_authors = self._split_species_and_authors(clean)
                    acc_key = f"{accepted_genus} {accepted_species}" + (f" var. {accepted_variety}" if accepted_variety else "")
                    self.species_dict[acc_key] = self._format_record(
                        accepted_genus, accepted_species, accepted_authors,
                        accepted_genus, accepted_species, accepted_authors,
                        "accepted",
                        variety=accepted_variety,
                        variety_original_author=accepted_var_orig,
                        variety_combination_author=accepted_var_comb
                    )

                # Handle synonym line in accepted context
                elif is_synonym_line and accepted_genus:
                    syn_g, syn_s, syn_var, syn_var_orig, syn_var_comb, syn_auth = self._split_species_and_authors(clean)
                    syn_key = f"{syn_g} {syn_s}" + (f" var. {syn_var}" if syn_var else "")
                    self.species_dict[syn_key] = self._format_record(
                        syn_g, syn_s, syn_auth,
                        accepted_genus, accepted_species, accepted_authors,
                        "synonym",
                        variety=syn_var,
                        variety_original_author=syn_var_orig,
                        variety_combination_author=syn_var_comb
                    )

    def lookup(self, name):
        return self.species_dict.get(name)

    def all_names(self):
        return list(self.species_dict.keys())

    def to_json(self, filepath=None):
        json_data = json.dumps(self.species_dict, indent=2, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_data)
        return json_data

    def grouped_by_accepted(self):
        grouped = {}
        for record in self.species_dict.values():
            acc_name = record["accepted_name"]
            acc_auth = record["accepted_authors"]
            if record["relationship"] == "accepted":
                grouped[acc_name] = {
                    "authors": acc_auth,
                    "synonyms": []
                }
            elif record["relationship"] == "synonym":
                if acc_name not in grouped:
                    grouped[acc_name] = {
                        "authors": acc_auth,
                        "synonyms": []
                    }
                grouped[acc_name]["synonyms"].append(record)
        return grouped

    def to_dataframe(self):
        def extract_gbif_authors(author_str):
            if not author_str:
                return ""
            # Extract only surname from each part, keep parenthetical authors and outer ones
            parts = re.findall(r'\(([^)]+)\)', author_str)
            remaining = re.sub(r'\([^)]*\)', '', author_str).strip()
            final = []
            if parts:
                final.append(f"({parts[0].split()[-1]})")
            if remaining:
                final.append(remaining.split()[-1])
            return " ".join(final)

        records = []
        for name, record in self.species_dict.items():
            row = {"name": name}
            for key in ["scientific_name", "genus", "species", "authors", "accepted_name",
                        "accepted_authors", "relationship", "variety", "variety_original_author", "variety_combination_author"]:
                row[key] = record.get(key, None)
            row["gbif_scientific_name"] = f"{record['genus']} {record['species']} {extract_gbif_authors(record['authors'])}".strip()
            row["accepted_gbif_name"] = f"{record['genus']} {record['species']} {extract_gbif_authors(record['accepted_authors'])}".strip()
            records.append(row)
        return pd.DataFrame(records)

    def accepted_with_synonyms(self):
        grouped = {}
        for name, record in self.species_dict.items():
            if record["relationship"] == "accepted":
                grouped[record["scientific_name"]] = []
        for name, record in self.species_dict.items():
            if record["relationship"] == "synonym":
                acc_name = record["accepted_name"]
                if acc_name not in grouped:
                    grouped[acc_name] = []
                grouped[acc_name].append(record["scientific_name"])
        return grouped

    def names_with_accepted(self):
        name_map = {}
        for name, record in self.species_dict.items():
            name_map[name] = record["accepted_name"]
        return name_map

    def filter_by(self, relationship=None, genus=None):
        return {
            name: rec for name, rec in self.species_dict.items()
            if (relationship is None or rec['relationship'] == relationship)
            and (genus is None or rec['genus'].lower() == genus.lower())
        }

    def fuzzy_lookup(self, query):
        query_norm = query.strip().lower().replace("var.", "var")
        return {
            name: rec for name, rec in self.species_dict.items()
            if query_norm in name.lower().replace("var.", "var")
        }

    def normalize_name(self, genus, species, variety=None):
        genus = genus.capitalize()
        species = species.lower()
        if variety:
            variety = variety.lower()
            return f"{genus} {species} var. {variety}"
        else:
            return f"{genus} {species}"
