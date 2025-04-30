import re
import json
import pandas as pd

class Nomenclator:
    def __init__(self, filepath):
        self.species_dict = {}
        self._parse_file(filepath)

    def _split_species_and_authors(self, name_str):
        """
        Split into genus, species, and authors, applying capitalization rules:
        - Genus: Capitalize first letter only
        - Species: all lowercase
        """
        parts = name_str.strip().split()
        if len(parts) < 2:
            return None, None, name_str.strip()
        genus = parts[0].capitalize()
        species = parts[1].lower()
        authors = " ".join(parts[2:]) if len(parts) > 2 else ""
        return genus, species, authors


    def _format_record(self, genus, species, authors, acc_genus, acc_species, acc_authors, relationship):
        return {
            "scientific_name": f"{genus} {species}",
            "genus": genus,
            "species": species,
            "authors": authors,
            "accepted_name": f"{acc_genus} {acc_species}",
            "accepted_authors": acc_authors,
            "relationship": relationship
        }

    def _parse_file(self, filepath):
        accepted_genus = accepted_species = accepted_authors = None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if not line.strip():
                    continue

                indent = len(line) - len(line.lstrip())
                stripped = line.strip()
                is_synonym_line = stripped.startswith(('=', '+'))
                clean = stripped.lstrip('=+').strip()

                if '=' in clean and is_synonym_line:
                    parts = [p.strip() for p in re.split(r'\s*=\s*', clean)]
                    if len(parts) == 2:
                        syn_full, acc_full = parts
                        syn_g, syn_s, syn_auth = self._split_species_and_authors(syn_full)
                        acc_g, acc_s, acc_auth = self._split_species_and_authors(acc_full)

                        if syn_g and acc_g:
                            self.species_dict[f"{syn_g} {syn_s}"] = self._format_record(
                                syn_g, syn_s, syn_auth,
                                acc_g, acc_s, acc_auth,
                                "synonym"
                            )
                            if f"{acc_g} {acc_s}" not in self.species_dict:
                                self.species_dict[f"{acc_g} {acc_s}"] = self._format_record(
                                    acc_g, acc_s, acc_auth,
                                    acc_g, acc_s, acc_auth,
                                    "accepted"
                                )
                    continue

                if indent > 0 and accepted_genus:
                    syn_g, syn_s, syn_auth = self._split_species_and_authors(clean)
                    self.species_dict[f"{syn_g} {syn_s}"] = self._format_record(
                        syn_g, syn_s, syn_auth,
                        accepted_genus, accepted_species, accepted_authors,
                        "synonym"
                    )
                    continue

                if not is_synonym_line:
                    accepted_genus, accepted_species, accepted_authors = self._split_species_and_authors(clean)
                    self.species_dict[f"{accepted_genus} {accepted_species}"] = self._format_record(
                        accepted_genus, accepted_species, accepted_authors,
                        accepted_genus, accepted_species, accepted_authors,
                        "accepted"
                    )

                elif is_synonym_line and accepted_genus:
                    syn_g, syn_s, syn_auth = self._split_species_and_authors(clean)
                    self.species_dict[f"{syn_g} {syn_s}"] = self._format_record(
                        syn_g, syn_s, syn_auth,
                        accepted_genus, accepted_species, accepted_authors,
                        "synonym"
                    )

    def lookup(self, name):
        return self.species_dict.get(name)

    def all_names(self):
        return list(self.species_dict.keys())

    def to_json(self, filepath=None):
        """
        Return or save the dictionary as JSON.
        
        Args:
            filepath (str or None): if provided, writes to this file.
        
        Returns:
            str: JSON string if filepath is None
        """
        json_data = json.dumps(self.species_dict, indent=2, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_data)
        return json_data

    def grouped_by_accepted(self):
        """
        Return a nested dictionary of accepted names and their synonyms.
        
        Returns:
            dict: {accepted_name: {'authors': str, 'synonyms': [dict, ...]}}
        """
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
        """
        Return a pandas DataFrame from the species dictionary,
        with 'name' as a column instead of the index.
        
        Returns:
            pandas.DataFrame
        """
        records = []
        for name, record in self.species_dict.items():
            row = {"name": name}
            row.update(record)
            records.append(row)
        return pd.DataFrame(records)

    def accepted_with_synonyms(self):
        """
        Return a dict mapping each accepted scientific name to a list of its synonyms.
        
        Returns:
            dict: {accepted_name: [synonym_name_1, synonym_name_2, ...]}
        """
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
        """
        Return a flat dict mapping each name (synonym or accepted) to its accepted name.
        
        Returns:
            dict: {name: accepted_name}
        """
        name_map = {}
        for name, record in self.species_dict.items():
            name_map[name] = record["accepted_name"]
        return name_map