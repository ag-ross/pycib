"""
Utility functions for CIB data import and export.

This module provides functions to load and save CIB matrices and descriptors
in CSV and JSON formats for interoperability with other tools.
"""

from __future__ import annotations

import csv
import json
from typing import Dict, List, Tuple

from cib.core import CIBMatrix


def load_from_csv(
    descriptors_path: str, impacts_path: str
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str, str, str], float]]:
    """
    Load descriptors and impacts from CSV files.

    Args:
        descriptors_path: Path to CSV file containing descriptor definitions.
            Expected format: Descriptor,State1,State2,State3,...
        impacts_path: Path to CSV file containing impact values.
            Expected format: Source_Descriptor,Source_State,Target_Descriptor,
            Target_State,Impact

    Returns:
        Tuple of (descriptors dictionary, impacts dictionary).

    Raises:
        FileNotFoundError: If CSV files are not found.
        ValueError: If CSV format is invalid.
    """
    descriptors: Dict[str, List[str]] = {}
    impacts: Dict[Tuple[str, str, str, str], float] = {}

    try:
        with open(descriptors_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                desc_name = row["Descriptor"]
                states = [
                    row[key]
                    for key in row.keys()
                    if key != "Descriptor" and row[key]
                ]
                if states:
                    descriptors[desc_name] = states
    except FileNotFoundError:
        raise FileNotFoundError(f"Descriptors file not found: {descriptors_path}")
    except KeyError as e:
        raise ValueError(f"Invalid CSV format: missing column {e}")

    try:
        with open(impacts_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_desc = row["Source_Descriptor"]
                src_state = row["Source_State"]
                tgt_desc = row["Target_Descriptor"]
                tgt_state = row["Target_State"]
                impact = float(row["Impact"])

                key = (src_desc, src_state, tgt_desc, tgt_state)
                impacts[key] = impact
    except FileNotFoundError:
        raise FileNotFoundError(f"Impacts file not found: {impacts_path}")
    except KeyError as e:
        raise ValueError(f"Invalid CSV format: missing column {e}")
    except ValueError as e:
        raise ValueError(f"Invalid impact value: {e}")

    return descriptors, impacts


def save_to_csv(
    matrix: CIBMatrix, descriptors_path: str, impacts_path: str
) -> None:
    """
    Save descriptors and impacts to CSV files.

    Args:
        matrix: CIB matrix to save.
        descriptors_path: Path to save descriptor definitions.
        impacts_path: Path to save impact values.

    Raises:
        IOError: If files cannot be written.
    """
    max_states = max(matrix.state_counts) if matrix.state_counts else 0

    with open(descriptors_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["Descriptor"] + [
            f"State{i+1}" for i in range(max_states)
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for desc_name, states in matrix.descriptors.items():
            row = {"Descriptor": desc_name}
            for i, state in enumerate(states):
                row[f"State{i+1}"] = state
            for i in range(len(states), max_states):
                row[f"State{i+1}"] = ""
            writer.writerow(row)

    with open(impacts_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "Source_Descriptor",
            "Source_State",
            "Target_Descriptor",
            "Target_State",
            "Impact",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key, value in matrix.iter_impacts():
            src_desc, src_state, tgt_desc, tgt_state = key
            writer.writerow(
                {
                    "Source_Descriptor": src_desc,
                    "Source_State": src_state,
                    "Target_Descriptor": tgt_desc,
                    "Target_State": tgt_state,
                    "Impact": value,
                }
            )


def load_from_json(
    path: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str, str, str], float]]:
    """
    Load descriptors and impacts from a JSON file.

    Args:
        path: Path to JSON file containing both descriptors and impacts.
            Expected format:
            {
                "descriptors": {"Desc1": ["State1", "State2"], ...},
                "impacts": {
                    "Desc1|State1|Desc2|State1": 2,
                    ...
                }
            }

    Returns:
        Tuple of (descriptors dictionary, impacts dictionary).

    Raises:
        FileNotFoundError: If JSON file is not found.
        ValueError: If JSON format is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    if "descriptors" not in data:
        raise ValueError("JSON missing 'descriptors' key")
    if "impacts" not in data:
        raise ValueError("JSON missing 'impacts' key")

    descriptors = data["descriptors"]
    impacts_raw = data["impacts"]

    impacts: Dict[Tuple[str, str, str, str], float] = {}
    for key_str, value in impacts_raw.items():
        parts = key_str.split("|")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid impact key format: {key_str}. "
                "Expected 'Desc1|State1|Desc2|State1'"
            )
        src_desc, src_state, tgt_desc, tgt_state = parts
        impacts[(src_desc, src_state, tgt_desc, tgt_state)] = float(value)

    return descriptors, impacts


def save_to_json(matrix: CIBMatrix, path: str) -> None:
    """
    Save descriptors and impacts to a JSON file.

    Args:
        matrix: CIB matrix to save.
        path: Path to save JSON file.

    Raises:
        IOError: If file cannot be written.
    """
    data = {
        "descriptors": matrix.descriptors,
        "impacts": {},
    }

    for key, value in matrix.iter_impacts():
        src_desc, src_state, tgt_desc, tgt_state = key
        key_str = f"{src_desc}|{src_state}|{tgt_desc}|{tgt_state}"
        data["impacts"][key_str] = value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
