import os
import csv
import shutil
import random
import tempfile
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from ordered_set import OrderedSet
from scripts.identify_ptms import identify_ptms

# TODO: Fix this unittests later


class TestIdentifyPTMs(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test.
        """

        # Create a temporary directory
        self.temp_dir = Path(tempfile.mkdtemp())
        self.files = []
        # Create two project folders
        self.project1_dir = self.temp_dir / "PROJECT1"
        self.project2_dir = self.temp_dir / "PROJECT2"
        os.makedirs(self.project1_dir, exist_ok=True)
        os.makedirs(self.project2_dir, exist_ok=True)

        # Create some CSV files paths to be used as ipc file
        for f in ("file1.csv", "file2.csv", "file3.csv", "file4.csv"):
            path = self.project1_dir / f
            self.files.append(path)
            self._fs_write_synthetic_data(n_samples=200, file_path=path)

    def tearDown(self):
        # Restore the original logging configuration
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _get_peptide_from_modified_peptide(modified_peptide):

        assert modified_peptide is not None, modified_peptide

        glycan_masses = (
            "1152",
            "215",
            "143",
            "130",
            "1809",
            "1518",
            "1493",
            "1006",
            "2013",
            "147",
        )
        peptide = modified_peptide
        # This is a naive implementation
        for glycan_mass in glycan_masses:
            peptide = peptide.replace(f"N[{glycan_mass}]", "N")

        return peptide

    @staticmethod
    def _get_synthetic_data_header():

        return [
            "index",
            "peptide",
            "modified_peptide",
            "precursor_mz",
            "precursor_charge",
            "mz",
            "intensity",
            "rt",
            "delta_mass",
        ]

    def _generate_synthetic_data(self, n_samples):
        """
        Generate synthetic data for the CSV files using realistic peptide sequences and PTMs.

        Args:
            n_samples (int): Number of rows to generate.

        Returns:
            pd.DataFrame: DataFrame with the required fields.

        """

        glycosylated_peptide_samples = (
            "GLVSGGVYNSHVGCLYTIPPECEHVN[1152]GSRRPCTEGDTR",
            "GLVSGGVYNSHVGCLYTIPPECEHVN[215]GSR",
            "KGLVSGGVYNSHVGCLYTIPPECEHVN[215]GSR",
            "HNN[143]DTQHWEVSDSNESFVADR",
            "HNN[130]DTQHWEVSDSNESFVADR",
            "KGLVSGGVYNSHVGCLYTIPPECEHVN[1809]GSRRPCTEGDTR",
            "KGLVSGGVYNSHVGCLYTIPPECEHVN[1518]GSRRPCTEGDTR",
            "LCVVALDFEQEMATAASSSSLEK",  # No PTM
            "VSINTVN[1493]LTAGQPMEVTVFR",
            "GLVSGGVYNSHVGCLYTIPPECEHVN[1809]GSRRPCTEGDTR",
            "GLVSGGVYNSHVGCLYTIPPECEHVN[1006]GSRRPCTEGDTR",
            "GLVSGGVYNSHVGCLYTIPPECEHVN[2013]GSR",
            "VSINTVN[147]LTAGQPMEVTVFR",
        )

        precursor_mz_mean, precursor_mz_std = 1107.484847, 219.590290
        precursor_charge_mean, precursor_charge_std = 3.456838, 1.048360
        rt_mean, rt_std = 12813.013395, 4852.915073
        delta_mass_mean, delta_mass_std = 0.324101, 0.654346

        peptides = []
        modified_peptides = []

        for i in range(n_samples):
            sample = random.choice(glycosylated_peptide_samples)
            modified_peptides.append(sample)
            peptides.append(self._get_peptide_from_modified_peptide(sample))

        # TODO: np.identify bellow should be removed and np.clip should be used instead
        return [
            {
                "index": i,
                "peptide": peptides[i % len(peptides)],
                "modified_peptide": modified_peptides[i % len(modified_peptides)],
                "precursor_mz": np.random.normal(precursor_mz_mean, precursor_mz_std),
                "precursor_charge": np.round(
                    np.random.normal(precursor_charge_mean, precursor_charge_std)
                ),
                "mz": np.random.uniform(100, 2000),
                "intensity": np.random.uniform(1e3, 1e6),
                "rt": np.random.normal(rt_mean, rt_std),
                "delta_mass": np.random.normal(delta_mass_mean, delta_mass_std),
            }
            for i in range(n_samples)
        ]

    def _fs_write_synthetic_data(
        self,
        n_samples,
        file_path: str,
    ):
        data = self._generate_synthetic_data(n_samples)
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self._get_synthetic_data_header())

            # Write header
            writer.writeheader()

            # Write rows
            writer.writerows(data)

    @patch("identify_ptms.logging")
    def test_identify_ptms_no_ptms(self, mock_read_feather):
        """
        Test identify_ptms when there are no PTMs in the data.
        """
        # Create a DataFrame with no PTMs
        df_no_ptms = pd.DataFrame(
            {"modified_peptide": ["PEPTIDE", None, "ANOTHER"], "index": [0, 1, 2]},
            index=[0, 1, 2],
        )

        mock_read_feather.return_value = df_no_ptms

        # Call the function
        result = identify_ptms(self.files, ptm_examples_limit=2)

        # Expected output: empty dictionary since no PTMs were found
        expected = {}
        self.assertEqual(result, expected)

    @patch("identify_ptms.logging")
    def test_identify_ptms_example_limit(self, mock_read_feather):
        """
        Test that identify_ptms respects the ptm_examples_limit parameter.
        """
        # Create a DataFrame with multiple instances of the same glycan mass
        df_limit_test = pd.DataFrame(
            {
                "modified_peptide": [
                    "PEPTIDE[N123]",  # Glycan mass 123
                    "ANOTHER[N123]",  # Same glycan mass
                    "YETANOTHER[N123]",  # Same glycan mass (should be ignored due to limit)
                ],
                "index": [0, 1, 2],
            },
            index=[0, 1, 2],
        )

        mock_read_feather.return_value = df_limit_test

        # Call the function with a limit of 2 examples per PTM
        result = identify_ptms(self.files, ptm_examples_limit=2)

        # Expected output: only 2 examples for glycan mass 123
        expected = {
            "123": OrderedSet(
                [
                    ("123", "project1", "file1.feather", 0, 0),
                    ("123", "project1", "file1.feather", 1, 1),
                ]
            )
        }

        self.assertEqual(result, expected)
        self.assertEqual(len(result["123"]), 2)  # Ensure the limit is respected

    @patch("identify_ptms.logging")
    def test_identify_ptms_duplicate_examples(self, mock_read_feather):
        """
        Test that identify_ptms skips duplicate peptide sequences.
        """
        # Create a DataFrame with duplicate modified peptides
        df_duplicates = pd.DataFrame(
            {
                "modified_peptide": [
                    "PEPTIDE[N123]",  # First occurrence
                    "PEPTIDE[N123]",  # Duplicate (should be skipped)
                    "ANOTHER[N123]",  # Different peptide, same glycan mass
                ],
                "index": [0, 1, 2],
            },
            index=[0, 1, 2],
        )

        mock_read_feather.return_value = df_duplicates

        # Call the function with a limit of 3 examples per PTM
        result = identify_ptms(self.files, ptm_examples_limit=3)

        # Expected output: only 2 examples, as the duplicate is skipped
        expected = {
            "123": OrderedSet(
                [
                    ("123", "project1", "file1.feather", 0, 0),
                    ("123", "project1", "file1.feather", 2, 2),
                ]
            )
        }

        self.assertEqual(result, expected)
        self.assertEqual(len(result["123"]), 2)  # Ensure the duplicate was skipped


if __name__ == "__main__":
    unittest.main()
