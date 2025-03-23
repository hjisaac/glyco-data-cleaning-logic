#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Script to identify PTMs (Post Translation Modifications) in a AA (Amino Acid) sequences 
i.e., proteins. The PTMs distribution is pretty much like a gaussian distribution.
"""

import os
import re
import sys
import logging

# https://stackoverflow.com/questions/17935130/which-module-should-contain-logging-config-dictconfigmy-dictionary-what-about
import logging.config  # noqa
import itertools
import pandas as pd
from tqdm import tqdm
from enum import Enum
from ordered_set import OrderedSet

# Temporary fix for imports, investigate later
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from common.utils import get_ipc_files
from common.constants import BASE_RAW_DATA_DIR, BASE_PTMS_DIR
from common.logger import get_logger_config


# In[2]:


logger_config = get_logger_config(subdir="scripts")
logging.config.dictConfig(logger_config)
logger = logging.getLogger(__name__)


class PTMSitesEnum(str, Enum):
    """
    Potential amino acid on which we may have a ptms.
    """

    # Asparagine (N) # noqa
    N_GLYCOSYLATION = "N"
    # Threonine (T) and Serine (S)
    O_GLYCOSYLATION = "ST"
    GLYCOSYLATION = N_GLYCOSYLATION + O_GLYCOSYLATION
    PHOSPHORYLATION = "XYZ"  # FIXME


GLYCOSYLATION_REGEX_TEMPLATE = r"(?P<aa>[{sites}])\[(?P<glycan_mass>\d+)\]"
# Glycosylation only happens on Asparagine (Asn-{Any}-Ser)
N_GLYCOSYLATION_REGEX = GLYCOSYLATION_REGEX_TEMPLATE.format(sites="N")

ipc_files = get_ipc_files(BASE_RAW_DATA_DIR)


def identify_ptms(
    ipc_files: list, ptm_examples_limit: int = 5, return_df=True  # noqa
) -> dict[str : OrderedSet[tuple]]:
    """

    Identify post-translational modifications (PTMs) from a list of IPC files.

    This function processes a list of IPC files containing peptide sequences
    and extracts PTMs. If a PTM is found, its occurrence is tracked efficiently
    using an OrderedSet just to keep the insertion the processing order. The function
    limits the number of stored examples per PTM to avoid excessive memory consumption.

    Note: After analyzing the data, we can see that the number of ptm is not huge.
    So if the number of examples is not huge, this algo will be efficient in
    terms of memory usage. The data is like few ptms occurs a very huge number
    of times.


    Args:
        ipc_files (list): A list of file paths to IPC files in Feather format.
        ptm_examples_limit (int, optional): The maximum number of examples to store for each PTM.
            Default to 5.

    Returns:
        dict: A dictionary where keys are glycan mass values and values are OrderedSets
        containing tuples of (glycan_mass, project_name, file_name, spectrum_id, ipc_index).
    """

    # Seen ptms
    seen_ptms = {}

    # As glob list files folder by folder, keeping track of the
    # previous project helps us to know when we change a project.
    current_project_name = None

    for ipc_file in tqdm(ipc_files, desc="Processing IPC files", unit="file"):

        # Group the files per project could help to avoid doing this
        *_, project_name, file_name = str(ipc_file).split("/")

        if current_project_name != project_name:
            logger.info(f"Start processing the ipc file  of project {project_name}")
            current_project_name = project_name

        df = pd.read_feather(ipc_file)

        added_examples_count = 0

        for peptide_sequence in df.itertuples(name="PeptideSequence"):
            if peptide_sequence.modified_peptide is None:
                continue

            ptms = re.findall(  # noqa
                N_GLYCOSYLATION_REGEX, peptide_sequence.modified_peptide
            )

            if not ptms:
                # Do nothing if there is no ptm
                continue

            for aa, glycan_mass in ptms:
                if glycan_mass in seen_ptms:
                    if len(seen_ptms[glycan_mass]) == ptm_examples_limit:
                        continue
                    if peptide_sequence.modified_peptide in seen_ptms[glycan_mass]:
                        # This is a known/seen example, so continue
                        continue

                    seen_ptms[glycan_mass].add(
                        # A hashable datastructures is necessary for OrderSet to work.
                        # Index and index of peptide respectively represent df index and spectrum index
                        # ({glycan_mass}, {project name}, {file_name}, {spectrum_id}, {ipc index})
                        (
                            glycan_mass,
                            project_name,
                            file_name,
                            peptide_sequence.index,
                            peptide_sequence.Index,
                        )
                    )

                else:
                    seen_ptms[glycan_mass] = OrderedSet(
                        [
                            (
                                glycan_mass,
                                project_name,
                                file_name,
                                peptide_sequence.index,
                                peptide_sequence.Index,
                            )
                        ]
                    )

                added_examples_count += 1

        logger.info(
            f"Successfully parsed {project_name}/{file_name} ipc file and added {added_examples_count} ptms new examples."
        )

    return (
        pd.DataFrame(
            itertools.chain.from_iterable(ptms.values()),
            columns=(
                "glycan_mass",
                "project_name",
                "file_name",
                "spectrum_id",
                "ipc_index",
            ),
        )
        if return_df
        else seen_ptms
    )


# In[ ]:


if __name__ == "__main__":
    ptms_df = identify_ptms(ipc_files)
    ptms_df.to_csv(
        f"{BASE_PTMS_DIR}/identified_glyco_ptms_with_5_examples.csv", index=False
    )
