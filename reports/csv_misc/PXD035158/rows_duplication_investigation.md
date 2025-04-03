# Duplicate Analysis

## Goal
Understand how the data is like and at which level is the dataset high quality. Investigate at which point is the data is duplicated

## Motivation
Training a model on a dataset make of the same entry means nothing. And as the distribution of peptides and modified peptides show high peak, 
it means some peptides or their modified versions are repeated many times. Are our entries repeated? If not at which level
do we have repetition?

## Approach
Check duplicates by progressively combining the columns of the dataset we care about.
    - Do this at project level understand the pattern that emerges
    - Save the peptides c
Check duplication of peptides/modified peptides across projects

## Result

Number of lines in the project PXD035158 dataset: 312275. This project will be our reference project.

- Many tuples `(peptide, modified_peptide, precursor_charge,)` has different `precursor_mz` meaning that `precursor_mz` reduce the duplication.
- Also `(peptide, modified_peptide)`, `(modified_peptide)` and `(peptide, modified_peptide, precursor_charge,)` are pretty much distributed the same way




