# Sharp bounds for paired F1 comparisons with missing evaluation labels

Authors: Tripti Tiwari (first); Mohit Tiwari (second and corresponding)

Tripti affiliation: Department of Management Studies, Bharati Vidyapeeth (Deemed to be University) Institute of Management and Research, New Delhi, India

Tripti email: tripti.imr@gmail.com; ORCID 0000-0002-9016-1988

Mohit affiliation: Department of Computer Science and Engineering, Bharati Vidyapeeth's
College of Engineering, New Delhi, India

Correspondence: mohit.t.bvcoe@gmail.com

Journal package: Pattern Analysis and Applications

## Status

Computational and manuscript package prepared. Funding, competing interests,
actual human contributions, exclusive submission, institutional approval where
applicable, and final author approval remain to be certified. No submission has
been made. The manuscript's pending declaration paragraph and the cover letter's
approval reservation must be finalized after that certification.

## Reproduce

Use Python 3.12 with the pinned versions in requirements.txt. The input datasets
are bundled with scikit-learn. Feature data are not downloaded by the scripts.
From this directory run:

```bash
python code/validate.py
python code/experiment.py
python code/reanalyze.py
python code/lp_benchmark.py
python code/check_evidence.py
python code/figures.py
```

The optional generic LP timing experiment can take longer than the main study.
Hardware-dependent timings can differ. Predictions can vary across numerical
environments; saved predictions permit auditing the reported analysis without
refitting. Seed sequences and all hyperparameters are specified in the protocol
and experiment script. Reanalyze.py adds the stronger marginal comparator and
constructed example after the initial run, as disclosed in the manuscript.

## Interpretation

Labels equal to -1 are missing to the solver. Binary labels 0 and 1 are observed.
An optional q interval constrains the missing positive count, not the full-pool
positive count. Outputs are rational sharp interval-hull endpoints and
attainable four-cell witnesses. They are not confidence intervals. Oracle
count regimes in the benchmark use hidden truth only as sensitivity analyses.
The independent comparator can use different q values for the two scores. The
count-matched comparator shares q but relaxes individual-label consistency.
The generic LP solves exactly the joint feasible problem.

## Files

- code/bounds.py: specialized solver and both marginal comparators.
- code/validate.py: exhaustive label-level and fixed-count LP verification.
- code/experiment.py: model fitting, controlled masking, primary comparisons.
- code/reanalyze.py: strengthened comparator and summaries.
- code/lp_benchmark.py: equivalent whole-envelope generic LP comparison.
- code/check_evidence.py: reconstructs and scores all endpoint witnesses.
- code/figures.py: regenerates the three scientific figures.
- results/comparisons.csv: all 7,290 cases, including witnesses and timings.
- results/prediction_records/: 90 NPZ files with split indices, complete test
  outcomes, frozen predictions, and each missingness mask.
- results/provenance.json: dataset shapes, recoding and array hashes.
- results/validation.json and evidence_audit.json: executed check results.
- results/fits.csv, scaling.csv, lp_benchmark.csv and summary.csv: raw and
  aggregated computational results.
- audit/PROTOCOL.md: locally fixed initial study specification; not an external
  preregistration.
- audit/Review_and_Revision.md: ten simulated perspectives and editor screen;
  private internal review, not external peer review.
- submission/: manuscript, cover letter, author form, metadata and highlights.

Online Resource 1 (ESM_1.zip) contains the reproducibility files, not the private
simulated reviews or personal author-approval form. Its accompanying caption is:
“Executable solver, validation, benchmark records, masking indices, endpoint
witnesses, protocol and reproduction instructions for paired finite-pool F1
bounds.”

## Data attribution and reuse

The data originate from the UCI Machine Learning Repository and are loaded from
scikit-learn. Original dataset citations appear in manuscript references 11–13.
The UCI dataset records list CC BY 4.0, which permits adaptation with attribution:
https://creativecommons.org/licenses/by/4.0/ . The supplied records transform
targets into the stated binary tasks and add predictions and controlled masks.
No clinical use, new collection, or writer-independent resampling is claimed.
The original feature matrices are not included. This archive does not claim
ownership of third-party data or software; their original licenses apply.

## AI disclosure

ChatGPT assisted with question formulation, derivations, programming, literature
retrieval, tool-driven computation, drafting and internal review. No AI-generated
labels are used as empirical truth. The named human authors must review and take
responsibility before submission. Neither simulated review nor successful tests
guarantee journal acceptance.
