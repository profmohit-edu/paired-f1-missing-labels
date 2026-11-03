# Protocol fixed before the benchmark run

This is a local prospective specification, not an externally preregistered study.

Primary contribution: an exact finite-pool, paired F1 difference interval and
attainable endpoint witnesses, using prediction-cell capacities and a supplied
interval for the number of positive missing labels. The quantity is a
deterministic identification interval, not a confidence interval.

Primary comparator: subtract individually sharp single-classifier intervals
computed under exactly the same missing-positive count constraint. No claim
of superiority to probabilistic imputation or weak-supervision estimators,
whose assumptions and estimands differ, will be made.

Validation: enumerate all paired prediction vectors and ternary label vectors
through pool size three, with every feasible contiguous missing-positive count
interval; add 1000 random pools up to size twelve; compare 200 fixed-count
instances with a generic linear program in both directions. Include empty
pools, zero denominators, equality, swapped models and attainable witnesses.

Benchmark: scikit-learn bundled breast cancer, wine and digits. Binary labels:
original class zero for cancer and wine, digits at least five for digits.
Thirty seeds 0 to 29; stratified 70/30 train/test split. Fit logistic regression
with standardization, C=1, max_iter=2000; tree max_depth=3; forest 100 trees,
max_depth=5, n_jobs=1. Default remaining scikit-learn 1.8 parameters. No
hyperparameter tuning or reuse of test outcomes for fitting.

After fitting, hide floor(r*n_test) outcomes for r in {0.2,0.5,0.8}, using
weighted sampling without replacement: uniform weights, 4:1 weights for
logistic predicted positives, or 4:1 weights for true positives. The last
mechanism intentionally depends on held-out truth to simulate outcome
dependent missingness. No claim that these mechanisms reflect a deployment.

All three classifier pairs use each mask. Count regimes: unrestricted [0,m];
an oracle sensitivity band of plus/minus ceil(0.1*m) around the true hidden
positive count; and the oracle exact count. The latter two are not available
from partial labels alone and must never be represented as deployable
information or estimated prevalence confidence intervals.

Report interval widths, certificate fractions, extra certificates, and wrong
complete-case signs as descriptive summaries. Repeated splits, classifier
pairs and masks share observations; they are not independent experimental
replicates. No significance testing or dataset-general population inference.
All cases and zero-width intervals retained. Three datasets illustrate the
algorithm, not clinical validity or state-of-the-art model performance.

Scaling: ten runs at 100,1000,10000,100000 missing records, using four fixed
cell proportions. Report elapsed wall-clock medians as local measurements.
