# Judge Bias Observations

| Bias | Measurement | Result | Interpretation | Mitigation |
|---|---:|---:|---|---|
| Position bias | A wins when listed first | 10/30 (33.3%) | No strong bias observed | Use swap-and-average for every pairwise call. |
| Length bias | B wins when B is longer | 3/9 (33.3%) | No strong length bias observed | Rubric explicitly rewards concise, grounded answers. |

## Conclusion

The judge pipeline keeps swap-and-average enabled by default. Absolute scoring is used as a second signal so pairwise results do not overfit to answer position or verbosity.
