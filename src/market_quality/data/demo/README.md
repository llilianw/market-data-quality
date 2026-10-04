# Derived multi-contract sample

`multi_contract_sample.parquet` concatenates unchanged original rows from the
supplied `sample/minute/CME/ES/ESZ25.parquet` and `ESH26.parquet` files for the
America/Chicago trading sessions closing on October 16–17, 2025.

It contains 2,760 ESZ25 rows and 629 ESH26 rows, preserving every source column.
It exists only to exercise multi-contract application behavior; it is not a
synthetic market scenario. Inferred gaps remain conditional on the configured
cadence/session assumptions and these bounded observations.

With the default `1min` cadence and America/Chicago overnight session schedule:

| Contract | Unexpected gap intervals | Missing expected timestamps |
|---|---:|---:|
| ESZ25 | 0 | 0 |
| ESH26 | 296 | 2,128 |

The stored classification value is `unexpected_gap`; filter with
`GapClassification.UNEXPECTED_GAP.value`. ESZ25 is absent from an unexpected-gap
grouping because it has no matching segments. These counts do not prove that
source records were lost.
