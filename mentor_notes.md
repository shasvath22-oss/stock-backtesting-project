# Notes from mentor feedback

These are my notes on the guidance I received from a Data Warehouse Architect at
New York University. They are a summary, not direct quotations.

## Starting the project

Start with SPY and keep the first experiment small enough to check. Store the
prices in SQL and make sure downloading the same dates again does not create
duplicates. In the implementation, the key also includes the snapshot ID so
separate experiments can keep their own data.

## Checking the results

Study the swing highs and lows before adding more modeling. Historical tests
should use only information that was available on the signal date. Compare AMZN
with SPY, and check losses and returns as well as prediction accuracy.
