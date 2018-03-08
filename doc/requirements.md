# Requirements
This is a list of use cases for which one should be able to use `snpop`

## SALT2 SNIa 
use SNPop to represent a population of Ia following the SALT2 model with different distributions describing the populations in the following cases:
- Simulate a set of SN drawing from the population distribution with known sets of observations (eg. LSST OpSim)
- Re-simulate an observed (or simulated) set of SN with known ids and redshifts from a different source, but changing the SN parameters like X0, x1, c etc.

This should be doable without Milky Way extinction as well as milky way extinction.
- The latter implies a method to provide coordinates 

