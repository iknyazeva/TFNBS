Usage
=====

This toolbox presents a Python implementation to use the TFNBS for fMRI and EEG functional connectivity data with the 
flexibility to use structured data (i.e. block diagonal matrices) corresponding to frequency range and connectivity matrix. 
While other network-oriented toolkits are composed of higher computational complexity, TFNBS is based on the construction 
of a null distribution which is permuted across many thresholds to identify statistical significance therefore optimizing 
performance. Our implementation of TFNBS follows efficient algorithms and allows usage of parallel computing cores, therefore 
massively reducing required computing resources over conventional approaches.