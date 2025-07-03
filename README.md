
# Threshold-Free Network-Based Statistics in Neuroscience

``TFNBS`` 

<!--- [pypi version] -->
[license]https://img.shields.io/github/license/:user/TFNBS?color=darkgreen
[size]https://img.shields.io/github/repo-size/iknyazeva/tfnbs


TFNBS Toolbox is a Python package for computation and generation of network-based statistics for neuroscience data 
(i.e. fMRI, EEG data). The core concept is based on eliminating the use of a hardcoded threshold using threshold-free 
cluster enhancement (TFCE) scores to assess statistical significance. It works on the principle of networks, where TFCE 
statistical values are computed across a range of thresholds over n cycles of permutations to uncover possible significance in the data. 
Our implementation of TFNBS follows efficient computing and allows for computations to be performed on parallel cores therefore 
massively reducing computation time and resources. 

Overview.png

![Overview of TFNOS](docs/TFNBS_Pipeline.png)

## Installation 
TFNOS toolbox can be installed using: 

```bash
    pip install tfnbs
```

## Documentation

For more information on TFNBS's features, modules and usage, please refer to the [official documentation](). 

Examples of usage on fMRI and EEG data are avaialble in [notebooks]() and [data](). 


## Citing the toolbox 
To cite the toolbox, you can use: [doi]() and refer to the paper [paper_doi]()
```base
    [doi]
```

For further discussions or reports on bugs, please contact [test@email.com]()
