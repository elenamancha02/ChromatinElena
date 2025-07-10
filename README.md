# ChromatinElena

This repository allows the user to see a collection of scripts and notebooks for analyzing chromatin‐fiber dynamics via Langevin simulations, KL‐divergence Markov tests, barrier‐crossing statistics and diffusion metrics.


*Contents*
                                              
mix.ipynb                         | Final combined Jupyter notebook with the full analysis pipeline: time series, 2D distributions, KDE, MSD, PCA.    
taudetection.py                   | Compute KL divergence vs. lag τ for a range of diffusion coefficients and extract the memory time τₘ by two methods: smoothing‐min and small‐relative threshold. 
simulations_with_threshold.py     | Same KL‐divergence analysis but using a fixed threshold (ε=0.03) to define τₘ; plots KL curves and τₘ(D).         
Crossing_Histograms_and_MFPT.py   | Run ensembles of Langevin trajectories, detect forward/backward barrier crossings, plot histograms, and compute MFPT vs. diffusion coefficient. 
alltogether.py                    | Combination of all the MFPT analysis, KL divergence curves, tau automatic detections for a simulation.                   


