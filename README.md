## Exploring Reservoir Computing for Brain Function and Dysfunction

This repository contains the code necessary to repeat the experiments in the master's thesis "Exploring Reservoir Computing for Brain Function and Dysfunction". The aim of this thesis is to increase our understanding of plasticity and conduction delays through computational modelling.

To achieve this, we added three mechanisms to the Conn2res toolbox [3]: plasticity, distance-based delays, and artificial lesions. The plasticity mechanism is an adapted version of the work of Falandays <i>et al.</i> (2024) and the distance-based delays were implemented according to the code written by Iacob and Dambre (2024). It is necessary to load the Conn2res toolbox to run this code, Conn2res can be installed following the instructions on [Conn2res GitHub repository](https://github.com/netneurolab/conn2res).

### Tutorial

The file "Plasticity_and_delay_tutorial.ipynb" provides a guide on how to implement the plasticity and delay mechanisms. The workflow remains similar to the original workflow developed by Suárez <i>et al.</i> (2024).

### Structure

This repository is structured around the objectives of the thesis. Each objective corresponds to an experiment. The folders contain all the code necessary to reproduce these experiments and a jupyter notebook that guides you through it. The objectives are:
- Parameter study
- Objective 1: Understanding the effect of plasticity and delays on memory capacity
- Objective 2: Investigating the effect of specific lesions
- Objective 3: Assessing the clinical relevance of the model(s)


### References
[1] Falandays, J.B., Yoshimi, J., Warren, W.H. et al. A potential mechanism for Gibsonian resonance: behavioral entrainment emerges from local homeostasis in an unsupervised reservoir network. Cogn Neurodyn 18, 1811–1834 (2024). https://doi.org/10.1007/s11571-023-09988-2

[2] Iacob S, Dambre J. Exploiting Signal Propagation Delays to Match Task Memory Requirements in Reservoir Computing. Biomimetics (Basel). 2024 Jun 14;9(6):355. doi: 10.3390/biomimetics9060355. PMID: 38921237; PMCID: PMC11201534.

[3] Suárez, L.E., Mihalik, A., Milisav, F. et al. Connectome-based reservoir computing with the conn2res toolbox. Nat Commun 15, 656 (2024). https://doi.org/10.1038/s41467-024-44900-4

[4] Hellyer PJ, Scott G, Shanahan M, Sharp DJ, Leech R. Cognitive Flexibility through Metastable Neural Dynamics Is Disrupted by Damage to the Structural Connectome. J Neurosci. 2015 Jun 17;35(24):9050-63. doi: 10.1523/JNEUROSCI.4648-14.2015. PMID: 26085630; PMCID: PMC4469735.
