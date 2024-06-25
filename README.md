# nmf_bioacoustic

Hello!

This is a repository presenting the Nonnegative Matrix Factorization (NMF) method for unsupervised bioacoustic signal processing. This work is introduced at the JJBA conference. You can find the poster here: https://ax-le.github.io/assets/pdf/posters/JJBA_2024_naive.pdf.

This repository focuses on presenting the use of NMF for several bioacoustics tasks, such as:
- Blind Source Separation,
- Unsupervised Estimation of the Number of Sources.

In particular, these tasks have been tested with the same method for both amphibean and whales acoustic signals, highlighting the ability of NMF to be relevant in very different conditions.

For now, this repository is mainly a Proof-Of-Concpet that NMF-like methods can be used. This is intended to be developed, and future work should be carried to consolidate the first conclusions.

This repository does not include the NMF code, which is instead maintained in the github project `nn_fac` of the current corresponding author (https://gitlab.imt-atlantique.fr/a23marmo/nonnegative-factorization/).
## Installation
You should first clone/fork/dowload this repository.

You should then install the requirements using the command
> `pip install -r requirements.txt`

And finally install the code using 
> `pip install -e <path_to_the_code_nmf_bioacoustic>`

This is the recommended way for installing the code. You may use another method, but note that the project was probably neither designed nor tested on a different way.

# Contact

Don't hesitate to contact me at the following mail adress if you have any question: axel.marmoret@imt-atlantique.fr