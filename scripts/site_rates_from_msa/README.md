# Script for site rate calculation 


estimate_site_rates_musel_freq_clean_codon_freq_from_prop.py: script that calculate rates from a MSA

```console 

python estimate_site_rates_musel_freq_clean_codon_freq_from_prop.py -l SEQ_mafft.fasta -k 1.4 -p 0.2

with -l SEQ_mafft.fasta setting the MSA
-k setting the kappa value
-p setting the rho value

This outputs a file called SEQ_mafft.fasta.site_rates_from_Q_and_pi.dat
```
