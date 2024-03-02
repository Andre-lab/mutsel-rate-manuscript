'''This scripts calculates an Q matrix and site rates from a sequence
    alignment using the mutation-selection model. Amino acid frequencies
    are estimated based on frequencies in the alignment while proposal
    codon frequencies are taken from the DNA model.'''

import argparse
import matplotlib.pyplot as plt

import random
import numpy as np
from Bio import SeqIO

# Read command line arguments
parser = argparse.ArgumentParser(
         formatter_class = argparse.RawDescriptionHelpFormatter,
         description = 'Calculate Qmatrix from MSA')
# Rdd arguments
parser.add_argument('-l', metavar = '<FASTA>', type = str, help = 'fasta sequence')
parser.add_argument('-k', metavar = '<kappa>', type = float, help = 'kappa')
parser.add_argument('-p', metavar = '<p_transition>', type = float, help = 'fraction of whole-codon mutations')
parser.add_argument('-r', metavar = '<reference rates>', type = str, help = 'reference rate data')
args = parser.parse_args()

# Setup default argumetns
if args.l is None:
   parser.print_help()
   parser.error("Name of fasta files not found!")

else:
    filename = args.l

# Setup default argumetns
#if args.o is None:
#   parser.print_help()
#   parser.error("Output name string not found!")

#else:
#    outstring = args.o
reference_rate_file = ""
if args.r is not None:
    reference_rate_file = args.r

if args.k is None:
    kappa = 2.0
else:
   kappa = args.k

if args.p is None:
    p_transition = 0.1
else:
   p_transition = args.p

# Hard-coded variables
neutral_scaling = True
usePiCodonBias=False
use_codon_proposal_freqs = True
alignment_size_cutoff = False
pseudo_count = False
gap_control = False
use_sequence_weights = False
print("usePiCodonBias: ", usePiCodonBias)
print("use_codon_proposal_freqs: ", use_codon_proposal_freqs)
print("alignment_size_cutoff: ", alignment_size_cutoff)
print("pseudo_count: ", pseudo_count)
print("gap_control: ", gap_control)
print("use_sequence_weights: ", use_sequence_weights)

# codon tables and lookup tables

# dictionary that stores codons and their respective amino acids
gencode = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}

gencode_reverse = {
    'I' : { 1: 'ATA', 2: 'ATC', 3: 'ATT'},
    'M' : { 1 : 'ATG'},
    'T' : { 1: 'ACA', 2 : 'ACC', 3 :'ACG', 4: 'ACT'},
    'N' : { 1: 'AAC', 2 : 'AAT'},
    'K' : { 1: 'AAA', 2 : 'AAG'},
    'L' : { 1: 'CTA', 2 : 'CTC', 3 : 'CTG', 4: 'CTT', 5: 'TTA', 6: 'TTG'},
    'P' : { 1: 'CCA', 2 : 'CCC', 3: 'CCG', 4: 'CCT'},
    'H' : { 1: 'CAC', 2 : 'CAT'},
    'Q' : { 1: 'CAA', 2 : 'CAG'},
    'R' : { 1: 'AGG', 2 : 'AGA', 3: 'CGA', 4 : 'CGC', 5: 'CGG', 6: 'CGT'},
    'V' : { 1: 'GTA', 2 : 'GTC', 3: 'GTG', 4: 'GTT'},
    'A' : { 1: 'GCA', 2 : 'GCC', 3: 'GCG', 4: 'GCT'},
    'D' : { 1: 'GAC', 2 : 'GAT'},
    'E' : { 1: 'GAA', 2 : 'GAG'},
    'G' : { 1: 'GGA', 2 : 'GGC', 3: 'GGG', 4: 'GGT'},
    'S' : { 1: 'AGC', 2 : 'AGT', 3: 'TCA', 4 : 'TCC', 5: 'TCG', 6: 'TCT'},
    'F' : { 1: 'TTC', 2 : 'TTT'},
    'Y' : { 1: 'TAC', 2 : 'TAT'},
    'C' : { 1: 'TGC', 2 : 'TGT'},
    'W' : { 1: 'TGG'},
    '_' : { 1: 'TAA', 2: 'TAG'}}

# codon frequencies from E coli. Not used and only here for comparison
codon_frequencies = {
'TTT' : '0.58',
'TTC' : '0.42',
'TTA' : '0.14',
'TTG' : '0.13',
'TAT' : '0.59',
'TAC' : '0.41',
'TAA' : '0.61',
'TAG' : '0.09',
'CTT' : '0.12',
'CTC' : '0.1',
'CTA' : '0.04',
'CTG' : '0.47',
'CAT' : '0.57',
'CAC' : '0.43',
'CAA' : '0.34',
'CAG' : '0.66',
'ATT' : '0.49',
'ATC' : '0.39',
'ATA' : '0.11',
'ATG' : '1',
'AAT' : '0.49',
'AAC' : '0.51',
'AAA' : '0.74',
'AAG' : '0.26',
'GTT' : '0.28',
'GTC' : '0.2',
'GTA' : '0.17',
'GTG' : '0.35',
'GAT' : '0.63',
'GAC' : '0.37',
'GAA' : '0.68',
'GAG' : '0.32',
'TCT' : '0.17',
'TCC' : '0.15',
'TCA' : '0.14',
'TCG' : '0.14',
'TGT' : '0.46',
'TGC' : '0.54',
'TGA' : '0.3 ',
'TGG' : '1',
'CCT' : '0.18',
'CCC' : '0.13',
'CCA' : '0.2',
'CCG' : '0.49',
'CGT' : '0.36',
'CGC' : '0.36',
'CGA' : '0.07',
'CGG' : '0.11',
'ACT' : '0.19',
'ACC' : '0.4',
'ACA' : '0.17',
'ACG' : '0.25',
'AGT' : '0.16',
'AGC' : '0.25',
'AGA' : '0.07',
'AGG' : '0.04',
'GCT' : '0.18',
'GCC' : '0.26',
'GCA' : '0.23',
'GCG' : '0.33',
'GGT' : '0.35',
'GGC' : '0.37',
'GGA' : '0.13',
'GGG' : '0.15',
}

# list of amino acids in specific order
aas = [ "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V" ]
aas_to_pos = {'A' : 0,'R': 1,'N' :2,'D' : 3,'C' : 4,'Q' : 5,'E' : 6, 'G' : 7 ,'H' : 8 ,'I' : 9, 'L' : 10,'K' : 11,'M' : 12, 'F' : 13, 'P' : 14,'S' : 15 ,'T' : 16 ,'W' : 17, 'Y' : 18 ,'V' : 19}

# list of codons
bases = ['T', 'C', 'A', 'G']
codon_labels = [a + b + c for a in bases for b in bases for c in bases]
codon_labels_nostop = [ c for c in codon_labels if gencode[c] != '_']
aas_from_codon_labels_nostop = np.array([ gencode[c] for c in codon_labels_nostop ])

codon2index = {}
codon2indexnostop = {}
for i, c in enumerate(codon_labels):
    codon2index[c] = i
for i, c in enumerate(codon_labels_nostop):
    codon2indexnostop[c] = i


### Iniatialize for Q64 to Q20 conversion
aas_stop = "ARNDCQEGHILKMFPSTWYV*"
aas_nostop = 'ARNDCQEGHILKMFPSTWYV'

amino_acids_codon_no_stop = np.array([x for x in 'FFLLSSSSYYCCWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'])
amino_acids_codon = np.array([x for x in 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'])
stop_idx = np.where(amino_acids_codon == '*')[0]

codon_count_list = np.array([4, 6, 2, 2, 2, 2, 2, 4, 2, 3, 6, 2, 1, 2, 4, 6, 4, 1, 2, 4])

to_c = np.tile(codon_count_list, (20, 1))
from_c = to_c.transpose()
relative_codon_rate = to_c
proposal_rates = codon_count_list / float(np.sum(codon_count_list))

aa2no = {}
no2aa = {}
for i, aa in enumerate(aas_stop):
    aa2no[aa] = i
    no2aa[i] = aa


F20_2_F61_idxs = np.empty((20, 20), dtype=object)

for j, J in enumerate(aas_nostop):
    for i, I in enumerate(aas_nostop):
        if j == i:  # We only considerkn the off diagonal elements...
            continue
        J_codons = np.where(aas_from_codon_labels_nostop == J)[0]
        I_codons = np.where(aas_from_codon_labels_nostop == I)[0]
        idxs_from = []
        idxs_to = []

        # print("correct list")
        for u in I_codons:
            for v in J_codons:
                idxs_from.append(u)
                idxs_to.append(v)
        idxs = (np.array(idxs_from), np.array(idxs_to))
        F20_2_F61_idxs[i][j] = idxs


################################ Read in the LG matrix ##############################################
def read_paml_ratemat_file(rrfile):
    R = np.zeros((20,20))
    pi = np.zeros(20)
    with open(rrfile, 'r') as f_open:
        for line_count,line in enumerate(f_open):
            if line_count < 20:
                data = line.split()
                #print data
                R[line_count,0:line_count] = data
            if line_count == 21:
                pi = [float(x) for x in line.split()]
    for i in range(0,20):
        for j in range(i,20):
            R[i][j] = R[j][i]

    # Diagonalize LG freq_vector
    diag_freq = np.zeros((20, 20))
    d_index = np.diag_indices(20)
    diag_freq[d_index] = pi

    # Find LG Q
    Q = np.dot(R, diag_freq)
    # 1) Set the diagonal, so that the rows sum to 0
    # 2) find the normalizing constant
    norm_constant = 0
    for i in range(0, 20):
        Q[i][i] = -np.sum(Q[i][:])
        norm_constant -= Q[i][i] * pi[i]
    # normalize Q
    Q = Q / norm_constant

    return R, pi, Q

def write_paml_ratemat_file(Rsym, pi, file):
    f = open(file, 'w')
    for i in range(20):
        line = ""
        for j in range(i):
            val = "%.6f" %(Rsym[i][j])
            line = line + val + ' '
        f.write(line + '\n')
        print(line)

    print()
    f.write('\n')
    line = ""
    for p in pi:
        val = "%.6f" %(p)
        line = line + val + ' '
    print(line)
    f.write(line)
    f.close()

###################################
def calculate_neutral_scaling(Q_nt_params):
#    Q = np.zeros((61, 61))
    pi = np.repeat(1. / 61.0, 61)

    Q61_proposal = make_Q_proposal(Q_nt_params)
    Q = calc_instantaneous_rate(Q61_proposal, pi)

    scale_matrix = Q_norm(Q,pi)

    d_index = np.diag_indices(61)
    Q[d_index] = 0.0
    row_sums = np.nansum(Q, axis=1)
    Q[d_index] = -row_sums

    diagonals = np.diagonal(Q)
    scale = -np.sum(diagonals * pi)

    return scale

def Q_norm(Q, pi):
    size = Q.shape[0]

    d_index = np.diag_indices(size)
    Q[d_index] = 0.0
    row_sums = np.nansum(Q, axis=1)
    Q[d_index] = -row_sums

    diagonals = np.diagonal(Q)
    rate = -np.sum(diagonals * pi)

    return Q / rate

def Q_norm_neutral_scaling(Q, pi, scaling):
    size = Q.shape[0]

    d_index = np.diag_indices(size)
    Q[d_index] = 0.0
    row_sums = np.nansum(Q, axis=1)
    Q[d_index] = -row_sums

    return Q / scaling

###################################


def Qnorm_from_Q(Q20, skip_list=None):
    if skip_list is not None:
        delete_list = []
        for aa in skip_list:
            delete_list.append(aa2no[aa])
        Q20 = np.delete(Q20, delete_list, axis=1)
        Q20 = np.delete(Q20, delete_list, axis=0)

    size = Q20.shape[0]
    # Set row sum
    d_index = np.diag_indices(size)
    Q20[d_index] = 0.0
    row_sums = np.nansum(Q20, axis=1)
    Q20[d_index] = -row_sums

    # Find pi and norm the matrix
    pi = np.ones(size).dot(np.linalg.inv(Q20 + np.ones((size, size))))
    relative_rate = -np.sum([Q20[i][i] * pi[i] for i in range(0, size)])
    Q20_normed = Q20 / relative_rate

    R_asym = Q20_normed / pi
    R_sym = (R_asym + R_asym.T) / 2
    R_sym[np.diag_indices(size)] = 0.0

    return Q20_normed, pi, R_sym

###################################

def Q64_2_Q61(Q64):

    # Getting the equilibrium frequencies directly from Q64

    Q61 = np.delete(Q64, stop_idx, axis=1)
    Q61 = np.delete(Q61, stop_idx, axis=0)

    d_index = np.diag_indices(61)
    Q61[d_index] = 0.0
    row_sums = np.nansum(Q61, axis=1)
    Q61[d_index] = -row_sums
    return Q61

def get_F20_pyvolve(Q61, pi_codons):
    F20 = np.zeros((20,20))
    F61 = pi_codons[:, np.newaxis] * Q61
    idx = np.array(12), np.array(44)

    for j, J in enumerate(aas_nostop):
        for i, I in enumerate(aas_nostop):
            if j == i:  # We only consider the off diagonal elements...
                continue
            F20[i][j] = F61[F20_2_F61_idxs_pyvolve[i][j]].sum()

    site_flux = F20.sum()

    d_index = np.diag_indices(20)
    F20[d_index] = 0.0
    row_sums = np.sum(F20, axis=1)
    F20[d_index] = -row_sums

    return F20, site_flux

def get_F20(Q61, pi_codons):
    F20 = np.zeros((20,20))
    F61 = pi_codons[:, np.newaxis] * Q61

    for j, J in enumerate(aas_nostop):
        for i, I in enumerate(aas_nostop):
            if j == i:  # We only consider the off diagonal elements...
                continue
            F20[i][j] = F61[F20_2_F61_idxs[i][j]].sum()

    site_flux = F20.sum()

    d_index = np.diag_indices(20)
    F20[d_index] = 0.0
    row_sums = np.sum(F20, axis=1)
    F20[d_index] = -row_sums

    return F20, site_flux

def p61_2_p20(p61):
    p20 = np.zeros((20))
    for i, aa in enumerate(aas_nostop):
        idxs = np.where(aas_from_codon_labels_nostop == aa)
        for idx in idxs[0]:
            p20[i] += p61[idx]
    return p20

def Q2R_pi(Q20):
    # Set row sum
    d_index = np.diag_indices(20)
    Q20[d_index] = 0.0
    row_sums = np.nansum(Q20, axis=1)
    Q20[d_index] = -row_sums

    # Find pi and norm the matrix
    pi = np.ones(20).dot(np.linalg.inv(Q20 + np.ones((20, 20))))
    relative_rate = -np.sum([Q20[i][i] * pi[i] for i in range(0, 20)])
    Q20_normed = Q20 / relative_rate

    R_asym = Q20_normed / pi
    R_sym = (R_asym + R_asym.T) / 2
    return R_sym, pi

def make_Q_proposal(Q_nt_params, usePiCodonBias=False):
    bases = ['T', 'C', 'A', 'G']
    probability_of_mutation = Q_nt_params['p_transition'] # 0.001
    codons = [a + b + c for a in bases for b in bases for c in bases]
    codon_connect = np.zeros((64, 64))

    for i, from_codon in enumerate(codons):
        for j, to_codon in enumerate(codons):
            codon_distance = hamm_dist(from_codon, to_codon)
            if codon_distance == 1:
                if 'K80' in Q_nt_params:
                    codon_connect[i][j] = probability_of_mutation + K80_rate(from_codon, to_codon, Q_nt_params)
                elif 'GTR' in Q_nt_params:
                    codon_connect[i][j] = probability_of_mutation + GTR_rate(from_codon, to_codon, Q_nt_params)
                elif 'foster' in Q_nt_params:
                    codon_connect[i][j] = foster_rate(from_codon, to_codon)
            elif codon_distance == 2:
                 codon_connect[i][j] = probability_of_mutation #* K80_rate(from_codon, to_codon, Q_nt_params)
            elif codon_distance == 3:
                 codon_connect[i][j] = probability_of_mutation #* probability_of_mutation #* K80_rate(from_codon, to_codon, Q_nt_params)

    d_index = np.diag_indices(64)
    codon_connect[d_index] = 0.0
    row_sums = np.nansum(codon_connect, axis=1)
    codon_connect[d_index] = -row_sums

    # The effective number of codons might not be the same as the actual number of codons
    recap_codon_usage_bias = usePiCodonBias
    if recap_codon_usage_bias and 'K80' in Q_nt_params:
        Q20_normed, pi, R_sym = Qnorm_from_Q(codon_connect)

        pi64_effective_count_biased = find_pi_codon_usage_bias()

        diag_freq = np.zeros((64, 64))
        d_index = np.diag_indices(64)
        diag_freq[d_index] = pi64_effective_count_biased
        Q = np.dot(R_sym, diag_freq)
        codon_connect, pi, R_sym = Qnorm_from_Q(Q)

    codon_connect = np.delete(codon_connect, stop_idx, axis=1)
    codon_connect = np.delete(codon_connect, stop_idx, axis=0)

    return codon_connect

def calc_instantaneous_rate(Q61_proposal, codon_pi):

    fixation_rate = np.zeros((61,61))

    for i, from_codon in enumerate(codon_labels_nostop):
        for j, to_codon in enumerate(codon_labels_nostop):
            index = codon2indexnostop[from_codon]
            pi_i = codon_pi[index]
            index = codon2indexnostop[to_codon]
            pi_j = codon_pi[index]

            mu_ij = Q61_proposal[i][j]
            mu_ji = Q61_proposal[j][i]

            if from_codon is to_codon:
                mu_ij = 0.
                mu_ji = 0.

            # If either frequency is equal to 0, then the rate is 0.
            if abs(pi_i) <= 0. or abs(pi_j) <= 0. or abs(mu_ij) <= 0. or abs(mu_ji) <= 0.:
                fixation_rate[i][j] = 0.

            # Otherwise, compute scaled selection coefficient as np.log( pi_mu ) = np.log( (mu_ji*pi_j)/(mu_ij*pi_i) )
            else:
                pi_mu = (mu_ji * pi_j) / (mu_ij * pi_i)

                # If pi_mu == 1, L'Hopitals gives fixation rate of 1 (substitution probability is the forward mutation rate)
                if abs(1. - pi_mu) <= 0.:
                    fixation_rate[i][j] = 1.
                    fixation_rate[i][j] = fixation_rate[i][j] * mu_ij
                else:
                    fixation_rate[i][j] = np.log(pi_mu) / (1. - 1. / pi_mu)
                    fixation_rate[i][j] = fixation_rate[i][j] * mu_ij
#
    return fixation_rate


def Q61_from_codon_frequencies(codon_pi, Q_nt_params, usePiCodonBias=False):

    Q61_proposal = make_Q_proposal(Q_nt_params, usePiCodonBias=usePiCodonBias)
    fixp_M61 = calc_instantaneous_rate(Q61_proposal, codon_pi)
    if neutral_scaling:
        scaling = calculate_neutral_scaling(Q_nt_params)
        fixp_M61_norm = Q_norm_neutral_scaling(fixp_M61,codon_pi, scaling)
    else:
         fixp_M61_norm = Q_norm(fixp_M61,codon_pi)

    return fixp_M61_norm

def flux2Q(M_pi, Fs, rates, Q_nt_params, outstr, cutoff=1e-300):
    pi_mean_unfiltered = np.nansum(M_pi, axis=0)
    pi_mean_unfiltered = pi_mean_unfiltered / np.nansum(pi_mean_unfiltered)
    nans = []
    n_rates = 0
    n_no_rate = 0
    # D2Vs = []
    pi_mean = np.zeros(20)
    F_distribution = np.zeros((len(Fs), 20, 20))
    for i, F in enumerate(Fs):
        if rates[i] > cutoff:
            n_rates += 1
            F_normed = F / rates[i]
            F_distribution[i, :, :] = F_normed
            pi_mean += M_pi[i]
        else:
            n_no_rate += 1
            nans.append(i)
    pi_mean = pi_mean / np.sum(pi_mean)

#    make_pi_correlation_plot(pi_mean, pi_mean_unfiltered, 'effect_of_invariant_on_pi_' + outstr + '.png',
#                             xlabel='pi_unfiltered', ylabel='pi_filtered', make_plot=False)

    Q_mean = np.nansum(F_distribution, axis=0) / pi_mean[:, np.newaxis ]
    if neutral_scaling:
        scaling = calculate_neutral_scaling(Q_nt_params)
        Q_mean_normed = Q_norm_neutral_scaling(Q_mean, pi_mean, scaling)
    else:
        Q_mean_normed = Q_norm(Q_mean, pi_mean)

    return pi_mean_unfiltered, pi_mean, Q_mean_normed, F_distribution, n_no_rate, n_rates
#########################################################
# Plotting functions
#########################################################

def make_correlation_plot(M_y, M_x, outname, standard_error = None, xlabel='Mut-Sel', ylabel='Nature (LG)', codon_colors=False, make_plot=True, set_lim=False, skip_list=[], half_plot=True, is_print=False, show_pcc=True, is_cys_none_colored=False, annotate=None):
    # Plot correlation between Rosetta rates and experimental rates
    X = []
    Y = []
    labels = []
    colors = []
    std_err = []
    for i in range(0, 20):
        k = i if half_plot else 0
        for j in range(k, 20):
            from_aa = no2aa[i]
            to_aa = no2aa[j]
            if from_aa in skip_list or to_aa in skip_list:
                continue
            if i == j:
                continue
            if M_x[i][j] < 1e-80:
                continue
            if standard_error is not None:
                if standard_error[i][j]/M_x[i][j] > 10:
                    continue
                #print(from_aa, to_aa, standard_error[i][j], M_x[i][j], M_y[i][j])
                std_err.append(standard_error[i][j])

            X.append(M_x[i][j])
            Y.append(M_y[i][j])
            labels.append(from_aa + "2" + to_aa)
            d = shortest_hamm_dist(aa2codons(from_aa), aa2codons(to_aa))
            if codon_colors:
                colors.append(d)
            else:
                if is_cys_none_colored and (from_aa == 'C' or to_aa == 'C'):
                    colors.append('None')
                else:
                    colors.append('black')
    PCC = np.corrcoef(np.log(X), np.log(Y))

    if make_plot:
        fig, ax = plt.subplots()
        if show_pcc:
            ax.set_title("pcc =" + str(round(PCC[0][1],2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if set_lim:
            ax.set_ylim([0.001,100])
            ax.set_xlim([0.001,100])
        if standard_error is None:
            if codon_colors:
                scat = ax.scatter(X, Y, c=colors, cmap='viridis', s=40, lw = 0)
                fig.colorbar(scat)
            else:
                scat = ax.scatter(X, Y, s=40, lw = 1, c=colors)
        else:
            scat = ax.scatter(X, Y, s=10, lw=1, c=colors)
            scat = ax.errorbar(X, Y, std_err, linestyle='None')
        for i, txt in enumerate(labels):
            if is_print:
                print(txt, X[i], Y[i])
            if annotate is not None and txt in annotate:
                ax.annotate(txt, (X[i]*1.05, Y[i]*1.05), fontsize=15, zorder=10)
            elif annotate is None:
                ax.annotate(txt, (X[i]*1.05, Y[i]*1.05), fontsize=15, zorder=10)
        fig.subplots_adjust(bottom=0.2)
        fig.subplots_adjust(left=0.2)
        ax.plot([0.001, 100], [0.001, 100], ls="--", c="black")

        plt.savefig('results/figs_LG_comparison/' + outname, dpi=300, transparent=False)

        plt.clf()
    return PCC[0][1]

def codon2relpi(codon):
    # https://openwetware.org/wiki/Escherichia_coli/Codon_usage
    d = {'GGG': 0.15, 'GGA': 0.11, 'GGT': 0.34, 'GGC': 0.4, 'GAG': 0.31, 'GAA': 0.69, 'GAT': 0.63, 'GAC': 0.37, 'GTG': 0.37,
     'GTA': 0.15, 'GTT': 0.26, 'GTC': 0.22, 'GCG': 0.36, 'GCA': 0.21, 'GCT': 0.16, 'GCC': 0.27, 'AGG': 0.02,
     'AGA': 0.04, 'CGG': 0.1, 'CGA': 0.06, 'CGT': 0.38, 'CGC': 0.4, 'AAG': 0.23, 'AAA': 0.77, 'AAT': 0.45, 'AAC': 0.55,
     'ATG': 1, 'ATA': 0.07, 'ATT': 0.51, 'ATC': 0.42, 'ACG': 0.27, 'ACA': 0.13, 'ACT': 0.17, 'ACC': 0.44, 'TGG': 1,
     'TGT': 0.45, 'TGC': 0.55, 'TAG': 0.07, 'TAA': 0.64, 'TGA': 0.29, 'TAT': 0.57, 'TAC': 0.43, 'TTT': 0.57,
     'TTC': 0.43, 'AGT': 0.15, 'AGC': 0.28, 'TCG': 0.15, 'TCA': 0.12, 'TCT': 0.15, 'TCC': 0.15, 'CAG': 0.65,
     'CAA': 0.35, 'CAT': 0.57, 'CAC': 0.43, 'TTG': 0.13, 'TTA': 0.13, 'CTG': 0.5, 'CTA': 0.04, 'CTT': 0.1, 'CTC': 0.1,
     'CCG': 0.52, 'CCA': 0.19, 'CCT': 0.16, 'CCC': 0.12}
    return d[codon]

def find_effective_frequency():
    effective_count = []
    for aa in aas_stop:
        codons_aa = aa2codons(aa)
        H = 0
        for codon in codons_aa:
            p = codon2relpi(codon)
            H += p * np.log(p)
        effective_count.append(np.exp(-H))
    return effective_count/np.sum(effective_count)

def find_pi_codon_usage_bias():
    effective_n_codons = np.array(find_effective_frequency())
    actual_n_codons = np.array([4, 6, 2, 2, 2, 2, 2, 4, 2, 3, 6, 2, 1, 2, 4, 6, 4, 1, 2, 4, 3])
    actual_n_codons = actual_n_codons/float(actual_n_codons.sum())
    overrepresentation = actual_n_codons / effective_n_codons

    pi64 = []
    bases = ['T', 'C', 'A', 'G']
    codons = [a + b + c for a in bases for b in bases for c in bases]
    amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
    for aa, codon in zip(amino_acids, codons):
        pi64.append(overrepresentation[aa2no[aa]])
    pi64 = np.array(pi64) / np.sum(pi64)

    return pi64

def hamm_dist(codon1, codon2):
    dist = 0
    for nt1, nt2 in zip(codon1, codon2):
        if nt1 != nt2:
            dist += 1
    return dist

def shortest_hamm_dist(codon_list1, codon_list2):
    dist_min = 3
    for c1 in codon_list1:
        for c2 in codon_list2:
            dist = 0
            for nt1, nt2 in zip(c1, c2):
                if nt1 != nt2:
                    dist += 1
            if dist < dist_min:
                dist_min = dist
    return dist_min

def K80_rate(c1, c2, Q_nt_params):
    k = Q_nt_params['K80']
    nt2no = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    #             A  G  C  T
    Q = np.array([[0, k, 1, 1],
                 [k, 0, 1, 1],
                 [1, 1, 0 ,k],
                 [1, 1, k, 0]])
    product_rate = 1
    for nt1, nt2 in zip(c1, c2):
        if nt1 != nt2:
            product_rate *= Q[nt2no[nt1]][nt2no[nt2]]

    return product_rate

def foster_rate(c1, c2):
    nt2no = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    #             A  G  C  T
    Q = np.array([[0, 49, 38, 17],
                  [82, 0, 17, 30],
                  [30, 17, 0, 82],
                  [17, 38, 49, 0]])
    if hamm_dist(c1, c2) == 1:
        for nt1, nt2 in zip(c1, c2):
            if nt1 != nt2:
                return Q[nt2no[nt1]][nt2no[nt2]]
    else:
        raise( "This function is not meant for more than a single nt difference")

def GTR_rate(c1, c2, Q_nt_params):
    pi_A = Q_nt_params['GTR'][0] / np.sum(Q_nt_params['GTR'][0:4])
    pi_G = Q_nt_params['GTR'][1] / np.sum(Q_nt_params['GTR'][0:4])
    pi_C = Q_nt_params['GTR'][2] / np.sum(Q_nt_params['GTR'][0:4])
    pi_T = Q_nt_params['GTR'][3] / np.sum(Q_nt_params['GTR'][0:4])
    a = Q_nt_params['GTR'][4]
    b = Q_nt_params['GTR'][5]
    c = Q_nt_params['GTR'][6]
    d = Q_nt_params['GTR'][7]
    e = Q_nt_params['GTR'][8]
    f = Q_nt_params['GTR'][9]

    nt2no = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    #             A  T  C  G
    Q = np.array([[0,      a*pi_G, b*pi_C, c*pi_T],
                 [ a*pi_A, 0,      d*pi_C, e*pi_T],
                 [ b*pi_A, d*pi_G, 0 ,     f*pi_T],
                 [ c*pi_A, e*pi_G, f*pi_C, 0]])
    if hamm_dist(c1, c2) == 1:
        for nt1, nt2 in zip(c1, c2):
            if nt1 != nt2:
                return Q[nt2no[nt1]][nt2no[nt2]]
    else:
        raise( "This function is not meant for more than a single nt difference")

def relative_codon_frequencies_from_proposal_model(Q_nt_params):
    Q61_prop = make_Q_proposal(Q_nt_params)
    pi_codon_proposal = np.ones(61).dot(np.linalg.inv(Q61_prop + np.ones((61, 61))))
    pi_sum_aa = p61_2_p20(pi_codon_proposal)

    pi_codon_proposal_relative = np.zeros(61)
    for c in range(61):
        codon = codon_labels_nostop[c]
        aa = gencode[codon]
        aa_pos = aas_to_pos[aa]
        pi_codon_proposal_relative[c] = pi_codon_proposal[c] / pi_sum_aa[aa_pos]

    return pi_codon_proposal_relative

def aa2codons(target_aa):
    bases = ['T', 'C', 'A', 'G']
    codons = [a + b + c for a in bases for b in bases for c in bases]
    amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
    codons_of_aa = []
    for aa, codon in zip(amino_acids, codons):
        if aa == target_aa:
            codons_of_aa.append(codon)
    return codons_of_aa

def read_bm(fasta):
    bm_file = fasta + '.bm'
    bm_dict = {}
    sum_w = 0
    with open(bm_file, 'r') as f_open:
        for line_count,line in enumerate(f_open):
            ls = line.split()
            sequence_name = ls[0]
            weight = ls[1]
            bm_dict[sequence_name] = float(weight)
            sum_w += float(weight)
    return bm_dict


def rate_from_alignment(records, outstr=""):
    F_vec = []
    pi_vec = []
    aln_length = len(records[0].seq)  # Assuming the sequence has the same lenght...Not always true perhaps
    aa_sequence = []
    # Sum the counts of the different amino acids in each column
    counts = []
    freq_codons = []
    freq_gaps = []
    pos = -1
    for aln_pos in range(aln_length):
        aa_pos = []
        aa_counts = [0.0] * 20
        if records[0].seq[aln_pos] == '-':
            #            print("Skipping :", aln_pos+1, records[0].seq[aln_pos])
            continue
        else:
            pos = pos + 1
            aa_sequence.append(records[0].seq[aln_pos])
        # Count the total number of non-gapped items
        non_gaps_items = 0
        sum_w = 0
        for rec in records:
            aa_pos.append(rec.seq[aln_pos])
            aa_i = rec.seq[aln_pos]
            if aa_i not in aas:
                continue
            non_gaps_items += 1
            apos = aas_to_pos[aa_i]
            w = 1.0
            if use_sequence_weights:
                w = sequence_weights[rec.name]
                sum_w += w
            aa_counts[apos] += w
        if use_sequence_weights:
            # We want the same number of counts after recounting. We also have to consider
            # that gaps can make the total weights not sum to 1.0.
            aa_counts = [c * non_gaps_items / sum_w for c in aa_counts]
        else:
            aa_counts = [aa_pos.count(a) for a in aas]
        counts.append(aa_counts)
        if sum(aa_counts) == 0:
            print("No amino acids at this position...Exiting", aln_pos, pos, aa_counts)
            exit()
        if pseudo_count:
            aa_counts = [c + 0.5 for c in aa_counts]
        freq_est = np.array(aa_counts) / sum(aa_counts)
        freq_codons.append(freq_est)
        gap_count = aa_pos.count('-')
        freq_gaps.append(gap_count / (gap_count + sum(aa_counts)))

    site_rates_no_internal = []
    pos_array = []
    protein_length = len(freq_codons)

    for pos in range(protein_length):
        codon_pi = []
        for c in range(61):
            codon = codon_labels_nostop[c]
            aa = gencode[codon]
            aa_pos = aas_to_pos[aa]
            pi_aa = freq_codons[pos][aa_pos]
            if use_codon_proposal_freqs:
                pi_codon = pi_aa * pi_codon_proposal_relative[c]
            else:
                pi_codon = pi_aa * float(codon_frequencies[codon])
            codon_pi.append(pi_codon)

        codon_pi = np.array(codon_pi)
        rate_matrix_new = Q61_from_codon_frequencies(codon_pi, Q_nt_params, usePiCodonBias)

        F, rate = get_F20(rate_matrix_new, codon_pi)
        site_rates_no_internal.append(rate)
        pos_array.append(pos+1)
        if gap_control and freq_gaps[pos] > 0.5:
            continue
        F_vec.append(F)
        pi_vec.append(p61_2_p20(codon_pi))

    np.savetxt(outstr + filename[:-5] + "site_rates_from_Q_and_pi.dat",
               list(zip(pos_array,aa_sequence, site_rates_no_internal, freq_gaps)), header="POS,AA,rate,%gap", fmt="%s")

    return site_rates_no_internal

def compare_to_reference(site_rates_no_internal, rates_ref, outstr=""):

    rates_ref = np.array(rates_ref) / sum(rates_ref)
    z = np.polyfit(site_rates_no_internal, rates_ref, 1)
    p = np.poly1d(z)
    r = np.corrcoef(site_rates_no_internal, rates_ref)

    #fig, ax = plt.subplots()
    #ax.scatter(site_rates_no_internal, rates_ref)
    #ax.plot(site_rates_no_internal, p(site_rates_no_internal), label='correlation='"%3.2f,%s" % (r[0][1], outstr))
    #ax.set_xlabel('Predicted rate')
    #ax.set_ylabel('True rate')
    #ax.legend()
    #ax.set_title('Predicted mutsel rates vs true rates')
    ##plt.show()
    ##plt.savefig(outstr+ "true_rates_vs_freq_rates.png")
    #plt.close()
    return r[0][1]

### Start calculations

Q_nt_params = {'K80': kappa, 'p_transition': p_transition}

if use_codon_proposal_freqs:
    pi_codon_proposal_relative = relative_codon_frequencies_from_proposal_model(Q_nt_params)

# Read fasta sequence alignment
print("Processing: ", filename)
records = list(SeqIO.parse(filename, "fasta"))
num_records = len(records)

if alignment_size_cutoff and num_records < 30:
    print("To few sequences in alignment:", num_records)
    exit()

# Read the frequency sequence weights
sequence_weights = {}
if use_sequence_weights:
    sequence_weights = read_bm(filename)
    if len(sequence_weights) == 0:
        print("No weights for this alignment...Skipping")
        exit()

rates_ref = []
if reference_rate_file != "":
    file_reference = open(reference_rate_file)
    rate_data_ref = file_reference.read().splitlines()

    rates_ref = []
    for r in rate_data_ref:
        rates_ref.append(float(r))

random_selection = False

if random_selection:
    repeats = 5
    r_vector = []
    selection_vector = []
    for selection_num in range(10, num_records, 10):
        r_repeat_vector = []
        for repeat in range(repeats):
            random_selection = random.sample(records, selection_num)
            outstr = str(selection_num) + '.' + str(repeat) + '.'
            site_rates_no_internal = rate_from_alignment(random_selection,outstr)
            r = compare_to_reference(site_rates_no_internal, rates_ref, outstr)
            r_repeat_vector.append(r)
            SeqIO.write(random_selection,outstr + "random_selection.fasta", "fasta")
        r_vector.append(r_repeat_vector)
        selection_vector.append(selection_num)
        print(selection_num, np.mean(r_repeat_vector))
    yvals = [np.mean(v) for v in r_vector ]

    fig, ax = plt.subplots()
    ax.scatter(selection_vector, yvals)
    ax.set_xlabel('Size of aligment')
    ax.set_ylabel('Pearson correlation to true rates')
    # plt.show()
    plt.savefig("Correlation_vs_alignment_lenght.png")

else:
    site_rates_no_internal = rate_from_alignment(records)
    if reference_rate_file != "":
       compare_to_reference(site_rates_no_internal, rates_ref)
