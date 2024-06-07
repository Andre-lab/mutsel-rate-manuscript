import numpy as np
import pickle
from simanneal import Annealer

aas = [ "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V" ]

def draw_random(pi):
    r = np.random.uniform()
    cdf = [  np.sum(pi[:i+1]) for i in range(len(pi)) ]
    cdf = np.array(cdf)
    index = np.argmax(cdf>r)
#    print(cdf, r, index)
    return index

def draw_random_n(pi,n):
    draws = []
    for i in range(n):
        draws.append(draw_random(pi))
    return draws

def exponential_probs(lamda, size):
    pi = [ np.exp(-i*lamda) for i in range(size) ]
    pi = pi/sum(pi)
    return pi

def simulate_site(pi, lamda, iterations=1000, total_draws=100):
    num_states = len(pi)
    exp_pi = exponential_probs(lamda,num_states)
    freq_vectors = []
    for i in range(iterations):
        seq_pos = []
        pi_mod = [ pi[i] for i in range(len(pi)) ]
        num_draws = [ int(total_draws*exp_pi[j]) for j in range(len(exp_pi)) ]
        #print("new: ")
        for n in num_draws:
            pi_mod = pi_mod/sum(pi_mod)
            draw = draw_random(pi_mod)
            l_add = [aas[draw]] * n
            seq_pos = seq_pos + l_add
            pi_mod[draw] = 0.0
        freq_vector = [ seq_pos.count(aas[i])/len(seq_pos) for i in range(len(aas)) ]
        freq_vectors.append(freq_vector)
    freq_vectors = np.array(freq_vectors)
    return freq_vectors

def sample_lamda():
    shape, scale = 2.138573, 1/2.708630
    lamda = np.random.default_rng().gamma(shape, scale, 1)
    return lamda

class BinaryAnnealer(Annealer):

    def move(self):
        # choose a random entry
        ensemble_selection = self.state['ensemble_selection']
        ensemble_selection.shape[0]
        i = np.random.randint(0,ensemble_selection.shape[0])
        # change the index
        sampled_vectors = self.state['sampled_vectors']
        j = np.random.randint(0,sampled_vectors.shape[0])
        ensemble_selection[i] = j
        self.state['ensemble_selection'] = ensemble_selection

    def energy(self):
        # evaluate the function to minimize
        sampled_vectors = self.state['sampled_vectors']
        ensemble_selection = self.state['ensemble_selection']
        pi = self.state['pi']
        ensemble = []
        for pos in ensemble_selection:
            ensemble.append(sampled_vectors[pos,:])
        ensemble = np.array(ensemble)
        means = [ ensemble[:,i].mean() for i in range(20) ]
        func = np.sum( ( means - pi )**2 )
        return func

# prior from LG matrix equilibrium frequency.
pi_lg = [0.079066, 0.055941, 0.041977, 0.053052, 0.012937, 0.040767, 0.071586, 0.057337, 0.022355, 0.062157,
                           0.099081, 0.064600, 0.022951, 0.042302, 0.044040, 0.061197, 0.053287, 0.012066, 0.034155, 0.069147]
pi = np.array(pi_lg)

sampled_vectors = []
sampled_lamda = []
for i in range(10000):
    lamda = sample_lamda()
    freq_vectors = simulate_site(pi, lamda)
    rand = np.random.randint(freq_vectors.shape[0])
    freq_vec = freq_vectors[rand]
    sampled_vectors.append(freq_vec)
    sampled_lamda.append(lamda)
sampled_vectors = np.array(sampled_vectors)
sampled_lamda = np.array(sampled_lamda)

pickle.dump( sampled_vectors, open( "results/sampled_vectors/sampled_vectors.10000.p", "wb" ) )
pickle.dump( sampled_lamda, open( "results/sampled_vectors/sampled_lamda.10000.p", "wb" ) )
sampled_vectors = pickle.load( open( "results/sampled_vectors/sampled_vectors.10000.p", "rb" ) )
sampled_lamda = pickle.load( open( "results/sampled_vectors/sampled_lamda.10000.p", "rb" ) )

total_size = sampled_vectors.shape[0]
num_samples = 1000
rints = np.random.randint( total_size, size=num_samples)
params = {'ensemble_selection' : rints, 'pi': pi, 'sampled_vectors': sampled_vectors }

ensemble = []
for pos in rints:
    ensemble.append(sampled_vectors[pos,:])
ensemble = np.array(ensemble)
means_sim_before = [ ensemble[:,i].mean() for i in range(20) ]
func_before = np.sum( ( means_sim_before - pi )**2 )

opt = BinaryAnnealer(params)
auto_schedule = opt.auto(minutes=5)
opt.set_schedule(auto_schedule)
param_out, diff = opt.anneal()
final_ensemble = param_out['ensemble_selection']

ensemble = []
for pos in final_ensemble:
    ensemble.append(sampled_vectors[pos,:])
ensemble = np.array(ensemble)
means_sim = [ ensemble[:,i].mean() for i in range(20) ]
pickle.dump( final_ensemble, open( "results/sampled_vectors/lg.ensemble_ids.1000.p", "wb" ) )
pickle.dump( ensemble, open( "results/sampled_vectors/lg.ensemble.1000.p", "wb" ) )

means = [ sampled_vectors[:,i].mean() for i in range(len(aas)) ]
print("Before opt:")
print(means_sim_before)
print("Func before: ", func_before )
print("After opt:")
print(means_sim)
print("Func after: ", diff )
print("LG       :", pi_lg)
