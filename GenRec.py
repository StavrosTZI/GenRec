import numpy as np 
import pandas as pd
import os
import scipy.linalg
import random
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from datetime import datetime


RESULTS_FILE = "ga_experiments.csv"


def quick_norm10(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return np.zeros_like(arr)
    arr = (arr - mean) / std  # Z-score
    arr = np.clip(arr, -3, 3)  # Clip outliers
    arr = 5 * arr + 5  # Scale to 0-10 range
    return np.round(arr)


def predict_ratings1(Wu, Wl, Rmi, Rtrue):
    if np.linalg.matrix_rank(Wu) < Wu.shape[0]:
        Ri_pred = Wu @ Wl @ Rmi
    else:
        try:
            Wu_inv = np.linalg.inv(Wu)
            Ri_pred = Wu_inv @ Wl @ Rmi
        except Exception as e:
            print(f"Error inverting Wu: {e}")
            Ri_pred = Wu @ Wl @ Rmi
    Ri_pred = quick_norm10(Ri_pred)
    #print(Ri_pred)
    Ri_error = 0
    for index, value in enumerate(Ri_pred):
        Ri_error += (value - Rtrue[index])**2 #mean squared error
    avg_error = Ri_error / len(Ri_pred)
    return avg_error
    
def predict_ratings2(Wu, Wl, Rmi, Rtrue,lambda_reg=0.01):
    identity = np.eye(Wu.shape[0])
    try:
        # Regularized inversion: (I - Wu + λI)^(-1)
        Ri_pred = scipy.linalg.solve(identity - Wu + lambda_reg * identity, Wl @ Rmi)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse for rank-deficient matrices
        Ri_pred = np.linalg.pinv(identity - Wu) @ (Wl @ Rmi)
    Ri_pred = quick_norm10(Ri_pred)
    #print(Ri_pred)
    Ri_error = 0
    for index, value in enumerate(Ri_pred):
        Ri_error += (value - Rtrue[index])**2 #mean squared error
    avg_error = Ri_error / len(Ri_pred)
    return avg_error
    
from scipy.sparse.linalg import minres

def predict_ratings3(Wu, Wl, Rmi, Rtrue, lambda_reg=0.01):
    identity = np.eye(Wu.shape[0])
    A = identity - Wu + lambda_reg * identity
    b = Wl @ Rmi
    try:
        Ri_pred = minres(A, b, maxiter=200, rtol=1e-3)[0]
    except:
        Ri_pred = np.linalg.lstsq(A, b, rcond=None)[0]
    Ri_pred = quick_norm10(Ri_pred)
    Ri_error = np.mean((Ri_pred - Rtrue) ** 2)  # Vectorized MSE
    return Ri_error

def precompute_item_split(split_ratio,dataframe):
    final_items = final_df["item"].unique()
    final_items_num = len(final_items)
    sorted_final_items = np.sort(final_items)
    final_items_dict = dict(zip(sorted_final_items,list(range(0,final_items_num))))
    items_group_df=final_df.groupby("item")
    item_users_dict={}
    for item in final_items:
       item_index=final_items_dict[item]
       item_users=set(items_group_df.get_group(item_index)["user"])
       item_users_dict[item_index]=item_users
    
    item_split_dict = {}
    cntr=0
    for item, users in item_users_dict.items():
       users = np.array(list(users))
       np.random.shuffle(users)
       split_idx = int(split_ratio * len(users))  # Fixed 50-50 split
       known,unknown = (users[:split_idx], users[split_idx:])
       itemdf = dataframe[dataframe['item'] == item]
       ratings_that_exist = itemdf[itemdf['user'].isin(known)]['rating'].to_numpy()
       Rtrue = itemdf[itemdf['user'].isin(unknown)]['rating'].to_numpy()
       if len(Rtrue) !=len(unknown) or len(ratings_that_exist) != len(known):
           print("size error while splitting", item, len(Rtrue), len(unknown), len(ratings_that_exist), len(known))
           cntr+=1
           
       item_split_dict[item] = (known,unknown, ratings_that_exist, Rtrue)
    print("number of errors:",cntr)
    #aslo computing numver of unique users for size of W
    final_users = final_df["user"].unique()
    return item_split_dict, final_users.size
    

def evaluate_individual(W,item_subset,lamda_reg=0.01):
    total_error=0.0
    count=0
    error_sum1 = 0
    error_sum2 = 0
    for item,(known,unknown,ratings_that_exist, Rtrue) in item_subset.items():
        W_unknown = W[np.ix_(unknown, unknown)]
        W_known_to_unkonown = W[np.ix_(unknown, known)]
        if ratings_that_exist.size != W_known_to_unkonown.shape[1]:
            print(f"Mismatch: ratings_that_exist size {ratings_that_exist.size}, W_known_to_unknown columns {W_known_to_unkonown.shape[1]}")
            ratings_that_exist = ratings_that_exist[:W_known_to_unkonown.shape[1]]
            #print(ratings_that_exist)
            #print(W_known_to_unkonown)
            #break
            error_sum1+=1
            
        if Rtrue.size != W_unknown.shape[0]:
            print(f"Mismatch: Rtrue size {Rtrue.size}, W_unknown rows {W_unknown.shape[0]}")
            Rtrue = Rtrue[:W_unknown.shape[0]]
            error_sum2+=1
        error = predict_ratings3(W_unknown, W_known_to_unkonown, ratings_that_exist, Rtrue,lambda_reg=lamda_reg)
        total_error += error 
        count+=1
    return total_error/count if count > 0 else np.inf  

#uniform crossover ensuring symmetric offspring
def uniform_crossover(parent1, parent2):
    # Allow asymmetric exploration by not enforcing symmetry during crossover
    mask = np.random.rand(*parent1.shape) < 0.5  # Random mask without symmetry
    offspring1 = parent1 * mask + parent2 * (1 - mask)
    offspring2 = parent2 * mask + parent1 * (1 - mask)
    # Symmetrize offspring after crossover
    offspring1 = (offspring1 + offspring1.T) / 2
    offspring2 = (offspring2 + offspring2.T) / 2
    return offspring1, offspring2

#symmetric mutation of individuals
def mutation1(individual, mutation_rate=0.5, scale=0.1):
    # Generate symmetric noise
    noise = np.random.normal(scale=scale, size=individual.shape)
    noise = (noise + noise.T) / 2  # Enforce symmetry
    # Apply mutation to upper triangle and mirror to lower
    mask = np.triu(np.random.rand(*individual.shape) < mutation_rate, k=0)
    mask = mask | mask.T  # Symmetric mask
    mutated = individual + mask * noise
    # Clip values to maintain valid range (e.g., 0-1)
    mutated = np.clip(mutated, 0.001, 0.999)
    return mutated

def mutation2(individual, mutation_rate=0.1, scale=0.1):  # Reduced from 0.5 to 0.1
    # Asymmetric noise during evolution
    noise = np.random.normal(scale=scale, size=individual.shape)
    mask = np.random.rand(*individual.shape) < mutation_rate
    mutated = individual + mask * noise
    # Symmetrize only after mutation
    mutated = (mutated + mutated.T) / 2  # Ensure symmetry post-mutation
    mutated = np.clip(mutated, 0.001, 0.999)
    return mutated



#simple random and symmetric for use at the start 
def create_initial_pop_1(size,final_users_num):
    population = {}
    for i in range(size):
        # Generate a random matrix and symmetrize it
        random_matrix = np.random.uniform(0.001, 0.999, (final_users_num, final_users_num))
        symmetric_matrix = (random_matrix + random_matrix.T) / 2
        population[i] = symmetric_matrix
    return population

def compute_user_similarity_matrix(final_df):
    # Create user-item rating matrix (rows=users, columns=items)
    user_item_matrix = final_df.pivot_table(
        index="user", columns="item", values="rating"
    ).fillna(0)
    
    # Compute symmetric cosine similarity matrix
    user_similarity = cosine_similarity(user_item_matrix)
    return user_similarity

def create_heuristic_symmetric_population(size, user_similarity, noise_scale=0.2):
    population = []
    for _ in range(size):
        if np.random.rand() < 0.5:  # 50% heuristic, 50% random
            # Heuristic individual
            noise = np.random.normal(scale=noise_scale, size=user_similarity.shape)
            noise = (noise + noise.T) / 2
            heuristic_matrix = user_similarity + noise
        else:
            # Random symmetric individual
            random_matrix = np.random.uniform(0.001, 0.999, user_similarity.shape)
            heuristic_matrix = (random_matrix + random_matrix.T) / 2
        
        heuristic_matrix = np.clip(heuristic_matrix, 0.001, 0.999)
        heuristic_matrix = (heuristic_matrix + heuristic_matrix.T) / 2
        population.append(heuristic_matrix)
    return population


#penalize too similar encouraging divwrsity
def calculate_diversity(population):
    diversity_penalty = {}
    for i, ind1 in enumerate(population):
        similarity = 0
        for ind2 in population:
            if np.array_equal(ind1, ind2):
                continue
            # Compute cosine similarity between flattened matrices
            similarity += np.dot(ind1.flatten(), ind2.flatten()) / (np.linalg.norm(ind1) * np.linalg.norm(ind2))
        diversity_penalty[tuple(ind1.flatten())] = similarity
    return diversity_penalty

#set of different heuristic populations functions
#kernel population
from scipy.spatial.distance import pdist, squareform
from sklearn.gaussian_process.kernels import RBF
def create_kernel_population(size, user_item_matrix):
    # Compute RBF kernel
    kernel = RBF(length_scale=1.0)
    K = kernel(user_item_matrix)
    # Add noise for diversity
    population = []
    for _ in range(size):
        noise = np.random.normal(scale=0.1, size=K.shape)
        heuristic_matrix = K + noise
        heuristic_matrix = (heuristic_matrix + heuristic_matrix.T) / 2  # Enforce symmetry
        population.append(heuristic_matrix)
    return population
#svd population
from scipy.sparse.linalg import svds
def create_svd_population(size, user_item_matrix, rank=10):
    population = []
    # Compute SVD
    U, S, Vt = svds(user_item_matrix, k=rank)
    W_svd = U @ np.diag(S)  # Low-rank user features
    for _ in range(size):
        # Perturb SVD-based matrix
        noise = np.random.normal(scale=0.1, size=(W_svd.shape[0], W_svd.shape[0]))
        heuristic_matrix = W_svd @ W_svd.T + noise  # Symmetric by construction
        heuristic_matrix = np.clip(heuristic_matrix, 0.001, 0.999)
        population.append(heuristic_matrix)
    return population
#nmf population
from sklearn.decomposition import NMF
def create_nmf_population(size, user_item_matrix, rank=10):
    model = NMF(n_components=rank)
    W_nmf = model.fit_transform(user_item_matrix)  # User-factor matrix
    H_nmf = model.components_                     # Factor-item matrix
    return [W_nmf @ W_nmf.T for _ in range(size)]  # Symmetric affinity matrices
#spectral population
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

def create_spectral_population(size, user_item_matrix):
    # Compute affinity matrix using cosine similarity
    affinity = cosine_similarity(user_item_matrix)
    # Compute graph Laplacian
    L = laplacian(affinity, normed=True)
    # Compute spectral embedding
    _, eigenvectors = eigsh(L, k=10, which='SM')  # 10 smallest eigenvectors
    spectral_embedding = eigenvectors[:, 1:]      # Skip first trivial eigenvector
    # Create population
    population = []
    for _ in range(size):
        noise = np.random.normal(loc=0.2, scale=0.1, size=(spectral_embedding.shape[0], spectral_embedding.shape[0]))
        heuristic_matrix = spectral_embedding @ spectral_embedding.T + noise
        heuristic_matrix = np.clip(heuristic_matrix, 0.001, 0.999)
        population.append(heuristic_matrix)
    return population

def create_lowrank_population(size, n_users, rank=10):
    population = []
    for _ in range(size):
        U = np.random.randn(n_users, rank)
        W = U @ U.T  # Symmetric by construction
        W = np.clip(W, 0.001, 0.999)
        population.append(W)
    return population


def create_hybrid_population(size, user_item_matrix):
    size_per_method = size // 6  # Split population evenly across methods
    populations = [
        create_nmf_population(size_per_method, user_item_matrix),
        create_spectral_population(size_per_method, user_item_matrix),
        create_svd_population(size_per_method, user_item_matrix),
        create_kernel_population(size_per_method, user_item_matrix),
        create_heuristic_symmetric_population(size_per_method, user_item_matrix),
        create_lowrank_population(size_per_method, user_item_matrix.shape[0])
    ]
    flat_population = [matrix for sublist in populations for matrix in sublist]
    return flat_population[:size]  # Trim to requested size





def GenRec1(dataset,population_size, generations,sample_ratio=0.2,similarity_penalty=0.25,elitism=5,lamda_reg=0.01,split_ratio=0.5,noise_scale=0.1,mutation_rate=0.1,scale=0.1):
   
    """Genetic algorithm for collaborative filtering with symmetric matrix factorization.Accepts a dataset and parameters for the genetic algorithm. Returns the best individual found by the algorithm."""
    initial_items,final_users_num = precompute_item_split(split_ratio,dataset)
    all_items=list(initial_items.keys())
    user_similarity=compute_user_similarity_matrix(dataset)
    
    # Precompute fixed validation subset once (e.g., 20% of items)
    fixed_validation_size = int(sample_ratio * len(all_items))
    fixed_validation_items = random.sample(all_items, fixed_validation_size)  # Fixed seed for reproducibility
    current_items_subset = {k: initial_items[k] for k in fixed_validation_items}  # Use this for all generations

    populations = {}
    initial_pop=create_hybrid_population(population_size,user_similarity)
    populations["initial"]=initial_pop
    pop = initial_pop
    #initial_gen_items=item_selector(initial_items,10)
    
    with Pool(processes=os.cpu_count() -1) as pool:   
        
        for generation in range(1,generations):
            #randomly select items for evaluation
            #selected_items=random.sample(all_items, int(sample_ratio * len(all_items)))
            #current_items_subset = {k: initial_items[k] for k in selected_items}
       
            args = [(ind, current_items_subset,lamda_reg) for ind in pop]
            fitness={}
            fitness_vector=pool.starmap(evaluate_individual, args)
            fitness = {tuple(ind.flatten()): err for ind, err in zip(pop, fitness_vector)}
            diversity = calculate_diversity(pop)

            # Combine fitness and diversity (weighted sum)
            # Weight for diversity penalty=similarit_penalty
            combined_fitness = {
                k: fitness[k] + similarity_penalty * diversity[k]  # Penalize similarity
                for k in fitness
            }
            
            
            sorted_individuals = sorted(fitness.keys(), key=lambda x: combined_fitness[x])
            #elitism keeping top n individuals
            
            new_pop = [np.array(ind).reshape(final_users_num, final_users_num) 
                    for ind in sorted_individuals[:elitism]]
            while len(new_pop) < population_size:
                parent1, parent2 = random.choices(pop[:population_size], k=2)  # Tournament selection
                offspring1,offspring2 = uniform_crossover(parent1, parent2)
                offspring1 = mutation1(offspring1,mutation_rate,scale)#with parameters
                offspring2 = mutation1(offspring2,mutation_rate,scale)#with parameters
                new_pop.extend((offspring1,offspring2))
            pop = new_pop[:population_size]
            
            populations[generation]=pop
            print(f"Generation {generation} complete,average fitness:,{np.average(np.array(list(fitness.values())))}")
    best_fitness=min(np.array(list(fitness.values())))
    avg_fitness=np.average(np.array(list(fitness.values())))
    print("Last generation complete, best fitness:{0}avg_fitness{1}".format(best_fitness,avg_fitness))
    return best_fitness,avg_fitness

#loaded functions

#item_split=precompute_item_split(0.5,final_df)
#error_vector=evaluate_individual(W,item_split)
#print("Error vector calculated:",error_vector)



def log_results(params, best_fitness, avg_fitness):#function to log results to a csv file
    """Log parameters and results to a DataFrame and save to CSV."""
    # Create a results row
    result_row = {
        **params,
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Append to file
    df = pd.DataFrame([result_row])
    if not os.path.isfile(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, index=False)
    else:
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)


if __name__ == '__main__':
    print("Script started")
    datafolder = "datafiles"
    pickle_file = os.path.normpath(os.path.join(datafolder, "final_df.pkl"))
    
    try:
        final_df = pd.read_pickle(pickle_file)
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()
    

    try:
        W=np.load("datafiles\\W.npy")
        print("W loaded")
    except Exception as e:
        print(f"Error loading W: {e}")

    try:
        unique_items = final_df["item"].unique()
        print(f"Unique items: {unique_items.shape}")
        users_that_rated = {item: final_df[final_df["item"] == item]['user'].values for item in unique_items}
        print("Users that rated initialized successfully")
    except Exception as e:
        print(f"Failed to initialize users_that_rated: {e}")
    
    param_combinations =[ 
        {
            "population_size": 50,
            "generations": 50,
            "sample_ratio": 0.4,
            "similarity_penalty": 0.1,
            "elitism":4,
            "lamda_reg":0.1,
            "split_ratio": 0.8,
            "noise_scale": 0.4,
            "mutation_rate": 0.4,
            "scale":0.1 
        }
    ]
    
    
    
    
    for params in param_combinations:
        print(f"Testing parameters: {params}")
        print("Running GenRec1")
        best_fitness, avg_fitness = GenRec1(final_df, **params)
        log_results(params, best_fitness, avg_fitness)
        


    print("Script ended")


