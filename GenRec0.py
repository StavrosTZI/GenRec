import numpy as np 
import pandas as pd
import os
import random
from multiprocessing import Pool
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster._spectral import SpectralClustering


RESULTS_FILE = "GenRec0Results.csv"



#In this section a different aproach is used for the entire system using low-rank user embeddings
#lets run them side by side and compare results
RANK = 30  # Hyperparameter (e.g., 10-50)

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





def create_initial_population0(size, n_users, rank=RANK):
    return [np.random.randn(n_users, rank) for _ in range(size)]

def predict_ratings0(U, known_users, unknown_users, R_known):
    # Compute affinity between unknown and known users
    W_subset = U[unknown_users] @ U[known_users].T
    # Normalize rows to sum to 1 (softmax)
    W_subset = np.exp(W_subset) / np.sum(np.exp(W_subset), axis=1, keepdims=True)
    # Predict as weighted average of known ratings
    return W_subset @ R_known


def evaluate_individual0(U, item_subset):
    total_error = 0.0
    count = 0
    for item, (known, unknown, R_known, R_true) in item_subset.items():
        # Skip items with no known/unknown users
        if len(known) == 0 or len(unknown) == 0:
            continue
        # Predict and calculate error
        R_pred = predict_ratings0(U, known, unknown, R_known)
        error = np.mean((R_pred - R_true) ** 2)+0.01 
        total_error += error
        count += 1
    return total_error / count if count > 0 else np.inf

def crossover0(parent1, parent2):
    # Column-wise crossover
    mask = np.random.rand(RANK) < 0.5
    child1 = parent1 * mask + parent2 * (~mask)
    child2 = parent2 * mask + parent1 * (~mask)
    # Add clipping after crossover
    child1 = np.clip(child1, -1e3, 1e3)
    child2 = np.clip(child2, -1e3, 1e3)
    return child1, child2

def mutation0(individual, mutation_rate=0.1, scale=0.1):
    noise = np.random.normal(scale=scale, size=individual.shape)
    mask = np.random.rand(*individual.shape) < mutation_rate
    mutated=individual + mask * noise
    mutated = np.clip(mutated, -1e3, 1e3)  # Constrain to [-1000, 1000]
    return  mutated

def GenRec0(dataset, population_size, generations,
            sample_ratio=0.2, similarity_penalty=0.25,
            elitism=5, lamda_reg=0.01, split_ratio=0.5,
            noise_scale=0.1, mutation_rate=0.1, scale=0.1):
    
    # Precompute data splits
    item_split_dict, n_users = precompute_item_split(split_ratio, dataset)
    all_items = list(item_split_dict.keys())
    
    # Fixed validation subset
    fixed_validation = random.sample(all_items, int(sample_ratio * len(all_items)))
    validation_subset = {k: item_split_dict[k] for k in fixed_validation}
    test_items = set(all_items) - set(fixed_validation)
    test_subset = {k: item_split_dict[k] for k in test_items}

    # Initialize population
    population = create_initial_population0(population_size, n_users)
    graph_data = []
    
    # Evolution loop
    with Pool(processes=os.cpu_count()-1) as pool:
        for gen in range(generations):
            # Evaluate fitness
            args = [(ind, validation_subset) for ind in population]
            fitness_results = pool.starmap(evaluate_individual0, args)
            fitness = {tuple(ind.flatten()): score for ind, score in zip(population, fitness_results)}
            
            # Selection
            sorted_pop = [ind for _, ind in sorted(zip(fitness_results, population), key=lambda x: x[0])]
            
            # New population
            new_pop = sorted_pop[:elitism]  # Keep top elites
            
            # Breed remaining population
            while len(new_pop) < population_size:
                parents = random.choices(sorted_pop[:population_size//2], k=2)
                child1, child2 = crossover0(parents[0], parents[1])
                new_pop.extend([
                    mutation0(child1, mutation_rate, scale),
                    mutation0(child2, mutation_rate, scale)
                ])
            
            population = new_pop[:population_size]
            best_embeding = population[0]# Get the individual with the lowest fitness value
            # Logging
            best_fitness = min(fitness.values())
            avg_fitness = np.mean(list(fitness.values()))
            graph_data.append((gen, avg_fitness))
            print(f"Gen {gen}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
    
    #creating W from the low rank embeding matrix U
    best_embeding = np.nan_to_num(best_embeding, nan=0.0, posinf=1e3, neginf=-1e3)
    best_embeding = np.clip(best_embeding, -1e3, 1e3)

    
    
    uxu=best_embeding@best_embeding.T
    best_individual = np.exp(uxu) / np.sum(np.exp(uxu), axis=1, keepdims=True)
    
    best_individual = np.nan_to_num(best_individual, nan=0.0)
    best_individual = np.clip(best_individual, -1, 1)

    if np.isnan(best_individual).any() or np.isinf(best_individual).any():
        print("WARNING: NaN/Inf detected in best_individual! Replacing with zeros.")
        best_individual = np.nan_to_num(best_individual, nan=0.0, posinf=1.0, neginf=0.0)
        

    #ploting avg fitness
    plt.figure(figsize=(8, 5))
    plt.plot(*zip(*graph_data), marker='o', linestyle='-', color='b', label="Avg Fitness") 
    plt.ylim(0, max([item[1] for item in graph_data]))
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Generation vs Average Fitness")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    folder = "figures"  # Change this to your desired folder
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    file_path = os.path.join(folder, f"Fitness_plot{timestamp}.png")
    plt.savefig(file_path)

    test_score = evaluate_individual0(population[0], test_subset)
    print("Last generation complete, best fitness:{0}avg_fitness:{1}test score:{2}".format(best_fitness,avg_fitness,test_score))
    return best_individual,best_fitness, avg_fitness, test_score

def log_results(params, best_fitness, avg_fitness, test_score, runtime):#function to log results to a csv file
    """Log parameters and results to a DataFrame and save to CSV."""
    # Create a results row
    result_row = {
        **params,
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "test_score": test_score,
        "timestamp": timestamp,
        "runtime":runtime
    }
    
    # Append to file
    df = pd.DataFrame([result_row])
    if not os.path.isfile(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, index=False)
    else:
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)

def cluster_graph(W,threshold=0.2):
    # Approximate kernel embeddings
    W_thresholded = np.where(W > threshold, W, 0)
    G = nx.from_numpy_array(W_thresholded)
    
    # Add node metadata (example: cluster labels)
    clusters = SpectralClustering(n_clusters=5,affinity='precomputed').fit_predict(W_thresholded)  # Optional
    nx.set_node_attributes(G, dict(enumerate(clusters)), 'cluster')
    
    
    folder = "figures"  # Change this to your desired folder
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    file_path = os.path.join(folder, f"GRAPH{timestamp}.gexf")
    
    nx.write_gexf(G,file_path)

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
        unique_items = final_df["item"].unique()
        print(f"Unique items: {unique_items.shape}")
        unique_users= final_df["user"].unique()
        users_that_rated = {item: final_df[final_df["item"] == item]['user'].values for item in unique_items}
        print("Users that rated initialized successfully")
    except Exception as e:
        print(f"Failed to initialize users_that_rated: {e}")
    
    param_combinations =[ {
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },
        {
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.8,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.4,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.1,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.3,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.6,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.001,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.01,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.1,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.06,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":2,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":5,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":8,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":10,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":15,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        },{
            "population_size": 50,
            "generations": 100,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.4,
            "elitism":4,
            "lamda_reg":0.01,
            "split_ratio": 0.8,
            "noise_scale": 0.2,
            "mutation_rate": 0.03,
            "scale":0.1 
        }]
      
    
    
    
    
    for params in param_combinations:
        print(f"Testing parameters: {params}")
        print("Running GenRec0")
        timestamp1=datetime.now()
        timestamp=timestamp1.strftime("%Y-%m-%d %H_%M_%S")
        best_individual,best_fitness, avg_fitness,test_score = GenRec0(final_df, **params)
        cluster_graph(best_individual)

        runtime=datetime.now()-timestamp1
        hours, remainder = divmod(runtime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_runtime = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        log_results(params, best_fitness, avg_fitness,test_score,formatted_runtime)
        


    print("Script ended")