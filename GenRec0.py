import numpy as np 
import pandas as pd
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool
from datetime import datetime
import matplotlib.pyplot as plt

RESULTS_FILE = "results.csv"


#In this section a different aproach is used for the entire system using low-rank user embeddings
#lets run them side by side and compare results
RANK = 50  # Hyperparameter (e.g., 10-50)

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
        error = np.mean((R_pred - R_true) ** 2)
        total_error += error
        count += 1
    return total_error / count if count > 0 else np.inf

def crossover0(parent1, parent2):
    # Column-wise crossover
    mask = np.random.rand(RANK) < 0.5
    child1 = parent1 * mask + parent2 * (~mask)
    child2 = parent2 * mask + parent1 * (~mask)
    return child1, child2

def mutation0(individual, mutation_rate=0.1, scale=0.1):
    noise = np.random.normal(scale=scale, size=individual.shape)
    mask = np.random.rand(*individual.shape) < mutation_rate
    return individual + mask * noise

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
            
            # Logging
            best = min(fitness.values())
            avg = np.mean(list(fitness.values()))
            graph_data.append((gen, avg))
            print(f"Gen {gen}: Best={best:.2f}, Avg={avg:.2f}")
    
    best_fitness = min(fitness.values())
    avg_fitness = np.mean(list(fitness.values()))

    #ploting avg fitness
    plt.figure(figsize=(8, 5))
    plt.plot(*zip(*graph_data), marker='o', linestyle='-', color='b', label="Avg Fitness") 
    plt.ylim(0, max([item[1] for item in graph_data]))
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Generation vs Average Fitness")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    timestamp= datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    folder = "figures"  # Change this to your desired folder
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
    file_path = os.path.join(folder, f"Fitness_plot{timestamp}.png")
    plt.savefig(file_path)

    test_score = evaluate_individual0(population[0], test_subset)
    return best_fitness, avg_fitness, test_score

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
    
    param_combinations =[ {
            "population_size": 100,
            "generations": 20,
            "sample_ratio": 0.6,
            "similarity_penalty": 0.25,
            "elitism":10,
            "lamda_reg":0.01,
            "split_ratio": 0.5,
            "noise_scale": 0.2,
            "mutation_rate": 0.21,
            "scale":0.1 
        }]
      
    
    
    
    
    for params in param_combinations:
        print(f"Testing parameters: {params}")
        print("Running GenRec0")
        best_fitness, avg_fitness,test_score = GenRec0(final_df, **params)
        print("Test score: {0}".format(test_score))
        log_results(params, best_fitness, avg_fitness)
        


    print("Script ended")