import numpy as np


def encode_binary(x, num_bits):
    max_val = 10
    int_val = int((x / max_val) * (2 ** num_bits - 1))
    binary_string = format(int_val, f'0{num_bits}b')
    return binary_string


def decode_binary(binary_string, num_bits):
    max_val = 10
    int_val = int(binary_string, 2)
    x = (int_val / (2 ** num_bits - 1)) * max_val
    return x


def initialize_population(pop_size, num_bits):
    return [np.random.randint(2, size=num_bits) for _ in range(pop_size)]


def fitness_function(individual, num_bits, target_decimal):
    x = decode_binary("".join(map(str, individual)), num_bits)
    return -abs(x - target_decimal)


def select_parents(population, fitnesses, num_parents):
    parents = []
    for _ in range(num_parents):
        parent_idx = np.argmax(fitnesses)
        parents.append(population[parent_idx])
        fitnesses[parent_idx] = -np.inf
    return parents


def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return offspring1, offspring2


def mutate(individual, mutation_rate):
    mutated_individual = np.copy(individual)
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            mutated_individual[i] = 1 - individual[i]
    return mutated_individual


def genetic_algorithm(num_bits, pop_size, num_generations, num_parents, mutation_rate, target_decimal):
    population = initialize_population(pop_size, num_bits)
    best_individual = None
    best_fitness = -np.inf

    for gen in range(num_generations):
        fitnesses = [fitness_function(ind, num_bits, target_decimal) for ind in population]
        parents = select_parents(population, fitnesses, num_parents)

        for i in range(0, len(parents), 2):
            offspring1, offspring2 = crossover(parents[i], parents[i + 1])
            population.extend([mutate(offspring1, mutation_rate), mutate(offspring2, mutation_rate)])

        population.sort(key=lambda ind: fitness_function(ind, num_bits, target_decimal), reverse=True)
        population = population[:pop_size]
        best_current_individual = population[0]
        best_current_fitness = fitness_function(best_current_individual, num_bits, target_decimal)

        if best_current_fitness > best_fitness:
            best_individual = best_current_individual
            best_fitness = best_current_fitness

    return best_individual


num_bits = 8
pop_size = 100
num_generations = 50
num_parents = 10
mutation_rate = 0.1

target_decimal = 6.5
target_binary_string = encode_binary(target_decimal, num_bits)
print(f"Target binary representation of x: {target_binary_string}")

best_individual = genetic_algorithm(num_bits, pop_size, num_generations, num_parents, mutation_rate, target_decimal)

binary_string = "".join(map(str, best_individual))
print(f"Binary representation of x: {binary_string}")

decoded_x = decode_binary(binary_string, num_bits)
print(f"Decoded value of x: {decoded_x}")
