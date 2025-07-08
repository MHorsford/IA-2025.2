import numpy as np
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# Configurações do algoritmo genético
POPULATION_SIZE = 20
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.03
MAX_GENERATIONS = 1000
ELITISM_COUNT = 2  # Top 2 indivíduos preservados por elitismo


def binary_to_state(individual):
    """Converte representação binária para o estado das rainhas."""
    state = []
    for i in range(0, 24, 3):
        # Extrai 3 bits e converte para inteiro (0-7)
        gene = individual[i:i + 3]
        row = gene[0] * 4 + gene[1] * 2 + gene[2] * 1
        state.append(row)
    return np.array(state)


def calculate_collisions(state):
    """Calcula o número de colisões (ataques) entre rainhas."""
    collisions = 0
    n = len(state)

    for i in range(n):
        for j in range(i + 1, n):
            # Verifica mesma linha ou mesma diagonal
            if state[i] == state[j] or abs(i - j) == abs(state[i] - state[j]):
                collisions += 1

    return collisions


def fitness(individual):
    """Calcula o fitness de um indivíduo (minimização de colisões)."""
    state = binary_to_state(individual)
    collisions = calculate_collisions(state)
    return 28 - collisions  # Máximo de pares não-atacantes = 28


def initialize_population():
    """Inicializa a população com indivíduos aleatórios."""
    return [np.random.randint(0, 2, size=24).tolist() for _ in range(POPULATION_SIZE)]


def roulette_selection(population, fitnesses):
    """Seleção de pais usando a estratégia da roleta."""
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choices(population, k=2)

    probabilities = [f / total_fitness for f in fitnesses]
    return random.choices(population, weights=probabilities, k=2)


def crossover(parent1, parent2):
    """Cruzamento com ponto de corte único."""
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    # Escolhe um ponto de corte entre 1 e 22
    cut_point = random.randint(1, 22)
    child1 = parent1[:cut_point] + parent2[cut_point:]
    child2 = parent2[:cut_point] + parent1[cut_point:]
    return child1, child2


def mutate(individual):
    """Aplica mutação por bit flip."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            mutated[i] = 1 - mutated[i]  # Flip do bit
    return mutated


def genetic_algorithm():
    """Executa o algoritmo genético para o problema das 8 rainhas."""
    population = initialize_population()
    best_fitness_history = []
    avg_fitness_history = []
    generations = 0
    start_time = time.time()

    for gen in range(MAX_GENERATIONS):
        generations = gen + 1
        # Avalia a população
        fitnesses = [fitness(ind) for ind in population]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / POPULATION_SIZE

        # Armazena histórico para análise
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)

        # Critério de parada (solução ótima)
        if best_fitness == 28:
            break

        # Seleciona elite
        elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
        elite = [population[i] for i in elite_indices]

        # Gera nova população
        new_population = elite.copy()

        while len(new_population) < POPULATION_SIZE:
            # Seleção de pais
            parent1, parent2 = roulette_selection(population, fitnesses)

            # Cruzamento
            child1, child2 = crossover(parent1, parent2)

            # Mutação
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    # Encontra o melhor indivíduo
    fitnesses = [fitness(ind) for ind in population]
    best_idx = np.argmax(fitnesses)
    best_individual = population[best_idx]
    best_state = binary_to_state(best_individual)
    best_collisions = calculate_collisions(best_state)
    elapsed_time = time.time() - start_time

    return {
        'best_state': best_state,
        'collisions': best_collisions,
        'generations': generations,
        'time': elapsed_time,
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history
    }


def run_ga_experiments(n_runs=50):
    """Executa múltiplas vezes o algoritmo genético e coleta estatísticas."""
    generations_list = []
    times_list = []
    solutions = defaultdict(list)

    for _ in range(n_runs):
        result = genetic_algorithm()
        generations_list.append(result['generations'])
        times_list.append(result['time'])
        solutions[result['collisions']].append(tuple(result['best_state']))

    return generations_list, times_list, solutions


def plot_evolution(result):
    """Plota a evolução do fitness ao longo das gerações."""
    plt.figure(figsize=(10, 6))
    plt.plot(result['best_fitness_history'], label='Melhor Fitness')
    plt.plot(result['avg_fitness_history'], label='Fitness Médio')
    plt.axhline(y=28, color='r', linestyle='--', label='Solução Ótima')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Evolução do Fitness no Algoritmo Genético')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Executa 50 experimentos
    generations_list, times_list, solutions = run_ga_experiments()

    # Calcula estatísticas
    avg_generations = np.mean(generations_list)
    std_generations = np.std(generations_list)
    avg_time = np.mean(times_list)
    std_time = np.std(times_list)

    # Mostra resultados
    print("\nEstatísticas do Algoritmo Genético:")
    print(f"Média de gerações: {avg_generations:.2f} ± {std_generations:.2f}")
    print(f"Média de tempo: {avg_time:.6f}s ± {std_time:.6f}s")

    print("\nTop 5 melhores soluções distintas encontradas:\n")
    distinct_solutions = []
    seen = set()

    # Ordena soluções por número de colisões (melhor = menos colisões)
    for collisions in sorted(solutions.keys()):
        for solution in solutions[collisions]:
            if solution not in seen:
                distinct_solutions.append((solution, collisions))
                seen.add(solution)
            if len(distinct_solutions) >= 5:
                break
        if len(distinct_solutions) >= 5:
            break

    for i, (solution, collisions) in enumerate(distinct_solutions[:5], 1):
        print(f"Solução #{i} (Colisões: {collisions})")
        print("Vetor:", list(solution))
        board = np.zeros((8, 8), dtype=str)
        for col, row in enumerate(solution):
            board[row][col] = 'Q'

        for row in board:
            print(' '.join(['Q' if cell == 'Q' else '.' for cell in row]))
        print()  # Linha em branco entre soluções

    # Total de soluções com 0 colisões
    total_zero_col = len(solutions[0])
    print(f"Total de soluções com 0 colisões encontradas nas 50 execuções: {total_zero_col}")

    # Plota a evolução de uma execução exemplo
    example_run = genetic_algorithm()
    plot_evolution(example_run)


if __name__ == "__main__":
    main()