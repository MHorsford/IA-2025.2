import numpy as np
import random
import time
from collections import defaultdict


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


def stochastic_hill_climbing(max_no_improve=500):
    """Executa o algoritmo Stochastic Hill Climbing para o problema das 8 rainhas."""
    # Estado inicial aleatório
    current_state = np.random.randint(0, 8, size=8)
    current_collisions = calculate_collisions(current_state)
    steps = 0
    iterations_without_improve = 0

    while iterations_without_improve < max_no_improve:
        steps += 1
        better_neighbors = []

        # Gera todos os vizinhos
        for col in range(8):
            for row in range(8):
                if row == current_state[col]:
                    continue  # Ignora a mesma posição atual

                # Cria vizinho
                neighbor = current_state.copy()
                neighbor[col] = row
                neighbor_collisions = calculate_collisions(neighbor)

                # Armazena vizinhos melhores
                if neighbor_collisions < current_collisions:
                    better_neighbors.append((neighbor, neighbor_collisions))

        # Seleciona aleatoriamente um vizinho melhor
        if better_neighbors:
            chosen_neighbor, chosen_collisions = random.choice(better_neighbors)
            current_state = chosen_neighbor
            current_collisions = chosen_collisions
            iterations_without_improve = 0
        else:
            iterations_without_improve += 1

    return current_state, current_collisions, steps


def run_experiments(n_runs=50):
    """Executa múltiplas vezes o algoritmo e coleta estatísticas."""
    steps_list = []
    times_list = []
    solutions = defaultdict(list)

    for _ in range(n_runs):
        start_time = time.time()
        solution, collisions, steps = stochastic_hill_climbing()
        elapsed = time.time() - start_time

        steps_list.append(steps)
        times_list.append(elapsed)
        solutions[collisions].append(tuple(solution))

    return steps_list, times_list, solutions


def main():
    # Executa 50 experimentos
    steps_list, times_list, solutions = run_experiments()

    # Calcula estatísticas
    avg_steps = np.mean(steps_list)
    std_steps = np.std(steps_list)
    avg_time = np.mean(times_list)
    std_time = np.std(times_list)

    # Mostra resultados
    print("\nEstatísticas:")
    print(f"Média de iterações: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Média de tempo: {avg_time:.6f}s ± {std_time:.6f}s")

    # Mostra as 5 melhores soluções distintas
    print("\nTop 5 melhores soluções distintas encontradas:\n")
    distinct_solutions = []
    seen = set()

    # Ordena soluções por número de colisões
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

    # Opcional: mostrar quantas soluções perfeitas (0 colisões) foram encontradas
    total_zero_col = len(solutions[0])
    print(f"Total de soluções com 0 colisões encontradas nas 50 execuções: {total_zero_col}")


if __name__ == "__main__":
    main()