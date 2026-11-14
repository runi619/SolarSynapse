
import numpy as np
import random
import time

from solar_data import angles, energy


def fitness(angle):
    return np.interp(angle, angles, energy)


def run_ga(
    population_size=20,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.2,
    mutation_sigma=5,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()

    population = [
        random.uniform(0, 90)
        for _ in range(population_size)
    ]

    history = []
    diversity_history = []

    best_angle = None
    best_energy = -np.inf

    for generation in range(generations):

        scores = [fitness(a) for a in population]

        gen_best = np.argmax(scores)

        if scores[gen_best] > best_energy:
            best_energy = scores[gen_best]
            best_angle = population[gen_best]

        history.append(best_energy)
        diversity_history.append(np.std(population))

        new_population = [best_angle]

        def select_parent():
            candidates = random.sample(
                list(zip(population, scores)), 3
            )
            return max(candidates, key=lambda x: x[1])[0]

        while len(new_population) < population_size:

            p1 = select_parent()
            p2 = select_parent()

            if random.random() < crossover_rate:
                alpha = random.random()
                c1 = alpha * p1 + (1-alpha) * p2
                c2 = alpha * p2 + (1-alpha) * p1
            else:
                c1, c2 = p1, p2

            if random.random() < mutation_rate:
                c1 += random.gauss(0, mutation_sigma)

            if random.random() < mutation_rate:
                c2 += random.gauss(0, mutation_sigma)

            c1 = np.clip(c1, 0, 90)
            c2 = np.clip(c2, 0, 90)

            new_population.append(c1)

            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population

    runtime = time.time() - start_time

    return {
        "best_angle": best_angle,
        "best_energy": best_energy,
        "history": history,
        "diversity": diversity_history,
        "runtime": runtime
    }
