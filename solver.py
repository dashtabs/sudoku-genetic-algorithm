import random
from collections import Counter
import math
import itertools



class SudokuSolver:
    def __init__(self, model, population_size, selection_rate, rand_selection_rate, n_children, restart_after_gen,
                 max_fitness, mutation, fitness_function):
        self.model = model
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.rand_selection_rate = rand_selection_rate
        self.n_children = n_children
        self.restart_after_gen = restart_after_gen
        self.current_generation = 0
        self.best_fitness = 0
        self.generations_without_improvement = 0
        self.max_fitness = max_fitness
        self.mutation = mutation
        self.fitness_function = fitness_function
        self.not_null_indices = self.get_not_null_indices()

    # Save the indexes of the filled numbers at the beginning of the game
    def get_not_null_indices(self):
        return [(i, j) for i, row in enumerate(self.model) for j, num in enumerate(row) if num != 0]

    # Intelligent fill-in: each row has unique numbers
    def generate_individual(self):
        individual = [row[:] for row in self.model]
        fill_in = set(range(1, 10))

        for row in individual:
            # Remove numbers already present in the row
            missing_numbers = list(fill_in - set(row))
            random.shuffle(missing_numbers)  # Shuffle the list once

            # Fill in the blanks in the row
            for i, value in enumerate(row):
                if value == 0:
                    row[i] = missing_numbers.pop()  # Pop the last element
        return individual

    def generate_initial_population(self):
        # Generate an initial population of individuals
        return [self.generate_individual() for _ in range(self.population_size)]

    # Use to evaluate the number of duplicates in grids
    def get_3x3_grids(self, matrix):
        return [
            [matrix[i][j] for i in range(row_start, row_start + 3) for j in range(col_start, col_start + 3)]
            for row_start in range(0, 9, 3)
            for col_start in range(0, 9, 3)
        ]

    # A primarily used fitness function based on the number of duplicates in columns and grids
    def fitness(self, individual):
        total_duplicates_count = 0

        # Directly iterate over rows, columns, and 3x3 grids
        for group in itertools.chain(individual, zip(*individual), self.get_3x3_grids(individual)):
            total_duplicates_count += sum(v - 1 for v in Counter(group).values() if v > 1)

        return self.max_fitness - total_duplicates_count

    # A worse-performing fitness function based on the amount of correctly constructed columns and grids
    # Change max_fitness to 18 for use
    def other_fitness(self, individual):
        transposed_individual = list(zip(*individual))

        # Count correct columns
        correct_columns = sum(
            all(Counter(column)[num] == 1 for num in range(1, 10)) for column in transposed_individual)

        # Get 3x3 grids from the 9x9 matrix
        grids_3x3 = self.get_3x3_grids(individual)

        # Count correct grids
        correct_grids = sum(all(Counter(grid)[num] == 1 for num in range(1, 10)) for grid in grids_3x3)

        total_correct = correct_columns + correct_grids
        return min(self.max_fitness, total_correct)

    # Choose rows randomly from parents
    @staticmethod
    def crossover(parent1, parent2):
        return [parent1[i][:] if random.random() < 0.5 else parent2[i][:] for i in range(9)]

    # This is ordinary mutation. Use for comparison with better ones
    def other_mutate(self, individual):
        if random.random() < 0.5:
            individual = self.generate_individual()
        else:
            for _ in range(random.randint(1, 4)):  # Multiple mutations per individual
                row = random.randint(0, 8)

                # Identify changeable (mutable) indices in the selected row
                changeable_indices = [col for col in range(9) if (row, col) not in self.not_null_indices]

                # Ensure there are at least two changeable indices to perform a swap
                if len(changeable_indices) < 2:
                    continue  # If not, skip to the next iteration

                # Randomly select two distinct indices from the changeable ones
                col1, col2 = random.sample(changeable_indices, 2)

                # Swap the values at these indices
                individual[row][col1], individual[row][col2] = individual[row][col2], individual[row][col1]
        return individual

    def dynamic_mutation_rate(self):
        # Adjust mutation rate based on generations
        return max(0.1, 1.0 - self.best_fitness / self.max_fitness)

    # Modify mutate method to use dynamic mutation rate (good-performing!)
    def mutate(self, individual):
        mutation_rate = self.dynamic_mutation_rate()

        if random.random() < mutation_rate or self.fitness(individual) - self.best_fitness == 0:
            individual = self.generate_individual()
        else:
            for _ in range(random.randint(1, 4)):
                row = random.randint(0, 8)
                changeable_indices = [col for col in range(9) if (row, col) not in self.not_null_indices]

                # swap several numbers in a row
                for _ in range(random.randint(1, 3)):
                    col1, col2 = random.sample(changeable_indices, 2)
                    individual[row][col1], individual[row][col2] = individual[row][col2], individual[row][col1]

        return individual

    def select_parents(self, population, fitness_scores):
        selected_parents = []

        # Select the best individuals based on selection_rate
        num_best_individuals = int(self.selection_rate * self.population_size)
        best_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[
                       :num_best_individuals]
        best_parents = [population[i] for i in best_indices]
        selected_parents.extend(best_parents)

        # Exclude best individuals when selecting random individuals
        remaining_population = [ind for idx, ind in enumerate(population) if idx not in best_indices]

        num_random_individuals = int(self.rand_selection_rate * self.population_size)
        random_parents = random.sample(remaining_population, num_random_individuals)
        selected_parents.extend(random_parents)

        return selected_parents

    #  Normal genetic algorithm
    def genetic_algorithm(self):
        population = self.generate_initial_population()

        best_solution = None
        for generation in range(self.restart_after_gen):
            if self.fitness_function == 0:
                fitness_scores = [self.fitness(ind) for ind in population]
            else:
                fitness_scores = [self.other_fitness(ind) for ind in population]

            best_index = max(range(self.population_size), key=lambda i: fitness_scores[i])
            best_individual = population[best_index]
            best_fitness = fitness_scores[best_index]

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.generations_without_improvement = 0
                best_solution = best_individual
            else:
                self.generations_without_improvement += 1

            # Select parents based on fitness scores
            selected_parents = self.select_parents(population, fitness_scores)

            # Introduce Elitism: Keep the best individuals without changes
            num_elites = int(0.5 * self.population_size)
            elites = selected_parents[:num_elites]

            # Performance is better if some of the worse individuals are also kept
            # num_baddies = int(0.5 * self.population_size)
            # baddies = selected_parents[num_baddies:]
            #
            children = elites.copy()
            # children.extend(baddies)

            # Generate rest of the children through crossover and mutation
            while len(children) < self.population_size:
                parent1, parent2 = random.sample(selected_parents, 2)

                for _ in range(self.n_children):
                    child = self.crossover(parent1, parent2)
                    if self.mutation == 0:
                        child = self.mutate(child)
                    else:
                        child = self.other_mutate(child)
                    children.append(child)

            # Replace old population with children
            population = children

            # Print or save information about the current generation
            print(f"Generation: {generation + 1}, Best Fitness: {best_fitness}")

            # Check for termination conditions
            if best_fitness == self.max_fitness or self.generations_without_improvement >= self.restart_after_gen:
                print("Final Solution:")
                for row in best_solution:
                    print(row)
                self.save_solution_to_file("output.txt", best_solution, generation + 1, best_fitness)
                break

            print("Final Solution:")
            for row in best_solution:
                print(row)

    # Implement simulated annealing to explore the solution space (hybrid algorithm)
    def simulated_annealing(self, individual, temperature):
        new_individual = self.mutate(individual)
        delta_fitness = self.fitness(new_individual) - self.fitness(individual)

        if delta_fitness > 0 or random.random() < math.exp(delta_fitness / temperature):
            return new_individual
        else:
            return individual

    # Hybrid algorithm has the best performance for simple sudoku tasks
    def hybrid_algorithm(self):
        population = self.generate_initial_population()

        best_solution = None
        temperature = 1.0  # Initial temperature for simulated annealing

        for generation in range(self.restart_after_gen):
            fitness_scores = [self.fitness(ind) for ind in population]

            best_index = max(range(self.population_size), key=lambda i: fitness_scores[i])
            best_individual = population[best_index]
            best_fitness = fitness_scores[best_index]

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.generations_without_improvement = 0
                best_solution = best_individual
            else:
                self.generations_without_improvement += 1

            selected_parents = self.select_parents(population, fitness_scores)

            elites = selected_parents[:int(0.5 * self.population_size)]
            children = elites.copy()
            # baddies = selected_parents[int(0.5 * self.population_size):]
            # children.extend(baddies)
            # children = []

            while len(children) < self.population_size:
                parent1, parent2 = random.sample(selected_parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.simulated_annealing(child, temperature)
                children.append(child)

            population = children

            # Decrease temperature for simulated annealing
            temperature *= 0.99

            print(f"Generation: {generation + 1}, Best Fitness: {best_fitness}")

            if best_fitness == self.max_fitness or self.generations_without_improvement >= self.restart_after_gen:
                print("Final Solution:")
                for row in best_solution:
                    print(row)
                    # Save the final solution to a text file
                self.save_solution_to_file("output.txt", best_solution, generation + 1, best_fitness)
                break

    def save_solution_to_file(self, output_file, solution,  last_generation, last_fitness):
        with open(output_file, 'w') as file:
            file.write(f"Last Generation: {last_generation}, Best Fitness: {last_fitness}\n\n")
            for row in solution:
                file.write(" ".join(map(str, row)) + "\n")
