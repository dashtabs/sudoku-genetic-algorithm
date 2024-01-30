# Sudoku with Genetic Algorithms
## By Daria Tabunshchyk

Sudoku is a classic logic-based combinatorial number-placement puzzle. In this project, genetic algorithms are employed to efficiently solve Sudoku puzzles.

The project is developed as part of the Genetic Algorithms course at TU Ilmenau.

## Parts of the project:
- SudokuSolver
The SudokuSolver class is the core component implementing the genetic algorithm for solving Sudoku puzzles. It includes functions for generating an initial population, calculating fitness, performing mutation and crossover, and running the genetic algorithm. The implementation can be found in `solver.py` 
- GUI
The GUI (`SudokuSolverGUI`) provides a very simple interface for interecting with the genetic algorithm's parameters and view the results.

# Features

- Two different fitness аunctions
The algorithm supports two fitness functions: "Count Duplicates" penalizes duplicates in columns and blocks (set max_fitness to 162), and "Count Correct" evaluates the number of correctly built columns and blocks (set max_fitness to 18).

- Two different mutation approaches
    - The standard mutation swaps numbers in rows or generates a new individual with the same probability. 
    - The dynamic mutation rate adjusts based on the generation and performance.
- Elitism
The algorithm employs elitism, preserving the best individuals (and it's possible to also include the worst) from each generation.
- Simulated Annealing
Simulated Annealing is utilized as part of the hybrid algorithm, contributing to better performance on simpler Sudoku tasks.
This project is created in terms of the Genetic Algorithms Course at the TU Ilmenau.

## Running the project
1. Clone the repository.
2. Ensure you have Python installed.
3. Run the program by executing `main.py`

## Terms
##### Genetic Algorithm:
An optimization algorithm inspired by the process of natural selection, used to find approximate solutions to optimization problems.
##### Elitism:
A genetic algorithm strategy where the best individuals from each generation are directly passed on to the next generation.
##### Simulated Annealing:
A probabilistic optimization algorithm that explores the solution space by accepting worse solutions with a certain probability.
## SudokuSolver class
Methods:
generate_initial_population(): Generates the initial population of Sudoku boards.
fitness(individual) (or other_fitness(individual): Evaluates the fitness of an individual Sudoku board.
mutate(individual) (or other_mutate(individual)): Performs mutation on an individual board.
crossover(parent1, parent2): Performs crossover between two parents.
genetic_algorithm(): Runs the standard genetic algorithm.
hybrid_algorithm(): Incorporates simulated annealing into the genetic algorithm for improved performance on simpler tasks.

## SudokuSolverGUI
Input Parameters:
Model, Population Size, Selection Rate, Random Selection Rate, Number of Children, Restart After Generation, Maximum Fitness, Fitness Function, Mutation.

Algorithm Selection:
Choose between Genetic Algorithm and Hybrid Algorithm (Genetic Algorithm + Simulated Annealing).

Output:
Displays the progress and final results of the algorithm.
## Results
Example results for 10 executions for Simulated Annealing:
| Generations | Fitness |
| ------ | ------ |
| 52 | 162 |
| 34 | 162 |
| 38 | 162 |
| 1000 | 160 |
| 45 | 162 |
| 35 | 162 |
| 33 | 162 |
| 34 | 162 |
| 40 | 162 |
| 34 | 162 |

## Pros and Cons
Pros:
Efficiently solves simple Sudoku puzzles.

Cons:
Often gets stuck in the local maximum for more complicated problems.

## Sources

[1] Wikipedia (Sudoku)
[2] [Solving Sudoku Puzzles with Genetic Algorithm (article)](https://nidragedd.github.io/sudoku-genetics/)
[3] https://github.com/ctjacobs/sudoku-genetic-algorithm 
[4] https://github.com/MojTabaa4/genetic-algorithm



