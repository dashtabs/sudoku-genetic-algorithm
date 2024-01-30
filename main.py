from solver import SudokuSolver
from PyQt5.QtWidgets import QApplication
from gui import SudokuSolverGUI
import sys

''' Hard model '''
# model = [
#     [0, 0, 2, 0, 0, 1, 0, 4, 9],
#     [0, 8, 0, 0, 0, 0, 5, 0, 1],
#     [3, 0, 0, 0, 0, 9, 0, 0, 0],
#     [0, 0, 0, 0, 5, 0, 0, 0, 0],
#     [0, 0, 0, 3, 9, 6, 2, 0, 0],
#     [6, 0, 1, 0, 0, 4, 0, 0, 3],
#     [0, 0, 3, 0, 0, 0, 0, 0, 2],
#     [9, 0, 4, 0, 6, 0, 0, 0, 0],
#     [0, 6, 0, 0, 2, 0, 0, 0, 0],
# ]
''' Middle model '''
# model = [
#     [2, 0, 0, 0, 0, 4, 0, 0, 0],
#     [0, 4, 5, 0, 0, 8, 1, 0, 6],
#     [1, 3, 0, 0, 9, 0, 0, 4, 0],
#     [0, 0, 0, 8, 4, 0, 0, 0, 9],
#     [0, 0, 0, 0, 0, 0, 0, 0, 4],
#     [4, 8, 9, 0, 0, 0, 7, 3, 1],
#     [6, 2, 0, 0, 0, 0, 0, 1, 7],
#     [0, 0, 0, 0, 0, 1, 6, 9, 0],
#     [7, 0, 1, 0, 3, 6, 0, 0, 0],
# ]

''' Easy model '''
model = [
    [0, 5, 6, 8, 9, 1, 0, 4, 0],
    [0, 4, 0, 7, 0, 0, 0, 0, 1],
    [8, 0, 1, 0, 0, 5, 2, 0, 0],
    [1, 0, 2, 0, 7, 0, 0, 0, 0],
    [5, 9, 8, 0, 6, 0, 0, 0, 4],
    [0, 6, 4, 0, 3, 8, 1, 0, 0],
    [0, 2, 7, 0, 0, 3, 0, 5, 0],
    [0, 0, 0, 2, 0, 4, 0, 0, 7],
    [9, 0, 0, 0, 8, 0, 4, 3, 2],
]


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create and show the GUI
    sudoku_solver_gui = SudokuSolverGUI()

    sys.exit(app.exec_())

# Use with GUI or comment the GUI and uncomment solver with either genetic_algorithm or hybrid_algorithm

# solver = SudokuSolver(model, population_size=1000, selection_rate=0.25,
#                       rand_selection_rate=0.25, n_children=4, restart_after_gen=1000, max_fitness=162,
#                       mutation='Dynamic Mutation', fitness_function='Count Duplicates')
# solver.genetic_algorithm()
# solver.hybrid_algorithm()
