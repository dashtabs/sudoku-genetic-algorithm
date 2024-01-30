from PyQt5.QtWidgets import QRadioButton, QComboBox, QApplication, QWidget, QVBoxLayout, QCheckBox, QLabel, QLineEdit, QPushButton, QTextEdit
from solver import SudokuSolver


class SudokuSolverGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        # LineEdits for setting parameters
        self.model_input = QLineEdit(self)
        self.population_size_input = QLineEdit(self)
        self.selection_rate_input = QLineEdit(self)
        self.rand_selection_rate_input = QLineEdit(self)
        self.n_children_input = QLineEdit(self)
        self.restart_after_gen_input = QLineEdit(self)
        self.max_fitness_input = QLineEdit(self)
        default_model = [
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
        self.model_input.setText(str(default_model))
        self.population_size_input.setText("1000")
        self.selection_rate_input.setText("0.25")
        self.rand_selection_rate_input.setText("0.25")
        self.n_children_input.setText("4")
        self.restart_after_gen_input.setText("1000")
        self.max_fitness_input.setText("162")

        # Dropdown for fitness function
        self.fitness_label = QLabel('Fitness Function:')
        self.fitness_dropdown = QComboBox(self)
        self.fitness_dropdown.addItems(['Count Duplicates', 'Count Correct'])

        # Dropdown for mutation
        self.mutation_label = QLabel('Mutation:')
        self.mutation_dropdown = QComboBox(self)
        self.mutation_dropdown.addItems(['Mutation', 'Dynamic Mutation'])

        # Checkboxes for algorithm selection
        self.choose_algorithm_label = QLabel('Choose Algorithm:')
        self.ga_radio = QRadioButton('Genetic Algorithm', self)
        self.hybrid_radio = QRadioButton('Hybrid Algorithm', self)
        self.ga_radio.setChecked(True)

        # Connect toggled signals to slot methods
        self.ga_radio.toggled.connect(self.update_radio_buttons)
        self.hybrid_radio.toggled.connect(self.update_radio_buttons)

        # Button to start the algorithm
        self.start_button = QPushButton('Start Algorithm', self)
        self.start_button.clicked.connect(self.start_algorithm)
        # TextEdit for displaying output
        self.output_text_edit = QTextEdit(self)
        self.output_text_edit.setReadOnly(True)  # Make it read-only

        # Add components to the layout
        layout.addWidget(QLabel('Model (2D list):'))
        layout.addWidget(self.model_input)
        layout.addWidget(QLabel('Population Size:'))
        layout.addWidget(self.population_size_input)
        layout.addWidget(QLabel('Selection Rate:'))
        layout.addWidget(self.selection_rate_input)
        layout.addWidget(QLabel('Random Selection Rate:'))
        layout.addWidget(self.rand_selection_rate_input)
        layout.addWidget(QLabel('Number of Children:'))
        layout.addWidget(self.n_children_input)
        layout.addWidget(QLabel('Restart After Generation:'))
        layout.addWidget(self.restart_after_gen_input)
        layout.addWidget(QLabel('Max Fitness:'))
        layout.addWidget(self.max_fitness_input)
        layout.addWidget(self.fitness_label)
        layout.addWidget(self.fitness_dropdown)
        layout.addWidget(self.mutation_label)
        layout.addWidget(self.mutation_dropdown)
        layout.addWidget(self.choose_algorithm_label)
        layout.addWidget(self.ga_radio)
        layout.addWidget(self.hybrid_radio)
        layout.addWidget(self.start_button)
        layout.addWidget(QLabel('Output:'))
        layout.addWidget(self.output_text_edit)

        self.setLayout(layout)
        self.setWindowTitle('Sudoku Solver GUI')
        self.show()

    def update_radio_buttons(self):
        # If Genetic Algorithm is checked, uncheck Hybrid Algorithm, and vice versa
        if self.ga_radio.isChecked():
            self.hybrid_radio.setChecked(False)
        if self.hybrid_radio.isChecked():
            self.ga_radio.setChecked(False)

    def start_algorithm(self):
        # Get values from UI components
        model_str = self.model_input.text()
        model = eval(model_str) if model_str else None
        population_size = int(self.population_size_input.text())
        selection_rate = float(self.selection_rate_input.text())
        rand_selection_rate = float(self.rand_selection_rate_input.text())
        n_children = int(self.n_children_input.text())
        restart_after_gen = int(self.restart_after_gen_input.text())
        max_fitness = int(self.max_fitness_input.text())

        # Determine which algorithm to use based on checkbox selection
        use_ga = self.ga_radio.isChecked()
        use_hybrid = self.hybrid_radio.isChecked()

        # Determine fitness function and mutation type based on dropdowns
        fitness_function = 0 if self.fitness_dropdown.currentText() == 'Count Duplicates' else 1
        mutation_type = 0 if self.mutation_dropdown.currentText() == 'Mutation' else 1

        # Initialize SudokuSolver with user-input parameters
        self.sudoku_solver = SudokuSolver(model=model,
        population_size=population_size,
        selection_rate=selection_rate,
        rand_selection_rate=rand_selection_rate,
        n_children=n_children,
        restart_after_gen=restart_after_gen,
        max_fitness=max_fitness, fitness_function=fitness_function, mutation=mutation_type)

        # Start the algorithm
        if use_ga:
            self.sudoku_solver.genetic_algorithm()
        elif use_hybrid:
            self.sudoku_solver.hybrid_algorithm()

        with open('C:/Users/dasha/PycharmProjects/genAlg/output.txt', 'r') as file:
            text = file.read()
            self.output_text_edit.setPlainText(text)
