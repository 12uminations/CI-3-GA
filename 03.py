import numpy as np
import random
import matplotlib.pyplot as plt

# Class for Data Handling
class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.X, self.y = self._load_and_preprocess()
    
    def _load_and_preprocess(self):
        data = np.genfromtxt(self.filepath, delimiter=',', dtype=str)
        X = np.array(data[:, 2:], dtype=float) 
        y = np.where(data[:, 1] == 'M', 1, 0)   
        X = (X - X.mean(axis=0)) / X.std(axis=0)  
        return X, y

    @staticmethod
    def split_k_fold(X, y, k=10):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        fold_size = len(indices) // k
        return [(np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]]), 
                 indices[i * fold_size:(i + 1) * fold_size]) for i in range(k)]

# Class for the MLP Neural Network
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X, weights):
        activations = [X]
        start = 0
        for layer in range(len(self.layers) - 2):
            end = start + self.layers[layer] * self.layers[layer + 1]
            W = weights[start:end].reshape((self.layers[layer], self.layers[layer + 1]))
            start = end
            end = start + self.layers[layer + 1]
            b = weights[start:end].reshape((1, self.layers[layer + 1]))
            start = end
            z = np.dot(activations[-1], W) + b
            activations.append(self.sigmoid(z))
        
        W_out = weights[start:start + self.layers[-2] * self.layers[-1]].reshape((self.layers[-2], self.layers[-1]))
        b_out = weights[start + self.layers[-2] * self.layers[-1]:].reshape((1, self.layers[-1]))
        z_out = np.dot(activations[-1], W_out) + b_out
        return self.softmax(z_out)

# Class for Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, neural_network, pop_size=100, mut_rate=0.02, gens=250, early_stop=25):
        self.nn = neural_network
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.gens = gens
        self.early_stop = early_stop
        self.history = []

    def init_population(self, weight_size):
        return [np.random.randn(weight_size) * 0.1 for _ in range(self.pop_size)]

    @staticmethod
    def select_population(population, fitness_scores, num_select):
        max_fit = np.max(fitness_scores)
        min_fit = np.min(fitness_scores)
        fit_range = max_fit - min_fit
        
        scaled_fitness = (fitness_scores - min_fit) / fit_range if fit_range != 0 else np.ones_like(fitness_scores) / len(fitness_scores)
        selected_indices = np.argsort(scaled_fitness)[-num_select:]
        return [population[i] for i in selected_indices]

    @staticmethod
    def crossover(parent1, parent2):
        point = np.random.randint(1, len(parent1) - 1)
        return np.concatenate((parent1[:point], parent2[point:]))

    def mutate(self, weights):
        for i in range(len(weights)):
            if np.random.rand() < self.mut_rate:
                weights[i] += np.random.randn() * 0.05
        return weights

    def evaluate(self, weights, X, y):
        preds = np.argmax(self.nn.forward(X, weights), axis=1)
        return np.mean(preds == y)

    def evolve(self, X_train, y_train, X_val, y_val):
        weight_size = sum([self.nn.layers[i] * self.nn.layers[i + 1] + self.nn.layers[i + 1] for i in range(len(self.nn.layers) - 1)])
        population = self.init_population(weight_size)
        best_fitness = -np.inf
        no_improve_count = 0

        for gen in range(self.gens):
            fitness_scores = [self.evaluate(weights, X_val, y_val) for weights in population]
            population = self.select_population(population, fitness_scores, self.pop_size // 2)
            new_pop = population.copy()

            while len(new_pop) < self.pop_size:
                parents = random.sample(population, 2)
                child = self.crossover(parents[0], parents[1])
                new_pop.append(self.mutate(child))
            
            population = new_pop
            best_gen_fit = max(fitness_scores)

            if best_gen_fit > best_fitness:
                best_fitness = best_gen_fit
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            self.history.append(best_gen_fit)
            print(f"Gen {gen + 1}/{self.gens}, Best fitness: {best_fitness:.4f}")

            if no_improve_count >= self.early_stop:
                print("Early stopping.")
                break

        return max(population, key=lambda w: self.evaluate(w, X_val, y_val))

    def plot_progress(self):
        plt.plot(self.history, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Genetic Algorithm Progress')
        plt.legend()
        plt.show()

# Function to plot predicted vs actual values for each fold
def plot_predicted_vs_actual(all_actuals, all_predictions):
    num_folds = len(all_actuals)
    fig, axs = plt.subplots(2, 5, figsize=(18, 10))  # 2 rows, 5 columns layout for 10 folds
    fig.suptitle("Predicted vs Actual Values for Each Fold")

    for i in range(num_folds):
        row = i // 5
        col = i % 5
        axs[row, col].scatter(all_actuals[i], all_predictions[i], alpha=0.6, color="blue")
        axs[row, col].plot([min(all_actuals[i]), max(all_actuals[i])], [min(all_actuals[i]), max(all_actuals[i])], 'r--', label="Ideal Line")
        axs[row, col].set_title(f"Fold {i + 1}")
        axs[row, col].set_xlabel("Actual Values")
        axs[row, col].set_ylabel("Predicted Values")
        axs[row, col].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Function to plot accuracy of each fold
def plot_accuracy_summary(accuracies):
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(accuracies) + 1), accuracies, color='skyblue')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Each Fold')
    plt.xticks(range(1, len(accuracies) + 1))
    plt.ylim([0, 100])
    plt.grid(axis='y')
    plt.show()

# Class for Cross-validation and Evaluation
class Evaluator:
    def __init__(self, data_handler, neural_network, pop_size=100, gens=250):
        self.data_handler = data_handler
        self.neural_network = neural_network
        self.pop_size = pop_size
        self.gens = gens

    def run_cross_validation(self):
        X, y = self.data_handler.X, self.data_handler.y
        folds = DataHandler.split_k_fold(X, y, k=10)
        accuracies = []
        fitness_hist = []
        all_actuals = []     # List to store actual values for each fold
        all_predictions = [] # List to store predicted values for each fold

        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"Fold {fold + 1}/10")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            ga = GeneticAlgorithm(self.neural_network, pop_size=self.pop_size, gens=self.gens)
            best_weights = ga.evolve(X_train, y_train, X_val, y_val)
            
            y_pred = np.argmax(self.neural_network.forward(X_val, best_weights), axis=1)
            accuracy = np.mean(y_pred == y_val) * 100
            accuracies.append(accuracy)
            fitness_hist.append(ga.history)

            # Store actual and predicted values for each fold
            all_actuals.append(y_val)
            all_predictions.append(y_pred)

            print(f'Fold {fold + 1} Accuracy: {accuracy:.2f}%')

        print(f'Mean Accuracy: {np.mean(accuracies):.2f}%')
        return accuracies, fitness_hist, all_actuals, all_predictions

# Main execution
if __name__ == '__main__':
    data_path = 'wdbc.txt'
    data_handler = DataHandler(data_path)
    
    num_layers = int(input("Enter number of hidden layers: "))
    hidden_nodes = [int(input(f"Nodes for hidden layer {i + 1}: ")) for i in range(num_layers)]
    layers = [30] + hidden_nodes + [2]  # Input layer, hidden layers, output layer
    neural_network = NeuralNetwork(layers)

    evaluator = Evaluator(data_handler, neural_network)
    accuracies, fitness_history, all_actuals, all_predictions = evaluator.run_cross_validation()
    
    plot_predicted_vs_actual(all_actuals, all_predictions)  # Call the new plotting function
    for fitness in fitness_history:
        plt.plot(fitness)  # Plot fitness over generations for each fold
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations for Each Fold')
    plt.legend([f'Fold {i+1}' for i in range(len(fitness_history))])
    plt.show()
    
    plot_accuracy_summary(accuracies)  # Plot summary of accuracy for each fold
