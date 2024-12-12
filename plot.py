import matplotlib.pyplot as plt

def plot_r2_scores(model_names, train_r2_scores, test_r2_scores):
    # Seaborn style grid
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plotting the R2 Scores for different models using line plot
    plt.figure(figsize=(10, 6))

    # Plotting the training R² scores
    plt.plot(model_names, train_r2_scores, marker='o', label='Training R²', color='b', linestyle='-', markersize=8)
    
    # Plotting the testing R² scores
    plt.plot(model_names, test_r2_scores, marker='o', label='Testing R²', color='r', linestyle='-', markersize=8)
    
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('Comparison of R² Scores for Different Models')
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig('plot.png')

def main():
    # Assume we have R² scores for 3 models
    model_names = ['Linear Regression', 'XGBoost', 'Neural Networks']
    train_r2_scores = [0.4816948612990217, 0.955414814079146, 0.8]
    test_r2_scores = [0.43440329978112124, 0.5778546471399888, 0.45]
    
    plot_r2_scores(model_names, train_r2_scores, test_r2_scores)
    
if __name__ == "__main__":
    main()