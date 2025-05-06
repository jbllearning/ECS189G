import pickle
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Data_Preprocessor import prepare_data


def train_and_evaluate(dataset_name, data_path):
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Initialize model
    model = Method_CNN(f"CNN_{dataset_name}", f"CNN for {dataset_name}", dataset_name)

    # Prepare data loaders
    train_loader = prepare_data(data['train'], dataset_name, model.batch_size)
    test_loader = prepare_data(data['test'], dataset_name, model.batch_size)

    # Train model
    best_accuracy = model.train_model(train_loader, test_loader)

    # Final evaluation
    _, metrics = model.evaluate(test_loader)
    print(f"\nFinal Evaluation for {dataset_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    return metrics

# Example usage:
ORL_results = train_and_evaluate('ORL', '/Users/jessie/Documents/GitHub/ECS189G/data/stage_3_data/ORL')
# MNIST_results = train_and_evaluate('MNIST', 'MNIST')
# CIFAR10_results = train_and_evaluate('CIFAR10', 'CIFAR10')