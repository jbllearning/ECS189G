import pickle
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Data_Preprocessor import prepare_data

# Get the absolute path to the project root folder
PROJECT_ROOT = Path(__file__).parent.parent.parent

def get_data_path(dataset_name):
    data_path = PROJECT_ROOT / 'data' / 'stage_3_data' / dataset_name
    # print(f"Resolved data path: {data_path}")  # Debug
    return str(data_path)

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
    metrics = model.evaluate(test_loader)

    print(f"\nFinal Evaluation for {dataset_name}:")
    print(f"Loss: {metrics['loss']:.4f}")  # Now includes loss
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    return metrics

# Example usage:

ORL_results = train_and_evaluate('ORL', get_data_path('ORL'))
MNIST_results = train_and_evaluate('MNIST', get_data_path('MNIST'))
CIFAR10_results = train_and_evaluate('CIFAR10', get_data_path('CIFAR'))