import pickle
from pathlib import Path

from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Data_Preprocessor import prepare_data

# Get the absolute path to the project root folder
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULT_DIR = PROJECT_ROOT / 'result' / 'stage_3_result'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def get_data_path(dataset_name):
    data_path = PROJECT_ROOT / 'data' / 'stage_3_data' / dataset_name
    return str(data_path)

#save both training and testing
def save_results(dataset_name, test_metrics, epoch_count, train_metrics):
    result_path = RESULT_DIR / f'results_{dataset_name}.txt'
    with open(result_path, 'w') as f:
        f.write(f"Training performance for {dataset_name} (final epoch on training data):\n")
        f.write(f"Epochs Run: {epoch_count}\n")
        f.write(f"Train Loss: {train_metrics['loss']:.4f}\n")
        f.write(f"Accuracy: {train_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {train_metrics['precision']:.4f}\n")
        f.write(f"Recall: {train_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {train_metrics['f1']:.4f}\n\n")

        f.write(f"Testing Performance for {dataset_name} (final evaluation on test set):\n")
        f.write(f"Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {test_metrics['f1']:.4f}\n")

    print(f"[Saved results to {result_path}]")


def train_and_evaluate(dataset_name, data_path):
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Initialize model
    model = Method_CNN(f"CNN_{dataset_name}", f"CNN for {dataset_name}", dataset_name)

    # Prepare data loaders
    train_loader = prepare_data(data['train'], dataset_name, model.batch_size) #
    test_loader = prepare_data(data['test'], dataset_name, model.batch_size)

    # Train model
    #best_accuracy = model.train_model(train_loader, test_loader)
    #count epoch & store final metrics
    best_accuracy, epoch_count, train_metrics = model.train_model(train_loader, test_loader)

    # Final evaluation
    metrics = model.evaluate(test_loader)

    print(f"\nTraining Performance for {dataset_name} (final epoch on training data):")
    print(f"Epochs Run: {epoch_count}")
    print(f"Train Loss: {train_metrics['loss']:.4f}")
    print(f"Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall: {train_metrics['recall']:.4f}")
    print(f"F1 Score: {train_metrics['f1']:.4f}")
    print(f"\nTesting Performance for {dataset_name}:")

    # Save results
    save_results(dataset_name, metrics, epoch_count, train_metrics)

    return metrics

# Example usage:
ORL_results = train_and_evaluate('ORL', get_data_path('ORL'))
MNIST_results = train_and_evaluate('MNIST', get_data_path('MNIST'))
CIFAR10_results = train_and_evaluate('CIFAR10', get_data_path('CIFAR'))

