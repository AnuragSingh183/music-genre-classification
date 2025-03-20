from matplotlib import pyplot as plt
import numpy as np

def train_test_split_with_filenames(X, y, filenames, test_size=0.2, random_state=42):
    """Split data into training and testing sets, including filenames."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    filenames_train, filenames_test = filenames[train_indices], filenames[test_indices]
    
    return X_train, X_test, y_train, y_test, filenames_train, filenames_test

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    # Calculate confusion matrix
    num_classes = len(np.unique(y_test))
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(y_test)):
        confusion_mat[y_test[i], predictions[i]] += 1
    
    # Calculate precision, recall, and F1 score
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        true_positive = confusion_mat[i, i]
        false_positive = np.sum(confusion_mat[:, i]) - true_positive
        false_negative = np.sum(confusion_mat[i, :]) - true_positive
        
        precision[i] = true_positive / (true_positive + false_positive + 1e-10)
        recall[i] = true_positive / (true_positive + false_negative + 1e-10)
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-10)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_mat,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_precision': np.mean(precision),
        'avg_recall': np.mean(recall),
        'avg_f1_score': np.mean(f1_score)
    }

def plot_metrics(model_histories, model_names):
    """Plot training metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    for i, (history, name) in enumerate(zip(model_histories, model_names)):
        plt.plot(history[0], label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    for i, (history, name) in enumerate(zip(model_histories, model_names)):
        plt.plot(history[1], label=name)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_mat, classes, title='Confusion Matrix'):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, format(conf_mat[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()