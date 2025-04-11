import numpy as np
from ffnn import FFNN
import os
import matplotlib.pyplot as plt

def test_save_load():
    print("Testing Save and Load")
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/test_model.json"
    
    layer_sizes = [4, 8, 3]
    activation_funcs = ["sigmoid", "softmax"]
    model = FFNN(
        layer_sizes=layer_sizes,
        activation_funcs=activation_funcs,
        weight_init="xavier",
        loss_function="cce",
        reg_type="l2",
        lambda_param=0.01,
        seed=42
    )
    
    print("\nOriginal model:")
    model.print_model()
    

    X_train = np.random.randn(100, 4)
    y_true_labels = np.random.randint(0, 3, size=100)
    y_train = np.zeros((100, 3))
    for i, label in enumerate(y_true_labels):
        y_train[i, label] = 1
    
    # Train the model
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        batch_size=32,
        learning_rate=0.01,
        epochs=5,
        verbose=1
    )
    
    X_test = np.random.randn(10, 4)
    original_predictions = model.predict(X_test)
    print(f"First few predictions shape: {original_predictions.shape}")
    print(f"First prediction: {original_predictions[0]}")
    
    print("\n Saving model to:", model_path)
    model.save(model_path)
    

    # Load the model
    loaded_model = FFNN.load(model_path)
    loaded_model.print_model()
    loaded_predictions = loaded_model.predict(X_test)
    print(f"First few predictions shape: {loaded_predictions.shape}")
    print(f"First prediction: {loaded_predictions[0]}")
    
    # Compare predictions
    print("\nCompare predictions")
    prediction_diff = np.sum(np.abs(original_predictions - loaded_predictions))
    print(f"differences: {prediction_diff:.10f}")
    
    if prediction_diff < 1e-10:
        print("\nberhasil")
    else:
        print("\ngagal")
    
if __name__ == "__main__":
    test_save_load()