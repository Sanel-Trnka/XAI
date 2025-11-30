"""Configuration targeting the reduced 6-feature WDBC subset."""

dataset_csv="dataset/data.csv"        # Dataset file path
testset_csv="dataset/test.csv"        # Testset file path
target_column="diagnosis"             # Ground truth column in Dataset
# Tuned hyperparameters for a 6-feature input (reduced feature set)
learning_rate=0.0003                   # Lower LR for fine-tuning with fewer inputs
momentum=0.9                          # Momentum for SGD (unused by Adam)
training_split=0.8                    # Train/Validate split ratio
optimizer="Adam"                      # Adam generally converges faster/stabler here
# Smaller / shallower network for 6 input features
hidden_layers=[16, 16]                # Tuned architecture for 6 features
activation="LeakyReLU"                     # Nonlinearity
ignore_columns=[
    "id",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"]                 # Columns to ignore (present in WDBC)
batch_size=16                         # Mini-batch size (smaller batch for fewer features)
epochs=60                            # More epochs to let the smaller net converge
max_depth=20                        # Max depth of Decision Tree (reduced to avoid overfit)
min_samples_leaf=15