"""Baseline configuration for the full WDBC feature set."""

dataset_csv="dataset/data.csv"        # Dataset file path
testset_csv="dataset/test.csv"        # Testset file path
target_column="diagnosis"             # Ground truth column in Dataset
learning_rate=0.001                   # Adam default LR for tabular MLPs
momentum=0.9                          # Momentum for SGD (unused by Adam)
training_split=0.8                    # Train/Validate split ratio
optimizer="Adam"                      # Adam generally converges faster/stabler here
hidden_layers=[64, 32]                # Compact architecture for WDBC (30 features)
activation="LeakyReLU"                     # Nonlinearity
ignore_columns=["id"]                 # Columns to ignore (present in WDBC)
batch_size=32                         # Mini-batch size
epochs=24                            # Enough budget to converge; still fast
max_depth=20                       # Max depth of Decision Tree
min_samples_leaf=5