# ECS189G — Stage 5: Graph Convolutional Network (GCN)

This stage implements a 2-layer Graph Convolutional Network (GCN) for node classification using PyTorch. The model is trained and evaluated on three citation graph datasets: **Cora**, **Citeseer**, and **Pubmed**.

To train and evaluate the model, just run script/stage_5_script/script_GCN.py. 
The script will This script will:
- Load and preprocess each dataset
- Train the GCN model (called Method_GCN.py)
- Evaluate final test metrics
- Save result and the training convergence plot

## Key files:
- Method_GCN.py: Main model and training loop implementation
- Dataset_Loader_Node_Classification.py: provided by professor to load citation graph datasets into PyTorch tensors
- script_GCN.py: Run training and evaluation for all datasets

### Notes
Training plots are saved in script/stage_5_script. **Results of the training are saved in result/stage_5_result.**
