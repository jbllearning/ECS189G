# ECS189G
To train and evaluate the model, just run script_CNN.py
## Key Files
The following are the key files involved in this implementation:
* Method_CNN.py: CNN model implementation
* Data_Preprocessor.py: Handles dataset loading and preprocessing
* script_CNN.py: Main training/evaluation pipeline

For the path to data, this is defined in line 11 of script.CNN.py. This is based on the project root directory and the path to the stage 3 data. This can be adjusted if needed.

### Notes
Models will automatically stop training once their threshold is reached (defined line 139 of Method_CNN).

Training plots are saved in stage_3_script. **Results of the training are saved at stage_3_result.**

The calls to Method_CNN are found in the bottom of the script_CNN.py