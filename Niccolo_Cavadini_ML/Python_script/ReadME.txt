After downloading the dataset from Google Drive, the paths to the data folders must be updated according to the user’s local directory structure.
Users may either:

1. run the training script Train_VALIDATION_LOSS.py to train the U-Net model from scratch and     generate new weights, or

2. download the pre-trained model best_model_valLoss.pth from Google Drive and directly execute Test_MASK.py, ensuring that both the data paths and the model path are correctly adjusted.

The file U_Net_improved.py, which contains the U-Net architecture used for both training and inference, must be located in the same directory (or in a directory included in the Python import path) as the training and testing scripts, since it is imported via: from U_Net_improved import UNet

For reference, training the model on the hardware used by the author (GPU with 6 GB of dedicated memory and 24 GB of system RAM; ASUS Vivobook Pro) requires approximately 4 hours. Training time will vary depending on the computational resources available.