# RadarTrafficData
Final ML project for the Machine Learning course of Ecole des Mines de Nancy

Kaggle link: https://www.kaggle.com/vinayshanbhag/radar-traffic-data

## Prepare the environment
Install Anaconda 
 `conda env create -f environment.yaml` to create the env
 `conda activate dl_env` to activate the env 
 
 `python -m ipykernel install --user --name=dl_env` to add the configurated env as a jupyter kernel
 
## Start a training
To train a simple LSTM model on the radar data : 

python main.py --dataset Path_2_csv --radar Radar_name --lr learning_rate --epochs 300

Look for more argument in the file main.py