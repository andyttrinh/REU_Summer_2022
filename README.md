# REU Summer 2022
This project was funded by the NFS for UConn's REU 2022 Program during the summer. 
Main goal of the project is to implement Diffential Privacy (DP) into AdderNet Models.
We then compare the training results to standard CNN models with DP

# Dependencies
This project was worked using Anaconda. The enviroment.yaml file is used to install the conda enviroment needed to run the code. Run the code below in the directory that has the enviroment.yaml to install dependencies.

```bash
conda env create --file enviroment.yaml
```

# Training Models
To train models run the code as follow:
For AdderNet Models use the RunDPAdderNet.py file instead.

```bash
python3 RunDPCNN.py 0.5 2
```

The first argument represents the noise_scale, the amount of noise added when training the model. Increasing this value will increase the amount of noised added, increasing security but decreasing the accuracy of the model.

The second argument represents the noise_bound, the gradient clipping bound.

Gaussian noise with standard deviation noise_scale * norm_bound will be addded to each clipped gradient.

# Libraries
Blurnn - Machine Learning with Differential Privacy Library was used to implement DP.
https://github.com/ailabstw/blurnn
