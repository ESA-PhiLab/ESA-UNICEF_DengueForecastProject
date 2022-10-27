# Installation guide

**This guide can be ignore for people running the code in google colaboratory**


## 1. Python and Anaconda Installation

You can rely on the original installation guide:

- [Windows installation](https://docs.anaconda.com/anaconda/install/windows/)
- [MacOS installation](https://docs.anaconda.com/anaconda/install/mac-os/)
- [Linux installation](https://docs.anaconda.com/anaconda/install/linux/)

For other Operation Systems please refer to [Anaconda Install](https://docs.anaconda.com/anaconda/install/index.html).

## 2. Setup the Anaconda environment

### a. Clone the repository on you desktop

This operation can be done in two ways:

1. open yout terminal, navigate to you desktop and run 
   `git https://github.com/ESA-PhiLab/ESA-UNICEF_DengueForecastProject`
2. download the project on your desktop

### b. Create a new virtual environment
This allows to create a isolated environment and to control the installation of dependancies without affecting your python base installation.

Before creating a new environment, be sure that the default anaconda enviroment is not active. By opening a terminal, you just need to check if before you user name there is something like ` (base) username: $`. If so run this command first:

` conda deactivate base ` or ` conda deactivate <environmentname> `

otherwise run directly this command, that creates a new enviroment (change yourenvname with whaterver you prefer)

` conda create -n yourenvname python=x.x anaconda`

### c. Acticate the virtual environment

To work on the environment you need to activate it first

` conda activate yourenvname `

### d. Install dependancies

First thing first, you need to move to the project folder

` cd <path to the project folder> `, for example ` cd Desktop\ESA-UNICEF_DengueForecastProject`

then you need to install pip (python package manager) on the conda environment

` conda install pip `

Then you can install the remaining packages listed in *[requirements.txt](../code/requirements.txt)*

` pip install code\requirements.txt `
