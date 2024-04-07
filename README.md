
# Environment Setup

## Connect to server (Regular Connection)

* ssh command
* password

## SSH key (Connection without input the password anymore)

* Copy C:\Users\zhululu\.ssh\id_ed25519.pub to /root/.ssh/authorized_keys

## Set git

* github--Settings--SSH keys--generate new keys
* Copy /root/.ssh/id_ed25519.pub to github--Settings--SSH keys--Add new SSH Key--key

##  conda

* wget  [https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh)
* bash ./Anaconda3-2020.11-Linux-x86_64.sh
* (should be changed) echo 'export PATH="[path to conda]/bin:$PATH"' >> ~/.bashrc
* source ~/.bashrc
* conda create -n eeec  python=3.8
* conda activate eeec
* pip install setuptools==65.5.0
* pip install wheel==0.38.4
* pip install -r requirements.txt

  

##  Install Atari and MuJoCo
* wget  [https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
* tar xvf mujoco210-linux-x86_64.tar.gz && mkdir -p ~/.mujoco && mv mujoco210 ~/.mujoco/mujoco210
* wget https://www.roboti.us/file/mjkey.txt -O ~/.mujoco/mjkey.txt
* apt update && apt install unrar
* wget http://www.atarimania.com/roms/Roms.rar && unrar x Roms.rar && unzip Roms/ROMS.zip
* python -m atari_py.import_roms ROMS

## Other Requirements
* cuda 11.3
* Ubuntu 22.04
* GPU Memory > 10
