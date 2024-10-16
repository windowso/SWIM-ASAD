cd SWIM-ASAD/

conda create -n swim python=3.11 -y
conda activate swim

# Install dependencies
conda install nvidia::cuda-toolkit=12.1 -y
pip install --upgrade pip
pip install -e .