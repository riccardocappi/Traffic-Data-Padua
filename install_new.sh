PYTORCH_VERSION=2.3.1
TORCH_GEOMETRIC_VERSION=2.3.1

echo "Installing dependencies"

conda activate traffic_data_pd

pip install --no-cache-dir torch==${PYTORCH_VERSION} --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==${TORCH_GEOMETRIC_VERSION}
pip install PyYAML==6.0.1
pip install "numpy<2"
pip install matplotlib
pip install optuna
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install git+https://github.com/TorchSpatiotemporal/tsl.git
pip install torch-geometric-temporal
pip install h3 folium shapely
pip install overturemaps
pip install osmnx
pip install seaborn
pip install scikit-learn
pip install geopy
pip install statsmodels

git clone https://github.com/google-research/timesfm.git
cd timesfm

pip install flax --no-deps
pip install "jax[cpu]" --no-deps

pip install lancedb