pip install -r requirements.txt
pip install setuptools==59.5.0

cd decoder/utils/furthestPointSampling
python3 setup.py install

# https://github.com/sshaoshuai/Pointnet2.PyTorch
cd decoder/utils/Pointnet2.PyTorch/pointnet2
python3 setup.py install