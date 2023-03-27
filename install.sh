
pip install mmcv-full==1.4.0
pip install mmsegmentation==0.20.2
export PATH=/opt/compiler/gcc-8.2/bin/:$PATH
pip install git+https://github.com/open-mmlab/mmdetection.git@v2.24.1
pip install git+https://github.com/open-mmlab/mmdetection3d.git@v0.17.1
# pip install -r requirements.txt

python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes

pip install git+https://github.com/scikit-image/scikit-image.git@v0.17.2