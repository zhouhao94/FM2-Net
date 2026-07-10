# Frequency-Decoupled Multimodal Fusion and Modulation for Underwater Salient Object Detection
### [[Paper]](https://arxiv.org/abs/2603.06231)

> [**Frequency-Decoupled Multimodal Fusion and Modulation for Underwater Salient Object Detection**](https://arxiv.org/abs/2507.17342)            
> [**Hao Zhou, Xu Yang, Hai Huang, Min Liu， Jie-Ming Ma, Chao-Meng Chen, Xu-Yao Zhang, Fei Luo**  
> **arXiv preprint arXiv:2603.06231**

## 🛠️ Get started

### Set up a new virtual environment
```
conda create -n FM2-Net python=3.8
conda activate FM2-Net
```

### Install dependency packages
```
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r ./requirements.txt
```

### Install Mamba
- We follow the settings outlined in [VideoMamba](https://github.com/OpenGVLab/VideoMamba).
```
git clone git@github.com:OpenGVLab/VideoMamba.git
cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
```

### Some packages may be useful
```
pip install tensorboard
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install protobuf==3.20.3
```
