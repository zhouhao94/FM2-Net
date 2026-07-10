# Frequency-Decoupled Multimodal Fusion and Modulation for Underwater Salient Object Detection
### [[Paper]](https://arxiv.org/abs/2603.06231)

> [**Frequency-Decoupled Multimodal Fusion and Modulation for Underwater Salient Object Detection**](https://arxiv.org/abs/2507.17342)            
> [**Hao Zhou, Xu Yang, Hai Huang, Min Liu, Jie-Ming Ma, Chao-Meng Chen, Xu-Yao Zhang, Fei Luo**  
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
```

## 🕹️ Prepare the data
### Setup [USOD10K dataset](https://www.argoverse.org/av2.html)
````
   data
   |-- USOD10K
   |   |-- USOD10K-TR
   |   |-- |-- USOD10K-TR-RGB
   |   |-- |-- USOD10K-TR-GT
   |   |-- |-- USOD10K-TR-depth
   |   |-- |-- USOD10K-TR-Boundary
   |   |-- USOD10K-Val
   |   |-- |-- USOD10K-Val-RGB
   |   |-- |-- USOD10K-Val-GT
   |   |-- |-- USOD10K-Val-depth
   |   |-- |-- USOD10K-Val-Boundary
   |   |-- USOD10K-TE
   |   |-- |-- USOD10K-TE-RGB
   |   |-- |-- USOD10K-TE-GT
   |   |-- |-- USOD10K-TE-depth
````
