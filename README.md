# FlameBench
<img width="260" alt="Pictu2re1" src="https://github.com/user-attachments/assets/1853622a-e172-422c-8a3e-4e0e9198f7c1" /> 

FlameBench is a benchmarking framework designed for combustion machine learning to evaluate the performance of various algorithms, models, or systems in a standardized and reproducible manner. This repository contains the tools, scripts, and datasets required to run benchmarks, analyze results, and compare performance across different configurations.

 必需的第三方依赖包：

  1. numpy - 用于数值计算
  2. cantera - 用于化学反应和燃烧计算
  3. torch (PyTorch) - 用于神经网络框架
  4. PyYAML - 用于配置文件解析
  5. matplotlib - 用于数据可视化
  6. pandas - 用于数据处理
  7. tqdm - 用于进度条显示

  安装命令：

  pip install numpy cantera torch PyYAML matplotlib pandas tqdm

  重要注意事项：

  1. PyTorch 安装可能需要根据系统和CUDA版本选择特定版本，请参考 https://pytorch.org/get-started/locally/ 获取正确的安装命令。
  2. Cantera 安装可能需要额外的配置，请参考 https://cantera.org/install/index.html。
  3. 对于1D火焰模拟功能，还需要安装 OpenFOAM，因为代码中调用了Allrun脚本和reconstructPar命令。

  这些依赖包支持项目的核心功能，包括化学反应建模、数据采样、神经网络训练和结果可视化。