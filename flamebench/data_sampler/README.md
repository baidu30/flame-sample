# oneDflame_setup 模块说明

## 功能概述

`oneDflame_setup.py` 模块为1D层流火焰模拟提供必要的设置功能。该模块作为OneDSampler和OpenFOAM CFD模拟之间的桥梁，负责：

1. 使用Cantera计算层流火焰特性
2. 准备OpenFOAM模拟所需的配置参数
3. 生成必要的初始条件文件
4. 更新模拟配置

## 主要功能函数

### 1. calculate_laminar_flame_properties
计算层流火焰速度和火焰厚度
- 输入：化学机制文件路径、气体状态参数
- 输出：火焰速度、火焰厚度、火焰对象

### 2. update_case_parameters
更新OpenFOAM案例参数
- 根据火焰特性计算域尺寸和时间步长
- 返回完整的案例参数字典

### 3. update_one_d_sample_config
更新1D采样配置文件
- 修改OpenFOAM字典文件以适应当前模拟

### 4. create_0_species_files
创建初始物种文件
- 在OpenFOAM的0/目录下生成温度、压力和物种浓度文件

### 5. update_set_fields_dict
更新初始条件设置文件
- 配置system/setFieldsDict文件

### 6. update_cantera_mechanism
更新Chemtab机制文件路径
- 确保OpenFOAM能够找到正确的化学机制文件

## 使用示例

```python
import cantera as ct
from oneDflame_setup import *

# 配置参数
mechanism_path = "mechanisms/Burke2012_s9r23.yaml"
gas_state = {
    "initial_temperature": 300,
    "initial_pressure": 101325,
    "fuel_composition": "H2:1",
    "oxidizer_composition": "O2:0.21,N2:0.79",
    "equivalence_ratio": 1.0
}

# 计算火焰特性
flame_speed, flame_thickness, flame = calculate_laminar_flame_properties(
    mechanism_path, gas_state
)

# 更新案例参数
case_params = update_case_parameters(
    mechanism_path, gas_state, flame_speed, flame_thickness
)

# 更新配置文件
update_one_d_sample_config(case_params, gas_state)
create_0_species_files(case_params)
update_set_fields_dict(case_params)
update_cantera_mechanism(mechanism_path)
```

## 依赖关系

- Cantera: 用于化学反应和火焰计算
- NumPy: 数值计算
- Python标准库: os, pathlib