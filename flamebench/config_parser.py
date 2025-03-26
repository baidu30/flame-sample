import yaml
from dataclasses import dataclass

@dataclass
class BaseConfig:
    mechanism: str
    scenario_type: str

@dataclass
class Config0D(BaseConfig):
    config_type: str
    phi_range: tuple
    pressure_range: tuple
    temperature_range: tuple
    # other parameters 

@dataclass
class Config1D(BaseConfig):
    config_type: str
    phi_range: tuple
    pressure_range: tuple
    temperature_range: tuple
    # other parameters

class ConfigParser:
    @staticmethod
    def load_config(yaml_path):
        with open(yaml_path) as f:
            config_data = yaml.safe_load(f)
        
        config = config_data['config_type']
        if config == '0D':
            return Config0D(**config_data)