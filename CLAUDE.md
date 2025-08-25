# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlameBench is a benchmarking framework for combustion machine learning that evaluates the performance of various algorithms, models, or systems in a standardized and reproducible manner. The repository contains tools, scripts, and datasets required to run benchmarks, analyze results, and compare performance across different configurations.

## Codebase Architecture

The codebase is structured as follows:

1. **flamebench/** - Main Python package containing the core framework
   - **config_parser.py** - Configuration parsing for 0D and 1D scenarios
   - **data_sampler/** - Data collection modules for different scenarios
     - **base_sampler.py** - Abstract base class for samplers
     - **zeroD_sampler.py** - 0D scenario data sampler
     - **oneD_sampler.py** - 1D scenario data sampler (laminar flame simulation)
   - **dataset_tools/** - Data processing and management utilities
     - **container.py** - PyTorch Dataset container for managing data
     - **merger.py** - Data merging utilities
     - **augmenter.py** - Data augmentation tools
   - **nn_framework/** - Neural network models and training framework
     - **model.py** - MLP and base TorchModel implementations
     - **loss.py** - Loss function definitions
   - **utils/** - Utility functions and visualization tools
     - **utils.py** - Path resolution and helper functions
     - **visualiser.py** - Visualization tools for results

2. **config/** - Configuration files for different scenarios
   - **0d_config.yaml** - Configuration for 0D scenarios
   - **1d_config.yaml** - Configuration for 1D scenarios

3. **mechanisms/** - Chemical mechanism files in YAML format
   - Various chemical mechanisms like Burke2012, drm19, etc.

4. **main.ipynb** - Jupyter notebook demonstrating the workflow

## Common Development Commands

### Running the Framework
The main workflow is demonstrated in `main.ipynb` and follows these steps:
1. Configuration loading with `ConfigParser`
2. Data sampling using `ZeroDSampler` or `OneDSampler`
3. Data preprocessing with `DatasetMerger` and `DataAugmenter`
4. Model training using `MLPModel` and `Container`

### Key Components

1. **Data Sampling**: 
   - 1D sampler uses Cantera for laminar flame simulations
   - Requires OpenFOAM for CFD simulations
   - Data is saved as .npy files

2. **Data Processing**:
   - Container class manages PyTorch datasets
   - Supports train/test splitting and data shuffling

3. **Model Training**:
   - MLPModel implements multi-layer perceptron networks
   - Built on PyTorch with custom training loop
   - Supports custom loss functions and metrics

## Development Considerations

1. The codebase uses Python 3.8+ features like union types (str|None)
2. Dependencies include: torch, numpy, pandas, cantera, tqdm
3. The 1D sampler requires OpenFOAM installation for CFD simulations
4. Chemical mechanisms are defined in YAML format using Cantera syntax