"""
oneDflame_setup.py

This module provides functions for setting up 1D flame simulations in OpenFOAM.
It interfaces with Cantera for flame property calculations and prepares the
necessary OpenFOAM configuration files for CFD simulations.
"""

import cantera as ct
import numpy as np
import os
from pathlib import Path


def calculate_laminar_flame_properties(mechanism_path, gas_state):
    """
    Calculate laminar flame speed and thickness using Cantera.
    
    Args:
        mechanism_path (str): Path to the Cantera mechanism file
        gas_state (dict): Dictionary containing gas state parameters
            - initial_temperature: Initial temperature (K)
            - initial_pressure: Initial pressure (Pa)
            - fuel_composition: Fuel composition string
            - oxidizer_composition: Oxidizer composition string
            - equivalence_ratio: Equivalence ratio
    
    Returns:
        tuple: (flame_speed, flame_thickness, flame)
    """
    # Convert Path object to string if necessary
    if isinstance(mechanism_path, Path):
        mechanism_path = str(mechanism_path)
    
    # Validate inputs
    if not mechanism_path or not isinstance(mechanism_path, str):
        raise ValueError("Invalid mechanism_path: must be a non-empty string")
    
    if not isinstance(gas_state, dict):
        raise ValueError("Invalid gas_state: must be a dictionary")
    
    # Store original path for validation
    original_path = mechanism_path
    
    # Convert absolute path to relative path for Cantera compatibility
    if Path(mechanism_path).is_absolute():
        # Get the current working directory
        cwd = Path.cwd()
        # Convert to relative path
        try:
            relative_path = Path(mechanism_path).relative_to(cwd)
            mechanism_path = str(relative_path).replace('\\', '/')  # Ensure forward slashes
        except ValueError:
            # If not relative, try to extract just the filename and use mechanisms/ prefix
            filename = Path(mechanism_path).name
            mechanism_path = f"mechanisms/{filename}"
    
    # Check if mechanism file exists (after path conversion)
    if not Path(mechanism_path).exists():
        # If relative path doesn't exist, try the original absolute path
        if Path(original_path).exists():
            mechanism_path = original_path
        else:
            raise FileNotFoundError(f"Mechanism file not found: {mechanism_path} (also tried: {original_path})")
    
    # Validate required gas state parameters
    required_params = ['initial_temperature', 'initial_pressure', 'fuel_composition', 
                      'oxidizer_composition', 'equivalence_ratio']
    for param in required_params:
        if param not in gas_state:
            raise ValueError(f"Missing required gas_state parameter: {param}")
    
    try:
        # Create gas object
        gas = ct.Solution(mechanism_path)
        
        # Set initial state
        initial_temperature = gas_state.get('initial_temperature', 300)
        initial_pressure = gas_state.get('initial_pressure', 101325)
        fuel_composition = gas_state.get('fuel_composition', 'H2:1')
        oxidizer_composition = gas_state.get('oxidizer_composition', 'O2:1')
        equivalence_ratio = gas_state.get('equivalence_ratio', 1.0)
        
        # Validate parameter ranges
        if initial_temperature <= 0:
            raise ValueError("Initial temperature must be positive")
        if initial_pressure <= 0:
            raise ValueError("Initial pressure must be positive")
        if equivalence_ratio <= 0:
            raise ValueError("Equivalence ratio must be positive")
        
        gas.TP = initial_temperature, initial_pressure
        gas.set_equivalence_ratio(equivalence_ratio, fuel_composition, oxidizer_composition)
        
        # Create flame object
        flame = ct.FreeFlame(gas, width=0.01)
        flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1, prune=0.0)
        
        # Solve the flame
        flame.solve(loglevel=0, refine_grid=True)
        
        # Calculate flame properties
        flame_speed = flame.velocity[0]
        # 计算温度边界
        T_unburned = flame.T[0]   # 未燃温度
        T_burned = flame.T[-1]    # 已燃温度
        delta_T_total = T_burned - T_unburned
        # 计算火焰厚度（使用最大温度梯度的倒数）
        T_grad = np.gradient(flame.T, flame.grid)
        max_grad_idx = np.argmax(np.abs(T_grad))
        flame_thickness = delta_T_total / abs(T_grad[max_grad_idx]) if T_grad[max_grad_idx] != 0 else 1e-3
        
        print(f"Laminar Flame Speed      :   {flame_speed:.10f} m/s")
        print(f"Laminar Flame Thickness  :   {flame_thickness:.10f} m")
        
        return flame_speed, flame_thickness, flame
    
    except Exception as e:
        print(f"Error calculating flame properties: {str(e)}")
        raise


def update_case_parameters(mechanism_path, gas_state, flame_speed, flame_thickness):
    """
    Update case parameters for OpenFOAM simulation.
    
    Args:
        mechanism_path (str): Path to the Cantera mechanism file
        gas_state (dict): Gas state parameters
        flame_speed (float): Laminar flame speed
        flame_thickness (float): Flame thickness
    
    Returns:
        dict: Updated case parameters
    """
    # Validate inputs
    if not isinstance(mechanism_path, str) or not mechanism_path:
        raise ValueError("Invalid mechanism_path: must be a non-empty string")
    
    if not isinstance(gas_state, dict):
        raise ValueError("Invalid gas_state: must be a dictionary")
    
    if not isinstance(flame_speed, (int, float)) or flame_speed <= 0:
        raise ValueError("Invalid flame_speed: must be a positive number")
    
    if not isinstance(flame_thickness, (int, float)) or flame_thickness <= 0:
        raise ValueError("Invalid flame_thickness: must be a positive number")
    
    try:
        # Calculate domain parameters
        domain_width = flame_thickness * 5  # 5 times flame thickness
        domain_length = domain_width * 10   # 10 times domain width
        half_domain_length = domain_length / 2
        
        # Calculate time parameters
        target_time_step = 1e-6
        chemical_time_scale = flame_thickness / flame_speed
        sample_time_steps = 100
        estimated_time_step = chemical_time_scale / 10
        estimated_sim_time = estimated_time_step * sample_time_steps
        estimated_write_time_step = estimated_time_step
        
        # Create case parameters dictionary
        case_params = {
            'mechanism_path': mechanism_path,
            'gas_state': gas_state,
            'flame_speed': flame_speed,
            'flame_thickness': flame_thickness,
            'domain_width': domain_width,
            'domain_length': domain_length,
            'half_domain_length': half_domain_length,
            'target_time_step': target_time_step,
            'chemical_time_scale': chemical_time_scale,
            'sample_time_steps': sample_time_steps,
            'estimated_time_step': estimated_time_step,
            'estimated_sim_time': estimated_sim_time,
            'estimated_write_time_step': estimated_write_time_step
        }
        
        # Print parameters
        print(f"Flame Thickness: {flame_thickness:.2e}")
        print(f"Flame Speed: {flame_speed:.2e}")
        print(f"Domain Width: {domain_width:.2e}")
        print(f"Domain Length: {domain_length:.2e}")
        print(f"Half Domain Length: {half_domain_length:.2e}")
        print(f"Target Time Step: {target_time_step:.2e}")
        print(f"Chemical Time Scale: {chemical_time_scale:.2e}")
        print(f"Sample Time Steps: {sample_time_steps:.2e}")
        print(f"Estimated Time Step: {estimated_time_step:.2e}")
        print(f"Estimated Sim Time: {estimated_sim_time:.2e}")
        print(f"Estimated Write Time Step: {estimated_write_time_step:.2e}")
        
        return case_params
    
    except Exception as e:
        print(f"Error updating case parameters: {str(e)}")
        raise


def update_one_d_sample_config(case_params, gas_state):
    """
    Update 1D sample configuration file (system/controlDict).
    
    Args:
        case_params (dict): Case parameters
        gas_state (dict): Gas state parameters
    """
    # Validate inputs
    if not isinstance(case_params, dict):
        raise ValueError("Invalid case_params: must be a dictionary")
    
    if not isinstance(gas_state, dict):
        raise ValueError("Invalid gas_state: must be a dictionary")
    
    try:
        # Create system directory if it doesn't exist
        system_dir = Path("system")
        system_dir.mkdir(exist_ok=True)
        
        # Generate controlDict content
        control_dict_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     chemFoam;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         {case_params['estimated_sim_time']:.2e};

deltaT          {case_params['estimated_time_step']:.2e};

writeControl    timeStep;

writeInterval   {case_params['sample_time_steps']};

purgeWrite      0;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

adjustTimeStep  no;

maxCo           1;

maxDeltaT       1e-4;

functions
{{
}};

// ************************************************************************* //
"""
        
        # Write controlDict file
        control_dict_path = system_dir / "controlDict"
        with open(control_dict_path, "w") as f:
            f.write(control_dict_content)
        
        print("SUCCESS Updated 1D sample configuration (system/controlDict)")
    
    except Exception as e:
        print(f"Error updating 1D sample configuration: {str(e)}")
        raise


def create_0_species_files(case_params):
    """
    Create initial species files in the 0 directory.
    
    Args:
        case_params (dict): Case parameters
    """
    # Validate inputs
    if not isinstance(case_params, dict):
        raise ValueError("Invalid case_params: must be a dictionary")
    
    try:
        # Create 0 directory if it doesn't exist
        zero_dir = Path("oneDFlame/0")
        zero_dir.mkdir(exist_ok=True)
        
        # Extract parameters
        mechanism_path = case_params['mechanism_path']
        gas_state = case_params['gas_state']
        
        # Validate mechanism path
        if not Path(mechanism_path).exists():
            raise FileNotFoundError(f"Mechanism file not found: {mechanism_path}")
        
        # Create gas object to get species names
        gas = ct.Solution(mechanism_path)
        
        # Create temperature field file (0/T)
        t_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      T;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 1 0 0 0];

internalField   uniform {gas_state.get('initial_temperature', 300)};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {gas_state.get('initial_temperature', 300)};
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    walls
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
        
        with open(zero_dir / "T", "w") as f:
            f.write(t_content)
        
        # Create pressure field file (0/p)
        p_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform {gas_state.get('initial_pressure', 101325)};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {gas_state.get('initial_pressure', 101325)};
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    walls
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
        
        with open(zero_dir / "p", "w") as f:
            f.write(p_content)
        
        # Create species field files (0/Yi)
        initial_temperature = gas_state.get('initial_temperature', 300)
        initial_pressure = gas_state.get('initial_pressure', 101325)
        fuel_composition = gas_state.get('fuel_composition', 'H2:1')
        oxidizer_composition = gas_state.get('oxidizer_composition', 'O2:1')
        equivalence_ratio = gas_state.get('equivalence_ratio', 1.0)
        
        # Set gas state for species calculations
        gas.TP = initial_temperature, initial_pressure
        gas.set_equivalence_ratio(equivalence_ratio, fuel_composition, oxidizer_composition)
        
        # Get mass fractions
        mass_fractions = gas.Y
        
        for i, species in enumerate(gas.species_names):
            yi_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      Y{species};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   uniform {mass_fractions[i]:.6e};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {mass_fractions[i]:.6e};
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    walls
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
            
            with open(zero_dir / f"Y{species}", "w") as f:
                f.write(yi_content)
        
        print("SUCCESS Created 0/ species files")
    
    except Exception as e:
        print(f"Error creating 0/ species files: {str(e)}")
        raise


def update_set_fields_dict(case_params):
    """
    Update setFieldsDict for initial condition setup.
    
    Args:
        case_params (dict): Case parameters
    """
    # Validate inputs
    if not isinstance(case_params, dict):
        raise ValueError("Invalid case_params: must be a dictionary")
    
    try:
        # Create system directory if it doesn't exist
        system_dir = Path("oneDFlame/system")
        system_dir.mkdir(exist_ok=True)
        
        # Extract parameters
        domain_length = case_params['domain_length']
        half_domain_length = case_params['half_domain_length']
        flame_thickness = case_params['flame_thickness']
        flame_speed = case_params['flame_speed']
        mechanism_path = case_params['mechanism_path']
        
        # Validate mechanism path
        if not Path(mechanism_path).exists():
            raise FileNotFoundError(f"Mechanism file not found: {mechanism_path}")
        
        # Create gas object to get species names
        gas = ct.Solution(mechanism_path)
        
        # Generate setFieldsDict content
        set_fields_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      setFieldsDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Set values on a selected portion of the domain
defaultFieldValues
(
    volScalarFieldValue T 300
    volScalarFieldValue p 101325
"""

        # Add species field values
        gas_state = case_params['gas_state']
        initial_temperature = gas_state.get('initial_temperature', 300)
        initial_pressure = gas_state.get('initial_pressure', 101325)
        fuel_composition = gas_state.get('fuel_composition', 'H2:1')
        oxidizer_composition = gas_state.get('oxidizer_composition', 'O2:1')
        equivalence_ratio = gas_state.get('equivalence_ratio', 1.0)
        
        # Set gas state for species calculations
        gas.TP = initial_temperature, initial_pressure
        gas.set_equivalence_ratio(equivalence_ratio, fuel_composition, oxidizer_composition)
        
        # Get mass fractions for burned gas
        mass_fractions = gas.Y
        
        for i, species in enumerate(gas.species_names):
            set_fields_content += f"    volScalarFieldValue Y{species} {mass_fractions[i]:.6e}\n"
        
        set_fields_content += """);

regions
(
    boxToCell
    {{
        box (0 0 0) ({half_domain_length:.6e} 0.1 0.1);
        fieldValues
        (
            volScalarFieldValue T {initial_temperature + 1000}  // Higher temperature in reaction zone
        );
    }}

    boxToCell
    {{
        box ({half_domain_length:.6e} 0 0) ({domain_length:.6e} 0.1 0.1);
        fieldValues
        (
            volScalarFieldValue T {initial_temperature}  // Unburned gas temperature
        );
    }}
);

// ************************************************************************* //
"""
        
        # Write setFieldsDict file
        set_fields_path = system_dir / "setFieldsDict"
        with open(set_fields_path, "w") as f:
            f.write(set_fields_content)
        
        print("SUCCESS Updated setFieldsDict")
    
    except Exception as e:
        print(f"Error updating setFieldsDict: {str(e)}")
        raise


def update_cantera_mechanism(mechanism_path):
    """
    Update Cantera mechanism file path in OpenFOAM configuration.
    
    Args:
        mechanism_path (str): Path to the Cantera mechanism file
    """
    # Validate inputs
    if not mechanism_path or not isinstance(mechanism_path, str):
        raise ValueError("Invalid mechanism_path: must be a non-empty string")
    
    # Check if mechanism file exists
    if not Path(mechanism_path).exists():
        raise FileNotFoundError(f"Mechanism file not found: {mechanism_path}")
    
    try:
        # Create constant directory if it doesn't exist
        constant_dir = Path("oneDFlame/constant")
        constant_dir.mkdir(exist_ok=True)
        
        # Generate CanteraMechanismFile content
        cantera_mech_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      CanteraMechanismFile;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Path to Cantera mechanism file
mechanismFile    "{mechanism_path}";

// ************************************************************************* //
"""
        
        # Write CanteraMechanismFile
        cantera_mech_path = constant_dir / "CanteraMechanismFile"
        with open(cantera_mech_path, "w") as f:
            f.write(cantera_mech_content)
        
        print(f"SUCCESS Updated CanteraMechanismFile to {mechanism_path}")
    
    except Exception as e:
        print(f"Error updating CanteraMechanismFile: {str(e)}")
        raise


# Example usage
if __name__ == "__main__":
    try:
        # Example configuration
        mechanism_path = "mechanisms/Burke2012_s9r23.yaml"
        gas_state = {
            "initial_temperature": 300,
            "initial_pressure": 101325,
            "fuel_composition": "H2:1",
            "oxidizer_composition": "O2:0.21,N2:0.79",
            "equivalence_ratio": 1.0
        }
        
        print("Starting 1D flame setup...")
        
        # Calculate flame properties
        print("1. Calculating laminar flame properties...")
        flame_speed, flame_thickness, flame = calculate_laminar_flame_properties(
            mechanism_path, gas_state
        )
        
        # Update case parameters
        print("2. Updating case parameters...")
        case_params = update_case_parameters(
            mechanism_path, gas_state, flame_speed, flame_thickness
        )
        
        # Update configuration files
        print("3. Creating OpenFOAM configuration files...")
        update_one_d_sample_config(case_params, gas_state)
        create_0_species_files(case_params)
        update_set_fields_dict(case_params)
        update_cantera_mechanism(mechanism_path)
        
        print("SUCCESS 1D flame setup completed successfully!")
        
    except Exception as e:
        print(f"ERROR in 1D flame setup: {str(e)}")
        raise
