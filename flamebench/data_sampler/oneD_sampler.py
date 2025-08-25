import os
import subprocess
import yaml
import numpy as np
from pathlib import Path
from cantera import Solution

import sys
# 添加 FlameBench 根目录到 sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.utils import get_path_from_root, is_numeric_string
import flamebench.data_sampler.oneDflame_setup as odf

class OneDSampler:
    def __init__(self, config_path=None, verbose=True):
        self.verbose = verbose
        self.project_root = get_path_from_root()
        self.working_dir = get_path_from_root("oneDFlame")
        self.config_path = config_path or get_path_from_root("config", "1d_config.yaml")
        self.data = None
        self._load_config()

    def _log(self, message):
        if self.verbose:
            print(message)

    def _load_config(self):
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.fuel = self.config.get("fuel", "unknown")
        
        mechanism_filename = self.config.get("mechanism")
        self.mechanism_path = get_path_from_root("mechanisms", mechanism_filename)
        
        self.gas_state = self.config.get("gas_state", {})

        self._log(f"Loaded config for fuel: {self.fuel}")

    def sample(self):
        self._log("\n Starting sampling pipeline...")
        self._run_case_setup()
        self._collect_data()
        self._log("Sampling completed.")

    def _run_case_setup(self):
        # Store original working directory
        # original_cwd = Path.cwd()
        
        # Change to working directory for OpenFOAM operations
        os.chdir(self.project_root)

        self._log("Calculating steady flame properties with Cantera...")
        
        # Convert absolute path to relative path for Cantera compatibility
        # Since we changed working directory, we need to handle path conversion carefully
        mechanism_path_for_cantera = self.mechanism_path
        if Path(self.mechanism_path).is_absolute():
            # Try to convert to relative path from the original directory
            try:
                # First, go back to original directory to test relative path
                relative_path = Path(self.mechanism_path).relative_to(Path.cwd())
                mechanism_path_for_cantera = str(relative_path).replace('\\', '/')
                self._log(f"Converted to relative path for Cantera: {mechanism_path_for_cantera}")
            except ValueError:
                # If not relative, use mechanisms/ prefix and copy file if needed
                filename = Path(self.mechanism_path).name
                mechanism_path_for_cantera = f"mechanisms/{filename}"
                self._log(f"Using mechanisms/ prefix for Cantera: {mechanism_path_for_cantera}")
                
                # Copy mechanism file to local mechanisms directory if it doesn't exist
                mechanisms_dir = Path("mechanisms")
                mechanisms_dir.mkdir(exist_ok=True)
                target_file = mechanisms_dir / filename
                if not target_file.exists():
                    import shutil
                    shutil.copy2(self.mechanism_path, target_file)
                    self._log(f"Copied mechanism file to: {target_file}")
            
            # Change back to working directory
        self.mechanism_path_for_cantera = mechanism_path_for_cantera

        flame_speed, flame_thickness, _ = odf.calculate_laminar_flame_properties(
            self.mechanism_path_for_cantera, self.gas_state
        )

        case_params = odf.update_case_parameters(
            self.mechanism_path_for_cantera, self.gas_state, flame_speed, flame_thickness
        )

        odf.update_one_d_sample_config(case_params, self.gas_state)
        odf.create_0_species_files(case_params)
        odf.update_set_fields_dict(case_params)
        odf.update_cantera_mechanism(self.mechanism_path_for_cantera)

        os.chdir(self.working_dir)

        self._log("⚙️ Running Allrun script...")
        subprocess.run(["chmod", "+x", "Allrun"], check=True)
        #subprocess.run(["./Allrun"], check=True)
        try:
            subprocess.run(["./Allrun"], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self._log("Error running Allrun!")
            self._log(f"Return code: {e.returncode}")
            self._log(f"STDOUT:\n{e.stdout}")
            self._log(f"STDERR:\n{e.stderr}")
            raise


        self._log("Running reconstructPar...")
        temp_gas = Solution(self.mechanism_path_for_cantera)
        fields_list = ['T', 'p'] + temp_gas.species_names
        fields_str = "(" + " ".join(fields_list) + ")"
        #subprocess.run(["reconstructPar", "-fields", fields_str], check=True)
        subprocess.run(
            ["reconstructPar", "-fields", fields_str],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def _collect_data(self):
        self._log("Collecting flame data from time directories...")

        sample_dims = ['T', 'p'] + Solution(self.mechanism_path_for_cantera).species_names
        time_dirs = sorted([
            d for d in Path('.').iterdir()
            if d.is_dir() and is_numeric_string(d.name)
        ], key=lambda d: float(d.name))[1:]  # skip time 0

        data_collector = []
        data_shape = 0

        for time_dir in time_dirs:
            time_arrays = []
            for var in sample_dims:
                file_path = time_dir / var
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                sample_started = False
                uniform_bool = False

                for i, line in enumerate(lines):
                    if "internalField" in line:
                        sample_started = True
                    if " uniform" in line and sample_started:
                        uniform_value = float(line.strip().split()[-1][:-1])
                        uniform_bool = True
                        break
                    if "(" in line and sample_started:
                        start = i + 1
                    if ")" in line and sample_started:
                        end = i
                        break

                if uniform_bool:
                    dim_array = np.ones((data_shape, 1)) * uniform_value
                else:
                    dim_array = np.loadtxt(lines[start:end]).reshape(-1, 1)
                    data_shape = dim_array.shape[0]

                time_arrays.append(dim_array)

            time_array = np.concatenate(time_arrays, axis=1)
            data_collector.append(time_array)

        self.data = np.concatenate(data_collector, axis=0)
        self._log(f"Collected data with shape: {self.data.shape}")

    def get_data(self):
        if self.data is None:
            raise ValueError("Data not sampled yet. Call sample() first.")
        return self.data

    def save(self, output_dir=None):
        if self.data is None:
            raise ValueError("No data to save. Run `sample()` first.")

        output_dir = output_dir or get_path_from_root("1DFlameRawData")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.fuel}-1Dflame.npy"
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, self.data)
        self._log(f"Saved data to {output_path}")

    def clean(self):
        self._log("Cleaning up with Allclean...")
        subprocess.run(["./Allclean"], check=True)
        self._log("Allclean completed.")

if __name__ == "__main__":
    sampler = OneDSampler()
    sampler.sample()