import matplotlib.pyplot as plt
import cantera as ct
import numpy as np

class DataVisualizer:
    @staticmethod
    def plot_loss_curve(train_loss, val_loss):
        plt.plot(train_loss, label='Train')
        plt.plot(val_loss, label='Validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def extract_flame_data(final_flame):
        collected_data = final_flame.collect_data(cols=['grid', 'T', 'Y', 'heat_release_rate'])
        z = collected_data['grid']
        T = collected_data['T']
        Y = collected_data['Y']
        heat_release_rate = collected_data['heat_release_rate']
        return z,T,Y,heat_release_rate

    def plot_flame_data(final_flame):
        z,T,Y,heat_release_rate=extract_flame_data(final_flame)
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), dpi=600)
        # Plot temperature and heat release rate if provided
        ax[0].plot(z, T, label='Temperature (K)')
        ax1 = ax[0].twinx()
        ax1.plot(z, heat_release_rate, label='Heat Release Rate [W/m³]', color='C1', linestyle='--')
        ax[0].set_xlabel('Position (m)')
        ax[0].set_ylabel('Temperature (K)')
        ax[0].set_title('Flame Temperature Profile')
        ax[0].grid()
        ax[0].legend()

        # Plot species mass fractions if provided
        for i, species in enumerate(final_flame.species_names):
            ax[1].plot(z, Y[:, i] / np.max(Y[:, i]), label=species)
        ax[1].set_xlabel('Position (m)')
        ax[1].set_ylabel('Normalized Mass Fraction')
        ax[1].set_title('Flame Species Mass Fractions')
        ax[1].grid()
        ax[1].legend()
        plt.tight_layout()
        plt.show()
        return ax, ax1

    def extract_counterflow_data(p,final_flame,tin_f,tin_o,mdot_o,mdot_f,comp_o,comp_f,width,loglevel):
        z,T,Y,heat_release_rate=extract_flame_data(final_flame)
        """参数设置
        p = ct.one_atm  # pressure
        tin_f = 300.0  # fuel inlet temperature
        tin_o = 300.0  # oxidizer inlet temperature
        mdot_o = 0.72  # kg/m^2/s
        mdot_f = 0.24  # kg/m^2/s
        comp_o = 'O2:0.21,N2:0.79'  # air composition
        comp_f = 'H2:1'  # fuel composition
        width = 0.02  # Distance between inlets is 2 cm
        loglevel = 0  # amount of diagnostic output (0 to 5)
        """
        gas = ct.Solution('Burke2012_s9r23.yaml')
        gas.TP = gas.T, p
        f = ct.CounterflowDiffusionFlame(gas, width=width)
        f.fuel_inlet.mdot = mdot_f
        f.fuel_inlet.X = comp_f
        f.fuel_inlet.T = tin_f
        f.oxidizer_inlet.mdot = mdot_o
        f.oxidizer_inlet.X = comp_o
        f.oxidizer_inlet.T = tin_o
        f.boundary_emissivities = 0.0, 0.0
        f.radiation_enabled = False
        f.set_refine_criteria(ratio=4, slope=0.2, curve=0.3, prune=0.04)
        f.solve(loglevel, auto=True)
        flame_result = f.to_solution_array()
        collected_data = flame_result.collect_data(cols=['grid', 'T', 'Y', 'heat_release_rate'])
        return collected_data,gas

    def plot_conterflow(p,final_flame,tin_f,tin_o,mdot_o,mdot_f,comp_o,comp_f,width,loglevel):
        collected_data,gas=extract_counterflow_data(p,final_flame,tin_f,tin_o,mdot_o,mdot_f,comp_o,comp_f,width,loglevel)
        fig, ax = plt.subplots(dpi=600)
        ax1 = ax.twinx()
        ax.plot(
        collected_data['grid'], 
        (collected_data['T'] - min(collected_data['T'])) / (max(collected_data['T']) - min(collected_data['T'])), 
        linestyle='--'
        )   
        for i, species in enumerate(gas.species_names):
            ax1.plot(
                collected_data['grid'], 
                collected_data['Y'][:, i],
                label=species,
            )
        ax.set_title('Temperature of the flame')
        # ax.set(ylim=(0,2500), xlim=(0.000, 0.020))
        ax1.legend()
        plt.show()
        return fig,ax,ax1