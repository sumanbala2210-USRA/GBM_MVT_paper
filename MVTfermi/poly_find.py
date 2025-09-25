from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import os
from scipy.signal import savgol_filter
import numpy as np
import scipy.stats
import yaml
# GDT packages
from gdt.core.binning.unbinned import bin_by_time 
from gdt.core.background.binned import Polynomial
from gdt.core.background.fitter import BackgroundFitter

#from gdt.core.spectra.parameters import *
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.plot.lightcurve import Lightcurve
from gdt.missions.fermi.gbm.tte import GbmTte

from scipy.signal import correlate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#data_interval = 6



import numpy as np
import scipy.stats

import numpy as np

def get_channel_indices(ebounds, energy_range=(8.0, 900.0)):
    """
    Finds the indices of channels that fall within a given energy range.

    Args:
        ebounds (np.ndarray): The ebounds array for a detector, structured as an
                              array of (e_min, e_max) tuples.
        energy_range (tuple, optional): The (min_energy, max_energy) in keV.
                                        Defaults to (8.0, 900.0).

    Returns:
        np.ndarray: An array of integer indices for the channels within the range.
    """
    # Convert the array of tuples into a 2D NumPy array for easier slicing
    ebounds_array = np.array(list(ebounds))
    
    # Get the minimum and maximum energy for each channel
    e_min = ebounds_array[:, 0]
    e_max = ebounds_array[:, 1]
    
    # Create a boolean mask to find channels where BOTH the min and max energy
    # are within the desired range.
    mask = (e_min >= energy_range[0]) & (e_max <= energy_range[1])
    
    # Use np.where() to get the indices where the mask is True
    channel_indices = np.where(mask)[0]
    
    return channel_indices



def perform_f_test(chi2_simple, dof_simple, chi2_complex, dof_complex, alpha=0.05):
    """
    Performs an F-test to compare two nested models. Returns True if complex is better.
    """
    # F-test is not valid if the more complex model has a worse fit
    if chi2_complex >= chi2_simple:
        return False
        
    f_statistic = ((chi2_simple - chi2_complex) / (dof_simple - dof_complex)) / (chi2_complex / dof_complex)
    df1 = dof_simple - dof_complex
    df2 = dof_complex
    critical_value = scipy.stats.f.ppf(1 - alpha, df1, df2)
    
    return f_statistic > critical_value


def find_best_poly_order(backfitter, energy_range=None, det='n1', max_order=4, outpath='./'):
    """
    Finds the best polynomial order for a background fit using an F-test.

    Args:
        backfitter (BackgroundFitter): The gdt background fitter object.
        phaii_data (Phaii): The PHaii data object, used to get energy bounds.
        energy_range (tuple, optional): The (min, max) energy in keV to consider.
        max_order (int, optional): The maximum polynomial order to test. Defaults to 4.

    Returns:
        int: The best polynomial order found.
    """
    print(f"Finding best polynomial order up to {max_order}...")
    
    # Find the channel indices corresponding to the desired energy range
    # This assumes a single detector's ebounds are representative
    if energy_range is None:
         if det[0]=='n':
              energy_range = (8.0, 900.0)
         if det[0]=='b':
              energy_range = (300.0, 40000.0)

    data_all = np.load('gbm_detectors_ebounds.npz', allow_pickle=True)
    data = data_all['gbm_detectors_ebounds']
    ebounds_for_detector = data[data['detector'] == det]['ebounds'][0]
    indices = get_channel_indices(ebounds_for_detector, energy_range=energy_range)

    previous_results = {}
    best_order = 1

    for order in range(1, max_order + 1):
        print(f"\n--- Testing Order {order} ---")
        backfitter.fit(order=order)
        
        # We only care about the chi-squared in our energy range of interest
        chi2_full = backfitter.statistic
    
        #chi_2 = backfitter.statistic/backfitter.dof
        chi2_selected = np.sum(chi2_full[indices])

        
        # DOF = (Number of data points) - (Number of model parameters)
        # Here, number of data points = num_channels
        dof_selected = len(indices) - (order + 1)

        plt.plot(range(len(chi2_full)), chi2_full/backfitter.dof, label=f'Order {order}')
        plt.plot(range(len(chi2_full[indices])), chi2_full[indices]/dof_selected, label=f'Selected')
        plt.hlines(1, xmin=0, xmax=len(chi2_full[indices]), colors='r', linestyles='--')
        plt.xlabel('Channel')
        plt.ylabel('Chi_2/D.O.F')
        plt.legend()
        plt.savefig(f"{outpath}bkgd_fit_{order}.png")
        plt.close()

        if dof_selected <= 0:
            print("Not enough degrees of freedom to continue. Stopping.")
            break

        print(f"Chi-squared (in range): {chi2_selected:.2f}")
        print(f"DOF (in range): {dof_selected}")
        
        current_results = {
            'chi2': chi2_selected,
            'dof': dof_selected,
            'order': order
        }
        
        # For order 1, it's our baseline best fit so far
        if order == 1:
            previous_results = current_results
            continue
            
        # From order 2 onwards, perform F-test against the previous order
        is_better = perform_f_test(
            chi2_simple=previous_results['chi2'],
            dof_simple=previous_results['dof'],
            chi2_complex=current_results['chi2'],
            dof_complex=current_results['dof']
        )
        
        if is_better:
            print(f"Conclusion: Order {order} is a significant improvement over order {order-1}.")
            best_order = order
            previous_results = current_results
        else:
            print(f"Conclusion: Order {order} is NOT a significant improvement. Best order is {best_order}.")
            break # Stop testing as soon as the improvement is not significant
            
    return best_order



def main():
	data_interval = 1
	bw_low = 1.024

	#file_name = f'simulation_parameters{data_interval}.yaml'
	outpath = os.path.join(os.getcwd(), f'Poly_{data_interval}/')
	os.makedirs(outpath, exist_ok=True)


	t_start = 773154450
	t_stop = 773154560
	#bw_low = config['bin_width']
	det_list = ['n8']
	trange = [t_start-80, t_stop+80]
	src_interval = [t_start, t_stop]
	bkgd_intervals=[[t_start-65,t_start-10],[t_stop+10,t_stop+65]] 
	#view_range=(t_start-80,t_stop+80) 
	num = '13z'
	erange = (8,900)

	#tte_list_data = list(tte_list.values())

	tte_dict_m = []
	for i in det_list:
		#tte_time_selection = GbmTte.open(f'glg_tte_{i}_250702_{num}_v00.fit').slice_time(trange)
		tte_time_selection = GbmTte.open('glg_tte_n8_250702_13z_v00.fit').slice_time(trange)
		tte_dict_m.append(tte_time_selection)
	#final_tte = GbmTte.merge(tte_dict)
	final_tte = GbmTte.merge(tte_dict_m)

	bw = 1.024
	#tte_all = final_tte.slice_energy((10, 800))
	phai_all = final_tte.to_phaii(bin_by_time, bw)
	lc_tot = phai_all.to_lightcurve(energy_range=erange)
	lcplot_in = Lightcurve(data=lc_tot)
	plt.vlines(src_interval, ymin=0, ymax=max(lc_tot.counts)*1.2, colors='green', label='Source Interval')

	#plt.show()
	plt.close()

	src_lc = phai_all.to_lightcurve(time_range=src_interval, energy_range=erange)

	bkg_lc_plot1 = phai_all.to_lightcurve(time_range=bkgd_intervals[0], energy_range=erange)
	bkg_lc_plot2 = phai_all.to_lightcurve(time_range=bkgd_intervals[1], energy_range=erange)

	backfitter = BackgroundFitter.from_phaii(phai_all, Polynomial, time_ranges=bkgd_intervals)
	backfitter.fit(order=1)
	chi_2 = backfitter.statistic/backfitter.dof
	best_order = find_best_poly_order(backfitter, energy_range=None, det='n1', max_order=4, outpath=outpath)
	backfitter.fit(order=best_order)

	bkgd_fit = backfitter.interpolate_bins(phai_all.data.tstart, phai_all.data.tstop)
	bkgd_fit_lc = bkgd_fit.integrate_energy(*erange)
	lcplot = Lightcurve(data=lc_tot, background=bkgd_fit_lc)
	lcplot.add_selection(src_lc)
	lcplot.add_selection(bkg_lc_plot1)
	lcplot.add_selection(bkg_lc_plot2)
	#lcplot.color = 'y'
	lcplot.selections[0].color = 'green'
	lcplot.selections[1].color = 'pink'
	lcplot.selections[2].color = 'pink'
	plt.savefig(f"{outpath}selection_{data_interval}.png")


	
if __name__ == '__main__':
    main()
