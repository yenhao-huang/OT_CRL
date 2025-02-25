import numpy as np
from scipy.interpolate import interp1d

def build_error_model():
        # (ref) x1 unit: tck
        x1_values = np.array([4680, 9360, 23400, 46800, 93600, 234000, 468000])
        x2_values = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
        ref_function = interp1d(x1_values, x2_values, kind='linear', fill_value="extrapolate")
        
        # (rcd) x1 unit: ns
        x1_values = np.array([2.5, 5, 7.5, 10, 12.5])
        x2_values = np.array([2e-1, 1e-2, 1e-3, 1e-9, 0])
        rcd_function = interp1d(x1_values, x2_values, kind='linear', fill_value="extrapolate")
        
        # (rp) x1 unit: ns
        x1_values = np.array([5, 7.5, 10, 12.5])
        x2_values = np.array([0.43, 1e-3, 1e-7, 0])
        rp_function = interp1d(x1_values, x2_values, kind='linear', fill_value="extrapolate")
        
        # (ras) 1 unit: ns
        x1_values = np.array([20, 22, 25, 27.5, 30, 35])
        x2_values = np.array([1e-6, 6e-7, 3e-7, 3e-7, 1e-7, 1e-7])
        ras_function = interp1d(x1_values, x2_values, kind='linear', fill_value="extrapolate")
        
        def error_model(ref_interval, rcd, rp, ras):
            ref_err = np.minimum(np.maximum(ref_function(ref_interval), 0), 1)
            rcd_err = np.minimum(np.maximum(rcd_function(rcd), 0), 1)
            rp_err = np.minimum(np.maximum(rp_function(rp), 0), 1)
            ras_err = np.minimum(np.maximum(ras_function(ras), 0), 1)
            print(f"REF error: {ref_err}, RCD error: {rcd_err}, RP_error: {rp_err}, RAS_error: {ras_err}")
            return ref_err + rcd_err + rp_err + ras_err
        
        return error_model

def convert_to_model_format(default_cycle):
    refi = default_cycle[0]
    rcd = default_cycle[1] * tCK
    rp = default_cycle[2] * tCK
    ras = default_cycle[3] * tCK
    return refi, rcd, rp, ras


tCK = 0.83
n_row_per_bank = 32768
error_model = build_error_model()

# refi, rcd, rp, ras
default_cycle = [4680, 16, 16, 39]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
err = error_model(refi, rcd, rp, ras)
print("Default error rate: {}".format(err))

# RCD ERROR RATE RANGE
print("RCD...")
default_cycle = [4680, 12, 16, 39]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
min_err = error_model(refi, rcd, rp, ras)

default_cycle = [4680, 16, 16, 39]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
max_err = error_model(refi, rcd, rp, ras)
print("RCD MIN Error rate: {} Max Error rate: {}".format(max_err, min_err))

# RP ERROR RATE RANGE
print("RP...")
default_cycle = [4680, 16, 12, 39]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
min_err = error_model(refi, rcd, rp, ras)

default_cycle = [4680, 16, 16, 39]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
max_err = error_model(refi, rcd, rp, ras)
print("RP MIN Error rate: {} Max Error rate: {}".format(max_err, min_err))

# RAS ERROR RATE RANGE
print("RAS...")
default_cycle = [4680, 16, 16, 19]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
min_err = error_model(refi, rcd, rp, ras)

default_cycle = [4680, 16, 16, 39]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
max_err = error_model(refi, rcd, rp, ras)
print("RAS MIN Error rate: {} Max Error rate: {}".format(max_err, min_err))

# REFI ERROR RATE RANGE
print("REFI...")
default_cycle = [4680, 16, 16, 39]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
max_err = error_model(refi, rcd, rp, ras)

default_cycle = [18720, 16, 16, 39]
refi, rcd, rp, ras = convert_to_model_format(default_cycle)
min_err = error_model(refi, rcd, rp, ras)
print("REFI MIN Error rate: {} Max Error rate: {}".format(max_err, min_err))
