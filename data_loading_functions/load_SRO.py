import h5py
import numpy as np

def load_SRO_EOS(filepath, vars_to_load=None):
    """
    Loads the SRO EOS from a hdf5 file. The ranges are saved as [ye, log(T), log(rho)]
    :param filepath: (str)
    :param vars_to_load: None or list(stings) If None all are loaded. If a list of string, each var in the list is loaded
    :return: [list of dictionaries, domain]
    """
    EPS = 10 ** -12
    f_vars = [
        "Abar", "Albar", "Xa", "Xh", "Xl", "Xn", "Xp", "Zbar", "Zlbar", "cs2", 
        "dedt", "dpderho", "dpdrhoe", "entropy", "gamma", "logenergy", 
        "logpress", "meffn", "meffp", "mu_e", "mu_n", "mu_p", "muhat", 
        "munu", "r", "u"]
    domain = np.zeros([3, 2])
    with h5py.File(filepath, 'r') as file:
        # load and check ranges
        ye = file['ye'][...]
        logtemp = file['logtemp'][...]
        logrho = file['logrho'][...]
        # -- compute domain
        domain[0, 0] = ye[0]
        domain[1, 0] = logtemp[0]
        domain[2, 0] = logrho[0]
        domain[0, 1] = ye[-1]
        domain[1, 1] = logtemp[-1]
        domain[2, 1] = logrho[-1]
        # -- make sure it is evenly spaced
        assert (np.all(abs((ye[1:] - ye[:-1]) - (ye[1] - ye[0])) / abs(ye[1] - ye[0]) < EPS))
        assert (np.all(abs((logtemp[1:] - logtemp[:-1]) - (logtemp[1] - logtemp[0])) / abs(logtemp[1] - logtemp[0]) < EPS))
        assert (np.all(abs((logrho[1:] - logrho[:-1]) - (logrho[1] - logrho[0])) / abs(logrho[1] - logrho[0]) < EPS))

        # load data
        data = {}
        if vars_to_load is None:
            for key in f_vars:
                data[key] = {'f': file[key][...]}
        else:
            for key in vars_to_load:
                data[key] = {'f': file[key][...]}
    return data, domain


if __name__ == "__main__":
    dir_path = "/mnt/c/Users/jared/OneDrive - Michigan State University/Research/ANL/tables/"
    filename = "Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5"
    filepath = dir_path + filename

    data = load_SFHo_EOS(filepath)

    print("Done")
