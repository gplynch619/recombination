import os
from datetime import datetime
import numpy as np
import pickle
import classy as Classy
from IPython.display import clear_output
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import scipy.integrate
import scipy
import matplotlib.patches
import time

def square(f):
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)**2
    return wrapper

def renormalize(norm):
    def dec(f):
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)*norm
        return wrapper
    return dec

class FisherCalculator:
    """ A class to compute a recombination Fisher matrix.
    """
    def __init__(self, cosmo_parameters, fisher_settings, name):
        self.cosmo_params = cosmo_parameters
        self.fisher_settings = fisher_settings

        self.is_debug = False
        if "debug" in self.fisher_settings.keys():
            self.is_debug = self.fisher_settings["debug"]

        self.save_output = True
        if "save_output" in self.fisher_settings.keys():
            self.save_output = self.fisher_settings["save_output"]

        if self.save_output:
            self.basename = name
            self.filebase = self.create_outdir()

        self.set_parameters()

        self.perturbations = []
        self.class_z_array = []

        self.BoltzmannSolver = Classy.Class()
        
        ### fiducials
        self.compute_fiducial_cosmology()

        self.reset_boltzmann()
        if self.save_output:
            self.write_log()

        self.dCl = []

        self.Fisher = FisherMatrix()

        self.error_covariance = []
        self.error_covariance_inv = []

        self.run_counter = 0
        if(self.xe_pert_type=="basis"):
            self.total_runs = self.pivots.shape[0]*5
        elif(self.xe_pert_type=="control"):
            self.total_runs = self.xe_control_pivots.shape[0]*5

    def reset_boltzmann(self):
        self.BoltzmannSolver.empty()
        self.BoltzmannSolver.set(self.class_default_settings)

    def compute_fiducial_cosmology(self):
        self.BoltzmannSolver.empty()
        self.BoltzmannSolver.set(self.class_fiducial_settings)
        self.BoltzmannSolver.compute() ##Run CLASS
        Tcmb = self.BoltzmannSolver.T_cmb()*10**6 #cmb temp in micro kelvin
        self.muK2 = (Tcmb)**2
        self.tt_fid = self.muK2*self.BoltzmannSolver.lensed_cl(self.ll_max)['tt'][2:]
        self.te_fid = self.muK2*self.BoltzmannSolver.lensed_cl(self.ll_max)['te'][2:]
        self.ee_fid = self.muK2*self.BoltzmannSolver.lensed_cl(self.ll_max)['ee'][2:]
        Yp = self.BoltzmannSolver.get_current_derived_parameters(['YHe'])['YHe']
        self.xe_max = 1 + Yp/2*(1-Yp)
        self.xe_fid = interp1d(self.BoltzmannSolver.get_thermodynamics()["z"], self.BoltzmannSolver.get_thermodynamics()["x_e"])
    
    def partial_derivative(self, parameter_name, list_of_values, spacing):
        
        tt_array=[]
        te_array=[]
        ee_array=[]

        for trial in list_of_values:
            self.BoltzmannSolver.set({parameter_name: trial}) #sets perturbation
            self.BoltzmannSolver.compute()
            cls = self.BoltzmannSolver.lensed_cl(self.ll_max)
            tt = cls['tt'][2:]
            ee = cls['ee'][2:]
            te = cls['te'][2:]
            tt[:(self.ll_min_tt-2)]=[0]*(self.ll_min_tt-2)
            ee[:(self.ll_min_pol-2)]=[0]*(self.ll_min_pol-2)
            te[:(self.ll_min_pol-2)]=[0]*(self.ll_min_pol-2)
            tt_array.append(tt)
            te_array.append(te)
            ee_array.append(ee)
        
        tt_array = np.vstack(tt_array)
        te_array = np.vstack(te_array)
        ee_array = np.vstack(ee_array)
        tt_grad = np.gradient(tt_array, spacing, axis=0)[2]
        te_grad = np.gradient(te_array, spacing, axis=0)[2]
        ee_grad = np.gradient(ee_array, spacing, axis=0)[2]

        return [tt_grad, ee_grad, te_grad]

    def compute_Fisher(self, target_parameters):
        """ Main routine which computes the full Fisher matrix for this experiment. 
        """
        print("Computing Fisher matrix for variables: {}".format(target_parameters))
        tt_derivs = []
        te_derivs = []
        ee_derivs = []

        col_names = []

        for param in target_parameters:
            if param=="xe_control_points":
                counter = 1
                for redshift in self.xe_control_pivots[1:-1]: #first and last pivot anchor DeltaX to 0 at the endpoints of perturbation
                    # first get the index we are perturbing:
                    cp_name = "q_{}".format(counter, redshift)
                    print(cp_name)
                    #index_of_point=0
                    formatted_cp_strings = []
                    dp = self.delta_param[cp_name]
                    cp_values = np.linspace(-2*dp,2*dp,5)
                    #zz = list(np.full(5, redshift))
                    #cp_values_rescaled = self.rescale_pert_amp(cp_values, zz)
                    #for i,val in enumerate(self.rec_params["xe_control_pivots"].split(",")):
                    #    if float(val)==redshift:
                    #        index_of_point=i
                    temp_cp_array = np.zeros(self.xe_control_pivots.shape)
                    for p in cp_values:
                        temp_cp_array[counter] = p
                        temp_cp_array_str = ["{:.2f}".format(q) for q in temp_cp_array]
                        formatted_cp_strings.append(",".join(temp_cp_array_str))
                    self.reset_boltzmann() #resets boltzmann solver to defaults
                    cmb_response = self.partial_derivative(param, formatted_cp_strings, dp)
                    tt_derivs.append(cmb_response[0])
                    ee_derivs.append(cmb_response[1])
                    te_derivs.append(cmb_response[2])
                    col_names.append(cp_name)
                    counter+=1
            elif param=="xe_pert_amps":
                for z in self.pivots:
                    self.reset_boltzmann()
                    cmb_response = self.compute_cmb_perturbation_response(z)
                    tt_derivs.append(cmb_response[0])
                    ee_derivs.append(cmb_response[1])
                    te_derivs.append(cmb_response[2])
            else:
                print(param)
                self.reset_boltzmann()
                dp = self.delta_param[param]
                trial_values = np.linspace(-2*dp, 2*dp, 5) + self.cosmo_params[param]
                cmb_response = self.partial_derivative(param, trial_values, dp)
                tt_derivs.append(cmb_response[0])
                ee_derivs.append(cmb_response[1])
                te_derivs.append(cmb_response[2])
                col_names.append(param)

        tt_derivs = np.vstack(tt_derivs)    
        te_derivs = np.vstack(te_derivs)    
        ee_derivs = np.vstack(ee_derivs) 
        
        if self.save_output:
            np.savez(os.path.join(self.filebase,"tt_derivs"), data=tt_derivs, comment='')
            np.savez(os.path.join(self.filebase,"te_derivs"), data=te_derivs, comment='')
            np.savez(os.path.join(self.filebase,"ee_derivs"), data=ee_derivs, comment='')
            np.savez(os.path.join(self.filebase,"z"), self.BoltzmannSolver.get_thermodynamics()['z'])

        self.reset_boltzmann()

        self.dCl = self.muK2*np.stack([tt_derivs, ee_derivs, te_derivs], axis=1)

        ##at this point we have the derivatives

        self.error_covariance = self.compute_error_covariance()

        self.error_covariance_inv = np.linalg.inv(self.error_covariance)

        self.Fisher.from_array(np.einsum("iXl,lXY,jYl->ij", self.dCl, self.error_covariance_inv, self.dCl), col_names, self.class_default_settings, self.fisher_settings)

        if self.save_output:
            np.savez(os.path.join(self.filebase,"Fisher_full"), data=self.Fisher.Fisher, comment=col_names)


        #standard_block = self.Fisher[:divider, :divider]
        #cross_block = self.Fisher[:divider, divider:]
        #perturbation_block = self.Fisher[divider:, divider:]

        #self.Fisher_marginalized = perturbation_block - np.einsum("ai,ij,jk->ak", cross_block.T, np.linalg.inv(standard_block), cross_block)
        #self.Fisher_fixed = perturbation_block

        #standard_cols = target_parameters[:divider]
  
        #full_cols = "{},{}".format(",".join(["{}".format(s) for s in standard_cols]), extra_cols)
        #np.savez(os.path.join(self.filebase,"Fisher_full"), data=self.Fisher, comment=full_cols)
        #np.savez(os.path.join(self.filebase,"Fisher_marginalized"), data=self.Fisher_marginalized, comment = extra_cols)
        #np.savez(os.path.join(self.filebase,"Fisher_fixed"), data=self.Fisher_fixed, comment=extra_cols)

        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

        print("Fisher matrices calculation completed at {}".format(date_time))

        return self.Fisher



    def rescale_pert_amp(self, amp, zz):
        @np.vectorize
        def inner_func(amp1, zz1): 
            shift = scipy.optimize.fsolve(lambda aa: scipy.special.expit(aa)*self.xe_max - self.xe_fid(zz1), 0)
            return scipy.special.expit(amp1+shift)*self.xe_max - self.xe_fid(zz1)
        return inner_func(amp, zz)

    def compute_error_covariance(self):
        arcmin_to_radians = 1./3438
        T_channels = []
        P_channels = []
        if self.noise_parameters['use_143']:
            #convert from arcmin -> radians, then divide by 2.355 to get sigma from FWHM
            beam_width_143 = (self.noise_parameters['beam_FWHM_143_arcmin']*arcmin_to_radians)/np.sqrt(8*np.log(2))
            B2_l = np.exp(-self.ll*(self.ll+1)*beam_width_143**2)
            T_channels.append(self.noise_parameters['weight_inv_T_143']/B2_l)
            P_channels.append(self.noise_parameters['weight_inv_P_143']/B2_l)
        if self.noise_parameters['use_217']:
            beam_width_143 = (self.noise_parameters['beam_FWHM_217_arcmin']*arcmin_to_radians)/np.sqrt(8*np.log(2))
            B2_l = np.exp(-self.ll*(self.ll+1)*beam_width_143**2)
            T_channels.append(self.noise_parameters['weight_inv_T_217']/B2_l)
            P_channels.append(self.noise_parameters['weight_inv_P_217']/B2_l)
    
        if not T_channels:
            Nl_T = 0.0
        else:
            Nl_T = (np.sum(np.reciprocal(T_channels), axis=0))**(-1.)

        if not P_channels:
            Nl_P = 0.0
        else:
            Nl_P = (np.sum(np.reciprocal(P_channels), axis=0))**(-1.)

        row1 = np.stack([(self.tt_fid+Nl_T)**2, self.te_fid**2, (self.tt_fid+Nl_T)*self.te_fid], axis=1)
        row2 = np.stack([self.te_fid**2, (self.ee_fid+Nl_P)**2, self.te_fid*(self.ee_fid+Nl_P)], axis=1)
        row3 = np.stack([(self.tt_fid+Nl_T)*self.te_fid, self.te_fid*(self.ee_fid+Nl_P), 0.5*(self.te_fid**2 + (self.tt_fid+Nl_T)*(self.ee_fid+Nl_P))], axis=1)

        sigma = np.stack([row1, row2, row3], axis=1)
        for i, ell in enumerate(self.ll):
            sigma[i]*=(2./(2*ell + 1))/self.noise_parameters['fsky']

        return sigma

    def compute_cmb_perturbation_response(self, z):
        """ Computes the CMB response to a perturbation at redshift z.
            Perturbation amplitudes are defined by the self.amplitudes
            attribute, which is an array of amplitudes that can either 
            be passed as part of rec_params, or set to default values.
        """
        self.BoltzmannSolver.set({'xe_single_zi': z})
        amplitudes = np.linspace(-.1, .1, 5)
        tt_array = []
        te_array = []
        ee_array = []

        for qi in amplitudes:
            start = time.time()
            amp_str = "{}".format(qi)        #Setting up string of perturbation amplitudes
            self.BoltzmannSolver.set({'xe_pert_amps': amp_str}) #sets perturbation
            self.BoltzmannSolver.compute()
            cls = self.BoltzmannSolver.lensed_cl(self.ll_max)
            tt = cls['tt'][2:]
            ee = cls['ee'][2:]
            te = cls['te'][2:]
            tt[:(self.ll_min_tt-2)]=[0]*(self.ll_min_tt-2)
            ee[:(self.ll_min_pol-2)]=[0]*(self.ll_min_pol-2)
            te[:(self.ll_min_pol-2)]=[0]*(self.ll_min_pol-2)
            tt_array.append(tt)
            te_array.append(te)
            ee_array.append(ee)
            end = time.time()
            self.run_counter+=1
            clear_output(wait=True)
            print("Run {}/{} took {:.2f} seconds".format(self.run_counter, self.total_runs, end-start))
            if(qi==self.amplitudes[-1]):
                self.perturbations.append(self.BoltzmannSolver.get_thermodynamics()['xe_pert'])

        tt_array = np.vstack(tt_array)
        te_array = np.vstack(te_array)
        ee_array = np.vstack(ee_array)
        spacing = amplitudes[1]-amplitudes[0]
        tt_grad = np.gradient(tt_array, spacing, axis=0)
        te_grad = np.gradient(te_array, spacing, axis=0)
        ee_grad = np.gradient(ee_array, spacing, axis=0)

        return [tt_grad[2], ee_grad[2], te_grad[2]]

    def get_current_H0(self):
         return self.BoltzmannSolver.get_current_derived_parameters(["H0"])["H0"]

    def set_parameters(self):
        """ Sets up calculator given the supplied settings
        """
        
        #######################################################
        # Copying some parameters internally to make code
        # neater.
        #######################################################

        self.ll_max = self.fisher_settings["ll_max"]
        self.xe_pert_type = self.fisher_settings["xe_pert_type"]
        self.ll = np.arange(2, self.ll_max+1)
        self.zmin = self.fisher_settings["zmin_pert"]
        self.zmax = self.fisher_settings["zmax_pert"]

        #######################################################
        # These parameters control which ell ranges to use
        # in fisher matrix calculations
        ####################################################### 
        
        if "ll_min_tt" in self.fisher_settings:
            self.ll_min_tt = self.fisher_settings["ll_min_tt"]
        else:
            self.ll_min_tt = 2

        if "ll_min_pol" in self.fisher_settings:
            self.ll_min_pol = self.fisher_settings["ll_min_pol"]
        else:
            self.ll_min_pol = 2



        #######################################################
        # Default noise covariance matrix parameters
        # Overwritten by values passed by user
        #######################################################

        self.noise_parameters = {'beam_FWHM_143_arcmin' : 7.3,
                                'beam_FWHM_217_arcmin' : 4.90,
                                'weight_inv_T_143' : 0.36e-4,
                                'weight_inv_P_143' : 1.61e-4,
                                'weight_inv_T_217' : 0.78e-4,
                                'weight_inv_P_217' : 3.25e-4,
                                'fsky' : 0.8,
                                'use_143' : True,
                                'use_217' : True
                                }  

        if "noise_params" in self.fisher_settings:
            for key,value in self.fisher_settings["noise_params"].items():
                self.noise_parameters[key] = value
        
        #######################################################
        # Create two settings dictionaries for common uses
        # - 'default': considered a blanck slate, reset values
        # - 'fiducial': settings for the fidcucial cosmology 
        #######################################################   

        th_verbose = 0
        input_verbose = 0
        if "thermodynamics_verbose" in self.fisher_settings.keys():
            th_verbose = self.fisher_settings["thermodynamics_verbose"]
        if "input_verbose" in self.fisher_settings.keys():
            input_verbose = self.fisher_settings["input_verbose"]

        self.class_default_settings = {'output' : 'tCl,pCl,lCl',
                                    'thermodynamics_verbose': th_verbose,
                                    'input_verbose': input_verbose,
                                    'lensing': 'yes',
                                    }

        self.class_fiducial_settings = self.class_default_settings.copy()
        self.class_fiducial_settings.update({"xe_pert_type": "none"})

        self.class_default_settings["xe_pert_type"] = self.xe_pert_type
    
        #######################################################
        # Set internal settings for each perturbation type
        # - 'none': no class parameter updates needed
        # - 'basis': need to update class with required settings
        #    like number of perturbations and width
        # - 'control': need to update things like str_pivots
        #######################################################   

        if self.xe_pert_type=="none":
            self.Npert = 0
            self.pivots = np.array([0,0])
            self.dz = 0
            self.width= 0  #dz is 1/3 the FWHM
        elif self.xe_pert_type=="basis":
            self.Npert = self.fisher_settings['Npert']
            self.pivots = np.linspace(self.zmin, self.zmax, self.Npert)
            self.dz = (self.zmax - self.zmin)/self.Npert
            self.width=self.dz/2.355/3.  #dz is 1/3 the FWHM
            self.class_default_settings.update({
                                    'xe_pert_num': 1,
                                    'zmin_pert': self.zmin,
                                    'zmax_pert': self.zmax,
                                    'thermo_Nz_lin': self.fisher_settings['linear_sampling'],
                                    'xe_single_width': self.width,
                                    'xe_single_zi': self.zmin
                                    })
        elif self.xe_pert_type=="control":
            self.xe_control_pivots = np.array([float(z) for z in self.fisher_settings["xe_control_pivots"].split(",")])
            self.Npert = len(self.xe_control_pivots)
            control_points = np.zeros(shape=self.xe_control_pivots.shape)
            control_points_str = ",".join(["{:.1f}".format(p) for p in control_points])
            self.class_default_settings.update({"xe_pert_num": len(self.xe_control_pivots),
                                                "zmin_pert": self.zmin,
                                                "zmax_pert": self.zmax,
                                                "xe_control_points": control_points_str,
                                                "xe_control_pivots": self.fisher_settings["xe_control_pivots"],
                                                "start_sources_at_tau_c_over_tau_h": 0.004
                                                })

        for key,val in self.cosmo_params.items():
            self.class_default_settings[key] = val
            self.class_fiducial_settings[key] = val

        #######################################################
        # Creating dictionary of step sizes for the partial
        # derivatives in the Fisher calculation
        ####################################################### 

        self.delta_param = {}

        for param_name, param_value in self.cosmo_params.items():
            self.delta_param[param_name] = 0.05*param_value

        cp_name_list = ["q_{}".format(int(x)) for x in np.arange(1,self.Npert-1)]

        for cp in cp_name_list:
            self.delta_param[cp] = 0.2

        if "delta_param" in self.fisher_settings:
            for param_name, step_value in self.fisher_settings["delta_param"].items():
                self.delta_param[param_name] = step_value

        return 

    def write_log(self):
        now = datetime.now()
        #date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        class_param_name = os.path.join(self.filebase ,"{}.class_param".format(self.basename))
        rec_param_name = os.path.join(self.filebase ,"{}.rec_param".format(self.basename))
        np.save(os.path.join(self.filebase,class_param_name), self.class_default_settings)
        np.save(os.path.join(self.filebase,rec_param_name), self.fisher_settings)

    def create_outdir(self):
        """ Creates a unique output directory for products.
        """
        now = datetime.now()
        d1 = now.strftime("%b%d")
        filebase = os.path.join(os.path.dirname(os.getcwd()), 'data')+"/"+d1+"."+self.basename+".0"
        i=0
        while(os.path.exists(filebase)):
            split = filebase.split(".")
            filebase=".".join(split[:-1])+".{}".format(i)
            i+=1
        os.mkdir(filebase)
        print("Created directory {}".format(filebase))
        return filebase
            
    def debug_print(self, out):
        if(self.is_debug):
            print(out)
        
class FisherMatrix:

    def __init__(self):
        
        self.is_loaded=False
        self.path = ''

        self.class_parameters = {}
        self.other_parameters = {}

        self.cosmo_param_list = ["omega_b", "omega_cdm", "n_s", "tau_reio", "A_s", "ln10^{10}A_s", "100*theta_s", "H0", "sigma8"]
        self.varied_params = {}

        self.Fisher = []
        self.col_names = []

        return

    def from_file(self, path):
        
        self.path = path
        class_parameters, fisher_settings = self.read_log()

        self.class_parameters = class_parameters
        self.fisher_settings = fisher_settings

        self.varied_params = {}

        for key, item in self.class_parameters.items():
            if key in self.cosmo_param_list:
                self.varied_params[key] = item

        if self.fisher_settings["xe_pert_type"]=="control":
            for i in np.arange(1, self.class_parameters["xe_pert_num"]-1):
                self.varied_params["q_{}".format(i)] = 0.0


        self.Fisher = np.load(os.path.join(path, "Fisher_full.npz"))["data"]
        self.col_names = np.load(os.path.join(path, "Fisher_full.npz"))["comment"]

        self.is_loaded = True

    def from_array(self, matrix, col_names, class_params, fisher_settings):
        self.Fisher = matrix
        self.col_names = col_names
        self.class_parameters = class_params
        self.fisher_settings = fisher_settings
        
        self.varied_params = {}

        for key, item in self.class_parameters.items():
            if key in self.cosmo_param_list:
                self.varied_params[key] = item

        if self.fisher_settings["xe_pert_type"]=="control":
            for i in np.arange(1, self.class_parameters["xe_pert_num"]-1):
                self.varied_params["q_{}".format(i)] = 0.0

        self.is_loaded = True

    def get_standard_block(self):
        return self.Fisher[:6, :6]

    def get_marginalized_matrix(self, parameters_to_keep):
        indices_to_keep = np.where(np.isin(self.col_names, parameters_to_keep))[0]
        marg_cols = np.array(self.col_names)[indices_to_keep]
        sorted_ind = [int(np.where(marg_cols==n)[0]) for n in parameters_to_keep]
        cov = np.linalg.pinv(self.Fisher)
        marginalized = np.linalg.pinv(cov[indices_to_keep, :][:, indices_to_keep])
        marginalized_resort = marginalized[sorted_ind,:][:,sorted_ind]
        return marginalized_resort

    def ellipse2d(self, parameter_combo): #should be list of 2 parameter names
        center = [self.varied_params[parameter_combo[0]], self.varied_params[parameter_combo[1]]]
        mat = np.linalg.pinv(self.get_marginalized_matrix(parameter_combo))
        sigx2 = mat[0,0]
        sigy2 = mat[1,1]
        cross = mat[0,1]
        a = np.sqrt((sigx2 + sigy2)/2. + np.sqrt((sigx2 - sigy2)**2/4. + cross**2))
        b = np.sqrt((sigx2 + sigy2)/2. - np.sqrt((sigx2 - sigy2)**2/4. + cross**2))
        theta=0.5*np.arctan2(2*cross,sigx2-sigy2)*180./(np.pi)
        alpha=np.sqrt(scipy.special.chdtri(2,1-0.68))
        return matplotlib.patches.Ellipse((center[0], center[1]), alpha*2*a, alpha*2*b,
                     angle=theta, linewidth=2, fill=False, zorder=2, color="red")
   #def run_PCA(self, matrix):
   #     if self.has_perturbations:
   #         return PCA(matrix, self.PCA_parameters)
   #     else:
   #         print("Directory has no perturbations")
   #         return

    def get_logname_from_dir(self):
        fn = os.path.basename(self.path)
        identifier = fn.split(".")[-2]
        return os.path.join(self.path, "{}.class_param.npy".format(identifier)), os.path.join(self.path, "{}.rec_param.npy".format(identifier))

    def read_log(self):
        class_path, rec_path = self.get_logname_from_dir()
        return np.load(class_path, allow_pickle=True).item(), np.load(rec_path, allow_pickle=True).item()

class PCA:

    def __init__(self, matrix, parameters):
        self.matrix = matrix
            
        self.width = parameters["width"]
        self.zmin = parameters["zmin"]
        self.zmax = parameters["zmax"]
        self.pivots = parameters["pivots"]

        u, s, vh = np.linalg.svd(self.matrix)
        self.eigenvals = s**2
        self.eigenvecs = vh

        self.eigenvecs*=1./np.sqrt(2*np.pi*self.width*self.width)
        # making sure eigenvals are positive
        for i,vec in enumerate(self.eigenvecs):
            if self.eigenvals[i]<0:
                self.eigenvals[i]*=-1.
                vec*=-1.


    def mode(self, n):
        vec = self.eigenvecs[n]
        #orient = np.trapz(vec, x=self.pivots)
        #if(orient>0):
        #    vec*=-1.
        fun = CubicSpline(self.pivots, vec)
        lim = int(len(self.pivots)+1)
        norm = scipy.integrate.quad(square(fun), self.zmin, self.zmax, points=self.pivots, limit=lim)[0]

        return renormalize(1/np.sqrt(norm))(fun)

    def mode_points(self, n):
        vec = self.eigenvecs[n]
        #orient = np.trapz(vec, x=self.pivots)
        #if(orient>0):
        #    vec*=-1.
        fun = CubicSpline(self.pivots, vec)
        lim = int(len(self.pivots)+1)
        norm = scipy.integrate.quad(square(fun), self.zmin, self.zmax, points=self.pivots, limit=lim)[0]
        return vec/np.sqrt(norm)

    def eigenval(self, n):
        return self.eigenvals[n]
