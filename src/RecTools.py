import os
from datetime import datetime
import numpy as np
import classy as Classy
from IPython.display import clear_output
from scipy.interpolate import CubicSpline
import scipy.integrate
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

class RecFisher:
    """ A class to compute a recombination Fisher matrix.
    """
    def __init__(self, cosmo_parameters, rec_parameters, name):
        self.cosmo_params = cosmo_parameters
        self.rec_params = rec_parameters

        self.is_debug = False
        if "debug" in self.rec_params.keys():
            self.is_debug = self.rec_params["debug"]

        self.basename = name
        self.filebase = self.create_outdir()
        
        self.debug_print("Setting parameters")

        self.set_parameters()

        self.debug_print("Writing log")

        self.write_log()
    
        self.perturbations = []
        self.class_z_array = []

        self.debug_print("Initializing Boltzman Solver")

        self.BoltzmannSolver = Classy.Class()
        self.BoltzmannSolver.set(self.class_common_settings)

        self.debug_print("Solver Initialized")

        self.tt_fid = []
        self.ee_fid = []
        self.te_fid = []

        self.dCl = []

        self.Fisher = []
        self.Fisher_fixed = []
        self.Fisher_marginalized =[]

        self.error_covariance = []
        self.error_covariance_inv = []

        self.run_counter = 0
        self.total_runs = self.pivots.shape[0]*self.amplitudes.shape[0]
        self.inc_percent = 0.05

    def compute_Fisher(self):
        """ Main routine which computes the full Fisher matrix for this recombination history.
        """
        self.debug_print("Starting computation of Fisher matrix")
        tt_derivs = []
        te_derivs = []
        ee_derivs = []
       
        #if self.do_perturbations:
        #    self.BoltzmannSolver.set({'xe_single_zi': self.rec_params['zmin_pert']})
        #    self.BoltzmannSolver.set({'xe_pert_amps': '0'})

        self.debug_print("Beginning parameter variation")

        if self.do_perturbations:
            self.BoltzmannSolver.set({'xe_single_zi': self.rec_params['zmin_pert']})
            self.BoltzmannSolver.set({'xe_pert_amps': '0'})

        for param_name, fiducial_value in self.cosmo_params.items():
            self.BoltzmannSolver.set(self.cosmo_params)
            print("Varying parameter: {}".format(param_name))
            cmb_responses = self.compute_cmb_standard_response(param_name, fiducial_value, self.inc_percent)
            print("Current H0 is {}".format(self.get_current_H0()))
            tt_derivs.append(cmb_responses[0])
            te_derivs.append(cmb_responses[1])
            ee_derivs.append(cmb_responses[2])

        if self.do_perturbations:
            for z in self.pivots:
                self.BoltzmannSolver.set(self.cosmo_params)
                cmb_responses = self.compute_cmb_perturbation_response(z)
                tt_derivs.append(cmb_responses[0])
                te_derivs.append(cmb_responses[1])
                ee_derivs.append(cmb_responses[2])
        else:
            print("Skipping perturbations")

        tt_derivs = np.vstack(tt_derivs)    
        te_derivs = np.vstack(te_derivs)    
        ee_derivs = np.vstack(ee_derivs) 

        np.save(os.path.join(self.filebase,"tt_derivs"), tt_derivs)
        np.save(os.path.join(self.filebase,"te_derivs"), te_derivs)
        np.save(os.path.join(self.filebase,"ee_derivs"), ee_derivs)
        np.save(os.path.join(self.filebase,"pivots"), self.pivots)
        np.save(os.path.join(self.filebase,"z"), self.BoltzmannSolver.get_thermodynamics()['z'])

        if self.do_perturbations:
            self.BoltzmannSolver.set({'xe_single_width': self.width})
            self.BoltzmannSolver.set({'xe_single_zi': self.rec_params['zmin_pert']})
            self.BoltzmannSolver.set({'xe_pert_amps': '0'})

        for key,val in self.cosmo_params.items():
            self.BoltzmannSolver.set({key: val})

        self.BoltzmannSolver.compute() ##Run CLASS
        Tcmb = self.BoltzmannSolver.T_cmb()*10**6 #cmb temp in micro kelvin
        #Tcmb=1.0
        
        muK2 = (Tcmb)**2

        self.tt_fid = muK2*self.BoltzmannSolver.lensed_cl(self.ll_max)['tt'][2:]
        self.te_fid = muK2*self.BoltzmannSolver.lensed_cl(self.ll_max)['te'][2:]
        self.ee_fid = muK2*self.BoltzmannSolver.lensed_cl(self.ll_max)['ee'][2:]

        self.dCl = muK2*np.stack([tt_derivs, ee_derivs, te_derivs], axis=1)

        self.error_covariance = self.compute_error_covariance()

        self.error_covariance_inv = np.linalg.inv(self.error_covariance)

        self.Fisher = np.einsum("iXl,lXY,jYl->ij", self.dCl, self.error_covariance_inv, self.dCl)

        divider = len(self.cosmo_params.keys())

        standard_block = self.Fisher[:divider, :divider]
        cross_block = self.Fisher[:divider, divider:]
        perturbation_block = self.Fisher[divider:, divider:]

        self.Fisher_marginalized = perturbation_block - np.einsum("ai,ij,jk->ak", cross_block.T, np.linalg.inv(standard_block), cross_block)
        self.Fisher_fixed = perturbation_block

        np.save(os.path.join(self.filebase,"Fisher_full"), self.Fisher)
        np.save(os.path.join(self.filebase,"Fisher_marginalized"), self.Fisher_marginalized)
        np.save(os.path.join(self.filebase,"Fisher_fixed"), self.Fisher_fixed)

        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

        print("Fisher matrices calculation completed at {}".format(date_time))

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

    def compute_cmb_standard_response(self, parameter_name, parameter_value, percent=0.05):
        """ Computes the CMB response to varying a standard parameter.
            Parameters are incremented by a fixed percentage supplied
            by the percent argument.
        """
        if parameter_name=="A_s":
            ln10e10As = np.log(parameter_value*(10**10))
            dp = percent*ln10e10As
            trial_values = np.linspace(-2*dp, 2*dp, self.amplitudes.shape[0]) + ln10e10As
            trial_values = np.exp(trial_values)/(10**10)
        else:
            dp = percent*parameter_value
            trial_values = np.linspace(-2*dp, 2*dp, self.amplitudes.shape[0]) + parameter_value
        
        tt_array=[]
        te_array=[]
        ee_array=[]

        for trial in trial_values:
            self.BoltzmannSolver.set({parameter_name: trial}) #sets perturbation
            self.BoltzmannSolver.compute()
            cls = self.BoltzmannSolver.lensed_cl(self.ll_max)
            tt_array.append(cls['tt'][2:])
            te_array.append(cls['te'][2:])
            ee_array.append(cls['ee'][2:])
        
        tt_array = np.vstack(tt_array)
        te_array = np.vstack(te_array)
        ee_array = np.vstack(ee_array)
        tt_grad = np.gradient(tt_array, dp, axis=0)[self.middle]
        te_grad = np.gradient(te_array, dp, axis=0)[self.middle]
        ee_grad = np.gradient(ee_array, dp, axis=0)[self.middle]

        return [tt_grad, te_grad, ee_grad]

    def compute_cmb_perturbation_response(self, z):
        """ Computes the CMB response to a perturbation at redshift z.
            Perturbation amplitudes are defined by the self.amplitudes
            attribute, which is an array of amplitudes that can either 
            be passed as part of rec_params, or set to default values.
        """
        self.BoltzmannSolver.set({'xe_single_zi': z})

        tt_array = []
        te_array = []
        ee_array = []

        for qi in self.amplitudes:
            start = time.time()
            amp_str = "{}".format(qi)        #Setting up string of perturbation amplitudes
            self.BoltzmannSolver.set({'xe_pert_amps': amp_str}) #sets perturbation
            self.BoltzmannSolver.compute()
            self.debug_print("H0 value is {}".format(self.get_current_H0()))
            cls = self.BoltzmannSolver.lensed_cl(self.ll_max)
            tt_array.append(cls['tt'][2:])
            te_array.append(cls['te'][2:])
            ee_array.append(cls['ee'][2:])
            end = time.time()
            self.run_counter+=1
            clear_output(wait=True)
            print("Run {}/{} took {:.2f} seconds".format(self.run_counter, self.total_runs, end-start))
            if(qi==self.amplitudes[-1]):
                self.perturbations.append(self.BoltzmannSolver.get_thermodynamics()['xe_pert'])

        tt_array = np.vstack(tt_array)
        te_array = np.vstack(te_array)
        ee_array = np.vstack(ee_array)
        spacing = self.amplitudes[1]-self.amplitudes[0]
        tt_grad = np.gradient(tt_array, spacing, axis=0)
        te_grad = np.gradient(te_array, spacing, axis=0)
        ee_grad = np.gradient(ee_array, spacing, axis=0)

        return [tt_grad[self.middle], te_grad[self.middle], ee_grad[self.middle]]

    def get_current_H0(self):
         return self.BoltzmannSolver.get_current_derived_parameters(["H0"])["H0"]

    def set_parameters(self):
        """ Sets derived parameters to given or default values
        """
        
        self.ll_max = self.rec_params["ll_max"]

        if "amplitudes" in self.rec_params:
            self.amplitudes = self.rec_params['amplitudes']
        else:
            self.amplitudes = np.linspace(-.1, .1, 5)

        if "increment_percentage" in self.rec_params:
            self.inc_percent = self.rec_params["increment_percentage"]

        
        self.middle = int((self.amplitudes.shape[0]-1)/2)

        self.ll = np.arange(2, self.ll_max+1)

        if "noise_params" in self.rec_params:
            self.noise_parameters = self.rec_params["noise_params"]
        else:
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
    
        th_verbose = 0
        input_verbose = 0
        if "thermodynamics_verbose" in self.rec_params.keys():
            th_verbose = self.rec_params["thermodynamics_verbose"]
        if "input_verbose" in self.rec_params.keys():
            input_verbose = self.rec_params["input_verbose"]


        self.class_common_settings = {'output' : 'tCl,pCl,lCl',
                                    #Class run parameters
                                    'thermodynamics_verbose': th_verbose,
                                    'input_verbose': input_verbose,
                                    'lensing': 'yes',
                                    }

        self.do_perturbations = True

        self.class_common_settings["perturb_xe"] = self.rec_params["perturb_xe"]
            
        if self.class_common_settings["perturb_xe"].lower()=='no':
            self.do_perturbations = False
            self.Npert = 0
            self.pivots = np.array([0,0])
            self.dz = 0
            self.width= 0  #dz is 1/3 the FWHM
        else:
            self.Npert = self.rec_params['Npert']
            self.pivots = np.linspace(self.rec_params['zmin_pert'], self.rec_params['zmax_pert'], self.Npert)
            self.dz = (self.rec_params['zmax_pert'] - self.rec_params['zmin_pert'])/self.Npert
            self.width=self.dz/2.355/3.  #dz is 1/3 the FWHM
            self.class_common_settings.update({
                                    'xe_pert_num': 1,
                                    'zmin_pert': self.rec_params['zmin_pert'],
                                    'zmax_pert': self.rec_params['zmax_pert'],
                                    'thermo_Nz_lin': self.rec_params['linear_sampling'],
                                    'xe_single_width': self.width
                                    })



        for key,val in self.cosmo_params.items():
            self.class_common_settings[key] = val

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

    def write_log(self):
        """ Writes log file containing parameters for this run.
        """
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        logname = os.path.join(self.filebase ,"{}.log".format(self.basename))
        with open(logname, 'w') as file:
            now = datetime.now()
            file.write("# Files in this directory created {}\n".format(date_time))
            file.write("{\n")
            for k,v in self.class_common_settings.items():
                file.write("{0} {1}\n".format(k,v))
            file.write("}\n")
            if self.do_perturbations:
                file.write("zmin {}\n".format(self.rec_params['zmin_pert']))
                file.write("zmax {}\n".format(self.rec_params['zmax_pert']))
                file.write("Npert {}\n".format(self.Npert))
                zstr = ",".join(self.pivots.astype(str))
                file.write("pivots {}\n".format(zstr))
                file.write("dz {}\n".format(self.dz))
                file.write("width {}\n".format(self.width))
            
            file.write("do_perturbations {}\n".format(self.do_perturbations))
            
            file.write("ll_max {}\n".format(self.ll_max))
            file.write("linear_sampling {}\n".format(self.rec_params['linear_sampling']))
            
    def debug_print(self, out):
        if(self.is_debug):
            print(out)
        
class FisherData:

    def __init__(self, path):
        
        self.path = path
        class_parameters, other_parameters = self.read_log(self.get_logname_from_dir())

        self.class_parameters = class_parameters
        self.other_parameters = other_parameters

        cosmo_param_names = ["omega_b", "omega_cdm", "n_s", "tau_reio", "A_s", "ln10^{10}A_s", "100*theta_s", "H0", "sigma8"]
        self.cosmo_params = {}

        for key, item in self.class_parameters.items():
            if key in cosmo_param_names:
                self.cosmo_params[key] = item

        print(self.cosmo_params)
        assert len(self.cosmo_params)==6

        self.Fisher_full = np.load(os.path.join(path, "Fisher_full.npy"))
        self.Fisher_marginalized = np.load(os.path.join(path, "Fisher_marginalized.npy"))
        self.Fisher_fixed = np.load(os.path.join(path, "Fisher_fixed.npy"))

        if other_parameters["do_perturbations"]=="True":
            self.has_perturbations = True
        else:
            self.has_perturbations = False

        if self.has_perturbations:
            self.width = float(other_parameters["width"])
            self.pivots = np.load(os.path.join(path, "pivots.npy"))
            self.zmin = float(class_parameters["zmin_pert"])
            self.zmax = float(class_parameters["zmax_pert"])

            self.PCA_parameters = {'width': self.width, 
                                'zmin' : self.zmin, 
                                'zmax' : self.zmax, 
                                'pivots' : self.pivots
                                }


        return

    def get_standard_block(self):
        return self.Fisher_full[:6, :6]

    def get_reparameterized_standard_block(self, H0_fid=67.66, increment_percentage=0.05, 
                                            full=False, amplitudes=np.linspace(-.1, .1, 5)):

        internal_common_settings = {'output' : 'tCl',
                   'thermodynamics_verbose': 0,
                   'input_verbose': 0,
                   'perturb_xe': 'no'
                   }
        
        internal_cosmo_settings = {}
        
        for parameter_name, parameter_value in self.cosmo_params.items():
            if parameter_name=="100*theta_s":
                continue
            internal_cosmo_settings[parameter_name] = parameter_value
        
        internal_cosmo_settings["H0"] = H0_fid

        M = Classy.Class()
        M.set(internal_common_settings)
        M.set(internal_cosmo_settings)

        trial_arrays = {}
        print(internal_cosmo_settings.items())
        for parameter_name, parameter_value in internal_cosmo_settings.items():
            dp = float(increment_percentage)*float(parameter_value)
            trial_values = np.linspace(-2*dp, 2*dp, 20) + float(parameter_value)
            trial_arrays[parameter_name]= trial_values

        slopes=[]
        
        for parameter_name, trial_values in trial_arrays.items():
            M.set(internal_cosmo_settings)
            print("Varying {}".format(parameter_name))
            calculated_values = []
            for value in trial_values:
                M.set({parameter_name: value})
                M.compute(level=["thermodynamics"])
                calculated_values.append(M.get_current_derived_parameters(["100*theta_s"])["100*theta_s"])
            slopes.append(np.average(np.gradient(calculated_values, trial_values)))

        if full:
            
            M.set(internal_common_settings)
            M.set(internal_cosmo_settings)
            
            internal_pert_settings={'perturb_xe': 'yes',
                                    'xe_pert_num': 1,
                                    'zmin_pert': self.zmin,
                                    'zmax_pert': self.zmax,
                                    'xe_single_width': self.width
                                    }
            
            M.set(internal_pert_settings)

            for zi in self.pivots:
                M.set({'xe_single_zi': zi})
                calculated_values=[]
                for qi in amplitudes:
                    amp_str = "{}".format(qi)        
                    M.set({'xe_pert_amps': amp_str})
                    M.compute(level=["thermodynamics"])
                    calculated_values.append(M.get_current_derived_parameters(["100*theta_s"])["100*theta_s"])
                slopes.append(np.gradient(calculated_values, amplitudes)[2])
            
            J = np.zeros(self.Fisher_full.shape)
            np.fill_diagonal(J, 1.)
            J[5,:] = slopes

            print(np.array(slopes))

            final = np.einsum("ij, jk, kl -> il", J.T, self.Fisher_full, J)

        else:
            
            J = np.zeros([5,6])
            ones = np.fill_diagonal(J, 1)
            J = np.vstack([J, slopes])
            print(slopes)
            final =  np.einsum("ij, jk, kl -> il", J.T, self.Fisher_full[:6, :6], J)

        return final

    def run_PCA(self, matrix):
        if self.has_perturbations:
            return PCA(matrix, self.PCA_parameters)
        else:
            print("Directory has no perturbations")
            return

    def get_logname_from_dir(self):
        fn = os.path.basename(self.path)
        identifier = fn.split(".")[-2]
        return os.path.join(self.path, "{}.log".format(identifier))

    def read_log(self, path):
        pars = {}
        other = {}
        with open(path) as infile:
            copy_dict = False
            for line in infile:
                if line[0]=="#":
                    continue
                if line.strip() == "{":
                    copy_dict = True
                    continue
                elif line.strip() == "}":
                    copy_dict = False
                    continue
                elif copy_dict:
                    key,value = line.strip().split(" ")
                    pars[key] = value
                else:
                    key,value = line.strip().split(" ")
                    other[key] = value
        return pars,other

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
