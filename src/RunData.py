import os
import numpy as np

class RunData:

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

    def __init__(self, path):
        logname = ""
        for filename in os.listdir(path):
            if(filename.split(".")[-1]=="log"):
                logname=filename
        pars,other = self.read_log(os.path.join(path,logname))
        self.params = pars
        self.zmin = float(pars["zmin_pert"])
        self.zmax = float(pars["zmax_pert"])
        if("linear_sampling" in other):
            self.linear_sampling = int(other["linear_sampling"])
        self.Npert = other["Npert"]
        #self.pivots = [float(zi) for zi in other["pivots"].split(" ")[1].split(",")]
        self.dz = float(other["dz"])
        self.width = float(other["width"])
        self.plancklike = other["Planck_Noise"]
        s = os.path.basename(path).split(".")[0]
        self.name=s
        self.tt_derivs = np.load(os.path.join(path, "tt_derivs.npy"))
        self.te_derivs = np.load(os.path.join(path, "te_derivs.npy"))
        self.ee_derivs = np.load(os.path.join(path, "ee_derivs.npy"))
        self.pivots = np.load(os.path.join(path, "pivots.npy"))
        self.Fisher = np.load(os.path.join(path, "fisher.npy"))
        self.z = np.load(os.path.join(path, "z.npy"))
