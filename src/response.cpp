#include "ClassPlus.h"
#include "BasicIO.h"
//#include "common.h"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <string>
#include <stdexcept>

typedef float REAL;

int main(int argc, char** argv){

	const int l_max_scalars = 2500;
	
	const REAL h = 0.6774;
	const REAL omega_b = 0.02230118936;
	const REAL omega_cdm = 0.11880163976;
	const REAL sigma8 = 0.8159;
	const REAL n_s = 0.9667;
	const REAL tau_reio = 0.078;

	const int zmin_pert = 500;
	const int zmax_pert = 1700;

	ClassParams Params;

	Params.set("h", h);
	Params.set("omega_b", omega_b);
	Params.set("omega_cdm", omega_cdm);
	Params.set("sigma8", sigma8);
	Params.set("n_s", n_s);
	Params.set("tau_reio", tau_reio);
	Params.set("output", "tCl,lCl,pCl");
	Params.set("thermodynamics_verbose", 1);
	Params.set("l_max_scalars",l_max_scalars);
	Params.set("lensing", "yes");
	Params.set("perturb_xe", "yes");
	Params.set("input_verbose", 0);
	Params.set("zmin_pert", zmin_pert);
	Params.set("zmax_pert", zmax_pert);
	Params.set("xe_pert_num", 1);
	Params.set("xe_single_width", 13/2.335);
	Params.set("xe_single_zi", 1100);
	Params.set("xe_pert_amps", str(.1));

	ClassPlus CLASS = ClassPlus(Params, true );
	CLASS.printFC();

	CLASS.compute();
	
	std::vector<unsigned> lvec(CLASS.l_max_scalars()-1,1);
  	lvec[0]=2;
	std::partial_sum(lvec.begin(),lvec.end(),lvec.begin()); // easy way to fille vector with 2,3,4,5....lmax


	std::vector<double> cltt, clte, clee, clbb;

	CLASS.getCls(lvec, cltt, clte, clee, clbb);

	/* { */
		/* BasicIO Scribe = BasicIO("test.dat"); */
	/*  */
		/* Scribe.attach(lvec, "l"); */
		/* Scribe.attach(cltt, "TT"); */
		/* Scribe.attach(clte, "TE"); */
		/* Scribe.attach(clbb, "BB"); */
        /*  */
		/* Scribe.write(); */
	/* } */

	std::vector<double> z, tau, xe, fid, pert, kappa, exp_kappa;
	
	CLASS.getThermoVecs(z, tau, xe, fid, pert, kappa, exp_kappa);

	std::cout<<"size of xe "<<xe.size()<<std::endl;
	{
		BasicIO Scribe = BasicIO("thermo.dat");

		Scribe.attach(z, "z");
		Scribe.attach(xe, "x_e");
		Scribe.attach(pert, "xe_pert");

		Scribe.write();
	}

	return 1;
}
