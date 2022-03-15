#include "ClassPlus.h"
#include "common.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <numeric>
#include <cassert>

template<typename T> std::string str(const T &x){
  std::ostringstream os;
  os << x;
  return os.str();
}

template<> std::string str (const float &x){
  std::ostringstream os;
  os << setprecision(8) << x;
  return os.str();
}
template<> std::string str (const double &x){
  std::ostringstream os;
  os << setprecision(16) << x;
  return os.str();
}
template<> std::string str (const bool &x){
  { return x ? "yes" : "no"; }
}

template<> std::string str (const std::string &x) {return x;}

std::string str (const char* s){return string(s);}

template string str(const int &x);
template string str(const signed char &x);
template string str(const unsigned char &x);
template string str(const short &x);
template string str(const unsigned short &x);
template string str(const unsigned int &x);
template string str(const long &x);
template string str(const unsigned long &x);
template string str(const long long &x);
template string str(const unsigned long long &x);

ClassPlus::ClassPlus(ClassParams & CP, bool verbose): cl(0), dofree(true){

	size_t n=CP.params.size();
	
	//allocates FC arrays based on number of input params
	parser_init(&fc,n,(char*)"pipo",_errmsg);
	

	map<string, string>::iterator itr;

	int i = 0;
	for (itr = CP.params.begin(); itr!=CP.params.end(); ++itr){
		strcpy(fc.name[i],(itr->first).c_str());
		strcpy(fc.value[i],(itr->second).c_str());
		if (itr->first=="l_max_scalars") {
  			istringstream strstrm(itr->second);
  			strstrm >> _lmax;
		}
		i++;
	}

	if (input_read_from_file(&fc,&pr,&ba,&th,&pt,&tr,&pm,&hr,&fo,&le,&sd,&op,_errmsg) == _FAILURE_)
		throw invalid_argument(_errmsg);
	
	// Check that there are no unread parameters
	for (size_t i=0;i<CP.params.size();i++){
		if (fc.read[i] !=_TRUE_) throw invalid_argument(string("Invalid CLASS parameter: ")+fc.name[i]);
	}

}

ClassPlus::~ClassPlus() {
//dofree && freeStructs();

	if( pt.has_cl_cmb_temperature || pt.has_cl_cmb_polarization || pt.has_cl_lensing_potential ){
		delete [] cl;
	}
}

int ClassPlus::compute(){

  int status=this->class_main(&fc,&pr,&ba,&th,&pt,&tr,&pm,&hr,&fo,&le,&sd,&op,_errmsg);
  
  // Allocates an array on the heap 
  if( pt.has_cl_cmb_temperature || pt.has_cl_cmb_polarization || pt.has_cl_lensing_potential ){
    cl= new double[hr.ct_size];
  }
  
  return status;

}

void ClassPlus::getThermoVecs(std::vector<double>& z, 
		std::vector<double>& tau,
		std::vector<double>& xe, 
		std::vector<double>& xe_fid, 
		std::vector<double>& xe_pert,
		std::vector<double>& kappa,
		std::vector<double>& exp_kappa){

	if (!dofree) throw out_of_range("no ThermoVec because CLASS failed");

	char titles[_MAXTITLESTRINGLENGTH_]={0};
	double * data;
	int size_data, number_of_titles;

	thermodynamics_output_titles(&ba,&th,titles);

	number_of_titles = 0;
	for(int i=0; i<strlen(titles); i++){
		if(titles[i] == '\t') number_of_titles++;	
	}
	
	size_data = number_of_titles*(th.tt_size);
	data = (double*)malloc(sizeof(double)*size_data);
	std::cout<<"number of titles "<<number_of_titles<<std::endl;
	thermodynamics_output_data(&ba, &th, number_of_titles, data);

    for (int index_vertical=0; index_vertical<(th.tt_size); index_vertical++){
      	z.push_back(data[index_vertical*number_of_titles+0]);
      	tau.push_back(data[index_vertical*number_of_titles+1]);
      	/* xe.push_back(data[index_vertical*number_of_titles+th.index_th_xe]); */
          /* xe_fid.push_back(data[index_vertical*number_of_titles+th.index_th_xe_fid]); */
          /* xe_pert.push_back(data[index_vertical*number_of_titles+th.index_th_xe_pert]); */
          /* kappa.push_back(data[index_vertical*number_of_titles+th.index_th_dkappa]); */
      	/* exp_kappa.push_back(data[index_vertical*number_of_titles+th.index_th_exp_m_kappa]); */
		xe.push_back(data[index_vertical*number_of_titles+2]);
		xe_fid.push_back(data[index_vertical*number_of_titles+3]);
		xe_pert.push_back(data[index_vertical*number_of_titles+4]);
		kappa.push_back(data[index_vertical*number_of_titles+5]);
		exp_kappa.push_back(data[index_vertical*number_of_titles+6]);
	}
	free(data);
}

double ClassPlus::getCl(ClassPlus::CL type, const long &l){
	if (!dofree) throw out_of_range("no Cl available because CLASS failed");

	if (output_total_cl_at_l(&hr,&le,&op,static_cast<int>(l),cl) == _FAILURE_){
		cerr << ">>>fail getting Cl type=" << (int)type << " @l=" << l <<endl;
		throw out_of_range(hr.error_message);
	}

	double cl_val=-1;

	double tomuk=1e6*Tcmb();
	double tomuk2=tomuk*tomuk;
	// at this stage, cl array holds the value of each Cl at this l 
	switch(type){
		case TT:
		  (hr.has_tt==_TRUE_) ? cl_val=tomuk2*cl[hr.index_ct_tt] : throw invalid_argument("no ClTT available");
		  break;
		case TE:
		  (hr.has_te==_TRUE_) ? cl_val=tomuk2*cl[hr.index_ct_te] : throw invalid_argument("no ClTE available");
		  break;
		case EE:
		  (hr.has_ee==_TRUE_) ? cl_val=tomuk2*cl[hr.index_ct_ee] : throw invalid_argument("no ClEE available");
		  break;
		case BB:
		  (hr.has_bb==_TRUE_) ? cl_val=tomuk2*cl[hr.index_ct_bb] : throw invalid_argument("no ClBB available");
		  break;
		case PP:
		  (hr.has_pp==_TRUE_) ? cl_val=cl[hr.index_ct_pp] : throw invalid_argument("no ClPhi-Phi available");
		  break;
		case TP:
		  (hr.has_tp==_TRUE_) ? cl_val=tomuk*cl[hr.index_ct_tp] : throw invalid_argument("no ClT-Phi available");
		  break;
		case EP:
		  (hr.has_ep==_TRUE_) ? cl_val=tomuk*cl[hr.index_ct_ep] : throw invalid_argument("no ClE-Phi available");
		  break;
	}

	return cl_val;

}

void ClassPlus::getCls(const std::vector<unsigned>& lvec, //input
		      std::vector<double>& cltt,
		      std::vector<double>& clte,
		      std::vector<double>& clee,
		      std::vector<double>& clbb)
{
  cltt.resize(lvec.size());
  clte.resize(lvec.size());
  clee.resize(lvec.size());
  clbb.resize(lvec.size());

  for (size_t i=0;i<lvec.size();i++){
    try{
      cltt[i]=getCl(ClassPlus::TT,lvec[i]);
      clte[i]=getCl(ClassPlus::TE,lvec[i]);
      clee[i]=getCl(ClassPlus::EE,lvec[i]);
      clbb[i]=getCl(ClassPlus::BB,lvec[i]);
    }
    catch(exception &e){
      throw e;
    }
  }

}

int ClassPlus::param_update(ClassParams & CP){
	dofree && freeStructs();
	
	size_t n=CP.params.size();
	
	parser_init(&fc,n,(char*)"pipo",_errmsg);

	map<string, string>::iterator itr;

	int i = 0;
	for (itr = CP.params.begin(); itr!=CP.params.end(); ++itr){
		strcpy(fc.name[i],(itr->first).c_str());
		strcpy(fc.value[i],(itr->second).c_str());
		if (itr->first=="l_max_scalars") {
  			istringstream strstrm(itr->second);
  			strstrm >> _lmax;
		}
		i++;
	}

	if (input_read_from_file(&fc,&pr,&ba,&th,&pt,&tr,&pm,&hr,&fo,&le,&sd,&op,_errmsg) == _FAILURE_)
		throw invalid_argument(_errmsg);

	for (size_t i=0;i<CP.params.size();i++){
		if (fc.read[i] !=_TRUE_) throw invalid_argument(string("Invalid CLASS parameter: ")+fc.name[i]);
	}
	return _SUCCESS_;
}

void ClassPlus::printFC(){
  printf("FILE_CONTENT SIZE=%d\n",fc.size);
  for (int i=0;i<fc.size;i++) printf("%d : %s = %s\n",i,fc.name[i],fc.value[i]);
}

int ClassPlus::class_main(
			    struct file_content *pfc,
			    struct precision * ppr,
			    struct background * pba,
			    struct thermodynamics * pth,
			    struct perturbations * ppt,
			    struct transfer * ptr,
			    struct primordial * ppm,
			    struct harmonic * phr,
			    struct fourier * pfo,
			    struct lensing * ple,
			    struct distortions * psd,
			    struct output * pop,
			    ErrorMsg errmsg) {


  if (input_read_from_file(pfc,ppr,pba,pth,ppt,ptr,ppm,phr,pfo,ple,psd,pop,errmsg) == _FAILURE_) {
    printf("\n\nError running input_read_from_file \n=>%s\n",errmsg);
    dofree=false;
    return _FAILURE_;
  }

  if (background_init(ppr,pba) == _FAILURE_) {
    printf("\n\nError running background_init \n=>%s\n",pba->error_message);
    dofree=false;
    return _FAILURE_;
  }

  if (thermodynamics_init(ppr,pba,pth) == _FAILURE_) {
    printf("\n\nError in thermodynamics_init \n=>%s\n",pth->error_message);
    background_free(&ba);
    dofree=false;
    return _FAILURE_;
  }

  if (perturbations_init(ppr,pba,pth,ppt) == _FAILURE_) {
    printf("\n\nError in perturb_init \n=>%s\n",ppt->error_message);
    thermodynamics_free(&th);
    background_free(&ba);
    dofree=false;
    return _FAILURE_;
  }

  if (primordial_init(ppr,ppt,ppm) == _FAILURE_) {
    printf("\n\nError in primordial_init \n=>%s\n",ppm->error_message);
    perturbations_free(&pt);
    thermodynamics_free(&th);
    background_free(&ba);
    dofree=false;
    return _FAILURE_;
  }

  if (fourier_init(ppr,pba,pth,ppt,ppm,pfo) == _FAILURE_)  {
    printf("\n\nError in nonlinear_init \n=>%s\n",pfo->error_message);
    primordial_free(&pm);
    perturbations_free(&pt);
    thermodynamics_free(&th);
    background_free(&ba);
    dofree=false;
    return _FAILURE_;
  }

  if (transfer_init(ppr,pba,pth,ppt,pfo,ptr) == _FAILURE_) {
    printf("\n\nError in transfer_init \n=>%s\n",ptr->error_message);
    fourier_free(&fo);
    primordial_free(&pm);
    perturbations_free(&pt);
    thermodynamics_free(&th);
    background_free(&ba);
    dofree=false;
    return _FAILURE_;
  }

  if (harmonic_init(ppr,pba,ppt,ppm,pfo,ptr,phr) == _FAILURE_) {
    printf("\n\nError in spectra_init \n=>%s\n",phr->error_message);
    transfer_free(&tr);
    fourier_free(&fo);
    primordial_free(&pm);
    perturbations_free(&pt);
    thermodynamics_free(&th);
    background_free(&ba);
    dofree=false;
    return _FAILURE_;
  }

  if (lensing_init(ppr,ppt,phr,pfo,ple) == _FAILURE_) {
    printf("\n\nError in lensing_init \n=>%s\n",ple->error_message);
    harmonic_free(&hr);
    transfer_free(&tr);
    fourier_free(&fo);
    primordial_free(&pm);
    perturbations_free(&pt);
    thermodynamics_free(&th);
    background_free(&ba);
    dofree=false;
    return _FAILURE_;
  }

  if (distortions_init(ppr,pba,pth,ppt,ppm,psd) == _FAILURE_) {
    printf("\n\nError in distortions_init \n=>%s\n",psd->error_message);
    lensing_free(&le);
    harmonic_free(&hr);
    transfer_free(&tr);
    fourier_free(&fo);
    primordial_free(&pm);
    perturbations_free(&pt);
    thermodynamics_free(&th);
    background_free(&ba);
    dofree=false;
    return _FAILURE_;
  }

  dofree=true;
  return _SUCCESS_;
}

int ClassPlus::freeStructs(){
	
	if (distortions_free(&sd) == _FAILURE_) {
		printf("\n\nError in distortions_free \n=>%s\n",sd.error_message);
		return _FAILURE_;
	}
	
	if (lensing_free(&le) == _FAILURE_) {
		printf("\n\nError in lensing_free \n=>%s\n",le.error_message);
		return _FAILURE_;
	}

	if (fourier_free(&fo) == _FAILURE_) {
		printf("\n\nError in fourier_free \n=>%s\n",fo.error_message);
		return _FAILURE_;
	}

	if (harmonic_free(&hr) == _FAILURE_) {
		printf("\n\nError in harmonic_free \n=>%s\n",hr.error_message);
		return _FAILURE_;
	}

	if (primordial_free(&pm) == _FAILURE_) {
		printf("\n\nError in primordial_free \n=>%s\n",pm.error_message);
		return _FAILURE_;
	}

	if (transfer_free(&tr) == _FAILURE_) {
		printf("\n\nError in transfer_free \n=>%s\n",tr.error_message);
		return _FAILURE_;
	}

	if (perturbations_free(&pt) == _FAILURE_) {
		printf("\n\nError in perturb_free \n=>%s\n",pt.error_message);
		return _FAILURE_;
	}

	if (thermodynamics_free(&th) == _FAILURE_) {
		printf("\n\nError in thermodynamics_free \n=>%s\n",th.error_message);
		return _FAILURE_;
	}

	if (background_free(&ba) == _FAILURE_) {
		printf("\n\nError in background_free \n=>%s\n",ba.error_message);
		return _FAILURE_;
	}

	return _SUCCESS_;
}
