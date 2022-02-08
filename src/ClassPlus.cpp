#include "ClassPlus.h"

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
//specilization
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

//instanciations
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

ClassPlus::ClassPlus(ClassParams & CP, bool verbose): dofree(true){

//prepare fp structure
	size_t n=CP.params.size();
//
	parser_init(&fc,n,(char*)"pipo",_errmsg);

//config
	map<string, string>::iterator itr;

	int i = 0;
	for (itr = CP.params.begin(); itr!=CP.params.end(); ++itr){
		strcpy(fc.name[i],(itr->first).c_str());
		strcpy(fc.value[i],(itr->second).c_str());
/* if (pars.key(i)=="l_max_scalars") { */
  /* istringstream strstrm(pars.value(i)); */
  /* strstrm >> _lmax; */
/* } */
		i++;
	}
//if( verbose ) cout << __FILE__ << " : using lmax=" << _lmax <<endl;

//input
	if (input_read_from_file(&fc,&pr,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,&sd,&op,_errmsg) == _FAILURE_)
		throw invalid_argument(_errmsg);

	//Makes sure every supplied parameter was read. 
	for (size_t i=0;i<CP.params.size();i++){
		if (fc.read[i] !=_TRUE_) throw invalid_argument(string("Invalid CLASS parameter: ")+fc.name[i]);
	}

}
