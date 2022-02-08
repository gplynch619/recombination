////////////////////////////////////////////////////////////////////////////
//  ClassPlus: A simplified cpp wrapper for CLASS 
// 
//	Not very different from the contents of cpp/ written by Stephane Plaszczynski
//	in the CLASS distribution. Simplified features down make this suitable
//	for a basic interface. 
// 
////////////////////////////////////////////////////////////////////////////

#ifndef CLASSPLUS_H
#define CLASSPLUS_H

#include"class.h"

//STD
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <ostream>
#include <map>
#include <iterator>

using namespace std;

//general utility to convert safely numerical types to string
template<typename T> string str(const T &x);
//specialisations
template<> string str (const float &x);
template<> string str (const double &x);
template<> string str (const bool &x); //"yes" or "no"
template<> string str (const string &x);

string str(const char* x);
//////////////////////////////////////////////////////////////////////////
//class to encapsulate CLASS parameters from any type (numerical or string)
class ClassParams{
	public:	
		ClassParams(){};
		ClassParams( const ClassParams& o): params(o.params){};
		
		template<typename T> unsigned set(const string& key,const T& value){
			params[key] = str(value);
			return params.size();
		}
		
		void print(){
			
			if(params.empty()){
				cout<<"\n Current parameter list is empty"<<endl; 
				return;
			}
			
			map<string, string>::iterator itr;
			cout << "\nCurrent parameters are : \n";
			cout << "PARAM\tValue\n";
    		for (itr = params.begin(); itr != params.end(); ++itr) {
				cout << '\t' << itr->first << '\t' << itr->second << '\n';
    		}
			cout << endl;
		};

		inline unsigned size() const {return params.size();}
		
 		map<string, string> params;
};

class ClassPlus {

	public:
		//ClassPlus(bool verbose=true);
		ClassPlus( ClassParams& params, bool verbose=true );

	private:
		struct file_content fc;
		struct precision pr;        /* for precision parameters */
		struct background ba;      /* for cosmological background */
		struct thermodynamics th;           /* for thermodynamics */
		struct perturbations pt;         /* for source functions */
		struct transfer tr;        /* for transfer functions */
		struct primordial pm;       /* for primordial spectra */
		struct harmonic sp;          /* for output spectra */
		struct fourier nl;        /* for non-linear spectra */
		struct lensing le;          /* for lensed spectra */
		struct distortions sd;      /* for spectral distortions */
		struct output op;           /* for output files */

		ErrorMsg _errmsg;
		double * cl;

		//helpers
		bool dofree;
		int freeStructs();
};
#endif
