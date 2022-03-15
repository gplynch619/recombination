#ifndef BASICIO_H
#define BASICIO_H

#include <vector>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <typeinfo>
#include <fstream>
#include <iostream>
#include <iomanip>

class AbstractColumn{
	public:
		AbstractColumn(int n): nrows(n) {};
		virtual ~AbstractColumn() = 0;
		virtual void write(std::ofstream& outfile, int i) = 0;
		virtual void write_header(std::ofstream& outfile) = 0;
		int nrows;
};


template <class T>
class Column : public AbstractColumn {
	public:
		Column(std::vector<T>& v, std::string name): 
		AbstractColumn(v.size()), 
		data(v), 
		use_scientific(true), 
		prec(5),
		colname(name){
			std::cout<<"nrows constructor "<<nrows<<std::endl;
			const char* t = typeid(T).name();
			int padding = 4;
			switch(*t){
				case 'j':
					prec=4;
					width=prec;
					use_scientific = false;
					break;
				case 'i':
					prec=4;
					width=prec;
					use_scientific = false;
					break;
				case 'd':
					prec=10;
					width=prec+padding;
					use_scientific = true;
					break;
				case 'f':
					prec=10;
					width=prec+padding;
					use_scientific = true;
					break;
				default:
					prec=10;
					width=prec+padding;
					use_scientific = true;
				// more types here later
			}
		}	
		
		~Column();
		void write(std::ofstream& outfile, int i);
		void write_header(std::ofstream& outfile);
		std::vector<T> data;
		std::string colname;
		int width;
		int prec;
		bool use_scientific;
};

template <class T>
void Column<T>::write_header(std::ofstream& outfile){
	outfile<<std::right<<std::setw(width)<<colname<<"\t";
}

template <class T>
void Column<T>::write(std::ofstream& outfile, int i){
	if(use_scientific){
		outfile<<std::scientific<<std::setprecision(prec)<<std::setw(width)<<data.at(i)<<"\t";
	} else {
		outfile<<std::left<<std::setw(width)<<data.at(i)<<"\t";
	}
}

template <class T>
Column<T>::~Column() {}

class BasicIO {
	public:
		BasicIO(std::string outname) : fname(outname), ncols(0) {}
		~BasicIO();
		
		template <class T>
		void attach(std::vector<T>& v, std::string name="");
		void write();
		inline int get_ncols() {return ncols;}
		inline std::string filename() {return fname;}
	private:
		std::vector<AbstractColumn*> columns;
		std::string fname;
		int ncols;
};

template <class T>
void BasicIO::attach(std::vector<T>& v, std::string name){
	Column<T>* col = new Column<T>(v, name);
	columns.push_back(col);
}
#endif
