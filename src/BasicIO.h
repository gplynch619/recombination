#ifndef BASICIO_H
#define BASICIO_H

#include <vector>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <iostream>

class AbstractColumn{
	public:
		AbstractColumn(int n): nrows(n) {};
		virtual ~AbstractColumn() = 0;
		virtual void write(std::ofstream& outfile, int i) = 0;
		int nrows;
};


template <class T>
class Column : public AbstractColumn {
	public:
		Column(std::vector<T>& v): AbstractColumn(v.size()), data(v) {}
		~Column();
		void write(std::ofstream& outfile, int i);
		std::vector<T> data;
};

/* template <class T> */
/* Column<T>::Column(std::vector<T>& v){ */
/*     nrows = v.size(); */
/*     data = v; */
/* } */

template <class T>
void Column<T>::write(std::ofstream& outfile, int i){
	outfile<<data.at(i);
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
		std::vector<std::string> column_names;
		std::string fname;
		int ncols;
};

template <class T>
void BasicIO::attach(std::vector<T>& v, std::string name){
	Column<T>* col = new Column<T>(v);
	columns.push_back(col);
	column_names.push_back(name);
}
#endif
