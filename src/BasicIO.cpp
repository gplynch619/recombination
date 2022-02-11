#include "BasicIO.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <fstream>

AbstractColumn::~AbstractColumn() {}

BasicIO::~BasicIO(){
	for(int i=0; i<ncols; i++){
		delete columns[i];
	}
}

void BasicIO::write(){
	std::ofstream outfile;
	const int col_width = 15;
	outfile.open(fname);
	outfile<<std::setprecision(10);
	outfile<<"#";
	outfile<<std::left<<std::setw(5)<<column_names.at(0);
	for(int i=1; i<columns.size(); i++){
		outfile<<std::left<<std::setw(col_width)<<column_names.at(i);
	}
	outfile<<std::endl;
	std::cout<<"columns[0]->nrows is "<<columns.at(0)->nrows<<std::endl;
	for(int i = 0; i<columns.at(0)->nrows; i++){
		outfile<<std::scientific<<std::left;
		for(int j = 0; j<columns.size(); j++){
			outfile<<std::setw(col_width);
			columns.at(j)->write(outfile, i);
		}
		outfile<<std::endl;
	}
	outfile.close();
}
