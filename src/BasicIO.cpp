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
	outfile.open(fname);
	outfile<<"#";
	for(int i =0; i<columns.size(); i++){
		columns.at(i)->write_header(outfile);	
	}
	outfile<<std::endl;
	std::cout<<"nrows "<<columns.at(0)->nrows<<std::endl;
	for(int i = 0; i<columns.at(0)->nrows; i++){
		for(int j = 0; j<columns.size(); j++){
			columns.at(j)->write(outfile, i);
		}
		outfile<<std::endl;
	}
	outfile.close();
}
