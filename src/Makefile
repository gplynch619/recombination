#CXX = g++-11
CXX = /opt/homebrew/Cellar/gcc/11.2.0_3/bin/g++-11
#CXX = g++
CLASS_DIR = class_h0 
CFLAGS = -O2 -fopenmp -Iclass_h0/include -Iclass_h0/external/HyRec2020 -Iclass_h0/external/RecfastCLASS -Iclass_h0/external/heating -I.
CLASSMODULES = class_h0/build/arrays.o class_h0/build/background.o class_h0/build/common.o \
	class_h0/build/dei_rkck.o class_h0/build/distortions.o class_h0/build/energy_injection.o \
	class_h0/build/evolver_ndf15.o class_h0/build/evolver_rkck.o class_h0/build/growTable.o \
	class_h0/build/helium.o class_h0/build/history.o class_h0/build/hydrogen.o \
	class_h0/build/hyperspherical.o class_h0/build/hyrectools.o \
	class_h0/build/injection.o class_h0/build/input.o class_h0/build/lensing.o \
	class_h0/build/noninjection.o class_h0/build/fourier.o class_h0/build/output.o \
	class_h0/build/parser.o class_h0/build/perturbations.o class_h0/build/primordial.o \
	class_h0/build/quadrature.o class_h0/build/sparse.o class_h0/build/harmonic.o \
	class_h0/build/thermodynamics.o class_h0/build/transfer.o \
	class_h0/build/trigonometric_integrals.o class_h0/build/wrap_hyrec.o class_h0/build/wrap_recfast.o

all: response

response: response.o ClassPlus.o BasicIO.o
	$(CXX) $(CFLAGS) $(CLASSMODULES) ClassPlus.o BasicIO.o response.o -o response

response.o: response.cpp BasicIO.cpp
	$(CXX) $(CFLAGS) -c response.cpp  -o response.o

BasicIO.o: BasicIO.cpp BasicIO.h
	$(CXX) $(CFLAGS) -c BasicIO.cpp -o BasicIO.o

ClassPlus.o: ClassPlus.cpp ClassPlus.h
	$(CXX) $(CFLAGS) -c ClassPlus.cpp -o ClassPlus.o

clean:
	rm -rf *.o response
