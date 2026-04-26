all:
	g++ -O3 background.cpp -o background.o
	g++ -O3 perturbation.cpp -o perturbation.o
	g++ rescale.cpp -o rescale.o
	./background.o
	./perturbation.o
	./rescale.o

clean:
	rm -f *.dat
	rm -f *.o



