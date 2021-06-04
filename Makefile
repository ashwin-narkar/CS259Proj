cc=g++

rnncpu:
	mkdir -p build
	g++ -o build/rnncpu rnncpu.cpp

clean:
	rm -r build