COMPILER=nvcc
FLAGS=--std c++17 -Wno-deprecated-gpu-targets -I. -Isrc -O2

build:
	mkdir -p bin
	$(COMPILER) $(FLAGS) src/main.cu -o bin/image_processor.exe

run:
	./bin/image_processor.exe -i data/input -o data/output -l data/results.csv

run-blur:
	./bin/image_processor.exe -i data/input -o data/output -l data/results_blur.csv -b

clean:
	rm -f bin/image_processor.exe data/results*.csv
