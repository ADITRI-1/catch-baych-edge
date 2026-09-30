NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14
.PHONY: build run clean
build: edge_detect
edge_detect: src/main.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@
run: build
	bash scripts/run.sh
clean:
	rm -f edge_detect
