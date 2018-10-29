
# Environment

* Ubuntu 18.04.1 LTS
* libsdl2-dev = 2.0.8
* libsdl2-ttf-dev = 2.0.14

library installation method in Ubuntu 18.04.1:
```
$ sudo apt install -y libsdl2-dev libsdl2-ttf-dev
```
# How to use
```
$ mkdir build && cd build
$ cmake ..
$ make -j8 && make install
$ cd bin
$ ./NBodySimulation
```

# Release Notes

## cpu version - Oct. 28
Simple implementation with boundary detection. 60fps.


## gpu version 1.0 - Oct. 29
This is a naive version with limited scalability, which can process at most 1024 bodies.
That's because we simply map i/j in serial version to blockIdx and thredIdx, which has a
maximum of 1024.

Features:
* 60 frames when doing 1024 bodies simulation with trail effect
* support trails using frame decay, this part is writen a cuda kernel
* applied cuda kernel to acclerate rasterization
* implemented boundary detection
* some optimization details:
    * Avoid redundat cudaMalloc and cudaFree operation when doing rasterization
    * slightly optimized reduction in time step, when calculate acceleration

TODOs:
* using more flexible kernel to compute time step, which has better scalability (support more bodies)
* do more optimizations to achieve better effiency, e.g. AOS(array of structure) -> SOA
* add more features, e.g. carefully designed initial conditions, force fields, user interaction of effect control

