
# Environment

* Ubuntu 18.04.1 LTS
* cuda == 10.0
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
$ ./NBodySimulation [random | stable | symmetric | breathe]
```
(Arguments in `[]` is optional).

# Release Notes

## cpu version - Oct. 28
Simple implementation with boundary detection. 60fps.


## gpu version 1.0 - Oct. 29
This is a naive version with limited scalability, which can process at most 1024 bodies.
That's because we simply map i/j in serial version to blockIdx/thredIdx, which has a
maximum of 1024.

### Features:
* 60 fps when doing 1024 bodies simulation with trail effect
* support trails using frame decay, this part is writen with cuda kernel
* applied cuda kernel to acclerate rasterization
* implemented boundary detection
* some optimization details:
    * Avoid redundat cudaMalloc and cudaFree operation when doing rasterization
    * slightly optimized reduction in time step, when calculate acceleration

### TODOs:
* write more flexible kernel to compute time step, which has better scalability (support more bodies)
* do more optimizations to achieve better effiency, e.g. AOS(array of structure) -> SOA
* add more features, e.g. carefully designed initial conditions, force fields, user interaction of effect control


## gpu version 2.0 - Nov. 1st
This version is scalable and has high peroformance.

### Features
* 60fps when doing 8096 bidies simulation on 980Ti (12-14 fps on K620).
* support trail effect using frame decay (frame = beta * frame + newState) over time. This effect is implemented with
cuda kernel.
* applied cuda kernel to accllerate rasterization
* boundary detection and  bounce back effct
* elaborate initial state. Support command line arguments to specify pattern, i.e. choose different initial states
    * usage: ./NBodySimulation [random | stable | symmetric | breathe]


### Optimization details
* allocate/Free memory collectively, avoid redudantly frequent allocation and free
* keep a shared memory copy of input data to reduce global memory accessing
* surprisingly, we found rsqrt() is far more fast than pow() (at least about 6x - 8x)
* store intermediate results to save float-point computations.

### Acknowlegement
Thanks [Hu Wenbo](https://github.com/crisb-DUT) for providing a cmake project file for my referrence and pointing
out the problem of using pow() and rsqrt().
