

## cuda verison

### version 1.0 - Oct. 29

This is a naive version with limited scalability, which can process at most 1024 bodies.
That's because we simply map i/j in serial version to blockIdx and thredIdx, which has a
maximum of 1024.