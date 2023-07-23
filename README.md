## LBFGS

This repository contains a C++20 port of the [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) implementation from [LBFGS-Lite](https://github.com/ZJU-FAST-Lab/LBFGS-Lite), with a number of usability improvements:

- error handling using the [outcome library](https://ned14.github.io/outcome/)
- a simplified API similar to the [Eigen Levenberg-Marquardt module](https://eigen.tuxfamily.org/dox/unsupported/group__NonLinearOptimization__Module.html)
- the ability to specify the scalar type (e.g. `float` or `double`)


### Limitations

- no step bounds
- no progress report
- no cancellation

These will be addressed in the future.