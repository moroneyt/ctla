## Compile-time linear algebra in C++

This library provides the class template `ctla::matrix` and its associated operations for compile-time (constexpr) linear algebra.

### Key features

•	Header-only library

•	All functions constexpr, all computation done at compile time

•	Intuitive syntax for initialisation, indexing, augmenting

•	Matrix arithmetic including inverses and linear systems supported

•	Block matrices supported

•	Runtime printing in MATLAB-compatible syntax if required

•	Documentation…not written.  Available matrix operations can be found in matrix.h, and see the examples start_here.cpp, arithmetic.cpp, regression.cpp, blocks.cpp.

### Installation
Nothing to install.  Just #include "matrix.h" and off you go.

### Requires
C++17 conforming compiler.  Tested on Clang 5.0 and GCC 7.2.

### Example

```c++
// ********************************
// Simple linear regression example
// ********************************

#include "matrix.h"

using namespace ctla;

template<auto Val>
struct Print{
    // force a compile error, hopefully with a message that outputs Val
    char dummy[0*Val-1];
};

void regression() {

    // The input values
    constexpr auto x = seq<1,10>().T();

    // The response values
    constexpr matrix<double, 10, 1> y(
        {3.8180, 5.0613, 5.2806, 4.0659, 4.1211, 2.2983, 0.2743, -0.9785, -6.8954, -10.4222}
    );

    // Try a linear model y = c0 + c1*x
    constexpr auto A_linear = augc(ones<double,10,1>(), x);
    constexpr auto c_linear = A_linear % y;
    constexpr auto y_linear = A_linear * c_linear;
    
    // Compute the R^2 value
    constexpr auto SStot           = normsq(y - mean(y));
    constexpr auto SSres_linear    = normsq(y - y_linear);
    constexpr auto Rsquared_linear = 1 - SSres_linear / SStot;
    Print<int(Rsquared_linear*100)>();
    // compiler prints error about Print<76>  i.e.  R^2 == 76%
    
    // Try a quadratic model y = c0 + c1*x + c2*x^2
    constexpr auto A_quadratic = augc(A_linear, mul(x,x));
    constexpr auto c_quadratic = A_quadratic % y;
    constexpr auto y_quadratic = A_quadratic * c_quadratic;
    
    // See if we get a better R^2 value
    constexpr auto SSres_quadratic    = normsq(y - y_quadratic);
    constexpr auto Rsquared_quadratic = 1 - SSres_quadratic / SStot;
    Print<int(Rsquared_quadratic*100)>();
    // compiler prints error about Print<98>  i.e.  R^2 == 98%
    
}
```
