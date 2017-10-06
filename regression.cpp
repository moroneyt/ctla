// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#include "matrix.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wmissing-braces" // https://bugs.llvm.org/show_bug.cgi?id=21629
#endif

using namespace ctla;

// ************************************
// Simple linear regression application
// ************************************


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
