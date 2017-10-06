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

// **********************************
// Demonstration of matrix arithmetic
// **********************************

void arithmetic() {

    constexpr auto A = matrix<int, 6, 8>( [](auto pos){ return pos+1; } );

    constexpr auto B = div(
        2.5*mul(-A(range<1,3>, range<4,7>), A(range<0,2>, range<1,4>)),
        1.8 + A(range<3,5>, range<0,5>)/3.7 * (1 - A(all, range<0,3>)*0.2)
    );
    constexpr auto C =
        (A(range<0,2>, range<3,5>) + 1.6) /
    (
        -0.6*A(range<2,4>, range<1,3>).T() +
        div(2 , A(range<0,2>,0)-0.9) * A(0,range<0,2>) +
        eye<double, 3>()
    );

    constexpr auto D =
        (B + B(all,2) * B(0,all) + B(all,1) * B(0,all)).T() %
        (A(range<0,3>, range<2,4>) * C
    );

    constexpr auto E =
        -(eye<int,1,3>()/2.3 * A(range<2,4>, range<5,7>)) *
        div(2,D) * (A(range<3,5>, range<0,2>) * div(2.5 , (10+eye<int,3,1>())));

     // The answer, checked with MATLAB, is 238.0481 to seven digits.
     constexpr double tol    = 1.0e-7;
     constexpr double relerr = (E - 238.0481) / E;
     static_assert(-tol < relerr && relerr < tol);


// MATLAB code:     
//
//A = reshape(1:48, 8, 6)';
//B = 2.5*(-A(2:4, 5:8) .* +A(1:3, 2:5)) ./ (1.8 + A(4:6, 1:6)/3.7 * (1 - A(:, 1:4)*0.2));
//C = (A(1:3, 4:6) + 1.6) / (-0.6*A(3:5, 2:4)' + (2 ./ (A(1:3,1)-0.9)) * A(1,1:3) + eye(3));
//D = (B + B(:,3) * B(1,:) + B(:,2) * B(1,:))' \ (A(1:4, 3:5) * C);
//E = -(eye(1,3)/2.3 * A(3:5, 6:8)) * (2./D) * (A(4:6, 1:3) * (2.5 ./ (10+eye(3,1))))
//
}
