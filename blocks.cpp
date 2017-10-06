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

// ******************************************************
// Demonstration of block matrices (matrices of matrices)
// ******************************************************

void blocks() {

    // a is a 4x2 matrix of 2x3 blocks
    constexpr auto a = ones<matrix<double, 2, 3>, 4, 2>();
    
    // b is a 2x3 matrix of 3x4 blocks
    constexpr auto b = ones<matrix<double, 3, 4>, 2, 3>();

    // These matrices conform, so we can multiply them
    constexpr auto c = a * b;

    // The result is a 4x3 matrix of 2x4 blocks
    static_assert(c.rows() == 4);
    static_assert(c.cols() == 3);
    static_assert(c(0,0).rows() == 2);
    static_assert(c(0,0).cols() == 4);

    // Block matrices can be multiplied by scalars as usual
    constexpr auto d = 2 * c;

    // (No change to any of these compared to c)
    static_assert(d.rows() == 4);
    static_assert(d.cols() == 3);
    static_assert(d(0,0).rows() == 2);
    static_assert(d(0,0).cols() == 4);

    // But they can also be multiplied by conforming block scalars
    constexpr auto s1 = ones<double, 3, 2>();
    constexpr auto s2 = ones<double, 4, 3>();
    
    // Here, on the left:
    constexpr auto e = s1 * c;

    // Outer dimensions are unchanged, inner dimensions are now 3x4
    static_assert(e.rows() == 4);
    static_assert(e.cols() == 3);
    static_assert(e(0,0).rows() == 3);
    static_assert(e(0,0).cols() == 4);
    
    // And now, on the right:
    constexpr auto f = c * s2;
    
    // Outer dimensions are unchanged, inner dimensions are now 2x3
    static_assert(f.rows() == 4);
    static_assert(f.cols() == 3);
    static_assert(f(0,0).rows() == 2);
    static_assert(f(0,0).cols() == 3);

    // Now let's build up some new matrices starting with this 2x2 block
    using blk22 = matrix<double, 2, 2>;
    constexpr blk22 a00({1,2,
                         3,4});

    // We could build a 3x3 matrix of 2x2 blocks
    constexpr matrix<blk22, 3, 3> amat({  a00,        2*a00,     div(1,a00),
                                        1+a00,        a00/2,    -a00,
                                        mul(a00,a00), a00-1,     inv(a00)});
    
    // Or, we could augment the 2x2 blocks to build a 6x6 matrix of doubles
    constexpr auto bmat =     augr(augc(  a00,        2*a00,     div(1,a00)),
                                   augc(1+a00,        a00/2,    -a00       ),
                                   augc(mul(a00,a00), a00-1,     inv(a00)  ));

    // Since amat is a square matrix of square blocks, we can find its inverse.
    constexpr auto amati = inv(amat);

    // bmat is a square matrix of doubles, so naturally we can find its inverse too.
    constexpr auto bmati = inv(bmat);

    // Check they are correct to within epsilon
    // (this is using the squared norm, so the test is conservative)
    constexpr double eps = std::numeric_limits<double>::epsilon();
    static_assert(
        normsq(amat * amati - eye<blk22, amat.rows()>()) < eps
    );
    static_assert(
        normsq(bmat * bmati - eye<double, bmat.rows()>()) < eps
    );

    // Blocks notwithstanding, the matrices a and b are really the same.
    // Let's solve the same linear system with both, and confirm we get the same result.

    // Using a blocked right hand side
    using blk21 = matrix<double, 2, 1>;
    constexpr matrix<blk21, 3, 1> arhs({blk21({1, 2}), blk21({3, 4}), blk21({5, 6})});
    
    // Using a non-blocked right hand side
    constexpr matrix<double, 6,1> brhs({1, 2, 3, 4, 5, 6});

    // We computed inverses earlier, but forget about that for now and just solve the systems
    constexpr auto asol = amat % arhs;
    constexpr auto bsol = bmat % brhs;

    // Now unblock the asol solution
    constexpr auto asol_non_blocked = augr(asol(0), asol(1), asol(2));
    
    // The two solutions are the same
    static_assert(normsq(asol_non_blocked - bsol) < eps);

}
