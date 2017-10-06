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

// *********************************************
// Demonstration of some basic matrix operations
// *********************************************

void start_here() {

    // Matrix construction from list of values
    constexpr matrix<int, 4, 3> A({1,  2,  3,
                                   4,  5,  6,
                                   7,  8,  9,
                                  10, 11, 12});

    // Simple indexing
    constexpr auto entry_12 = A(1,2);
    static_assert(entry_12 == 6);
    
    // Row indexing
    constexpr auto row_2 = A(2, all);
    static_assert(
        row_2 == matrix<int, 1, 3>({7,8,9})
    );

    // Column indexing
    constexpr auto col_1 = A(all, 1);
    static_assert(
        col_1 == matrix<int, 4, 1>({ 2,
                                     5,
                                     8,
                                    11})
    );
    
    // Range indexing
    constexpr auto submat = A(range<1,3>, range<1,2>);
    static_assert(
        submat == matrix<int, 3, 2>({5, 6,
                                     8, 9,
                                    11,12})
    );

    constexpr auto rows_0_to_2 = A(range<0,2>, all);
    static_assert(
        rows_0_to_2 == matrix<int, 3, 3>({1,2,3,
                                          4,5,6,
                                          7,8,9})
    );
    
    constexpr auto cols_1_to_2 = A(all, range<1,2>);
    static_assert(
        cols_1_to_2 == matrix<int, 4, 2>({2, 3,
                                          5, 6,
                                          8, 9,
                                         11,12})
    );

    constexpr auto some_of_column_2 = A(range<1,3>, 2);
    static_assert(
        some_of_column_2 == matrix<int, 3, 1>({ 6,
                                                9,
                                               12})
    );

    constexpr auto some_of_row_3 = A(3, range<0,1>);
    static_assert(
        some_of_row_3 == matrix<int, 1, 2>({10, 11})
    );

    // Transpose
    constexpr auto AT = A.T();
    static_assert(
        AT == matrix<int, 3, 4>({1,4,7,10,
                                 2,5,8,11,
                                 3,6,9,12})
    );

    // Transform
    constexpr auto Aiseven = transform(A, [](auto val){return val % 2 == 0;});
    static_assert(
        Aiseven == matrix<bool, 4, 3>({0, 1, 0,
                                       1, 0, 1,
                                       0, 1, 0,
                                       1, 0, 1})
    );

    // Default construction is a zero matrix
    constexpr matrix<int, 2, 5> Z;
    static_assert(
       Z == matrix<int, 2, 5>({0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0})
    );
    static_assert(Z == zeros<int, 2, 5>());

    // Here is a matrix with all entries equal to the same value
    constexpr auto F = 3*ones<int, 2, 5>();
    static_assert(
        F == matrix<int, 2, 5>({3, 3, 3, 3, 3,
                                3, 3, 3, 3, 3})
    );
    static_assert(F == Z + 3);

    // Another option is to use the unary function constructor
    constexpr matrix<int, 2, 5> G([](auto){return 3;});
    static_assert(G == F);

    // Construction from a binary function (i,j) -> A(i,j)
    constexpr matrix<int, 4, 3> B([](auto i, auto j){return 10*i + j;});
    static_assert(
        B == matrix<int, 4, 3>({ 0,  1,  2,
                                10, 11, 12,
                                20, 21, 22,
                                30, 31, 32})
    );

    // Matrices can be augmented
    constexpr auto R = augr(Z, F); // augment as rows
    static_assert(
        R == matrix<int, 4, 5>({0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0,
                                3, 3, 3, 3, 3,
                                3, 3, 3, 3, 3})
    );

    constexpr auto C = augc(col_1, B, Aiseven); // augment as columns
    static_assert(
        C == matrix<int, 4, 7>({2,  0,  1,  2, 0, 1, 0,
                                5,  10, 11, 12, 1, 0, 1,
                                8,  20, 21, 22, 0, 1, 0,
                                11, 30, 31, 32, 1, 0, 1})
    );
    
    // More complex matrices can be built by augmenting simpler ones
    constexpr auto Q = augr(
                            augc(1, zeros<int,1,2>(), 1),
                            augc(eye<int,2>(), ones<int,2,2>()),
                            seq<1,4>()
                           );
    static_assert(
        Q == matrix<int, 4, 4>({1, 0, 0, 1,
                                1, 0, 1, 1,
                                0, 1, 1, 1,
                                1, 2, 3, 4})
    );

    // Find the sum of all entries in a row or column matrix
    constexpr auto rowsum = sum(Q(3,all));
    static_assert(rowsum == 10);
    
    constexpr auto colsum = sum(Q(all,2));
    static_assert(colsum == 5);

    // Find the product of all entries in a row or column matrix
    constexpr auto rowprod = prod(seq<1,10>());
    static_assert(rowprod == 3628800);
    
    constexpr auto colprod = prod(Q(range<1,3>,2));
    static_assert(colprod == 3);

    // Another example with sequences
    constexpr auto digits = seq<'0', '9'>();
    static_assert(
        digits == matrix<char, 1, 10>(
            {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        )
    );

}
