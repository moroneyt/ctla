// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifndef CTLA_DETAIL_INVERSE_H
#define CTLA_DETAIL_INVERSE_H

#include "range.h"
#include "traits.h"

#include <cstddef>
#include <tuple>

namespace ctla {

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N> class matrix;

namespace detail {

// Scalar inversion
template <typename T> constexpr auto do_inv(T t) { return 1 / t; }

// 1x1 inversion
template <typename U> constexpr auto do_inv(const matrix<U, 1, 1>& x) {
   return matrix<U, 1, 1>({do_inv(x(0, 0))});
}

// 2x2 block partitioned inversion
// Pre: a necessary condition is that the top left block be recursively block-
// invertible. Hence not all nonsingular matrices are invertible by this
// algorithm. Even fewer are accurately invertible since the algorithm isn't
// numerically stable.
template <typename AA, typename BB, typename CC, typename DD>
constexpr auto block_inverse(const AA& A, const BB& B, const CC& C,
                             const DD& D) {
   const auto invA = do_inv(A);
   const auto invS = do_inv(D - C * invA * B);

   const auto Q = invA + invA * B * invS * C * invA;
   const auto R = -invA * B * invS;
   const auto S = -invS * C * invA;
   const auto T = invS;

   return std::make_tuple(Q, R, S, T);
}

// 2x2 inversion
template <typename U> constexpr auto do_inv(const matrix<U, 2, 2>& a) {
   const auto A = a(0, 0);
   const auto B = a(0, 1);
   const auto C = a(1, 0);
   const auto D = a(1, 1);

   const auto [Q, R, S, T] = block_inverse(A, B, C, D);

   return matrix<U, 2, 2>({Q, R, S, T});
}

// NxN inversion
template <typename U, std::ptrdiff_t N>
constexpr auto do_inv(const matrix<U, N, N>& a) {
   const auto A = a(range<0, N / 2 - 1>, range<0, N / 2 - 1>);
   const auto B = a(range<0, N / 2 - 1>, range<N / 2, N - 1>);
   const auto C = a(range<N / 2, N - 1>, range<0, N / 2 - 1>);
   const auto D = a(range<N / 2, N - 1>, range<N / 2, N - 1>);

   const auto [Q, R, S, T] = block_inverse(A, B, C, D);

   return augr(augc(Q, R), augc(S, T));
}

template <typename U, typename V, std::ptrdiff_t N, std::ptrdiff_t K>
constexpr auto solve_left(const matrix<U, N, N>& a, const matrix<V, N, K>& b) {
   return do_inv(a) * b; // we'll just solve using the inverse matrix
}

template <typename U, typename V, std::ptrdiff_t N, std::ptrdiff_t K>
constexpr auto solve_right(const matrix<U, K, N>& b, const matrix<V, N, N>& a) {
   return b * do_inv(a); // we'll just solve using the inverse matrix
}

} // namespace detail

} // namespace ctla

#endif
