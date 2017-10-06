// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifndef CTLA_DETAIL_IS_CONFORMING
#define CTLA_DETAIL_IS_CONFORMING

#include "traits.h"

namespace ctla {

namespace detail {

// We want to enable/disable the arithmetic operators only when the operands
// are compatible.  This is particularly important when the element types of
// the operands are themselves matrices, i.e. block matrices.

// matrix matrix sum
// *****************
template <typename U, typename V> struct is_matrix_addition_conforming_impl;

template <typename U, typename V> struct is_matrix_addition_conforming {
   constexpr static bool value =
       is_matrix_addition_conforming_impl<collapse_t<U>, collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_matrix_addition_conforming_v =
    is_matrix_addition_conforming<U, V>::value;

// matrix of scalars plus matrix of scalars
template <typename U, typename V> struct is_matrix_addition_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// matrix of blocks plus matrix of scalars
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_matrix_addition_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// matrix of scalars plus matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_matrix_addition_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value = false;
};

// matrix of blocks plus matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_matrix_addition_conforming_impl<matrix<U, M, N>, matrix<V, L, K>> {
   constexpr static bool value =
       (M == L) && (N == K) &&
       is_matrix_addition_conforming_impl<collapse_t<U>, collapse_t<V>>::value;
};

// matrix matrix product
// *********************
template <typename U, typename V> struct is_matrix_multiply_conforming_impl;

template <typename U, typename V> struct is_matrix_multiply_conforming {
   constexpr static bool value =
       is_matrix_multiply_conforming_impl<collapse_t<U>, collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_matrix_multiply_conforming_v =
    is_matrix_multiply_conforming<U, V>::value;

// matrix of scalars times matrix of scalars
template <typename U, typename V> struct is_matrix_multiply_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// matrix of blocks times matrix of scalars
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_matrix_multiply_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// matrix of scalars times matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_matrix_multiply_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value = false;
};

// matrix of blocks times matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_matrix_multiply_conforming_impl<matrix<U, M, N>, matrix<V, L, K>> {
   constexpr static bool value =
       (N == L) &&
       is_matrix_multiply_conforming_impl<collapse_t<U>, collapse_t<V>>::value;
};

// matrix left division
// ********************
template <typename U, typename V>
struct is_matrix_left_division_conforming_impl;

template <typename U, typename V> struct is_matrix_left_division_conforming {
   constexpr static bool value =
       is_matrix_left_division_conforming_impl<collapse_t<U>,
                                               collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_matrix_left_division_conforming_v =
    is_matrix_left_division_conforming<U, V>::value;

// matrix of scalars solving matrix of scalars
template <typename U, typename V>
struct is_matrix_left_division_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// matrix of blocks solving matrix of scalars
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_matrix_left_division_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// matrix of scalars solving matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_matrix_left_division_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value = false;
};

// matrix of blocks solving matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_matrix_left_division_conforming_impl<matrix<U, M, N>,
                                               matrix<V, L, K>> {
   constexpr static bool value =
       (M == N) && (N == L) &&
       is_matrix_left_division_conforming_impl<collapse_t<U>,
                                               collapse_t<V>>::value;
};

// matrix right division
// *********************
template <typename U, typename V>
struct is_matrix_right_division_conforming_impl;

template <typename U, typename V> struct is_matrix_right_division_conforming {
   constexpr static bool value =
       is_matrix_right_division_conforming_impl<collapse_t<U>,
                                                collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_matrix_right_division_conforming_v =
    is_matrix_right_division_conforming<U, V>::value;

// matrix of scalars dividing matrix of scalars
template <typename U, typename V>
struct is_matrix_right_division_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// matrix of blocks dividing matrix of scalars
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_matrix_right_division_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// matrix of scalars dividing matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_matrix_right_division_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value = false;
};

// matrix of blocks dividing matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_matrix_right_division_conforming_impl<matrix<U, M, N>,
                                                matrix<V, L, K>> {
   constexpr static bool value =
       (N == L) && (L == K) &&
       is_matrix_right_division_conforming_impl<collapse_t<U>,
                                                collapse_t<V>>::value;
};

// scalar matrix product
// *********************
template <typename U, typename V>
struct is_left_scalar_multiply_conforming_impl;

template <typename U, typename V> struct is_left_scalar_multiply_conforming {
   constexpr static bool value =
       is_left_scalar_multiply_conforming_impl<collapse_t<U>,
                                               collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_left_scalar_multiply_conforming_v =
    is_left_scalar_multiply_conforming<U, V>::value;

// scalar times matrix of scalars
template <typename U, typename V>
struct is_left_scalar_multiply_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// block scalar times matrix of scalars
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_left_scalar_multiply_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// scalar times matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_left_scalar_multiply_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value =
       is_left_scalar_multiply_conforming_impl<U, collapse_t<V>>::value;
};

// block scalar times matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_left_scalar_multiply_conforming_impl<matrix<U, M, N>,
                                               matrix<V, L, K>> {
   constexpr static bool value =
       (N == L) &&
       is_matrix_multiply_conforming_impl<collapse_t<U>, collapse_t<V>>::value;
};

// matrix scalar product (scalar parameter is first)
// *********************
template <typename U, typename V>
struct is_right_scalar_multiply_conforming_impl;

template <typename U, typename V> struct is_right_scalar_multiply_conforming {
   constexpr static bool value =
       is_right_scalar_multiply_conforming_impl<collapse_t<U>,
                                                collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_right_scalar_multiply_conforming_v =
    is_right_scalar_multiply_conforming<U, V>::value;

// matrix of scalars times scalar
template <typename U, typename V>
struct is_right_scalar_multiply_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// matrix of scalars times block
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_right_scalar_multiply_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// matrix of blocks times scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_right_scalar_multiply_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value =
       is_right_scalar_multiply_conforming_impl<U, collapse_t<V>>::value;
};

// matrix of blocks times block scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_right_scalar_multiply_conforming_impl<matrix<U, M, N>,
                                                matrix<V, L, K>> {
   constexpr static bool value =
       (M == K) &&
       is_matrix_multiply_conforming_impl<collapse_t<V>, collapse_t<U>>::value;
};

// scalar matrix quotient
// **********************
template <typename U, typename V>
struct is_left_scalar_quotient_conforming_impl;

template <typename U, typename V> struct is_left_scalar_quotient_conforming {
   constexpr static bool value =
       is_left_scalar_quotient_conforming_impl<collapse_t<U>,
                                               collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_left_scalar_quotient_conforming_v =
    is_left_scalar_quotient_conforming<U, V>::value;

// scalar dividing matrix of scalars
template <typename U, typename V>
struct is_left_scalar_quotient_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// block scalar dividing matrix of scalars
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_left_scalar_quotient_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// scalar dividing matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_left_scalar_quotient_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value =
       (M == N) &&
       is_left_scalar_quotient_conforming_impl<U, collapse_t<V>>::value;
};

// block scalar dividing matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_left_scalar_quotient_conforming_impl<matrix<U, M, N>,
                                               matrix<V, L, K>> {
   constexpr static bool value =
       (L == K) && (N == L) &&
       is_matrix_right_division_conforming_impl<collapse_t<U>,
                                                collapse_t<V>>::value;
};

// matrix scalar quotient (scalar parameter is first)
// **********************
template <typename U, typename V>
struct is_right_scalar_quotient_conforming_impl;

template <typename U, typename V> struct is_right_scalar_quotient_conforming {
   constexpr static bool value =
       is_right_scalar_quotient_conforming_impl<collapse_t<U>,
                                                collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_right_scalar_quotient_conforming_v =
    is_right_scalar_quotient_conforming<U, V>::value;

// matrix of scalars dividing scalar
template <typename U, typename V>
struct is_right_scalar_quotient_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// matrix of scalars dividing block scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_right_scalar_quotient_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// matrix of blocks dividing scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_right_scalar_quotient_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value =
       is_right_scalar_quotient_conforming_impl<U, collapse_t<V>>::value;
};

// matrix of blocks dividing block scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_right_scalar_quotient_conforming_impl<matrix<U, M, N>,
                                                matrix<V, L, K>> {
   constexpr static bool value =
       (M == N) && (N == K) &&
       is_matrix_right_division_conforming_impl<collapse_t<V>,
                                                collapse_t<U>>::value;
};

// scalar matrix addition
// **********************
template <typename U, typename V> struct is_scalar_addition_conforming_impl;

template <typename U, typename V> struct is_scalar_addition_conforming {
   constexpr static bool value =
       is_scalar_addition_conforming_impl<collapse_t<U>, collapse_t<V>>::value;
};
template <typename U, typename V>
inline constexpr bool is_scalar_addition_conforming_v =
    is_scalar_addition_conforming<U, V>::value;

// scalar plus matrix of scalars
template <typename U, typename V> struct is_scalar_addition_conforming_impl {
   constexpr static bool value =
       std::is_convertible_v<U, V> || std::is_convertible_v<V, U>;
};

// block scalar plus matrix
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_scalar_addition_conforming_impl<matrix<U, M, N>, V> {
   constexpr static bool value = false;
};

// scalar plus matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct is_scalar_addition_conforming_impl<U, matrix<V, M, N>> {
   constexpr static bool value =
       is_scalar_addition_conforming_impl<U, collapse_t<V>>::value;
};

// block scalar plus matrix of blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t L, std::ptrdiff_t K>
struct is_scalar_addition_conforming_impl<matrix<U, M, N>, matrix<V, L, K>> {
   constexpr static bool value =
       (M == L) && (N == K) &&
       is_scalar_addition_conforming_impl<collapse_t<U>, collapse_t<V>>::value;
};

} // namespace detail

} // namespace ctla

#endif
