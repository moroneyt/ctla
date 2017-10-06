// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifndef CTLA_DETAIL_DO_THINGS_H
#define CTLA_DETAIL_DO_THINGS_H

#include "traits.h"

#include <cstddef>
#include <utility>

namespace ctla {

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N> class matrix;

// Implementations of the majority of matrix operations are found here.

namespace detail {

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_row(const matrix<U, M, N>& b, std::ptrdiff_t i,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   return matrix<U, 1, N>({b[i * N + Idx]...});
}

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_col(const matrix<U, M, N>& b, std::ptrdiff_t j,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   return matrix<U, M, 1>({b[Idx * N + j]...});
}

template <std::ptrdiff_t from, std::ptrdiff_t to, typename U, std::ptrdiff_t M,
          std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_row_subrange(const matrix<U, M, N>& b,
                               std::integer_sequence<std::ptrdiff_t, Idx...>) {
   constexpr auto sz = to - from + 1;
   return matrix<U, sz, N>({b[from * N + Idx]...});
}

template <std::ptrdiff_t from, std::ptrdiff_t to, typename U, std::ptrdiff_t M,
          std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_col_subrange(const matrix<U, M, N>& b,
                               std::integer_sequence<std::ptrdiff_t, Idx...>) {
   constexpr auto sz = to - from + 1;
   return matrix<U, M, sz>({b[from + (Idx / sz) * N + Idx % sz]...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr bool do_equal(const matrix<U, M, N>& left,
                        const matrix<V, M, N>& right,
                        std::integer_sequence<std::ptrdiff_t, Idx...>) {
   return (... && (left[Idx] == right[Idx]));
}

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_transpose(const matrix<U, M, N>& b,
                            std::integer_sequence<std::ptrdiff_t, Idx...>) {
   return matrix<U, N, M>(
       {traits<U>::transpose(b[(Idx % M) * N + Idx / M])...});
}

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, typename F,
          std::ptrdiff_t... Idx>
constexpr auto do_transform(const matrix<U, M, N>& b, F func,
                            std::integer_sequence<std::ptrdiff_t, Idx...>) {
   return matrix<U, M, N>({func(b[Idx])...});
}

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_sum(const matrix<U, M, N>& b,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   return (U() + ... + b[Idx]);
}

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_prod(const matrix<U, M, N>& b,
                       std::integer_sequence<std::ptrdiff_t, Idx...>) {
   return (traits<U>::one() * ... * b[Idx]);
}

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_normsq(const matrix<U, M, N>& b,
                         std::integer_sequence<std::ptrdiff_t, Idx...>) {
   return (0 + ... + traits<U>::normsq(b[Idx]));
}

template <typename U, typename V, std::ptrdiff_t M1, std::ptrdiff_t M2,
          std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_augr(const matrix<U, M1, N>& left,
                       const matrix<V, M2, N>& right,
                       std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<detail::promotion_t<U, V>, M1 + M2, N>;
   return result_type({(Idx < M1 * N ? left[Idx] : right[Idx - M1 * N])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t... Idx>
constexpr auto
do_augr_left_scalar(const U& left, const matrix<V, M, 1>& right,
                    std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<detail::promotion_t<U, V>, M + 1, 1>;
   return result_type({(Idx == 0 ? left : right[Idx - 1])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t... Idx>
constexpr auto
do_augr_right_scalar(const matrix<U, M, 1>& left, const V& right,
                     std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<detail::promotion_t<U, V>, M + 1, 1>;
   return result_type({(Idx < M ? left[Idx] : right)...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N1,
          std::ptrdiff_t N2, std::ptrdiff_t... Idx>
constexpr auto do_augc(const matrix<U, M, N1>& left,
                       const matrix<V, M, N2>& right,
                       std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<detail::promotion_t<U, V>, M, N1 + N2>;
   return result_type(
       {(Idx % (N1 + N2) < N1 ? left[Idx - (Idx / (N1 + N2)) * N2]
                              : right[Idx - (1 + Idx / (N1 + N2)) * N1])...});
}

template <typename U, typename V, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto
do_augc_left_scalar(const U& left, const matrix<V, 1, N>& right,
                    std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<detail::promotion_t<U, V>, 1, N + 1>;
   return result_type({(Idx == 0 ? left : right[Idx - 1])...});
}

template <typename U, typename V, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto
do_augc_right_scalar(const matrix<U, 1, N>& left, const V& right,
                     std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<detail::promotion_t<U, V>, 1, N + 1>;
   return result_type({(Idx < N ? left[Idx] : right)...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_mul_no_conform(const matrix<U, M, N>& left, const matrix<V, N, M>& right,
                  std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() * std::declval<V>()), M, N>;
   return result_type({(left[Idx] * right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_mul_with_left_transpose(const matrix<U, M, N>& left,
                           const matrix<V, M, N>& right,
                           std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<decltype(traits<U>::transpose(std::declval<U>()) *
                                       std::declval<V>()),
                              M, N>;
   return result_type({(traits<U>::transpose(left[Idx]) * right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_mul_with_right_transpose(const matrix<U, M, N>& left,
                            const matrix<V, M, N>& right,
                            std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<decltype(std::declval<U>() *
                                       traits<V>::transpose(std::declval<V>())),
                              M, N>;
   return result_type({(left[Idx] * traits<V>::transpose(right[Idx]))...});
}

template <typename U, typename V, std::ptrdiff_t K>
constexpr auto do_dot_no_transpose(const matrix<U, 1, K>& left,
                                   const matrix<V, K, 1>& right) {
   return sum(do_mul_no_conform(
       left, right, std::make_integer_sequence<std::ptrdiff_t, K>()));
}

template <typename U, typename V, std::ptrdiff_t M>
constexpr auto do_dot_left_transpose(const matrix<U, M, 1>& left,
                                     const matrix<V, M, 1>& right) {
   return sum(do_mul_with_left_transpose(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M>()));
}

template <typename U, typename V, std::ptrdiff_t N>
constexpr auto do_dot_right_transpose(const matrix<U, 1, N>& left,
                                      const matrix<V, 1, N>& right) {
   return sum(detail::do_mul_with_right_transpose(
       left, right, std::make_integer_sequence<std::ptrdiff_t, N>()));
}

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_pos(const matrix<U, M, N>& b,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<decltype(+std::declval<U>()), M, N>;
   return result_type({+b[Idx]...});
}

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, std::ptrdiff_t... Idx>
constexpr auto do_neg(const matrix<U, M, N>& b,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type = matrix<decltype(-std::declval<U>()), M, N>;
   return result_type({-b[Idx]...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_add_right_scalar(const matrix<U, M, N>& left, const V& right,
                    std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() + std::declval<V>()), M, N>;
   return result_type({(left[Idx] + right)...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_add_left_scalar(const U& left, const matrix<V, M, N>& right,
                   std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() + std::declval<V>()), M, N>;
   return result_type({(left + right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto do_add(const matrix<U, M, N>& left, const matrix<V, M, N>& right,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() + std::declval<V>()), M, N>;
   return result_type({(left[Idx] + right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_sub_right_scalar(const matrix<U, M, N>& left, const V& right,
                    std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() - std::declval<V>()), M, N>;
   return result_type({(left[Idx] - right)...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_sub_left_scalar(const U& left, const matrix<V, M, N>& right,
                   std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() - std::declval<V>()), M, N>;
   return result_type({(left - right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto do_sub(const matrix<U, M, N>& left, const matrix<V, M, N>& right,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() - std::declval<V>()), M, N>;
   return result_type({(left[Idx] - right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_mul_right_scalar(const matrix<U, M, N>& left, const V& right,
                    std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() * std::declval<V>()), M, N>;
   return result_type({(left[Idx] * right)...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_mul_left_scalar(const U& left, const matrix<V, M, N>& right,
                   std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() * std::declval<V>()), M, N>;
   return result_type({(left * right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto do_mul(const matrix<U, M, N>& left, const matrix<V, M, N>& right,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() * std::declval<V>()), M, N>;
   return result_type({(left[Idx] * right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_div_right_scalar(const matrix<U, M, N>& left, const V& right,
                    std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() / std::declval<V>()), M, N>;
   return result_type({(left[Idx] / right)...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto
do_div_left_scalar(const U& left, const matrix<V, M, N>& right,
                   std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() / std::declval<V>()), M, N>;
   return result_type({(left / right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t... Idx>
constexpr auto do_div(const matrix<U, M, N>& left, const matrix<V, M, N>& right,
                      std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() / std::declval<V>()), M, N>;
   return result_type({(left[Idx] / right[Idx])...});
}

template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t K, std::ptrdiff_t... Idx>
constexpr auto do_times(const matrix<U, M, K>& left,
                        const matrix<V, K, N>& right,
                        std::integer_sequence<std::ptrdiff_t, Idx...>) {
   using result_type =
       matrix<decltype(std::declval<U>() * std::declval<V>()), M, N>;
   return result_type({dot(left.row(Idx / N), right.col(Idx % N))...});
}

} // namespace detail

} // namespace ctla

#endif
