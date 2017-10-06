// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifndef CTLA_DETAIL_TRAITS_H
#define CTLA_DETAIL_TRAITS_H

#include <cstddef>
#include <type_traits>
#include <utility>

namespace ctla {

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N> class matrix;

namespace detail {

// some fundamental operations required on scalar types
template <typename T> struct traits {

   // the value "one" (n.b. T() is assumed to give "zero")
   constexpr static auto one() { return T(1); }

   // squared norm
   constexpr static auto normsq(const T& t) { return t * t; }

   // transpose is specialised for block scalars
   constexpr static auto transpose(const T& t) { return t; }

   // used to collapse a 1x1 block down
   constexpr static auto collapse(const T& t) { return t; }
};

template <typename T>
using collapse_t = decltype(traits<T>::collapse(std::declval<T>()));

template <typename T> constexpr auto collapse(const T& t) {
   return traits<T>::collapse(t);
}

// specialisation for block scalars
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
struct traits<matrix<U, M, N>> {

   // the identity matrix functions as "one" for blocks
   constexpr static auto one() {
      return matrix<U, M, N>(
          [](auto i, auto j) { return i == j ? traits<U>::one() : U(); });
   }
   constexpr static auto normsq(const matrix<U, M, N>& b) {
      return do_normsq(b, std::make_integer_sequence<std::ptrdiff_t, M * N>());
   }
   constexpr static auto transpose(const matrix<U, M, N>& b) { return b.T(); }
   constexpr static auto collapse(const matrix<U, M, N>& b) { return b; }
};

// specialisation for 1x1 blocks
template <typename U> struct traits<matrix<U, 1, 1>> {
   constexpr static auto one() { return matrix<U, 1, 1>({traits<U>::one()}); }
   constexpr static auto normsq(const matrix<U, 1, 1>& b) {
      return normsq(b(0, 0));
   }
   constexpr static auto transpose(const matrix<U, 1, 1>& b) { return b.T(); }
   constexpr static auto collapse(const matrix<U, 1, 1>& b) {
      return traits<U>::collapse(b(0, 0));
   }
};

// what type to promote to when (e.g.) augmenting differently-typed matrices
template <typename T, typename S> struct promotion {
   using type = std::remove_reference_t<decltype(true ? std::declval<T>()
                                                      : std::declval<S>())>;
};

template <typename T, typename S>
using promotion_t = typename promotion<T, S>::type;

// specialise promotion type for matching-size blocks
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
struct promotion<matrix<U, M, N>, matrix<V, M, N>> {
   using type = matrix<promotion_t<U, V>, M, N>;
};

// specialise promotion type for non-matching-size blocks, to have no type
template <typename U, typename V, std::ptrdiff_t M1, std::ptrdiff_t M2,
          std::ptrdiff_t N1, std::ptrdiff_t N2>
struct promotion<matrix<U, M1, N1>, matrix<V, M2, N2>> {};

} // namespace detail

} // namespace ctla

#endif