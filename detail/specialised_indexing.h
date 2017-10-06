// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifndef CTLA_DETAIL_SPECIALISED_INDEXING_H
#define CTLA_DETAIL_SPECIALISED_INDEXING_H

#include <cstddef>
#include <utility>

namespace ctla {

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N> class matrix;

namespace detail {

// matrix derives from specialised_indexing, which provides some
// additional indexing operators depending on the matrix dimensions

// primary template has nothing special
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
struct specialised_indexing {
   struct dummy {};
   void operator()(dummy) const {} // just so using base::operator() is valid
};

// row, column and single-element matrices support single-value indexing a(i)
// so we'll derive each of their specialisations from this base class
template <typename Matrix> class single_value_index {
 public:
   constexpr auto operator()(std::ptrdiff_t pos) const {
      return static_cast<const Matrix&>(*this)[pos];
   }

   template <std::ptrdiff_t from, std::ptrdiff_t to>
   constexpr auto operator()(detail::range_type<from, to>) const {
      return do_subrange<from, to>(
          static_cast<const Matrix&>(*this),
          std::make_integer_sequence<std::ptrdiff_t, to - from + 1>());
   }

 private:
   template <std::ptrdiff_t from, std::ptrdiff_t to, std::ptrdiff_t... Idx>
   static constexpr auto
   do_subrange(const Matrix& b, std::integer_sequence<std::ptrdiff_t, Idx...>) {
      return matrix<typename Matrix::value_type, to - from + 1, 1>(
          {b[from + Idx]...});
   }
};

// row matrices
template <typename U, std::ptrdiff_t N>
class specialised_indexing<U, 1, N>
    : public single_value_index<matrix<U, 1, N>> {};

// column matrices
template <typename U, std::ptrdiff_t M>
class specialised_indexing<U, M, 1>
    : public single_value_index<matrix<U, M, 1>> {};

// one-by-one matrices
template <typename U>
struct specialised_indexing<U, 1, 1>
    : public single_value_index<matrix<U, 1, 1>> {

   // one-by-one matrices can also convert to their value type
   constexpr operator U() const {
      return static_cast<const matrix<U, 1, 1>&>(*this)[0];
   }
};

} // namespace detail

} // namespace ctla

#endif
