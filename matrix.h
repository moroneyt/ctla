// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored                                               \
    "-Wmissing-braces" // https://bugs.llvm.org/show_bug.cgi?id=21629
#endif

#include "detail/all.h"
#include "detail/do_things.h"
#include "detail/inverse.h"
#include "detail/is_conforming.h"
#include "detail/range.h"
#include "detail/specialised_indexing.h"
#include "detail/traits.h"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace ctla {

// matrix is a statically-sized matrix type supporting compile-time (constexpr)
// linear algebra operations.  Its elements can be integer types, floating point
// types or other types that define the relevant (constexpr) operators,
// including matrix itself (i.e. block matrices).

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
class matrix : public detail::specialised_indexing<U, M, N> {
 public:
   using value_type = U;
   using difference_type = std::ptrdiff_t;
   using reference = value_type&;
   using const_reference = const value_type&;
   using pointer = value_type*;
   using const_pointer = const value_type*;
   using const_iterator = typename std::array<U, M * N>::const_iterator;
   using iterator = const_iterator;
   using const_reverse_iterator =
       typename std::array<U, M * N>::const_reverse_iterator;
   using reverse_iterator = const_reverse_iterator;

   // bring in some specialised indexing operators for row and column matrices
   using detail::specialised_indexing<U, M, N>::operator();

   // default constructor value-initialises
   constexpr matrix() : data_() {}

   // construct from an array of values in row-major orientation
   constexpr matrix(const std::array<U, M * N>& vals) : data_(vals) {}

   constexpr matrix(const matrix&) = default;
   constexpr matrix(matrix&&) = default;

   // converting constructor
   template <typename V,
             typename = std::enable_if_t<std::is_convertible_v<V, U>>>
   constexpr matrix(const matrix<V, M, N>& other)
       : data_(pos_functor<M * N>([&](auto pos) { return other[pos]; })) {}

   // construct each entry (i,j) from a binary function f as f(i,j)
   template <
       typename F,
       typename = std::enable_if_t<std::is_convertible_v<
           decltype(std::declval<F>()(std::ptrdiff_t(), std::ptrdiff_t())), U>>>
   constexpr explicit matrix(F func) : data_(ij_functor<M, N>(func)) {}

   // construct each entry at position pos (in row-major order) from a unary
   // function f as f(pos)
   template <typename F, typename = std::enable_if_t<std::is_convertible_v<
                             decltype(std::declval<F>()(std::ptrdiff_t())), U>>>
   constexpr explicit matrix(F func, void* = 0)
       : data_(pos_functor<M * N>(func)) {}

   constexpr matrix& operator=(const matrix&) = default;
   constexpr matrix& operator=(matrix&&) = default;

   // converting assignment
   template <typename V,
             typename = std::enable_if_t<std::is_convertible_v<U, V>>>
   constexpr matrix& operator=(const matrix<V, M, N>& other) {
      data_ = pos_functor<M * N>([&](auto pos) { return other[pos]; });
   }

   // A(i,j) returns entry (i,j)
   constexpr U operator()(std::ptrdiff_t i, std::ptrdiff_t j) const {
      return (*this)[i * N + j];
   }

   // A(i, all) returns row i; i.e. MATLAB's A(i, :)
   constexpr auto operator()(std::ptrdiff_t i, detail::all_type) const {
      return detail::do_row(*this, i,
                            std::make_integer_sequence<std::ptrdiff_t, N>());
   }

   // A(all, j) returns column j; i.e. MATLAB's A(:, j)
   constexpr auto operator()(detail::all_type, std::ptrdiff_t j) const {
      return detail::do_col(*this, j,
                            std::make_integer_sequence<std::ptrdiff_t, M>());
   }

   // A(range<a,b>, all) returns rows a..b; i.e. MATLAB's A(a:b, :)
   template <std::ptrdiff_t from, std::ptrdiff_t to>
   constexpr auto operator()(detail::range_type<from, to>,
                             detail::all_type) const {
      return detail::do_row_subrange<from, to>(
          *this,
          std::make_integer_sequence<std::ptrdiff_t, (to - from + 1) * N>());
   }

   // A(all, range<a,b>) returns columns a..b; i.e. MATLAB's A(:, a:b)
   template <std::ptrdiff_t from, std::ptrdiff_t to>
   constexpr auto operator()(detail::all_type,
                             detail::range_type<from, to>) const {
      return detail::do_col_subrange<from, to>(
          *this,
          std::make_integer_sequence<std::ptrdiff_t, (to - from + 1) * M>());
   }

   // A(range<a,b>, j) returns rows a..b of column j; i.e. MATLAB's A(a:b, j)
   template <std::ptrdiff_t from, std::ptrdiff_t to>
   constexpr auto operator()(detail::range_type<from, to>,
                             std::ptrdiff_t j) const {
      return (*this)(range<from, to>, all)(all, j);
   }

   // A(i, range<a,b>) returns columns a..b of row i; i.e. MATLAB's A(i, a:b)
   template <std::ptrdiff_t from, std::ptrdiff_t to>
   constexpr auto operator()(std::ptrdiff_t i,
                             detail::range_type<from, to>) const {
      return (*this)(all, range<from, to>)(i, all);
   }

   // A(range<a,b>, range<c,d>) returns the submatrix with rows a..b and columns
   // c..d; i.e. MATLAB's A(a:b, c:d)
   template <std::ptrdiff_t from1, std::ptrdiff_t to1, std::ptrdiff_t from2,
             std::ptrdiff_t to2>
   constexpr auto operator()(detail::range_type<from1, to1>,
                             detail::range_type<from2, to2>) const {
      return (*this)(range<from1, to1>, all)(all, range<from2, to2>);
   }

   // A.row(i) is another way to get the ith row; i.e. equivalent to A(i, all)
   constexpr auto row(std::ptrdiff_t i) const {
      return detail::do_row(*this, i,
                            std::make_integer_sequence<std::ptrdiff_t, N>());
   }

   // A.col(j) is another way to get the jth column; i.e. equivalent to A(all,
   // j)
   constexpr auto col(std::ptrdiff_t j) const {
      return detail::do_col(*this, j,
                            std::make_integer_sequence<std::ptrdiff_t, M>());
   }

   // A.T() returns the transpose of A (somewhat resembling the mathematical
   // notation A^T)
   constexpr auto T() const {
      return detail::do_transpose(
          *this, std::make_integer_sequence<std::ptrdiff_t, M * N>());
   }

   // the number of rows and columns
   constexpr std::ptrdiff_t rows() const { return M; }
   constexpr std::ptrdiff_t cols() const { return N; }

   // direct indexing using a single index; this is how all other indexing
   // operators access the data
   constexpr U operator[](std::ptrdiff_t pos) const { return data_[pos]; }

   // iterator support
   constexpr const_iterator begin() const { return data_.cbegin(); }

   constexpr const_iterator cbegin() const { return data_.cbegin(); }

   constexpr const_iterator end() const { return data_.cend(); }

   constexpr const_iterator cend() const { return data_.cend(); }

   constexpr const_iterator rbegin() const { return data_.crbegin(); }

   constexpr const_iterator crbegin() const { return data_.crbegin(); }

   constexpr const_iterator rend() const { return data_.crend(); }

   constexpr const_iterator crend() const { return data_.crend(); }

 private:
   std::array<U, M * N> data_;

   template <std::ptrdiff_t MM, std::ptrdiff_t NN, typename F>
   static constexpr auto ij_functor(F func) {
      return ij_functor_impl<NN>(
          func, std::make_integer_sequence<std::ptrdiff_t, MM * NN>());
   }

   template <std::ptrdiff_t NN, typename F, std::ptrdiff_t... Idx>
   static constexpr auto
   ij_functor_impl(F func, std::integer_sequence<std::ptrdiff_t, Idx...> idx) {
      return std::array<U, idx.size()>({func(Idx / NN, Idx % NN)...});
   }

   template <std::ptrdiff_t Sz, typename F>
   static constexpr auto pos_functor(F func) {
      return pos_functor_impl(func,
                              std::make_integer_sequence<std::ptrdiff_t, Sz>());
   }

   template <typename F, std::ptrdiff_t... Idx>
   static constexpr auto
   pos_functor_impl(F func, std::integer_sequence<std::ptrdiff_t, Idx...> idx) {
      return std::array<U, idx.size()>({func(Idx)...});
   }
};

// integer sequence of the form [a, a+1, ..., b-1, b]
template <auto a, auto b> constexpr auto seq() {
   return matrix<detail::promotion_t<decltype(a), decltype(b)>, 1, b - a + 1>(
       [](auto pos) { return a + pos; });
}

// matrix of zeros
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
constexpr auto zeros() {
   return matrix<U, M, N>();
}

// matrix of ones
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
constexpr auto ones() {
   return matrix<U, M, N>([](auto) { return detail::traits<U>::one(); });
}

// identity matrix
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N> constexpr auto eye() {
   return matrix<U, M, N>(
       [](auto i, auto j) { return i == j ? detail::traits<U>::one() : U(); });
}

// square identity matrix
template <typename U, std::ptrdiff_t N> constexpr auto eye() {
   return eye<U, N, N>();
}

// transpose as non-member
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
constexpr auto transpose(const matrix<U, M, N>& b) {
   return b.T();
}

// transform through unary function
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N, typename F>
constexpr auto transform(const matrix<U, M, N>& b, F func) {
   return detail::do_transform(
       b, func, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// row augment: augr(a, b) means [a;b] in MATLAB notation; i.e. one on top of
// the other
template <
    typename U, typename V, std::ptrdiff_t M1, std::ptrdiff_t M2,
    std::ptrdiff_t N,
    typename = std::enable_if_t<detail::is_matrix_addition_conforming_v<U, V>>>
constexpr auto augr(const matrix<U, M1, N>& left,
                    const matrix<V, M2, N>& right) {
   return detail::do_augr(
       left, right,
       std::make_integer_sequence<std::ptrdiff_t, (M1 + M2) * N>());
}

// row augment [scalar; column]
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              N == 1 && detail::is_scalar_addition_conforming_v<U, V>>>
constexpr auto augr(const U& left, const matrix<V, M, N>& right) {
   return detail::do_augr_left_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M + 1>());
}

// row augment [column; scalar]
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              N == 1 && detail::is_scalar_addition_conforming_v<V, U>>>
constexpr auto augr(const matrix<U, M, N>& left, const V& right) {
   return detail::do_augr_right_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M + 1>());
}

// variadic row augmenting for augr(a, b, ...)
template <typename U, typename V, typename... Args>
constexpr auto augr(const U& left, const V& right, const Args&... args) {
   return augr(left, augr(right, args...));
}

// column augment: augc(a, b) means [a b] in MATLAB notation; i.e. side by side
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N1,
    std::ptrdiff_t N2,
    typename = std::enable_if_t<detail::is_matrix_addition_conforming_v<U, V>>>
constexpr auto augc(const matrix<U, M, N1>& left,
                    const matrix<V, M, N2>& right) {
   return detail::do_augc(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M*(N1 + N2)>());
}

// column augment [scalar row]
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              M == 1 && detail::is_scalar_addition_conforming_v<U, V>>>
constexpr auto augc(const U& left, const matrix<V, M, N>& right) {
   return detail::do_augc_left_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, N + 1>());
}

// column augment [row scalar]
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              M == 1 && detail::is_scalar_addition_conforming_v<V, U>>>
constexpr auto augc(const matrix<U, M, N>& left, const V& right) {
   return detail::do_augc_right_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, N + 1>());
}

// variadic column augmenting for augc(a, b, ...)
template <typename U, typename V, typename... Args>
constexpr auto augc(const U& left, const V& right, const Args&... args) {
   return augc(left, augc(right, args...));
}

// sum of entries
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if<M == 1 || N == 1>>
constexpr auto sum(const matrix<U, M, N>& b) {
   return detail::do_sum(b,
                         std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// mean of entries
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if<M == 1 || N == 1>>
constexpr auto mean(const matrix<U, M, N>& b) {
   return sum(b) / (M * N);
}

// product of entries
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if<M == 1 || N == 1>>
constexpr auto prod(const matrix<U, M, N>& b) {
   return detail::do_prod(b,
                          std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// squared Frobenius norm
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
constexpr auto normsq(const matrix<U, M, N>& b) {
   return detail::do_normsq(
       b, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// dot product (single elements)
template <typename U, typename V>
constexpr auto dot(const matrix<U, 1, 1>& left, const matrix<V, 1, 1>& right) {
   return left(0, 0) * right(0, 0);
}

// dot product (row matrix by column matrix)
template <typename U, typename V, std::ptrdiff_t K>
constexpr auto dot(const matrix<U, 1, K>& left, const matrix<V, K, 1>& right) {
   return detail::do_dot_no_transpose(left, right);
}

// dot product (both operands are column matrices)
template <typename U, typename V, std::ptrdiff_t M>
constexpr auto dot(const matrix<U, M, 1>& left, const matrix<V, M, 1>& right) {
   return detail::do_dot_left_transpose(left, right);
}

// dot product (both operands are row matrices)
template <typename U, typename V, std::ptrdiff_t N>
constexpr auto dot(const matrix<U, 1, N>& left, const matrix<V, 1, N>& right) {
   return detail::do_dot_right_transpose(left, right);
}

// matrix == matrix
template <typename U, typename V, std::ptrdiff_t M1, std::ptrdiff_t N1,
          std::ptrdiff_t M2, std::ptrdiff_t N2>
constexpr bool operator==(const matrix<U, M1, N1>& left,
                          const matrix<V, M2, N2>& right) {
   if constexpr (M1 != M2 || N1 != N2) {
      return false;
   } else {
      return detail::do_equal(
          left, right, std::make_integer_sequence<std::ptrdiff_t, M1 * N1>());
   }
}

// scalar == matrix
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
constexpr bool operator==(const U& left, const matrix<V, M, N>& right) {
   if constexpr (M == 1 && N == 1) {
      return left == right(0, 0);
   } else {
      return false;
   }
}

// matrix == scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N>
constexpr bool operator==(const matrix<U, M, N>& left, const V& right) {
   if constexpr (M == 1 && N == 1) {
      return left(0, 0) == right;
   } else {
      return false;
   }
}

// matrix != matrix
template <typename U, typename V, std::ptrdiff_t M1, std::ptrdiff_t N1,
          std::ptrdiff_t M2, std::ptrdiff_t N2>
constexpr bool operator!=(const matrix<U, M1, N1>& left,
                          const matrix<V, M2, N2>& right) {
   return !(left == right);
}

// +matrix
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
constexpr auto operator+(const matrix<U, M, N>& b) {
   return detail::do_pos(b,
                         std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// -matrix
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
constexpr auto operator-(const matrix<U, M, N>& b) {
   return detail::do_neg(b,
                         std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// scalar + matrix
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
    typename = std::enable_if_t<detail::is_scalar_addition_conforming_v<U, V>>>
constexpr auto operator+(const U& left, const matrix<V, M, N>& right) {
   return detail::do_add_left_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// matrix + scalar
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
    typename = std::enable_if_t<detail::is_scalar_addition_conforming_v<V, U>>>
constexpr auto operator+(const matrix<U, M, N>& left, const V& right) {
   return detail::do_add_right_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// matrix + matrix
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
    typename = std::enable_if_t<detail::is_matrix_addition_conforming_v<U, V>>>
constexpr auto operator+(const matrix<U, M, N>& left,
                         const matrix<V, M, N>& right) {
   return detail::do_add(left, right,
                         std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// scalar - matrix
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
    typename = std::enable_if_t<detail::is_scalar_addition_conforming_v<U, V>>>
constexpr auto operator-(const U& left, const matrix<V, M, N>& right) {
   return detail::do_sub_left_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// matrix - scalar
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
    typename = std::enable_if_t<detail::is_scalar_addition_conforming_v<V, U>>>
constexpr auto operator-(const matrix<U, M, N>& left, const V& right) {
   return detail::do_sub_right_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// matrix - matrix
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
    typename = std::enable_if_t<detail::is_matrix_addition_conforming_v<U, V>>>
constexpr auto operator-(const matrix<U, M, N>& left,
                         const matrix<V, M, N>& right) {
   return detail::do_sub(left, right,
                         std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// scalar * matrix
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_left_scalar_multiply_conforming_v<U, V>>>
constexpr auto operator*(const U& left, const matrix<V, M, N>& right) {
   return detail::do_mul_left_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// matrix * scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_right_scalar_multiply_conforming_v<V, U>>>
constexpr auto operator*(const matrix<U, M, N>& left, const V& right) {
   return detail::do_mul_right_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// matrix * matrix (matrix multiplication)
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
    std::ptrdiff_t K,
    typename = std::enable_if_t<detail::is_matrix_multiply_conforming_v<U, V>>>
constexpr auto operator*(const matrix<U, M, K>& left,
                         const matrix<V, K, N>& right) {
   return detail::do_times(left, right,
                           std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// scalar / matrix (scalar * inverse matrix)
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              M == N && detail::is_left_scalar_quotient_conforming_v<U, V>>>
constexpr auto operator/(const U& left, const matrix<V, M, N>& right) {
   return left * inv(right);
}

// matrix / scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_right_scalar_quotient_conforming_v<V, U>>>
constexpr auto operator/(const matrix<U, M, N>& left, const V& right) {
   return detail::do_div_right_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// mul(scalar, matrix) is the same as scalar * matrix
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_left_scalar_multiply_conforming_v<U, V>>>
constexpr auto mul(const U& left, const matrix<V, M, N>& right) {
   return detail::do_mul_left_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// mul(matrix, scalar) is the same as matrix * scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_right_scalar_multiply_conforming_v<V, U>>>
constexpr auto mul(const matrix<U, M, N>& left, const V& right) {
   return detail::do_mul_right_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// mul(matrix, matrix) is element-wise multiplication
template <
    typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
    typename = std::enable_if_t<detail::is_matrix_multiply_conforming_v<U, V>>>
constexpr auto mul(const matrix<U, M, N>& left, const matrix<V, M, N>& right) {
   return detail::do_mul(left, right,
                         std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// variadic mul for mul(a1, a2, ...)
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename... Args>
constexpr auto mul(const matrix<U, M, N>& left, const matrix<V, M, N>& right,
                   const Args&... args) {
   return mul(left, mul(right, args...));
}

// div(scalar, matrix) is element-wise division
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_left_scalar_quotient_conforming_v<U, V>>>
constexpr auto div(const U& left, const matrix<V, M, N>& right) {
   return detail::do_div_left_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// div(matrix, scalar) is the same as matrix / scalar
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_right_scalar_quotient_conforming_v<U, V>>>
constexpr auto div(const matrix<U, M, N>& left, const V& right) {
   return detail::do_div_right_scalar(
       left, right, std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// div(matrix, matrix) is element-wise division
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_matrix_right_division_conforming_v<U, V>>>
constexpr auto div(const matrix<U, M, N>& left, const matrix<V, M, N>& right) {
   return detail::do_div(left, right,
                         std::make_integer_sequence<std::ptrdiff_t, M * N>());
}

// matrix % matrix is matrix "left division" i.e. solving a linear system a*x=b
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t K,
          typename = std::enable_if_t<
              detail::is_matrix_left_division_conforming_v<U, V>>>
constexpr auto operator%(const matrix<U, M, N>& a, const matrix<V, M, K>& b) {
   // We'll solve the normal equations.  This provides support for least-squares
   // problems, and also helps the matrix inversion algorithm by giving it
   // an SPD matrix (presuming a is rank N) to work with.
   // Having said that, this also squares the condition number, and the
   // inversion algorithm isn't stable anyhow, so don't expect 16 digits of
   // glorious accuracy.
   const auto aT = a.T();
   return detail::solve_left(aT * a, aT * b);
}

// matrix / matrix is matrix "right division" i.e. solving a linear system x*a=b
template <typename U, typename V, std::ptrdiff_t M, std::ptrdiff_t N,
          std::ptrdiff_t K,
          typename = std::enable_if_t<
              detail::is_matrix_right_division_conforming_v<U, V>>>
constexpr auto operator/(const matrix<U, K, N>& b, const matrix<V, M, N>& a) {
   // See comments for left division
   const auto aT = a.T();
   return detail::solve_right(b * aT, a * aT);
}

// matrix inverse
template <typename U, std::ptrdiff_t N,
          typename = std::enable_if_t<
              detail::is_matrix_left_division_conforming_v<U, U>>>
constexpr auto inv(const matrix<U, N, N>& a) {
   return a % eye<U, N>();
}

} // namespace ctla

#ifdef __clang__
#pragma clang diagnostic pop
#endif
