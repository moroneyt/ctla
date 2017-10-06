// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifndef CTLA_RUNTIME_PRINT_H
#define CTLA_RUNTIME_PRINT_H

#include <cstddef>
#include <ostream>

namespace ctla {

template <typename U, std::ptrdiff_t M, std::ptrdiff_t N> class matrix;

// Runtime printing in MATLAB style; the output is valid MATLAB syntax
// that can be used to read the matrix into MATLAB
template <typename U, std::ptrdiff_t M, std::ptrdiff_t N>
std::ostream& operator<<(std::ostream& os, const matrix<U, M, N>& a) {
   os << '[';
   for (std::ptrdiff_t i = 0; i < M; ++i) {
      for (std::ptrdiff_t j = 0; j < N; ++j) {
         os << a(i, j);
         os << (j == N - 1 ? "" : " ");
      }
      os << (i == M - 1 ? "" : ";");
   }
   os << ']';
   return os;
}

} // namespace ctla

#endif
