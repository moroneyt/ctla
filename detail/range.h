// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifndef CTLA_DETAIL_RANGE_H
#define CTLA_DETAIL_RANGE_H

#include <cstddef>

namespace ctla {

namespace detail {

template <std::ptrdiff_t from, std::ptrdiff_t to> struct range_type {};

} // namespace detail

// range is used to indicate ranges of rows or columns when indexing
template <std::ptrdiff_t from, std::ptrdiff_t to>
inline constexpr detail::range_type<from, to> range{};

} // namespace ctla

#endif
