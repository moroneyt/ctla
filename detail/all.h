// Copyright (C) 2017 Tim Moroney, Queensland University of Technology
// You may use, distribute and modify this code under the licensing terms
// specified in the file https://github.com/moroneyt/ctla/blob/master/LICENSE
// a copy of which should have been included with this code.

#ifndef CTLA_DETAIL_ALL_H
#define CTLA_DETAIL_ALL_H

namespace ctla {

// all is used to indicate all rows or columns when indexing.
// We use a reference to function type so that other functions called "all"
// (a common enough name) can co-exist with this "all", even
// in the presence of using namespace directives. Our function all will
// be chosen by overload resolution for expressions like A(n, all).

namespace detail {

struct all_dummy {};
using all_type = void (&)(all_dummy);

} // namespace detail

inline void all(detail::all_dummy) {}

} // namespace ctla

#endif
