// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayViewDumper.h                                           (C) 2000-2026 */
/*                                                                           */
/* Functions to dump values of Arccore array views (ArrayView, Span, ...)    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYVIEWDUMPER_H
#define ARCCORE_BASE_ARRAYVIEWDUMPER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Helper class to dump an array view on a stream.
 *
 * The method \a dumpArray() is templated on the stream type so we do not
 * need to include the header 'iostream' in this header. The goal is to reduce
 * compilation time. The user has to include 'iostream' if he needs to
 * dump the array.
 */
template <typename ViewType>
class ArrayViewDumper
{
 public:

  template <typename Stream> static void
  dumpArray(Stream& o, ViewType val, int max_print)
  {
    using size_type = typename ViewType::size_type;
    size_type n = val.size();
    if (max_print > 0 && n > max_print) {
      // Only displays the first (max_print/2) and the last (max_print/2)
      // otherwise if the array is very large it can generate enormous
      // output listings.
      size_type z = (max_print / 2);
      size_type z2 = n - z;
      o << "[0]=\"" << val[0] << '"';
      for (size_type i = 1; i < z; ++i)
        o << " [" << i << "]=\"" << val[i] << '"';
      o << " ... ... (skipping indexes " << z << " to " << z2 << " ) ... ... ";
      for (size_type i = (z2 + 1); i < n; ++i)
        o << " [" << i << "]=\"" << val[i] << '"';
    }
    else {
      for (size_type i = 0; i < n; ++i) {
        if (i != 0)
          o << ' ';
        o << "[" << i << "]=\"" << val[i] << '"';
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
