// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckedConvert.h                                            (C) 2000-2023 */
/*                                                                           */
/* Fonctions pour convertir un type en un autre avec vérification.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CHECKEDCONVERT_H
#define ARCANE_UTILS_CHECKEDCONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/BadCastException.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Convert.h"

#include <limits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions pour convertir un type en un autre avec vérification
 */
namespace Arcane::CheckedConvert
{
namespace impl
{
  inline Integer
  toInteger(Int64 v)
  {
    if (v > std::numeric_limits<Integer>::max() || v < std::numeric_limits<Integer>::min())
      ARCANE_THROW(BadCastException, "Invalid conversion from '{0}' to type Integer", v);
    return static_cast<Integer>(v);
  }
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Int64 en un \c Integer
inline Integer
toInteger(Real r)
{
  double v = Convert::toDouble(r);
  if (v > (double)std::numeric_limits<Integer>::max() || v < (double)std::numeric_limits<Integer>::min())
    ARCANE_THROW(BadCastException, "Invalid conversion from '{0}' to type Integer", v);
  return static_cast<Integer>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti \a v en le type \c Integer
inline Integer
toInteger(long long v)
{
  return impl::toInteger(static_cast<Int64>(v));
}
//! Converti \a v en le type \c Integer
inline Integer
toInteger(long v)
{
  return impl::toInteger(static_cast<Int64>(v));
}
//! Converti \a v en le type \c Integer
inline Integer
toInteger(int v)
{
  return impl::toInteger(static_cast<Int64>(v));
}
//! Converti \a v en le type \c Integer
inline Integer
toInteger(unsigned long long v)
{
  return impl::toInteger(static_cast<Int64>(v));
}
//! Converti \a v en le type \c Integer
inline Integer
toInteger(unsigned long v)
{
  return impl::toInteger(static_cast<Int64>(v));
}
//! Converti \a v en le type \c Integer
inline Integer
toInteger(unsigned int v)
{
  return impl::toInteger(static_cast<Int64>(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Int64 en un \c Int32

inline Int32
toInt32(Int64 v)
{
  if (v > std::numeric_limits<Int32>::max() || v < std::numeric_limits<Int32>::min())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Int32", v));
  return (Int32)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Int64 en un \c Int16

inline Int16
toInt16(Int64 v)
{
  if (v > std::numeric_limits<Int16>::max() || v < std::numeric_limits<Int16>::min())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Int16", v));
  return (Int16)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplie trois 'Integer' et vérifie que le résultat peut être contenu
 * dans un 'Integer'.
 */
inline Integer
multiply(Integer x, Integer y, Integer z)
{
  Int64 x2 = static_cast<Int64>(x);
  Int64 y2 = static_cast<Int64>(y);
  Int64 z2 = static_cast<Int64>(z);
  Int64 xyz = x2 * y2 * z2;
  return impl::toInteger(xyz);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplie deux 'Integer' et vérifie que le résultat peut être contenu
 * dans un 'Integer'.
 */
inline Integer
multiply(Integer x, Integer y)
{
  Int64 x2 = static_cast<Int64>(x);
  Int64 y2 = static_cast<Int64>(y);
  Int64 xy = x2 * y2;
  return impl::toInteger(xy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::CheckedConvert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
