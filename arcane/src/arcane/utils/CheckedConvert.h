// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckedConvert.h                                            (C) 2000-2018 */
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

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fonctions pour convertir un type en un autre avec vérification
 */
namespace CheckedConvert
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Int64 en un \c Integer

inline Integer
toInteger(Real r)
{
  double v = Convert::toDouble(r);
  if (v>(double)std::numeric_limits<Integer>::max() || v<(double)std::numeric_limits<Integer>::min())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Integer",v));
  return (Integer)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Int64 en un \c Integer

inline Integer
toInteger(Int64 v)
{
  if (v>std::numeric_limits<Integer>::max() || v<std::numeric_limits<Integer>::min())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Integer",v));
  return (Integer)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Int32 en un \c Integer

inline Integer
toInteger(Int32 v)
{
#ifdef ARCANE_64BIT
  if (v>std::numeric_limits<Integer>::max() || v<std::numeric_limits<Integer>::min())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Integer",v));
#endif
  return (Integer)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c unsigned int en un \c Integer

inline Integer
toInteger(unsigned int v)
{
  if (v>(unsigned int)std::numeric_limits<Integer>::max())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Integer",v));
  return (Integer)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c unsigned int en un \c Integer

inline Integer
toInteger(unsigned long v)
{
  if (v>(unsigned long)std::numeric_limits<Integer>::max())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Integer",v));
  return (Integer)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c unsigned int en un \c Integer

inline Integer
toInteger(unsigned long long v)
{
  if (v>(unsigned long long)std::numeric_limits<Integer>::max())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Integer",v));
  return (Integer)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Converti un \c Int64 en un \c Int32

inline Int32
toInt32(Int64 v)
{
  if (v>std::numeric_limits<Int32>::max() || v<std::numeric_limits<Int32>::min())
    throw BadCastException(A_FUNCINFO,
                           String::format("Invalid conversion from '{0}' to type Int32",v));
  return (Int32)v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplie trois 'Integer' et vérifie que le résultat peut être contenu
 * dans un 'Integer'.
 */
inline Integer
multiply(Integer x,Integer y,Integer z)
{
  Int64 xyz = ((Int64)x) * ((Int64)y) * ((Int64)z);
  return CheckedConvert::toInteger(xyz);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplie deux 'Integer' et vérifie que le résultat peut être contenu
 * dans un 'Integer'.
 */
inline Integer
multiply(Integer x,Integer y)
{
  Int64 xyz = ((Int64)x) * ((Int64)y);
  return CheckedConvert::toInteger(xyz);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace CheckedConvert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
