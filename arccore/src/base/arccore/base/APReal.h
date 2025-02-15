// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* APReal.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Réel en précision arbitraire.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_APREAL_H
#define ARCCORE_BASE_APREAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#ifndef ARCCORE_REAL_USE_APFLOAT
#include <iostream>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Emulation d'un réel en précision arbitraire.
 *
 * Si on souhaite une précision arbitraire, il faut utiliser la bibliothèque
 * 'apfloat'. Cette classe définit un type pour le cas où cette bibliothèque
 * n'est pas disponible mais n'assure pas une précision arbitraire.
 * apfloat Cette classe émule la clas
 */
#ifndef ARCCORE_REAL_USE_APFLOAT

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline bool
operator<(const APReal& a,const APReal& b)
{
  return a.v[0] < b.v[0];
}

inline bool
operator>(const APReal& a,const APReal& b)
{
  return a.v[0] > b.v[0];
}

inline bool
operator==(const APReal& a,const APReal& b)
{
  return a.v[0]==b.v[0];
}

inline bool
operator!=(const APReal& a,const APReal& b)
{
  return !operator==(a,b);
}

inline APReal
operator+(const APReal& a,const APReal& b)
{
  APReal result;
  result.v[0] = a.v[0] + b.v[0];
  return result;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<< (std::ostream& o,APReal t)
{
  o << t.v[0];
  return o;
}

inline std::istream&
operator>> (std::istream& i,APReal& t)
{
  i >> t.v[0];
  return i;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

