// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* APReal.h                                                    (C) 2000-2018 */
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

namespace Arccore
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

