// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueConvertInternal.h                                      (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir une chaîne de caractère en un type donné.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_VALUECONVERTINTERNAL_H
#define ARCANE_UTILS_INTERNAL_VALUECONVERTINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ValueConvert.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline bool
builtInGetArrayValueFromStream(Array<T>& v, std::istream& sbuf)
{
  T read_val = T();
  if (!sbuf.eof())
    sbuf >> ws;
  while (!sbuf.eof()) {
    sbuf >> read_val;
    if (sbuf.fail() || sbuf.bad())
      return true;
    v.add(read_val);
    sbuf >> ws;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline bool
builtInGetArrayValue(Array<T>& v, StringView s)
{
  impl::StringViewInputStream svis(s);
  std::istream& sbuf = svis.stream();
  return builtInGetArrayValueFromStream(v, sbuf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

