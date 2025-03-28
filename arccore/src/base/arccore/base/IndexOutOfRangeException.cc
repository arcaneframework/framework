// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexOutOfRangeException.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Exception lorsqu'un indice de tableau est invalide.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/IndexOutOfRangeException.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexOutOfRangeException::
IndexOutOfRangeException(const TraceInfo& where,const String& message,
                         Int64 index,Int64 min_value_inclusive,
                         Int64 max_value_exclusive)
: Exception("IndexOutOfRange",where,message)
, m_index(index)
, m_min_value_inclusive(min_value_inclusive)
, m_max_value_exclusive(max_value_exclusive)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IndexOutOfRangeException::
explain(std::ostream& m) const
{
  m << "Index '" << m_index << "' out of bounds ("
    << m_min_value_inclusive << " <= "
    << m_index << " < "
    << m_max_value_exclusive << ").\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

