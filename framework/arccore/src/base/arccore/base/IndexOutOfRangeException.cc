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
/* IndexOutOfRangeException.cc                                 (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'un indice de tableau est invalide.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/IndexOutOfRangeException.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexOutOfRangeException::
IndexOutOfRangeException(const TraceInfo& where,const String& message,
                         Int64 index,Int64 min_value,Int64 max_value)
: Exception("IndexOutOfRange",where,message)
, m_index(index)
, m_min_value(min_value)
, m_max_value(max_value)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndexOutOfRangeException::
IndexOutOfRangeException(const IndexOutOfRangeException& ex)
: Exception(ex)
, m_index(ex.m_index)
, m_min_value(ex.m_min_value)
, m_max_value(ex.m_max_value)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IndexOutOfRangeException::
explain(std::ostream& m) const
{
  m << "Array index out of bounds ("
    << " index=" << m_index
    << " min_value=" << m_min_value
    << " max_value=" << m_max_value
    << ").\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

