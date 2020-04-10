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
/* SerializeGlobal.cc                                          (C) 2000-2020 */
/*                                                                           */
/* Définitions globales de la composante 'Serialize' de 'Arccore'.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/ISerializer.h"

#include "arccore/base/Span.h"

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ISerializer::
putSpan(Span<const Real> values)
{
  put(values);
}

void ISerializer::
putSpan(Span<const Int16> values)
{
  put(values);
}

void ISerializer::
putSpan(Span<const Int32> values)
{
  put(values);
}

void ISerializer::
putSpan(Span<const Int64> values)
{
  put(values);
}

void ISerializer::
putSpan(Span<const Byte> values)
{
  put(values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ISerializer::
reserveSpan(Span<const Real> values)
{
  reserveSpan(DT_Real,values.size());
}

void ISerializer::
reserveSpan(Span<const Int16> values)
{
  reserveSpan(DT_Int64,values.size());
}

void ISerializer::
reserveSpan(Span<const Int32> values)
{
  reserveSpan(DT_Int32,values.size());
}

void ISerializer::
reserveSpan(Span<const Int64> values)
{
  reserveSpan(DT_Int64,values.size());
}

void ISerializer::
reserveSpan(Span<const Byte> values)
{
  reserveSpan(DT_Byte,values.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ISerializer::
getSpan(Span<Real> values)
{
  get(values.smallView());
}

void ISerializer::
getSpan(Span<Int16> values)
{
  get(values.smallView());
}

void ISerializer::
getSpan(Span<Int32> values)
{
  get(values.smallView());
}

void ISerializer::
getSpan(Span<Int64> values)
{
  get(values.smallView());
}

void ISerializer::
getSpan(Span<Byte> values)
{
  get(values.smallView());
}

} // End namespace Arccore
