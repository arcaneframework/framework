// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeGlobal.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Définitions globales de la composante 'Serialize' de 'Arccore'.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/BasicSerializer.h"

#include "arccore/base/Ref.h"
#include "arccore/base/Span.h"
#include "arccore/base/BasicDataType.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
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
  reserveSpan(DT_Real, values.size());
}

void ISerializer::
reserveSpan(Span<const Int16> values)
{
  reserveSpan(DT_Int16, values.size());
}

void ISerializer::
reserveSpan(Span<const Int32> values)
{
  reserveSpan(DT_Int32, values.size());
}

void ISerializer::
reserveSpan(Span<const Int64> values)
{
  reserveSpan(DT_Int64, values.size());
}

void ISerializer::
reserveSpan(Span<const Byte> values)
{
  reserveSpan(DT_Byte, values.size());
}

void ISerializer::
reserveSpan(Span<const Int8> values)
{
  reserveSpan(DT_Int8, values.size());
}

void ISerializer::
reserveSpan(Span<const Float16> values)
{
  reserveSpan(DT_Float16, values.size());
}

void ISerializer::
reserveSpan(Span<const BFloat16> values)
{
  reserveSpan(DT_BFloat16, values.size());
}

void ISerializer::
reserveSpan(Span<const Float32> values)
{
  reserveSpan(DT_Float32, values.size());
}

void ISerializer::
reserveSpan(Span<const Float128> values)
{
  reserveSpan(eBasicDataType::Float128, values.size());
}

void ISerializer::
reserveSpan(Span<const Int128> values)
{
  reserveSpan(eBasicDataType::Int128, values.size());
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ISerializer::
reserve(int dt, Int64 n)
{
  reserve(static_cast<eDataType>(dt), n);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ISerializer::
reserveSpan(int dt, Int64 n)
{
  reserveSpan(static_cast<eDataType>(dt), n);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializer>
createSerializer()
{
  auto* x = new BasicSerializer();
  return makeRef<ISerializer>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore
