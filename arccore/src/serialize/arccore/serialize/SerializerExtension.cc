// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializerExtension.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Extension of the serializer with support of multidimensional arrays.      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/SerializerExtension.h"

#include "arccore/base/Float128.h"
#include "arccore/base/Float16.h"
#include "arccore/base/Int128.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/MDSpan.h"
#include "arccore/common/NumArray.h"
#include "arccore/collections/Array2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializerExtension::
reserveArray(Span<const String> values) const
{
  // size of each String + size of array.
  m_serializer->reserveInt64(values.size() + 1);
  for (const String& elem : values) {
    m_serializer->reserveSpan(elem.bytes());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializerExtension::
putArray(Span<const String> values) const
{
  m_serializer->putInt64(values.size());
  for (const String& elem : values) {
    m_serializer->putInt64(elem.length());
    m_serializer->putSpan(elem.bytes());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializerExtension::
getArray(Array<String>& values) const
{
  const Int64 size_array = m_serializer->getInt64();
  values.resize(size_array);
  for (Int64 i = 0; i < size_array; ++i) {
    const Int64 len = m_serializer->getInt64();
    UniqueArray<Byte> bytes(len);
    m_serializer->getSpan(bytes);
    values[i] = String(bytes.span());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Explicit instantiation to check compilation
template void SerializerExtension::reserveSpan<Int64>(Span2<const Int64>) const;
template void SerializerExtension::reserveArray<Int64>(Span2<const Int64>) const;
template void SerializerExtension::putSpan<Int64>(Span2<const Int64>) const;
template void SerializerExtension::putArray<Int64>(Span2<const Int64>) const;
template void SerializerExtension::getSpan<Int64>(Span2<Int64>) const;
template void SerializerExtension::getArray<Int64>(Array2<Int64>&) const;

template void SerializerExtension::reserveSpan<Real, MDDim2>(MDSpan<const Real, MDDim2>) const;
template void SerializerExtension::reserveArray<Real, MDDim2>(MDSpan<const Real, MDDim2>) const;
template void SerializerExtension::putSpan<Real, MDDim2>(MDSpan<const Real, MDDim2>) const;
template void SerializerExtension::putArray<Real, MDDim2>(MDSpan<const Real, MDDim2>) const;
template void SerializerExtension::getSpan<Real, MDDim2>(MDSpan<Real, MDDim2>) const;
template void SerializerExtension::getArray<Real, MDDim2>(NumArray<Real, MDDim2>&) const;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
