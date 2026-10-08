// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializerExtension.h                                       (C) 2000-2026 */
/*                                                                           */
/* Extension of the serializer with support of multidimensional arrays.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_SERIALIZE_SERIALIZEREXTENSION_H
#define ARCCORE_SERIALIZE_SERIALIZEREXTENSION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/ISerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_SERIALIZE_EXPORT SerializerExtension
{
 public:

  SerializerExtension(ISerializer* serializer)
  : m_serializer(serializer)
  {}

 public:

  template <class Type>
  void reserveSpan(Span2<const Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void reserveSpan(MDSpan<const Type, Extents> values) const;

  template <class Type>
  void reserveArray(Span2<const Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void reserveArray(MDSpan<const Type, Extents> values) const;

  void reserveArray(Span<const String> values) const;

 public:

  template <class Type>
  void putSpan(Span2<const Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void putSpan(MDSpan<const Type, Extents> values) const;

  template <class Type>
  void putArray(Span2<const Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void putArray(MDSpan<const Type, Extents> values) const;

  void putArray(Span<const String> values) const;

 public:

  template <class Type>
  void getSpan(Span2<Type> values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void getSpan(MDSpan<Type, Extents> values) const;

  template <class Type>
  void getArray(Array2<Type>& values) const;

  template <class Type, class Extents>
  requires(Extents::rank() <= 4) void getArray(NumArray<Type, Extents>& values) const;

  void getArray(Array<String>& values) const;

 private:

  ISerializer* m_serializer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
