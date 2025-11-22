// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashAlgorithm.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Interface d'un algorithme de hashage.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HashAlgorithmValue::
setSize(Int32 size)
{
  ARCANE_FATAL_IF((size < 0), "Invalid negative size '{0}'", size);
  ARCANE_FATAL_IF((size > MAX_SIZE), "Invalid size '{0}' max value is '{1}'", size, MAX_SIZE);
  m_size = size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String IHashAlgorithm::
name() const
{
  ARCANE_THROW(NotImplementedException, "name() method");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 IHashAlgorithm::
hashSize() const
{
  ARCANE_THROW(NotImplementedException, "hashSize() method");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IHashAlgorithm::
computeHash64(Span<const Byte> input, ByteArray& output)
{
  computeHash(input.smallView(), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IHashAlgorithm::
computeHash64(Span<const std::byte> input, ByteArray& output)
{
  const Byte* x = reinterpret_cast<const Byte*>(input.data());
  computeHash64(Span<const Byte>(x, input.size()), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IHashAlgorithm::
computeHash(Span<const std::byte> input, HashAlgorithmValue& value)
{
  UniqueArray<Byte> legacy_bytes;
  computeHash64(input, legacy_bytes);
  Int32 n = legacy_bytes.size();
  value.setSize(n);
  SmallSpan<std::byte> value_as_bytes(value.bytes());
  for (Int32 i = 0; i < n; ++i)
    value_as_bytes[i] = static_cast<std::byte>(legacy_bytes[i]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IHashAlgorithmContext> IHashAlgorithm::
createContext()
{
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
