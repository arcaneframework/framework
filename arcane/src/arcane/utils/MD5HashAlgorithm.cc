// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MD5HashAlgorithm.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Calcule la fonction de hashage MD5.                                       */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StdHeader.h"
#include "arcane/utils/MD5HashAlgorithm_Licensed.h"
#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  void _computeHash64(Span<const std::byte> input, ByteArray& output)
  {
    unsigned char buf[16];
    _md5_buffer((const char*)input.data(), input.size(), buf);

    for (int i = 0; i < 16; ++i) {
      output.add(buf[i]);
    }
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MD5HashAlgorithm::
MD5HashAlgorithm()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MD5HashAlgorithm::
computeHash(ByteConstArrayView input, ByteArray& output)
{
  Span<const Byte> input64(input);
  return _computeHash64(asBytes(input64), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MD5HashAlgorithm::
computeHash64(Span<const Byte> input, ByteArray& output)
{
  Span<const std::byte> bytes(asBytes(input));
  return _computeHash64(bytes, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MD5HashAlgorithm::
computeHash64(Span<const std::byte> input, ByteArray& output)
{
  return _computeHash64(input, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
