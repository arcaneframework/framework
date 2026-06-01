// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MD5HashAlgorithm.h                                          (C) 2000-2023 */
/*                                                                           */
/* Calculates the MD5 hashing function.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MD5HASHALGORITHM_H
#define ARCANE_UTILS_MD5HASHALGORITHM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the MD5 hashing function of an array.
 *
 * For this algorithm, the key size is 16 bytes.
 */
class ARCANE_UTILS_EXPORT MD5HashAlgorithm
: public IHashAlgorithm
{
 public:

  MD5HashAlgorithm();

 public:

  String name() const override { return "MD5"; }
  Int32 hashSize() const override { return 16; }

 public:

  void computeHash(ByteConstArrayView input, ByteArray& output) override;
  void computeHash64(Span<const Byte> input, ByteArray& output) override;
  void computeHash64(Span<const std::byte> input, ByteArray& output) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
