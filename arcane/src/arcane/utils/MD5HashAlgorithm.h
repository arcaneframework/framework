// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MD5HashAlgorithm.h                                          (C) 2000-2023 */
/*                                                                           */
/* Calcule la fonction de hashage MD5.                                       */
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
 * \brief Calcule la fonction de hashage MD5 d'un tableau.
 *
 * Pour cet algorithme, la taille de la clé est de 16 octets.
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
