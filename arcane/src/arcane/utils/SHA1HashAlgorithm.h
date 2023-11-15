// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SHA1HashAlgorithm.h                                         (C) 2000-2023 */
/*                                                                           */
/* Calcule la fonction de hashage SHA-1.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SHA1HASHALGORITHM_H
#define ARCANE_UTILS_SHA1HASHALGORITHM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::SHA1Algorithm
{
class SHA1;
}

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de l'algorithme SHA-1.
 */
class ARCANE_UTILS_EXPORT SHA1HashAlgorithm
: public IHashAlgorithm
{
 public:

  void computeHash(Span<const std::byte> input, HashAlgorithmValue& value) override;
  void computeHash(ByteConstArrayView input, ByteArray& output) override;
  void computeHash64(Span<const Byte> input, ByteArray& output) override;
  void computeHash64(Span<const std::byte> input, ByteArray& output) override;
  String name() const override { return "SHA1"; }
  Int32 hashSize() const override { return 20; }
  Ref<IHashAlgorithmContext> createContext() override;
  bool hasCreateContext() const override { return true; }

 private:

  void _computeHash64(Span<const std::byte> input, ByteArray& output);
  void _computeHash(Span<const std::byte> input, HashAlgorithmValue& value);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
