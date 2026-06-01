// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SHA3HashAlgorithm.h                                         (C) 2000-2023 */
/*                                                                           */
/* Calculates the SHA-3 hashing function (224, 256, 384 or 512).             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SHA3HASHALGORITHM_H
#define ARCANE_UTILS_SHA3HASHALGORITHM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::SHA3Algorithm
{
class SHA3;
}

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for SHA-3 algorithms.
 */
class ARCANE_UTILS_EXPORT SHA3HashAlgorithm
: public IHashAlgorithm
{
 public:

  void computeHash(ByteConstArrayView input, ByteArray& output) final;
  void computeHash64(Span<const Byte> input, ByteArray& output) final;
  void computeHash64(Span<const std::byte> input, ByteArray& output) final;

 protected:

  virtual void _initialize(SHA3Algorithm::SHA3&) = 0;

 private:

  void _computeHash64(Span<const std::byte> input, ByteArray& output);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Hash for the SHA-3 256 algorithm
class ARCANE_UTILS_EXPORT SHA3_256HashAlgorithm
: public SHA3HashAlgorithm
{
 public:

  String name() const override { return "SHA3_256"; }
  Int32 hashSize() const override { return 32; }

 protected:

  void _initialize(SHA3Algorithm::SHA3&) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Hash for the SHA-3 224 algorithm
class ARCANE_UTILS_EXPORT SHA3_224HashAlgorithm
: public SHA3HashAlgorithm
{
 public:

  String name() const override { return "SHA3_224"; }
  Int32 hashSize() const override { return 28; }

 protected:

  void _initialize(SHA3Algorithm::SHA3&) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Hash for the SHA-3 384 algorithm
class ARCANE_UTILS_EXPORT SHA3_384HashAlgorithm
: public SHA3HashAlgorithm
{
 public:

  String name() const override { return "SHA3_384"; }
  Int32 hashSize() const override { return 48; }

 protected:

  void _initialize(SHA3Algorithm::SHA3&) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Hash for the SHA-3 512 algorithm
class ARCANE_UTILS_EXPORT SHA3_512HashAlgorithm
: public SHA3HashAlgorithm
{
 public:

  String name() const override { return "SHA3_512"; }
  Int32 hashSize() const override { return 64; }

 protected:

  void _initialize(SHA3Algorithm::SHA3&) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
