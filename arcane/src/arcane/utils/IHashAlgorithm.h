// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IHashAlgorithm.h                                            (C) 2000-2023 */
/*                                                                           */
/* Interface of a hashing algorithm.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IHASHALGORITHM_H
#define ARCANE_UTILS_IHASHALGORITHM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arccore/base/Span.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Hash algorithm return value.
 */
class ARCANE_UTILS_EXPORT HashAlgorithmValue
{
 public:

  static constexpr Int32 MAX_SIZE = 64;

 public:

  SmallSpan<std::byte> bytes()
  {
    return { m_value.data(), m_size };
  }
  SmallSpan<const std::byte> bytes() const
  {
    return { m_value.data(), m_size };
  }
  SmallSpan<const Byte> asLegacyBytes() const
  {
    return { reinterpret_cast<const Byte*>(m_value.data()), m_size };
  }
  void setSize(Int32 size);

 private:

  std::array<std::byte, MAX_SIZE> m_value = {};
  Int32 m_size = MAX_SIZE;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Context for calculating a hash incrementally.
 *
 * The same context can be used multiple times by calling reset() to
 * reset the instance.
 *
 * \code
 * IHashAlgorithm* algo = ...;
 * HashAlgorithmValue hash_value;
 * hash_value.reserve(algo->hashSize());
 * UniqueArray<std::byte> array1 = ...;
 * UniqueArray<std::byte> array2 = ...;
 *
 * Ref<IHashContext> context = algo->createContext();
 * context->updateHash(array1);
 * context->updateHash(array2);
 * context->computeHashValue(hash_value);
 * \endcode
 */
class ARCANE_UTILS_EXPORT IHashAlgorithmContext
{
 public:

  virtual ~IHashAlgorithmContext() = default;

 public:

  //! Resets the instance to calculate a new hash value.
  virtual void reset() = 0;

  //! Adds the array \a input to the calculated hash
  virtual void updateHash(Span<const std::byte> input) = 0;

  //! Calculates the hash value and returns it in hash_value.
  virtual void computeHashValue(HashAlgorithmValue& hash_value) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a hashing algorithm.
 */
class ARCANE_UTILS_EXPORT IHashAlgorithm
{
 public:

  virtual ~IHashAlgorithm() = default;

 public:

  //NOTE: for now (version 3.10) still pure virtual to remain
  // compatible with existing code
  //! Name of the algorithm
  virtual String name() const;

  //NOTE: for now (version 3.10) still pure virtual to remain
  // compatible with existing code. Throws FatalErrorException if not overridden
  //! Size (in bytes) of the hash key.
  virtual Int32 hashSize() const;

  /*!
   * \brief Calculates the hash value for the array \a input.
   *
   * The hash value is <strong >added</strong > to \a output.
   * The added length is equal to hashSize().
   */
  virtual void computeHash64(Span<const Byte> input, ByteArray& output);

  /*!
   * \brief Calculates the hash value for the array \a input.
   *
   * The hash value is <strong >added</strong > to \a output.
   * The added length is equal to hashSize().
   */
  virtual void computeHash64(Span<const std::byte> input, ByteArray& output);

  //NOTE: for now (version 3.10) still pure virtual to remain
  // compatible with existing code
  /*!
   * \brief Calculates the hash value for the array \a input.
   *
   * The hash value is positioned in \a value
   */
  virtual void computeHash(Span<const std::byte> input, HashAlgorithmValue& value);

  //NOTE: for now (version 3.11) still pure virtual to remain
  // compatible with existing code
  /*!
   * \brief Creates a context to calculate the hash value
   * incrementally.
   *
   * If the implementation does not support incremental mode (hasCreateContext()==false),
   * an exception is thrown.
   */
  virtual Ref<IHashAlgorithmContext> createContext();

  //NOTE: for now (version 3.11) still pure virtual to remain
  // compatible with existing code
  //! Indicates if the implementation supports incremental hashing
  virtual bool hasCreateContext() const { return false; }

 public:

  /*!
   * \brief Calculates the hash value for the array \a input.
   *
   * The hash value is <strong >added</strong > to \a output.
   * The length depends on the algorithm used.
   */
  ARCANE_DEPRECATED_REASON("Y2023: Use computeHash64(Span<const std::byte> input,ByteArray& output) instead")
  virtual void computeHash(ByteConstArrayView input, ByteArray& output) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
