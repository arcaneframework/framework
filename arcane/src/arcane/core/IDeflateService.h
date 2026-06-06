// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDeflateService.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface of a service allowing compression/decompression of data.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDEFLATESERVICE_H
#define ARCANE_CORE_IDEFLATESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Minimum size (in bytes) for an array to be compressed.
// Below this size, it is not compressed.
// If this size changes, the corresponding size must be changed
// in the VariableComparer in C#.
static const Integer DEFLATE_MIN_SIZE = 512;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a service allowing compression/decompression of data.
 *
 * \deprecated Use IDataCompressor instead.
 */
class ARCANE_CORE_EXPORT IDeflateService
{
 public:

  virtual ~IDeflateService() = default;

 public:

  virtual void build() = 0;

 public:

  /*!
   * \brief Compresses the data \a values and stores it in \a compressed_values.
   *
   * This operation may throw an IOException exception in case of an error.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This interface is deprecated. Use IDataCompressor instead")
  virtual void compress(ByteConstArrayView values, ByteArray& compressed_values) = 0;

  /*!
   * \brief Compresses the data \a values and stores it in \a compressed_values.
   *
   * This operation may throw an IOException exception in case of an error.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This interface is deprecated. Use IDataCompressor instead")
  virtual void compress(Span<const Byte> values, ByteArray& compressed_values);

  /*!
   * \brief Decompresses the data \a compressed_values and stores it in \a values.
   *
   * \a values must already have been allocated to the necessary size to contain
   * the decompressed data.
   * This operation may throw an IOException exception in case of an error.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This interface is deprecated. Use IDataCompressor instead")
  virtual void decompress(ByteConstArrayView compressed_values, ByteArrayView values) = 0;

  /*!
   * \brief Decompresses the data \a compressed_values and stores it in \a values.
   *
   * \a values must already have been allocated to the necessary size to contain
   * the decompressed data.
   * This operation may throw an IOException exception in case of an error.
   */
  ARCANE_DEPRECATED_REASON("Y2023: This interface is deprecated. Use IDataCompressor instead")
  virtual void decompress(Span<const Byte> compressed_values, Span<Byte> values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
