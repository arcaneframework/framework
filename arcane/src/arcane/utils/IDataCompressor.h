// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataCompressor.h                                           (C) 2000-2021 */
/*                                                                           */
/* Interface allowing data compression/decompression.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IDATACOMPRESSOR_H
#define ARCANE_UTILS_IDATACOMPRESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a service for compressing/decompressing data.
 */
class ARCANE_UTILS_EXPORT IDataCompressor
{
 public:
  
  virtual ~IDataCompressor() = default;

 public:

  virtual void build() =0;

 public:

  //! Algorithm name
  virtual String name() const =0;

 /*!
  * \brief Minimum array size below which compression is not useful.
  *
  * This can be used by the caller to avoid compressing/decompressing
  * certain arrays. This value is not used internally by this instance.
  *
  * If the caller uses this value, consistency must be guaranteed both during
  * compression and decompression (i.e.: do not call decompression for arrays
  * whose decompressed size is less than minCompressSize() if the compress()
  * method was not called for that array).
  */
  virtual Int64 minCompressSize() const =0;

  /*!
   * \brief Compresses the data \a values and stores it in \a compressed_values.
   *
   * This operation may throw an IOException exception in case of an error.
   */
  virtual void compress(Span<const std::byte> values,Array<std::byte>& compressed_values) =0;

  /*!
   * \brief Decompresses the data \a compressed_values and stores it in \a values.
   *
   * \a values must already have been allocated to the necessary size to contain
   * the decompressed data.
   * This operation may throw an IOException exception in case of an error.
   */
  virtual void decompress(Span<const std::byte> compressed_values,Span<std::byte> values) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
