// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITranscoder.h                                               (C) 2000-2005 */
/*                                                                           */
/* Interface of a converter to/from the UTF-16 format.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ITRANSCODER_H
#define ARCANE_UTILS_ITRANSCODER_H
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
 * \internal
 * \brief Interface of a converter to/from the UTF-16 format.
 */
class ITranscoder
{
 public:

  virtual ~ITranscoder() {} //<! Releases resources

 public:

  virtual void build() = 0;

 public:

  /*! \brief Translates the source \a src of length \a src_len to the UTF-16 format
   *
   * Stores the conversion in \a out, which must be pre-allocated
   * and of sufficient length.
   */
  virtual void transcodeToUtf16(const Byte* src, Integer src_len, UChar* out) = 0;

  /*! \brief Translates the source \a src of length \a src_len from the UTF-16 format
   *
   * Stores the conversion in \a out, which must be pre-allocated
   * and of sufficient length.
   */
  virtual void transcodeFromUtf16(const UChar* src, Integer src_len, Byte* out) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
