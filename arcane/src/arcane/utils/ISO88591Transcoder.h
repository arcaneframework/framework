// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISO88591Transcoder.h                                        (C) 2000-2005 */
/*                                                                           */
/* ISO-8859-1 Transcoder from/to UTF-16.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ISO88591TRANSCODER_H
#define ARCANE_UTILS_ISO88591TRANSCODER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITranscoder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief ISO-8859-1 Transcoder from/to UTF-16.
 */
class ISO88591Transcoder
: public ITranscoder
{
 public:

  virtual ~ISO88591Transcoder();

 public:

  virtual void build();

 public:

  virtual void transcodeToUtf16(const Byte* src, Integer src_len, UChar* out);
  virtual void transcodeFromUtf16(const UChar* src, Integer src_len, Byte* out);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
