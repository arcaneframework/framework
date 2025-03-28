// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicTranscoder.h                                           (C) 2000-2025 */
/*                                                                           */
/* Conversions utf8/utf16/iso-8859-1.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BASICTRANSCODER_H
#define ARCCORE_BASE_BASICTRANSCODER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> class CoreArray;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 */
class ARCCORE_BASE_EXPORT BasicTranscoder
{
 public:

  BasicTranscoder() = delete;

 public:

  static void transcodeFromUtf16ToUtf8(Span<const UChar> utf16,CoreArray<Byte>& utf8);
  static void transcodeFromUtf8ToUtf16(Span<const Byte> utf8,CoreArray<UChar>& utf16);

  static Int64 stringLen(const UChar* ustr);

  static void replaceWS(CoreArray<Byte>& ustr);
  static void collapseWS(CoreArray<Byte>& ustr);

  static void upperCase(CoreArray<Byte>& utf8);
  static void lowerCase(CoreArray<Byte>& utf8);

  static void substring(CoreArray<Byte>& utf8,Span<const Byte> rhs,Int64 pos,Int64 len);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

