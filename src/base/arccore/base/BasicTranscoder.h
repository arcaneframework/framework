// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicTranscoder.h                                           (C) 2000-2018 */
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

namespace Arccore
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

  static void transcodeFromISO88591ToUtf16(const std::string& s,CoreArray<UChar>& utf16);
  static void transcodeFromUtf16ToISO88591(Span<const UChar> utf16,std::string& s);

  static void transcodeFromISO88591ToUtf8(const char* str,Int64 len,CoreArray<Byte>& utf8);
  static void transcodeFromUtf8ToISO88591(Span<const Byte> utf8,std::string& s);

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

