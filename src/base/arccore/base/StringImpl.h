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
/* StringImpl.h                                                (C) 2000-2019 */
/*                                                                           */
/* Implémentation d'une chaîne de caractère UTf-8 ou UTF-16.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRINGIMPL_H
#define ARCCORE_BASE_STRINGIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/CoreArray.h"
#include "arccore/base/BaseTypes.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

class StringView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Implémentation de la classe String.
 *
 * \warning Cette classe est interne à %Arcane et ne doit pas êre utilisée
 * en dehors de %Arcane.
 *
 * Actuellement l'implémentation supporte deux encodages simultanés: UTF-8 et UTF-16.
 * L'encodage UTF-16 est obsolète et sera supprimé fin 2019.
 *
 * Lorsque le C++20 sera disponible, cette classe ne sera qu'une encapsulation
 * de std::u8string.
 */
class ARCCORE_BASE_EXPORT StringImpl
{
 public:
  StringImpl(std::string_view str);
  StringImpl(const UChar* str);
  StringImpl(const StringImpl& str);
  StringImpl(Span<const Byte> bytes);
 private:
  StringImpl();
 public:
  //TODO: rendre obsolète.
  UCharConstArrayView utf16();
  //! Vue sur l'encodage UTF-8 *AVEC* zéro terminal
  Span<const Byte> largeUtf8();
  //! idem largeUtf8() mais *SANS* le zéro terminal
  Span<const Byte> bytes();
  bool isEqual(StringImpl* str);
  bool isLessThan(StringImpl* str);
  bool isEqual(StringView str);
  bool isLessThan(StringView str);
  std::string_view toStdStringView();
  StringView view();
 public:
  void addReference();
  void removeReference();
  Int32 nbReference() { return m_nb_ref.load(); }
 public:
  void internalDump(std::ostream& ostr);
 public:
  StringImpl* clone();
  StringImpl* append(StringImpl* str);
  StringImpl* append(StringView str);
  StringImpl* replaceWhiteSpace();
  StringImpl* collapseWhiteSpace();
  StringImpl* toUpper();
  StringImpl* toLower();
  static StringImpl* substring(StringImpl* str,Int64 pos,Int64 len);

 public:
  bool null() { return false; }
  bool empty();
  bool hasUtf8() const { return (m_flags & eValidUtf8); }
  bool hasUtf16() const { return (m_flags & eValidUtf16); }
 private:
  enum
  {
    eValidUtf16 = 1 << 0,
    eValidUtf8 = 1 << 1
  };
  std::atomic<Int32> m_nb_ref;
  int m_flags;
  CoreArray<UChar> m_utf16_array;
  CoreArray<Byte> m_utf8_array;

  void _setUtf16(const UChar* src);
  void _createUtf16();
  void _setUtf8(const Byte* src);
  void _createUtf8();
  inline void _checkReference();
  void _invalidateUtf16();
  void _invalidateUtf8();
  void _setArray();
  void _setStrFromArray(Int64 ulen);
  void _printStrUtf16(std::ostream& o,Span<const UChar> str);
  void _printStrUtf8(std::ostream& o,Span<const Byte> str);
  void _appendUtf8(Span<const Byte> ref_str);
  inline void _initFromSpan(Span<const Byte> bytes);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

