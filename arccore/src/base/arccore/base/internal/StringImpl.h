// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringImpl.h                                                (C) 2000-2025 */
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

namespace Arcane
{

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
  friend class String;
  friend class StringBuilder;
 public:
  StringImpl(std::string_view str);
  StringImpl(const StringImpl& str);
  StringImpl(Span<const Byte> bytes);
 private:
  StringImpl();
  StringImpl(Span<const UChar> uchars);
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
  Impl::CoreArray<UChar> m_utf16_array;
  Impl::CoreArray<Byte> m_utf8_array;

  void _setUtf16(Span<const UChar> src);
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
  inline void _finalizeUtf8Creation();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

