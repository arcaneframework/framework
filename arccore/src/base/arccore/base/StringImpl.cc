// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringImpl.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'une chaîne de caractère UTf-8 ou UTF-16.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/internal/StringImpl.h"
#include "arccore/base/BasicTranscoder.h"
#include "arccore/base/CStringUtils.h"
#include "arccore/base/StringView.h"

#include <cstring>

//#define ARCCORE_DEBUG_UNISTRING

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool global_arccore_debug_string = false;

namespace
{
const char* const global_empty_string = "";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StringException
: public std::exception
{
 public:
  StringException(const char* where) : m_where(where) {}
  ~StringException() ARCCORE_NOEXCEPT {}
  virtual const char* what() const ARCCORE_NOEXCEPT
  {
    return m_where;
  }
 private:
  const char* m_where;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCCORE_DEBUG_UNISTRING
static void
_badStringImplReference(StringImpl* ptr)
{
  cerr << "** FATAL: Trying to use deleted StringImpl " << ptr << '\n';
}
#endif

inline void StringImpl::
_checkReference()
{
#ifdef ARCCORE_DEBUG_UNISTRING
  if (m_nb_ref.value()<=0){
    _badStringImplReference(this);
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void StringImpl::
_finalizeUtf8Creation()
{
  m_flags |= eValidUtf8;
  // \a m_utf8_array doit toujours avoir un zéro terminal.
  if (m_utf8_array.empty())
    m_utf8_array.add('\0');
  else if (m_utf8_array.back()!='\0')
    m_utf8_array.add('\0');
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void StringImpl::
_initFromSpan(Span<const Byte> bytes)
{
  m_utf8_array = bytes;
  _finalizeUtf8Creation();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl(std::string_view str)
: m_nb_ref(0)
, m_flags(0)
{
  auto b = reinterpret_cast<const Byte*>(str.data());
  _initFromSpan(Span<const Byte>(b,str.size()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl(Span<const UChar> uchars)
: m_nb_ref(0)
, m_flags(0)
{
  _setUtf16(uchars);
  m_flags = eValidUtf16;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl(const StringImpl& str)
: m_nb_ref(0)
, m_flags(str.m_flags)
, m_utf16_array(str.m_utf16_array)
, m_utf8_array(str.m_utf8_array)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl(Span<const Byte> bytes)
: m_nb_ref(0)
, m_flags(0)
{
  _initFromSpan(bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl()
: m_nb_ref(0)
, m_flags(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const Byte> StringImpl::
bytes()
{
  Span<const Byte> x = largeUtf8();
  Int64 size = x.size();
  if (size>0)
    return { x.data(), size-1 };
  // Ne devrait normalement pas arriver mais si c'est le cas on retourne
  // une vue sur la chaîne vide car cette méthode garantit qu'il y a un
  // zéro terminal à la fin.
  // NOTE: On ne lève pas d'exception car cette méthode est utilisée dans les
  // sorties via operator<< et cela peut être utilisé notamment dans
  // les destructeurs des objets.
  std::cerr << "INTERNAL ERROR: Null size in StringImpl::bytes()";
  return { reinterpret_cast<const Byte*>(global_empty_string), 0 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::string_view StringImpl::
toStdStringView()
{
  Span<const Byte> x = bytes();
  return std::string_view(reinterpret_cast<const char*>(x.data()),x.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView StringImpl::
view()
{
  return StringView(bytes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
addReference()
{
  ++m_nb_ref;
  _checkReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
removeReference()
{
  _checkReference();
  Int32 r = --m_nb_ref;
#ifndef ARCCORE_DEBUG_UNISTRING
  if (r==0)
    delete this;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<UChar> StringImpl::
utf16()
{
  _checkReference();
  _createUtf16();
  return m_utf16_array.view().smallView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const Byte> StringImpl::
largeUtf8()
{
  _checkReference();
  _createUtf8();
  return m_utf8_array.view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringImpl::
isEqual(StringImpl* str)
{
  _checkReference();
  _createUtf8();
  Span<const Byte> ref_array = str->largeUtf8();
  bool v = CStringUtils::isEqual((const char*)ref_array.data(),(const char*)m_utf8_array.data());
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringImpl::
isLessThan(StringImpl* str)
{
  _checkReference();
  _createUtf8();
  if (m_flags & eValidUtf8){
    Span<const Byte> ref_array = str->largeUtf8();
    bool v = CStringUtils::isLess((const char*)m_utf8_array.data(),(const char*)ref_array.data());
    return v;
  }
  ARCCORE_ASSERT((0),("InternalError in StringImpl::isEqual()"));
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringImpl::
isEqual(StringView str)
{
  _checkReference();
  _createUtf8();
  return str.toStdStringView() == toStdStringView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringImpl::
isLessThan(StringView str)
{
  _checkReference();
  _createUtf8();
  return toStdStringView() < str.toStdStringView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
clone()
{
  _checkReference();
  _createUtf8();
  StringImpl* n = new StringImpl(*this);
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringImpl::
empty()
{
  _checkReference();
  if (m_flags & eValidUtf8) {
    ARCCORE_ASSERT((!m_utf8_array.empty()),("Not 0 terminated utf8 encoding"));
    return m_utf8_array.size()<=1; // Décompte le 0 terminal
  }
  if (m_flags & eValidUtf16) {
    ARCCORE_ASSERT((!m_utf16_array.empty()),("Not 0 terminated utf16 encoding"));
    return m_utf16_array.size()<=1; // Décompte le 0 terminal
  }
  ARCCORE_ASSERT((0),("InternalError in StringImpl::empty()"));
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
append(StringImpl* str)
{
  _checkReference();
  _createUtf8();
  Span<const Byte> ref_str = str->largeUtf8();
  _appendUtf8(ref_str);
  return this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
append(StringView str)
{
  Span<const Byte> str_bytes = str.bytes();
  if (!str_bytes.data())
    return this;

  _checkReference();
  _createUtf8();

  _appendUtf8(Span<const Byte>(str_bytes.data(),str_bytes.size() + 1));;
  return this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_appendUtf8(Span<const Byte> ref_str)
{
  Int64 ref_size = ref_str.size();
  Int64 utf8_size = m_utf8_array.size();
  Int64 current_size = utf8_size - 1;

  ARCCORE_ASSERT((ref_size>0),("Bad ref_size"));
  ARCCORE_ASSERT((utf8_size>0),("Bad utf8_size"));
  ARCCORE_ASSERT((ref_str[ref_size-1]==0),("Bad ref null terminal"));
  ARCCORE_ASSERT((m_utf8_array[utf8_size-1]==0),("Bad ref null terminal"));

  m_utf8_array.resize(current_size + ref_size);
  std::memcpy(&m_utf8_array[current_size],ref_str.data(),ref_size);

  m_flags |= eValidUtf8;
  _invalidateUtf16();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
replaceWhiteSpace()
{
  _createUtf8();
  _invalidateUtf16();
  BasicTranscoder::replaceWS(m_utf8_array);
  return this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
collapseWhiteSpace()
{
  _createUtf8();
  _invalidateUtf16();
  BasicTranscoder::collapseWS(m_utf8_array);
  return this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
toUpper()
{
  _createUtf8();
  _invalidateUtf16();
  BasicTranscoder::upperCase(m_utf8_array);
  return this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
toLower()
{
  _createUtf8();
  _invalidateUtf16();
  BasicTranscoder::lowerCase(m_utf8_array);
  return this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
substring(StringImpl* str,Int64 pos,Int64 len)
{
  StringImpl* s = new StringImpl();
  BasicTranscoder::substring(s->m_utf8_array,str->largeUtf8(),pos,len);
  s->m_flags |= eValidUtf8;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_createUtf16()
{
  if (m_flags & eValidUtf16)
    return;

  if (m_flags & eValidUtf8){
    ARCCORE_ASSERT(m_utf16_array.empty(),("Not empty utf16_array"));
    BasicTranscoder::transcodeFromUtf8ToUtf16(m_utf8_array,m_utf16_array);
    m_flags |= eValidUtf16;
    return;
  }

  ARCCORE_ASSERT((0),("InternalError in StringImpl::_createUtf16()"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_createUtf8()
{
  if (m_flags & eValidUtf8)
    return;

  if (m_flags & eValidUtf16){
    ARCCORE_ASSERT(m_utf8_array.empty(),("Not empty utf8_array"));
    BasicTranscoder::transcodeFromUtf16ToUtf8(m_utf16_array,m_utf8_array);
    _finalizeUtf8Creation();
    return;
  }

  ARCCORE_ASSERT((0),("InternalError in StringImpl::_createUtf16()"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_setUtf16(Span<const UChar> src)
{
  m_utf16_array = src;
  if (m_utf16_array.empty())
    m_utf16_array.add(0);
  else if (m_utf16_array.back()!='\0')
    m_utf16_array.add(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_invalidateUtf16()
{
  m_flags &= ~eValidUtf16;
  m_utf16_array.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_invalidateUtf8()
{
  m_flags &= ~eValidUtf8;
  m_utf8_array.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_printStrUtf16(std::ostream& o,Span<const UChar> str)
{
  Int64 buf_size = str.size();
  o << "(bufsize=" << buf_size
    << " begin=" << str.data() << " - ";
  for( Int64 i=0; i<buf_size; ++i )
    o << (int)str[i] << ' ';
  o << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_printStrUtf8(std::ostream& o,Span<const Byte> str)
{
  Int64 buf_size = str.size();
  o << "(bufsize=" << buf_size << " - ";
  for( Int64 i=0; i<buf_size; ++i )
    o << (int)str[i] << ' ';
  o << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
internalDump(std::ostream& ostr)
{
  ostr << "(utf8=valid=" << ((m_flags & eValidUtf8)!=0)
       << ",len=" << m_utf8_array.size() << ",val=";
  _printStrUtf8(ostr,m_utf8_array);
  ostr << ")";

  ostr << "(utf16=valid=" << ((m_flags & eValidUtf16)!=0)
       << ",len=" << m_utf16_array.size() << ",val=";
  _printStrUtf16(ostr,m_utf16_array);
  ostr << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
