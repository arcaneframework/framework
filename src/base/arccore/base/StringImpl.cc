// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* StringImpl.cc                                               (C) 2000-2019 */
/*                                                                           */
/* Implémentation d'une chaîne de caractère unicode.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringImpl.h"
#include "arccore/base/BasicTranscoder.h"
#include "arccore/base/CStringUtils.h"

#include <cstring>

//#define ARCCORE_DEBUG_UNISTRING

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool global_arccore_debug_string = false;

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
_initFromSpan(Span<const Byte> bytes)
{
  m_flags = eValidUtf8;
  m_utf8_array = bytes;
  // \a m_utf8_array doit toujours avoir un zéro terminal.
  if (m_utf8_array.empty())
    m_utf8_array.add('\0');
  else if (m_utf8_array.back()!='\0')
    m_utf8_array.add('\0');
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl(std::string_view str)
: m_nb_ref(0)
, m_flags(0)
{
  auto b = reinterpret_cast<const Byte*>(str.data());
  _initFromSpan({b,str.size()});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl(const UChar* str)
: m_nb_ref(0)
, m_flags(0)
{
  _setUtf16(str);
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

std::string_view StringImpl::
toStdStringView()
{
  Span<const Byte> x = largeUtf8();
  Int64 size = x.size();
  ARCCORE_ASSERT((size>0),("Null size during conversion to std::string_view"));
  return std::string_view(reinterpret_cast<const char*>(x.data()),size-1);
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

ConstArrayView<Byte> StringImpl::
utf8()
{
  return largeUtf8().smallView();
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
#if 0
  cerr << "** COMPARE = UTF8 ";
  _printStrUtf8(cerr,ref_array);
  _printStrUtf8(cerr,m_utf8_array);
  cerr << " => " << v << '\n';
#endif
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
#if 0
    cerr << "** COMPARE < UTF8 ";
    _printStrUtf8(cerr,ref_array);
    _printStrUtf8(cerr,m_utf8_array);
    cerr << " => " << v << '\n';
#endif
    return v;
  }
  ARCCORE_ASSERT((0),("InternalError in StringImpl::isEqual()"));
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringImpl::
isEqual(const char* str)
{
  _checkReference();
  _createUtf8();
  // TODO: faire une version optimisée sans avoir à calculer la longueur
  // de \a str
  //Span<const Byte> ustr{reinterpret_cast<const Byte*>(str),std::strlen(str)};
  //bool v = CStringUtils::isEqual(m_local_str.c_str(),str);
  //bool v = (m_utf8_array == ustr);
  bool v = (std::strcmp(reinterpret_cast<const char*>(m_utf8_array.data()),str)==0);
  //std::cout << "COMPARE '" << str << "' '" << (const char*)m_utf8_array.data() << " V=" << v << "\n";
  //cerr << "** COMPARE LOCAL STR =" << str << "><" << m_local_str << "> => " << v << "\n";
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringImpl::
isLessThan(const char* str)
{
  _checkReference();
  _createUtf8();
  return (std::strcmp(reinterpret_cast<const char*>(m_utf8_array.data()),str)<0);
  //return m_local_str < str;
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
append(const char* str)
{
  if (!str)
    return this;

  _checkReference();
  _createUtf8();

  Int64 len = std::strlen(str);
  _appendUtf8(Span<const Byte>(reinterpret_cast<const Byte*>(str),len+1));
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
    BasicTranscoder::transcodeFromUtf16ToUtf8(m_utf16_array,m_utf8_array);
    m_flags |= eValidUtf8;
    return;
  }

  ARCCORE_ASSERT((0),("InternalError in StringImpl::_createUtf16()"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
_setUtf16(const UChar* src)
{
  ARCCORE_CHECK_PTR(src);
  Int64 len = BasicTranscoder::stringLen(src);
  m_utf16_array.resize(len+1);
  ::memcpy(m_utf16_array.data(),src,sizeof(UChar)*len);
  m_utf16_array[len] = 0;
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
  o << "(bufsize=" << buf_size
    << " - "
    << (const char*)str.data()
    << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringImpl::
internalDump(std::ostream& ostr)
{
  ostr << "(utf8=valid=" << (m_flags & eValidUtf8)
       << ",len=" << m_utf8_array.size() << ",val=";
  _printStrUtf8(ostr,m_utf8_array);
  ostr << ")";

  ostr << "(utf16=valid" << (m_flags & eValidUtf16)
       << ",len=" << m_utf16_array.size() << ",val=";
  _printStrUtf16(ostr,m_utf16_array);
  ostr << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
