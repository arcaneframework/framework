// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* StringImpl.cc                                               (C) 2000-2018 */
/*                                                                           */
/* Implémentation d'une chaîne de caractère unicode.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringImpl.h"
#include "arccore/base/BasicTranscoder.h"
#include "arccore/base/CStringUtils.h"

#include <cstring>

//#define ARCCORE_DEBUG_UNISTRING

//#define ARCCORE_USE_ICONV

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

StringImpl::
StringImpl(const char* str)
: m_nb_ref(0)
, m_flags(eValidLocal)
, m_local_str()
{
  if (str!=nullptr)
    m_local_str = str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl(const char* str,Int64 len)
: m_nb_ref(0)
, m_flags(eValidLocal)
, m_local_str(str,len)
{
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
, m_local_str(str.m_local_str)
, m_utf16_array(str.m_utf16_array)
, m_utf8_array(str.m_utf8_array)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl::
StringImpl(Span<const Byte> bytes)
: m_nb_ref(0)
, m_flags(eValidUtf8)
, m_utf8_array(bytes)
{
  // \a m_utf8_array doit toujours avoir un zéro terminal.
  if (m_utf8_array.empty())
    m_utf8_array.add('\0');
  else if (m_utf8_array.back()!='\0')
    m_utf8_array.add('\0');
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

const std::string& StringImpl::
local()
{
  _checkReference();
  _createLocal();
  return m_local_str;
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
  if (hasLocal() && str->hasLocal()){
    const std::string& ref_str = str->local();
    bool v = m_local_str==ref_str;
#if 0
    cerr << "** COMPARE LOCAL <" << ref_str << "><" << m_local_str << "> => " << v << "\n";
#endif
    return v;
  }
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
  if ((m_flags & eValidLocal) && str->hasLocal()){
    const std::string& ref_str = str->local();
    bool v = m_local_str < ref_str;

#if 0
    cerr << "** COMPARE < LOCAL A=" << m_local_str << " B= " << ref_str;
    cerr << " => " << v << '\n';
#endif

    return v;
  }
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
  _createLocal();
  //bool v = CStringUtils::isEqual(m_local_str.c_str(),str);
  bool v = (m_local_str == str);
  //cerr << "** COMPARE LOCAL STR =" << str << "><" << m_local_str << "> => " << v << "\n";
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool StringImpl::
isLessThan(const char* str)
{
  _checkReference();
  _createLocal();
  return m_local_str < str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
clone()
{
  _checkReference();
  //_createUtf16();
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
    ARCCORE_ASSERT((!m_utf8_array.empty()),("Not 0 terminated utf16 encoding"));
    return m_utf8_array.size()<=1; // Décompte le 0 terminal
  }
  if (m_flags & eValidUtf16) {
    ARCCORE_ASSERT((!m_utf16_array.empty()),("Not 0 terminated utf16 encoding"));
    return m_utf16_array.size()<=1; // Décompte le 0 terminal
  }
  if (m_flags & eValidLocal)
    return m_local_str.empty();
  ARCCORE_ASSERT((0),("InternalError in StringImpl::empty()"));
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
append(StringImpl* str)
{
  _checkReference();
  if ((m_flags & eValidLocal) && str->hasLocal()){
    const std::string& ref_str = str->local();
    m_local_str.append(ref_str);
    _invalidateUtf8();
    _invalidateUtf16();
    return this;
  }
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
  _checkReference();
  if (m_flags & eValidLocal){
    m_local_str.append(str);
    _invalidateUtf16();
    _invalidateUtf8();
    return this;
  }
  _createUtf8();

  CoreArray<Byte> buf;
  Int64 len = CStringUtils::largeLength(str);
  BasicTranscoder::transcodeFromISO88591ToUtf8(str,len,buf);
  _appendUtf8(buf);
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
  _invalidateLocal();
  _invalidateUtf16();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringImpl* StringImpl::
replaceWhiteSpace()
{
  _createUtf8();
  _invalidateLocal();
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
  _invalidateLocal();
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
  _invalidateLocal();
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
  _invalidateLocal();
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

  if (m_flags & eValidLocal){
    BasicTranscoder::transcodeFromISO88591ToUtf16(m_local_str,m_utf16_array);
    m_flags |= eValidUtf16;
    return;
  }

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

  if (m_flags & eValidLocal){
    Int64 len = arccoreCheckLargeArraySize(m_local_str.length());
    BasicTranscoder::transcodeFromISO88591ToUtf8(m_local_str.c_str(),len,m_utf8_array);
    m_flags |= eValidUtf8;
    return;
  }

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
_createLocal()
{
  if (m_flags & eValidLocal)
    return;

  if (m_flags & eValidUtf8){
    BasicTranscoder::transcodeFromUtf8ToISO88591(m_utf8_array,m_local_str);
    m_flags |= eValidLocal;
    return;
  }

  if (m_flags & eValidUtf16){
    BasicTranscoder::transcodeFromUtf16ToISO88591(m_utf16_array,m_local_str);
    m_flags |= eValidLocal;
    return;
  }

  ARCCORE_ASSERT((0),("InternalError in StringImpl::_createLocal()"));
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
_invalidateLocal()
{
  m_flags &= ~eValidLocal;
  m_local_str = std::string();
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

  ostr << "(local=valid="  << (m_flags & eValidLocal)
       << ",len=" << m_local_str.length() << ",val=";
  ostr << m_local_str;
  ostr << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
