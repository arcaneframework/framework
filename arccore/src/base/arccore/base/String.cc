// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* String.cc                                                   (C) 2000-2025 */
/*                                                                           */
/* Chaîne de caractères unicode.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/StringBuilder.h"
#include "arccore/base/CStringUtils.h"
#include "arccore/base/APReal.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/StringView.h"
#include "arccore/base/StringUtils.h"

#include "arccore/base/internal/StringImpl.h"

#include <iostream>
#include <cstring>
#include <limits>
#include <vector>

#define A_FASTLOCK(ptr)
/*!
 * \file StringUtils.h
 *
 * \brief Fonctions utilitaires sur les chaînes de caractères.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(const std::string& str)
: m_p(new StringImpl(str))
, m_const_ptr_size(-1)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(std::string_view str)
: m_p(new StringImpl(str))
, m_const_ptr_size(-1)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(StringView str)
: m_p(new StringImpl(str.toStdStringView()))
, m_const_ptr(nullptr)
, m_const_ptr_size(-1)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(const UCharConstArrayView& ustr)
: m_p(new StringImpl(ustr))
, m_const_ptr_size(-1)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(const Span<const Byte>& ustr)
: m_p(new StringImpl(ustr))
, m_const_ptr_size(-1)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(StringImpl* impl)
: m_p(impl)
, m_const_ptr_size(-1)
{
  if (m_p)
    m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(const char* str,Integer len)
: m_p(new StringImpl(std::string_view(str,len)))
, m_const_ptr_size(-1)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(char* str)
: m_p(new StringImpl(str))
, m_const_ptr_size(-1)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(const char* str,bool do_alloc)
: m_p(nullptr)
, m_const_ptr(nullptr)
, m_const_ptr_size(-1)
{
  if (do_alloc){
    m_p = new StringImpl(str);
    m_p->addReference();
  }
  else{
    m_const_ptr = str;
    if (m_const_ptr)
      m_const_ptr_size = std::strlen(str);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::
String(const String& str)
: m_p(str.m_p)
, m_const_ptr(str.m_const_ptr)
, m_const_ptr_size(str.m_const_ptr_size)
{
  if (m_p)
    m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String& String::
operator=(const String& str)
{
  if (str.m_p)
    str.m_p->addReference();
  _removeReferenceIfNeeded();
  _copyFields(str);
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String& String::
operator=(String&& str)
{
  _removeReferenceIfNeeded();

  _copyFields(str);
  str._resetFields();

  return (*this);

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String& String::
operator=(StringView str)
{
  return this->operator=(String(str));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String& String::
operator=(std::string_view str)
{
  return this->operator=(String(str));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String& String::
operator=(const std::string& str)
{
  return this->operator=(String(str));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const char* String::
localstr() const
{
  if (m_const_ptr)
    return m_const_ptr;
  if (m_p)
    return m_p->toStdStringView().data();
  return "";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void String::
_removeImplReference()
{
  m_p->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<UChar> String::
_internalUtf16BE() const
{
  A_FASTLOCK(this);
  if (!m_p){
    if (m_const_ptr){
      _checkClone();
    }
  }
  if (m_p)
    return m_p->utf16();
  return ConstArrayView<UChar>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<UChar> String::
utf16() const
{
  return _internalUtf16BE();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ByteConstArrayView String::
utf8() const
{
  if (m_p)
    return m_p->largeUtf8().smallView();
  if (m_const_ptr){
    Int64 ts = m_const_ptr_size+1;
    Int32 s = arccoreCheckArraySize(ts);
    return ByteConstArrayView(s,reinterpret_cast<const Byte*>(m_const_ptr));
  }
  return ByteConstArrayView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<const Byte> String::
bytes() const
{
  if (m_p)
    return m_p->bytes();
  if (m_const_ptr)
    return Span<const Byte>(reinterpret_cast<const Byte*>(m_const_ptr),m_const_ptr_size);
  return Span<const Byte>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool String::
null() const
{
  if (m_const_ptr)
    return false;
  return m_p==0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool String::
empty() const
{
  if (m_const_ptr)
    return m_const_ptr[0]=='\0';
  if (m_p)
    return m_p->empty();
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer String::
len() const
{
  auto x = this->toStdStringView();
  return arccoreCheckArraySize(x.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 String::
length() const
{
  auto x = this->toStdStringView();
  return x.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::string_view String::
toStdStringView() const
{
  if (m_const_ptr){
#ifdef ARCCORE_CHECK
    Int64 xlen = std::strlen(m_const_ptr);
    if (xlen!=m_const_ptr_size)
      ARCCORE_FATAL("Bad length (computed={0} stored={1})",xlen,m_const_ptr_size);
#endif
    return _viewFromConstChar();
  }
  if (m_p)
    return m_p->toStdStringView();
  return std::string_view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView String::
view() const
{
  if (m_const_ptr)
    return StringView(_viewFromConstChar());
  if (m_p)
    return m_p->view();
  return StringView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String::operator StringView() const
{
  return view();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void String::
_checkClone() const
{
  if (m_const_ptr || !m_p){
    std::string_view sview;
    if (m_const_ptr)
      sview = _viewFromConstChar();
    m_p = new StringImpl(sview);
    m_p->addReference();
    m_const_ptr = nullptr;
    m_const_ptr_size = -1;
    return;
  }
  if (m_p->nbReference()!=1){
    StringImpl* old_p = m_p;
    m_p = m_p->clone();
    m_p->addReference();
    old_p->removeReference();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
clone() const
{
  if (m_p)
    return String(m_p->clone());
  return String(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String& String::
_append(const String& str)
{
  if (str.null())
    return *this;
  _checkClone();
  if (str.m_const_ptr){
    m_p = m_p->append(str._viewFromConstChar());
  }
  else
    m_p = m_p->append(str.m_p);
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
replaceWhiteSpace(const String& s)
{
  String s2(s);
  s2._checkClone();
  s2.m_p = s2.m_p->replaceWhiteSpace();
  return s2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
collapseWhiteSpace(const String& s)
{
  if (s.null())
    return String();
  String s2(s);
  s2._checkClone();
  s2.m_p = s2.m_p->collapseWhiteSpace();
  return s2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
upper() const
{
  String s2(*this);
  s2._checkClone();
  s2.m_p = s2.m_p->toUpper();
  return s2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
lower() const
{
  String s2(*this);
  s2._checkClone();
  s2.m_p = s2.m_p->toLower();
  return s2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
operator+(std::string_view str) const
{
  if (str.empty())
    return (*this);
  String s2(*this);
  return s2._append(str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
operator+(const std::string& str) const
{
  if (str.empty())
    return (*this);
  String s2(*this);
  return s2._append(str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
operator+(const String& str) const
{
  String s2(*this);
  return s2._append(str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
fromNumber(unsigned long v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(unsigned int v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(double v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(long double v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(int v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(long v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(unsigned long long v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(long long v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(const APReal& v)
{
  return StringFormatterArg(v).value();
}

String String::
fromNumber(double v,Integer nb_digit_after_point)
{
  Int64 mulp = 1;
  for( Integer z=0; z<nb_digit_after_point; ++z )
    mulp *= 10;
  Int64 p = (Int64)(v * (Real)mulp);
  Int64 after_digit = p % mulp;
  Int64 before_digit = p / mulp;
  StringBuilder s(String::fromNumber(before_digit) + ".");
  {
    Integer nb_zero = 0;
    Int64 mv = mulp / 10;
    for( Integer i=0; i<(nb_digit_after_point-1); ++i ){
      if (after_digit>=mv){
        break;
      }
      ++nb_zero;
      mv /= 10;
    }
    for( Integer i=0; i<nb_zero; ++i )
      s += "0";
  }
  s += String::fromNumber(after_digit);
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
operator+(unsigned long v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

String String::
operator+(unsigned int v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

String String::
operator+(double v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

String String::
operator+(long double v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

String String::
operator+(int v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

String String::
operator+(long v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

String String::
operator+(unsigned long long v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

String String::
operator+(long long v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

String String::
operator+(const APReal& v) const
{
  String s2(*this);
  return s2._append(String::fromNumber(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool String::
isLess(const String& b) const
{
  if (m_const_ptr){
    if (b.m_const_ptr)
      return CStringUtils::isLess(m_const_ptr,b.m_const_ptr);
    if (b.m_p){
      if (b.m_p->isEqual(m_const_ptr))
        return false;
      return !(b.m_p->isLessThan(m_const_ptr));
    }
    // b est la chaîne nulle mais pas moi.
    return false;
  }

  if (b.m_const_ptr){
    if (m_p)
      return m_p->isLessThan(b.m_const_ptr);
    // Je suis la chaîne nulle mais pas b.
    return true;    
  }

  if (m_p){
    if (b.m_p)
      return m_p->isLessThan(b.m_p);
    // b est la chaine nulle mais pas moi
    return false;
  }

  // Je suis la chaine nulle
  return b.m_p==nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
fromUtf8(Span<const Byte> bytes)
{
  return String(new StringImpl(bytes));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void String::
internalDump(std::ostream& ostr) const
{
  ostr << "StringDump(m_const_ptr=" << (void*)m_const_ptr << ",m_p=" << m_p;
  if (m_p){
    ostr << ",";
    m_p->internalDump(ostr);
  }
  ostr << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 String::
hashCode() const
{
  Span<const Byte> s = bytes();
  Int32 h = 0;
  Int64 n = s.size();
  for( Int64 i=0; i<n; ++i ){
    h = (h << 5) - h + s[i];
  }
  return h;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringFormatterArg::
_formatReal(Real avalue)
{
  std::ostringstream ostr;
  ostr.precision(std::numeric_limits<Real>::digits10);
  ostr << avalue;
  m_str_value = ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StringFormatter
{
 public:
  StringFormatter(const String& format)
  : m_format(format), m_current_arg(0) {}
 public:
  void addArg(const String& ostr)
  {
    char buf[20];
    Integer nb_z = 0;
    Integer z = m_current_arg;
    ++m_current_arg;
    if (z>=100){
      // N'arrive normalement jamais car seul ce fichier a accès à cette
      // méthode.
      std::cerr << "Too many args (maximum is 100)";
      return;
    }
    else if (z>=10){
      nb_z = 2;
      buf[0] = (char)('0' + (z / 10));
      buf[1] = (char)('0' + (z % 10));
    }
    else{
      nb_z = 1;
      buf[0] = (char)('0' + z);
    }
    buf[nb_z] = '}';
    ++nb_z;
    buf[nb_z] = '\0';

    std::string str = m_format.localstr();
    const char* local_str = str.c_str();
    // TODO: ne pas utiliser de String mais un StringBuilder pour format
    const Int64 slen = str.length();
    for( Int64 i=0; i<slen; ++i ){
      if (local_str[i]=='{'){
        if (i+nb_z>=slen)
          break;
        bool is_ok = true;
        for( Integer j=0; j<nb_z; ++j )
          if (local_str[i+1+j]!=buf[j]){
            is_ok = false;
            break;
          }
        if (is_ok){
          std::string str1(local_str,local_str+i);
          std::string str2(local_str+i+1+nb_z);
          m_format = String(str1) + ostr + str2;
          // Il faut quitter tout de suite car str n'est plus valide
          // puisque m_format a changé.
          break;
        }
      }
    }
  }
 public:
  const String& value() const { return m_format; }
 public:
  String m_format;
  Integer m_current_arg;
 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str)
{
  return str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1,
       const StringFormatterArg& arg2)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  sf.addArg(arg2.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  sf.addArg(arg2.value());
  sf.addArg(arg3.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3,
       const StringFormatterArg& arg4)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  sf.addArg(arg2.value());
  sf.addArg(arg3.value());
  sf.addArg(arg4.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3,
       const StringFormatterArg& arg4,
       const StringFormatterArg& arg5)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  sf.addArg(arg2.value());
  sf.addArg(arg3.value());
  sf.addArg(arg4.value());
  sf.addArg(arg5.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3,
       const StringFormatterArg& arg4,
       const StringFormatterArg& arg5,
       const StringFormatterArg& arg6)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  sf.addArg(arg2.value());
  sf.addArg(arg3.value());
  sf.addArg(arg4.value());
  sf.addArg(arg5.value());
  sf.addArg(arg6.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3,
       const StringFormatterArg& arg4,
       const StringFormatterArg& arg5,
       const StringFormatterArg& arg6,
       const StringFormatterArg& arg7)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  sf.addArg(arg2.value());
  sf.addArg(arg3.value());
  sf.addArg(arg4.value());
  sf.addArg(arg5.value());
  sf.addArg(arg6.value());
  sf.addArg(arg7.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3,
       const StringFormatterArg& arg4,
       const StringFormatterArg& arg5,
       const StringFormatterArg& arg6,
       const StringFormatterArg& arg7,
       const StringFormatterArg& arg8)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  sf.addArg(arg2.value());
  sf.addArg(arg3.value());
  sf.addArg(arg4.value());
  sf.addArg(arg5.value());
  sf.addArg(arg6.value());
  sf.addArg(arg7.value());
  sf.addArg(arg8.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
format(const String& str,const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3,
       const StringFormatterArg& arg4,
       const StringFormatterArg& arg5,
       const StringFormatterArg& arg6,
       const StringFormatterArg& arg7,
       const StringFormatterArg& arg8,
       const StringFormatterArg& arg9)
{
  StringFormatter sf(str);
  sf.addArg(arg1.value());
  sf.addArg(arg2.value());
  sf.addArg(arg3.value());
  sf.addArg(arg4.value());
  sf.addArg(arg5.value());
  sf.addArg(arg6.value());
  sf.addArg(arg7.value());
  sf.addArg(arg8.value());
  sf.addArg(arg9.value());
  return sf.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
concat(const StringFormatterArg& arg1)
{
  return arg1.value();
}

String String::
concat(const StringFormatterArg& arg1,
       const StringFormatterArg& arg2)
{
  return arg1.value()+arg2.value();
}

String String::
concat(const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3)
{
  return arg1.value()+arg2.value()+arg3.value();
}

String String::
concat(const StringFormatterArg& arg1,
       const StringFormatterArg& arg2,
       const StringFormatterArg& arg3,
       const StringFormatterArg& arg4)
{
  return arg1.value()+arg2.value()+arg3.value()+arg4.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
plural(const Integer n, const String & str, const bool with_number)
{
  return String::plural(n, str, str+"s", with_number);
}

namespace
{
inline int
_abs(Integer a)
{
  return (a>0) ? a : (-a);
}
}

String String::
plural(const Integer n, const String & str, const String & str2, const bool with_number)
{
  if (with_number)
    return String::concat(n, " ", ((_abs(n) > 1)?str2:str) );
  else
    return ((_abs(n) > 1)?str2:str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool String::
contains(const String& arg1) const
{
  // Considère que la chaîne nulle est incluse dans toute chaîne
  if (arg1.null())
    return true;
  if (null())
    return false;
  std::string_view a = this->toStdStringView();
  std::string_view b = arg1.toStdStringView();
  return a.find(b) != std::string_view::npos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool String::
endsWith(const String& s) const
{
  Span<const Byte> v = bytes();
  Span<const Byte> ref = s.bytes();
  Int64 ref_size = ref.size();
  Int64 v_size = v.size();
  if (ref_size>v_size)
    return false;
  const Byte* v_begin = &v[v_size-ref_size];
  return std::memcmp(v_begin,ref.data(),ref_size)==0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool String::
startsWith(const String& s) const
{
  Span<const Byte> v = bytes();
  Span<const Byte> ref = s.bytes();
  Int64 ref_size = ref.size();
  Int64 v_size = v.size();
  if (ref_size>v_size)
    return false;
  return memcmp(v.data(),ref.data(),ref_size)==0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
substring(Int64 pos) const
{
  return substring(pos,length()-pos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
substring(Int64 pos,Int64 len) const
{
  if (pos<0)
    pos = 0;
  //TODO: normalement le _checkClone() n'est pas utile
  _checkClone();
  String s2(StringImpl::substring(m_p,pos,len));
  return s2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String String::
join(String delim,ConstArrayView<String> strs)
{
  StringBuilder sb;
  for( Integer i=0, n=strs.size(); i<n; ++i ){
    if (i!=0)
      sb += delim;
    sb += strs[i];
  }
  return sb.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" bool
operator==(const String& a,const String& b)
{
  //cout << "COMPARE String String a='" << a << "' b='" << b << "'\n";
  if (a.m_const_ptr){
    if (b.m_const_ptr)
      return CStringUtils::isEqual(a.m_const_ptr,b.m_const_ptr);
    if (b.m_p)
      return b.m_p->isEqual(a.m_const_ptr);
    // Si je suis ici, 'b' est la chaine nulle et pas 'a'
    return false;
  }

  if (b.m_const_ptr){
    if (a.m_p)
      return a.m_p->isEqual(b.m_const_ptr);
    return false;
  }
  if (a.m_p){
    if (b.m_p)
      return a.m_p->isEqual(b.m_p);
    // b est la chaine nulle mais pas moi
    return false;
  }

  // Je suis la chaine nulle
  return b.m_p==nullptr;
}

extern "C++" bool
operator==(const String& a,const char* b)
{
  return a==String(b);
}


extern "C++" bool
operator==(const char* a,const String& b)
{
  return String(a)==b;
}

extern "C++" bool
operator<(const String& a,const String& b)
{
  bool v = a.isLess(b);
  //cout << "IsLess a='" << a << "' b='" << b << "' v=" << v << '\n';
  /*cout << "COMPARE A=" << a << " B=" << b << " V=" << v;
  if (a.m_const_ptr)
    cout << " AConst = " << a.m_const_ptr;
  if (b.m_const_ptr)
    cout << " BConst = " << b.m_const_ptr;
    cout << '\n';*/
  return v;
}

extern "C++" String
operator+(const char* a,const String& b)
{
  return String(a)+b;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
bool global_write_utf8 = false;
}
std::ostream&
operator<<(std::ostream& o,const String& str)
{
  // A utiliser plus tard lorsque l'encodage par défaut sera UTF8
  if (global_write_utf8)
    str.writeBytes(o);
  else
    o << str.localstr();

  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void String::
writeBytes(std::ostream& o) const
{
  Span<const Byte> v = this->bytes();
  Int64 vlen = v.size();
  o.write((const char*)v.data(),vlen);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::istream&
operator>>(std::istream& i,String& str)
{
  std::string s;
  i >> s;
  str = s;
  return i;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StringUtilsImpl
{
 public:
  static std::vector<UChar>
  toUtf16BE(const String& str)
  {
    ConstArrayView<UChar> x{ str._internalUtf16BE() };
    Int32 n = x.size();
    if (n==0)
      return {};
    // x contient normalement toujours un zéro terminal que l'on ne met
    // pas dans le vecteur
    return { x.begin(), x.begin()+(n-1) };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::vector<UChar> StringUtils::
asUtf16BE(const String& str)
{
  return StringUtilsImpl::toUtf16BE(str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::wstring StringUtils::
convertToStdWString([[maybe_unused]] const String& str)
{
#ifdef ARCCORE_OS_WIN32
  ConstArrayView<UChar> utf16 = str.utf16();
  const wchar_t* wdata = reinterpret_cast<const wchar_t*>(utf16.data());
  const size_t wlen = utf16.size();
  std::wstring_view wstr_view(wdata,wlen);
  return std::wstring(wstr_view);
#else
  ARCCORE_THROW(NotSupportedException,"This function is only supported on Win32");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String StringUtils::
convertToArcaneString([[maybe_unused]] const std::wstring_view& wstr)
{
#ifdef ARCCORE_OS_WIN32
  const UChar* ux = reinterpret_cast<const UChar*>(wstr.data());
  Int32 len = arccoreCheckArraySize(wstr.length());
  ConstArrayView<UChar> buf(len, ux);
  return String(buf);
#else
  ARCCORE_THROW(NotSupportedException,"This function is only supported on Win32");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

