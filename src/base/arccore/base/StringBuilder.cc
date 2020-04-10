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
/* StringBuilder.cc                                            (C) 2000-2018 */
/*                                                                           */
/* Constructeur de chaîne de caractère unicode.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringBuilder.h"
#include "arccore/base/StringImpl.h"
#include "arccore/base/String.h"
#include "arccore/base/StringView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
StringBuilder(const std::string& str)
: m_p(new StringImpl(str))
, m_const_ptr(0)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
StringBuilder(const UCharConstArrayView& ustr)
: m_p(new StringImpl(ustr.data()))
, m_const_ptr(0)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
StringBuilder(const ByteConstArrayView& ustr)
: m_p(new StringImpl(ustr))
, m_const_ptr(0)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
StringBuilder(StringImpl* impl)
: m_p(impl)
, m_const_ptr(0)
{
  if (m_p)
    m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
StringBuilder(const char* str,Integer len)
: m_p(new StringImpl(std::string_view(str,len)))
, m_const_ptr(0)
{
  m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
StringBuilder(const char* str)
: m_p(0)
, m_const_ptr(0)
{
  bool do_alloc = true;
  if (do_alloc){
    m_p = new StringImpl(str);
    m_p->addReference();
  }
  else{
    m_p = 0;
    m_const_ptr = str;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
StringBuilder(const String& str)
: m_p(str.m_p)
, m_const_ptr(str.m_const_ptr)
{
  if (m_p)
    m_p->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
StringBuilder(const StringBuilder& str)
: m_p(nullptr)
, m_const_ptr(str.m_const_ptr)
{
  if (str.m_p){
    m_p = new StringImpl(*str.m_p);
    m_p->addReference();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
~StringBuilder()
{
  if (m_p)
    m_p->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const StringBuilder& StringBuilder::
operator=(const String& str)
{
  if (str.m_p)
    str.m_p->addReference();
  if (m_p)
    m_p->removeReference();
  m_p = str.m_p;
  m_const_ptr = str.m_const_ptr;
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const StringBuilder& StringBuilder::
operator=(const char* str)
{
  if (m_p)
    m_p->removeReference();
  m_p = 0;
  m_const_ptr = str;
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Copie \a str dans cette instance.
const StringBuilder& StringBuilder::
operator=(const StringBuilder& str)
{
  if (str!=(*this)){
    if (m_p)
      m_p->removeReference();
    m_const_ptr = str.m_const_ptr;
    if (str.m_p){
      m_p = new StringImpl(*str.m_p);
      m_p->addReference();
    }
  }
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringBuilder::
_checkClone() const
{
  if (m_const_ptr || !m_p){
    std::string_view sv;
    if (m_const_ptr)
      sv = std::string_view(m_const_ptr);
    m_p = new StringImpl(sv);
    m_p->addReference();
    m_const_ptr = nullptr;
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

StringBuilder StringBuilder::
clone() const
{
  if (m_p)
    return StringBuilder(m_p->clone());
  return StringBuilder(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String StringBuilder::
toString() const
{
  if (m_p)
    return String(m_p->clone());
  return String(StringView(m_const_ptr));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder::
operator String() const
{
  return toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder& StringBuilder::
append(const String& str)
{
  if (str.null())
    return *this;
  _checkClone();
  if (str.m_const_ptr){
    StringView sv{std::string_view(str.m_const_ptr,str.m_const_ptr_size)};
    m_p = m_p->append(sv);
  }
  else
    m_p = m_p->append(str.m_p);
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder& StringBuilder::
replaceWhiteSpace()
{
  _checkClone();
  m_p = m_p->replaceWhiteSpace();
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder& StringBuilder::
collapseWhiteSpace()
{
  _checkClone();
  m_p = m_p->collapseWhiteSpace();
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder& StringBuilder::
toUpper()
{
  _checkClone();
  m_p = m_p->toUpper();
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringBuilder& StringBuilder::
toLower()
{
  _checkClone();
  m_p = m_p->toLower();
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringBuilder::
operator+=(const char* str)
{
  append(str);
}

void StringBuilder::
operator+=(const String& str)
{
  append(str);
}

void StringBuilder::
operator+=(char v)
{
  char buf[2];
  buf[0] = v;
  buf[1] = '\0';
  append(buf);
}

void StringBuilder::
operator+=(unsigned long v)
{
  append(String::fromNumber(v));
}

void StringBuilder::
operator+=(unsigned int v)
{
  append(String::fromNumber(v));
}

void StringBuilder::
operator+=(double v)
{
  append(String::fromNumber(v));
}

void StringBuilder::
operator+=(long double v)
{
  append(String::fromNumber(v));
}

void StringBuilder::
operator+=(int v)
{
  append(String::fromNumber(v));
}

void StringBuilder::
operator+=(long v)
{
  append(String::fromNumber(v));
}

void StringBuilder::
operator+=(unsigned long long v)
{
  append(String::fromNumber(v));
}

void StringBuilder::
operator+=(long long v)
{
  append(String::fromNumber(v));
}

void StringBuilder::
operator+=(const APReal& v)
{
  append(String::fromNumber(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringBuilder::
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o,const StringBuilder& str)
{
  String s = str.toString();
  o << s.localstr();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

