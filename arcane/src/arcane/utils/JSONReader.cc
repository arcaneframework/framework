// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONReader.cc                                               (C) 2000-2019 */
/*                                                                           */
/* Lecteur au format JSON.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/JSONReader.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"

#define RAPIDJSON_HAS_STDSTRING 1
#include "arcane/utils/internal/json/rapidjson/document.h"
#include "arcane/utils/internal/json/rapidjson/stringbuffer.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class JSONKeyValue::Impl
{
 public:
  rapidjson::Value::Member* toMember() const
  {
    return (rapidjson::Value::Member*)(this);
  }
};

class JSONValue::Impl
{
 public:
  rapidjson::Value* toValue() const
  {
    return (rapidjson::Value*)(this);
  }
};

class JSONWrapperUtils
{
 public:
  static JSONKeyValue build(rapidjson::Value::Member* v)
  {
    return JSONKeyValue((JSONKeyValue::Impl*)(v));
  }
  static JSONKeyValue build(rapidjson::Value::ConstMemberIterator v)
  {
    return JSONKeyValue((JSONKeyValue::Impl*)&(*v));
  }
  static JSONValue build(rapidjson::Value* v)
  {
    return JSONValue((JSONValue::Impl*)(v));
  }
};

/*
 * Types définis par rapidjson.
 enum Type {
  kNullType = 0,      //!< null
  kFalseType = 1,     //!< false
  kTrueType = 2,      //!< true
  kObjectType = 3,    //!< object
  kArrayType = 4,     //!< array
  kStringType = 5,    //!< string
  kNumberType = 6     //!< number
};
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView JSONKeyValue::
name() const
{
  if (!m_p)
    return StringView();
  auto& x = m_p->toMember()->name;
  if (x.IsString())
    return StringView(Span<const Byte>((const Byte*)x.GetString(),x.GetStringLength()));
  return StringView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONValue JSONKeyValue::
value() const
{
  if (!m_p)
    return JSONValue();
  auto& x = m_p->toMember()->value;
  return JSONWrapperUtils::build(&x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView JSONValue::
valueAsString() const
{
  if (!m_p)
    return StringView();
  auto x = m_p->toValue();
  if (x->IsString())
    return StringView(Span<const Byte>((const Byte*)x->GetString(),x->GetStringLength()));
  return StringView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 JSONValue::
valueAsInt64() const
{
  if (!m_p)
    return 0;
  auto x = m_p->toValue();
  if (x->IsInt64())
    return x->GetInt64();
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real JSONValue::
valueAsReal() const
{
  if (!m_p)
    return 0;
  auto x = m_p->toValue();
  std::cout << "TYPE=" << x->GetType() << "\n";
  if (x->IsDouble())
    return x->GetDouble();
  if (x->GetType()==rapidjson::kStringType){
    // Convertie la chaîne de caractères en un réél
    StringView s = this->valueAsString();
    Real r = 0.0;
    if (!builtInGetValue(r,s))
      return r;
  }
  return 0.0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONKeyValue JSONValue::
keyValueChild(StringView name) const
{
  if (!m_p)
    return JSONKeyValue();
  auto d = m_p->toValue();
  rapidjson::Value::MemberIterator x = d->FindMember((const char*)(name.bytes().data()));
  if (x==d->MemberEnd())
    return JSONKeyValue();
  return JSONWrapperUtils::build(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONValue JSONValue::
child(StringView name) const
{
  return keyValueChild(name).value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONValueList JSONValue::
children() const
{
  if (!m_p)
    return JSONValueList();
  auto d = m_p->toValue();
  JSONValueList values;
  if (!d->IsObject())
    ARCANE_FATAL("The value has to be of type 'object'");
  for( auto& x : d->GetObject()){
    auto y = JSONWrapperUtils::build(&x);
    values.add(y.value());
  }
  return values;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONValueList JSONValue::
valueAsArray() const
{
  if (!m_p)
    return JSONValueList();
  auto d = m_p->toValue();
  JSONValueList values;
  if (!d->IsArray())
    ARCANE_FATAL("The value has to be of type 'array'");
  for( rapidjson::SizeType i = 0; i < d->Size(); ++i ){
    rapidjson::Value& x = (*d)[i];
    auto y = JSONWrapperUtils::build(&x);
    values.add(y);
  }
  return values;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool JSONValue::
isArray() const
{
  if (!m_p)
    return false;
  auto d = m_p->toValue();
  return d->IsArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool JSONValue::
isObject() const
{
  if (!m_p)
    return false;
  auto d = m_p->toValue();
  return d->IsObject();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONKeyValueList JSONValue::
keyValueChildren() const
{
  if (!m_p)
    return JSONKeyValueList();
  auto d = m_p->toValue();
  JSONKeyValueList values;
  for( auto& x : d->GetObject()){
    auto y = JSONWrapperUtils::build(&x);
    values.add(y);
  }
  return values;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class JSONDocument::Impl
{
 public:
  Impl() : m_document()
  {
  }
 public:
  rapidjson::Document m_document;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONDocument::
JSONDocument()
: m_p(nullptr)
{
  m_p = new Impl();
}

JSONDocument::
~JSONDocument()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JSONDocument::
parse(Span<const std::byte> bytes)
{
  rapidjson::Document& d = m_p->m_document;
  d.Parse((const char*)bytes.data(),bytes.size());
  if (d.HasParseError()){
    std::cout << "ERROR: " << d.GetParseError() << "\n";
    ARCANE_FATAL("Parsing error ret={0}",d.GetParseError());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JSONDocument::
parse(Span<const Byte> bytes)
{
  parse(asBytes(bytes));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONValue JSONDocument::
root() const
{
  rapidjson::Value& d = m_p->m_document;
  return JSONWrapperUtils::build(&d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

