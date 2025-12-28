// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONReader.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Lecteur au format JSON.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/JSONReader.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/CheckedConvert.h"
#include "arccore/base/internal/ConvertInternal.h"

#define RAPIDJSON_HAS_STDSTRING 1
#include "arccore/common/internal/json/rapidjson/document.h"
#include "arccore/common/internal/json/rapidjson/stringbuffer.h"

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
    return StringView(Span<const Byte>((const Byte*)x.GetString(), x.GetStringLength()));
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

String JSONValue::
value() const
{
  if (!m_p)
    return String();
  auto x = m_p->toValue();
  if (x->IsString())
    return String(Span<const Byte>((const Byte*)x->GetString(), x->GetStringLength()));
  return String();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView JSONValue::
valueAsStringView() const
{
  if (!m_p)
    return StringView();
  auto x = m_p->toValue();
  if (x->IsString())
    return StringView(Span<const Byte>((const Byte*)x->GetString(), x->GetStringLength()));
  return StringView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView JSONValue::
valueAsString() const
{
  return valueAsStringView();
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

bool JSONValue::
valueAsBool() const
{
  if (!m_p)
    return false;
  auto x = m_p->toValue();
  if (x->IsBool())
    return x->GetBool();
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 JSONValue::
valueAsInt32() const
{
  Int64 v = valueAsInt64();
  return CheckedConvert::toInt32(v);
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
  if (x->GetType() == rapidjson::kStringType) {
    // Convertie la chaîne de caractères en un réél
    StringView s = this->valueAsStringView();
    Real r = 0.0;
    if (!Convert::Impl::StringViewToIntegral::getValue(r, s))
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
  if (x == d->MemberEnd())
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

JSONValue JSONValue::
expectedChild(StringView name) const
{
  JSONKeyValue k = keyValueChild(name);
  if (!k)
    ARCCORE_FATAL("No key '{0}' found in json document", name);
  return k.value();
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
    ARCCORE_FATAL("The value has to be of type 'object'");
  for (auto& x : d->GetObject()) {
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
    ARCCORE_FATAL("The value has to be of type 'array'");
  for (rapidjson::SizeType i = 0; i < d->Size(); ++i) {
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
  for (auto& x : d->GetObject()) {
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

  Impl()
  : m_document()
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

namespace
{
  using namespace rapidjson;

  const char*
  _getErrorCodeString(ParseErrorCode c)
  {
    switch (c) {
    case kParseErrorNone:
      return "No error";
    case kParseErrorDocumentEmpty:
      return "The document is empty";
    case kParseErrorDocumentRootNotSingular:
      return "The document root must not follow by other values";
    case kParseErrorValueInvalid:
      return "Invalid value";
    case kParseErrorObjectMissName:
      return "Missing a name for object member";
    case kParseErrorObjectMissColon:
      return "Missing a colon after a name of object member";
    case kParseErrorObjectMissCommaOrCurlyBracket:
      return "Missing a comma or '}' after an object member";
    case kParseErrorArrayMissCommaOrSquareBracket:
      return "Missing a comma or ']' after an array element";
    case kParseErrorStringUnicodeEscapeInvalidHex:
      return "Incorrect hex digit after \\u escape in string";
    case kParseErrorStringUnicodeSurrogateInvalid:
      return "The surrogate pair in string is invalid";
    case kParseErrorStringEscapeInvalid:
      return "Invalid escape character in string";
    case kParseErrorStringMissQuotationMark:
      return "Missing a closing quotation mark in string";
    case kParseErrorStringInvalidEncoding:
      return "Invalid encoding in string";
    case kParseErrorNumberTooBig:
      return "Number too big to be stored in double";
    case kParseErrorNumberMissFraction:
      return "Miss fraction part in number";
    case kParseErrorNumberMissExponent:
      return "Miss exponent in number";
    case kParseErrorTermination:
      return "Parsing was terminated";
    case kParseErrorUnspecificSyntaxError:
      return "Unspecific syntax error";
    default:
      return "Unknown";
    }
    return "Unknown";
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JSONDocument::
parse(Span<const std::byte> bytes, StringView filename)
{
  using namespace rapidjson;
  Document& d = m_p->m_document;
  ParseResult r = d.Parse((const char*)bytes.data(), bytes.size());
  if (d.HasParseError()) {
    std::cout << "ERROR: " << d.GetParseError() << "\n";
    ARCCORE_FATAL("Parsing error file='{0}' ret={1} position={2} message='{3}'",
                  filename, d.GetParseError(),
                  r.Offset(), _getErrorCodeString(r.Code()));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JSONDocument::
parse(Span<const std::byte> bytes)
{
  parse(bytes, "(Unknown)");
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

void JSONDocument::
parse(Span<const Byte> bytes, StringView filename)
{
  parse(asBytes(bytes), filename);
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
