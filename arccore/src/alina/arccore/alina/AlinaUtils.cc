// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaUtils.cc                                               (C) 2000-2026 */
/*                                                                           */
/* Utility classes.                                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/CSRMatrixView.h"
#include "arccore/alina/AlinaPropertyTree.h"

// JSON reader and writer of the arccore 'common' component.
#include "arccore/common/JSONReader.h"
#include "arccore/common/JSONWriter.h"

#include "arccore/base/Convert.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using PropertyTreeImpl = detail::PropertyTreeImpl;

namespace detail
{
  const PropertyTreeImpl& empty_ptree()
  {
    static const PropertyTreeImpl p;
    return p;
  }

  // To access PropertyTree::m_property_tree as 'detail::PropertyTreeImpl'
  class PropertyWrapper
  {
   public:

    static const PropertyTreeImpl& toImpl(const PropertyTree& p)
    {
      return *(static_cast<const PropertyTreeImpl*>(p.m_property_tree));
    }
    static PropertyTreeImpl& toImpl(PropertyTree& p)
    {
      return *(static_cast<PropertyTreeImpl*>(p.m_property_tree));
    }
  };

} // namespace detail

namespace
{
  const PropertyTreeImpl& toImpl(const PropertyTree& p)
  {
    return detail::PropertyWrapper::toImpl(p);
  }
  const PropertyTreeImpl& toImpl(const PropertyTree* p)
  {
    return detail::PropertyWrapper::toImpl(*p);
  }
  PropertyTreeImpl& toImpl(PropertyTree* p)
  {
    return detail::PropertyWrapper::toImpl(*p);
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  std::string realToString(double v)
  {
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, res.ptr);
  }

  bool tryParseInt64(const std::string& s, Int64& v)
  {
    auto x = Convert::ScalarType<Int64>::tryParse(StringView(s));
    if (!x)
      return false;
    v = *x;
    return true;
  }

  bool tryParseDouble(const std::string& s, double& v)
  {
    auto x = Convert::ScalarType<Real>::tryParse(StringView(s));
    if (!x)
      return false;
    v = Convert::toDouble(*x);
    return true;
  }

  // 'true' and 'false' are accepted to ease the reading of boolean
  // parameters stored in a JSON file.
  Int64 parseInteger(const std::string& s, const char* name)
  {
    if (s == "true")
      return 1;
    if (s == "false")
      return 0;
    Int64 v = 0;
    if (!tryParseInt64(s, v))
      ARCANE_FATAL("Can not convert value '{0}' to integer for parameter '{1}'", s, name);
    return v;
  }

  Int32 parseInt32(const std::string& s, const char* name)
  {
    Int64 v = parseInteger(s, name);
    if (v < static_cast<Int64>(std::numeric_limits<Int32>::min()) ||
        v > static_cast<Int64>(std::numeric_limits<Int32>::max()))
      ARCANE_FATAL("Value '{0}' is out of range for parameter '{1}'", s, name);
    return static_cast<Int32>(v);
  }

  double parseReal(const std::string& s, const char* name)
  {
    if (s == "true")
      return 1.0;
    if (s == "false")
      return 0.0;
    double v = 0.0;
    if (!tryParseDouble(s, v))
      ARCANE_FATAL("Can not convert value '{0}' to real for parameter '{1}'", s, name);
    return v;
  }

  void* parsePointer(const std::string& s, const char* name)
  {
    if (s.empty())
      return nullptr;
    // The value is a hexadecimal number with an optional '0x' prefix.
    auto x = Convert::ScalarType<Int64>::tryParse(StringView(s));
    if (!x)
      ARCANE_FATAL("Can not convert value '{0}' to pointer for parameter '{1}'", s, name);
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(*x));
  }

  std::string pointerToString(const void* p)
  {
    std::ostringstream ostr;
    ostr << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(p);
    return ostr.str();
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  PropertyTreeImpl nodeFromJson(const JSONValue& v)
  {
    PropertyTreeImpl n;
    if (v.isObject()) {
      for (const auto& kv : v.keyValueChildren()) {
        std::string name(kv.name().toStdStringView());
        n.children.emplace_back(std::move(name), nodeFromJson(kv.value()));
      }
    }
    else if (v.isArray()) {
      for (const auto& e : v.valueAsArray())
        n.children.emplace_back("", nodeFromJson(e));
    }
    else if (v.isString()) {
      n.data = std::string(v.valueAsStringView().toStdStringView());
      n.has_data = true;
    }
    else if (v.isBool()) {
      n.data = v.valueAsBool() ? "true" : "false";
      n.has_data = true;
    }
    else if (v.isNumber()) {
      if (v.isInt64()) {
        n.data = std::to_string(v.valueAsInt64());
      }
      else if (v.isUint64()) {
        UInt64 u = v.valueAsUInt64();
        if (u <= static_cast<UInt64>(std::numeric_limits<Int64>::max()))
          n.data = std::to_string(static_cast<Int64>(u));
        else
          n.data = realToString(static_cast<double>(u));
      }
      else {
        n.data = realToString(v.valueAsReal());
      }
      n.has_data = true;
    }
    // A 'null' value produces an empty node.
    return n;
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  void writeKeyValue(JSONWriter& w, const StringView& key, const std::string& s)
  {
    if (s == "true") {
      w.write(key, true);
      return;
    }
    if (s == "false") {
      w.write(key, false);
      return;
    }
    Int64 i = 0;
    if (tryParseInt64(s, i)) {
      w.write(key, i);
      return;
    }
    double d = 0.0;
    if (tryParseDouble(s, d)) {
      w.write(key, d);
      return;
    }
    w.write(key, StringView(s));
  }

  void writeJsonNode(JSONWriter& w, const PropertyTreeImpl& n)
  {
    if (n.children.empty()) {
      w.writeValue(n.has_data ? StringView(n.data) : StringView());
      return;
    }
    bool is_array = true;
    for (const auto& c : n.children)
      if (!c.first.empty()) {
        is_array = false;
        break;
      }
    if (is_array) {
      w.beginArray();
      for (const auto& c : n.children)
        writeJsonNode(w, c.second);
      w.endArray();
    }
    else {
      w.beginObject();
      for (const auto& c : n.children) {
        const StringView key(c.first);
        if (c.second.children.empty()) {
          writeKeyValue(w, key, c.second.has_data ? c.second.data : std::string());
        }
        else {
          w.writeKey(key);
          writeJsonNode(w, c.second);
        }
      }
      w.endObject();
    }
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyTree::
PropertyTree()
: m_property_tree(new PropertyTreeImpl())
, m_is_own(true)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyTree::
PropertyTree(const PropertyTree& rhs)
{
  if (rhs.m_is_own) {
    m_property_tree = new PropertyTreeImpl(toImpl(rhs));
    m_is_own = true;
  }
  else {
    m_property_tree = rhs.m_property_tree;
    m_is_own = false;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyTree::
~PropertyTree()
{
  if (m_is_own)
    delete &toImpl(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyTree PropertyTree::
get_child_empty(const std::string& path) const
{
  const PropertyTreeImpl* child = toImpl(this).findNode(path);
  PropertyTree p;
  p.m_property_tree = const_cast<PropertyTreeImpl*>(child ? child : &detail::empty_ptree());
  p.m_is_own = false;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PropertyTree::
erase(const char* name)
{
  return toImpl(this).eraseChild(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

size_t PropertyTree::
count(const char* name) const
{
  return toImpl(this).count(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
read_json(const std::string& filename)
{
  std::ifstream is(filename, std::ios::binary);
  if (!is)
    ARCANE_FATAL("Can not read JSON file '{0}'", filename);
  std::string content((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());

  JSONDocument doc;
  Span<const Byte> bytes(reinterpret_cast<const Byte*>(content.data()), content.size());
  doc.parse(bytes, StringView(filename));

  toImpl(this) = nodeFromJson(doc.root());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 PropertyTree::get(const char* param_type, Int32 default_value) const
{
  const std::string* s = toImpl(this).getValue(param_type);
  if (!s)
    return default_value;
  return parseInt32(*s, param_type);
}
Int64 PropertyTree::get(const char* param_type, Int64 default_value) const
{
  const std::string* s = toImpl(this).getValue(param_type);
  if (!s)
    return default_value;
  return parseInteger(*s, param_type);
}
double PropertyTree::get(const char* param_type, double default_value) const
{
  const std::string* s = toImpl(this).getValue(param_type);
  if (!s)
    return default_value;
  return parseReal(*s, param_type);
}
double* PropertyTree::get(const char* param_type, double* default_value) const
{
  const std::string* s = toImpl(this).getValue(param_type);
  if (!s)
    return default_value;
  return static_cast<double*>(parsePointer(*s, param_type));
}
void* PropertyTree::get(const char* param_type, void* default_value) const
{
  const std::string* s = toImpl(this).getValue(param_type);
  if (!s)
    return default_value;
  return parsePointer(*s, param_type);
}
std::string PropertyTree::get(const char* param_type, const std::string& default_value) const
{
  const std::string* s = toImpl(this).getValue(param_type);
  if (!s)
    return default_value;
  return *s;
}

void PropertyTree::put(const std::string& path, Int32 value)
{
  toImpl(this).put(path, std::to_string(value));
}
void PropertyTree::put(const std::string& path, Int64 value)
{
  toImpl(this).put(path, std::to_string(value));
}
void PropertyTree::put(const std::string& path, double value)
{
  toImpl(this).put(path, realToString(value));
}
void PropertyTree::put(const std::string& path, const std::string& value)
{
  toImpl(this).put(path, value);
}
void PropertyTree::put(const std::string& path, double* value)
{
  toImpl(this).put(path, pointerToString(value));
}
void PropertyTree::put(const std::string& path, void* value)
{
  toImpl(this).put(path, pointerToString(value));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
_addChild(const std::string& path, const char* name,
          const PropertyTree& obj)
{
  toImpl(this).addChild(std::string(path) + name, toImpl(obj));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
check_params(const std::set<std::string>& names) const
{
  const auto& p = toImpl(this);
  bool has_error = false;
  for (const auto& n : names) {
    if (!p.count(n)) {
      ARCCORE_ALINA_PARAM_MISSING(n);
    }
  }
  for (const auto& v : p.children) {
    if (!names.count(v.first)) {
      std::cerr << "WARNING: unknown parameter " << v.first << "\n";
      has_error = true;
    }
  }
  if (has_error)
    ARCANE_FATAL("Invalid parameters");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
check_params(const std::set<std::string>& names,
             const std::set<std::string>& opt_names) const
{
  const auto& p = toImpl(this);
  bool has_error = false;

  for (const auto& n : names) {
    if (!p.count(n)) {
      ARCCORE_ALINA_PARAM_MISSING(n);
    }
  }
  for (const auto& n : opt_names) {
    if (!p.count(n)) {
      ARCCORE_ALINA_PARAM_MISSING(n);
    }
  }
  for (const auto& v : p.children) {
    if (!names.count(v.first) && !opt_names.count(v.first)) {
      std::cerr << "WARNING: unknown parameter " << v.first << "\n";
      has_error = true;
    }
  }
  if (has_error)
    ARCANE_FATAL("Invalid parameters");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyTree::
putKeyValue(const std::string& param)
{
  size_t eq_pos = param.find('=');
  if (eq_pos == std::string::npos)
    ARCANE_FATAL("param in put() should have \"key=value\" format (param='{0}')", param);
  toImpl(this).put(param.substr(0, eq_pos), param.substr(eq_pos + 1));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

detail::empty_params::
empty_params(const PropertyTree& ap)
{
  const auto& p = toImpl(ap);
  for (const auto& v : p.children) {
    std::cerr << "Alina: unknown parameter " << v.first << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& o, const PropertyTree& obj)
{
  JSONWriter writer(JSONWriter::FormatFlags::None);
  writeJsonNode(writer, toImpl(obj));
  o << writer.getBuffer();
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
