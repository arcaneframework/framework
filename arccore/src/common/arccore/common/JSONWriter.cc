// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONWriter.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Ecrivain au format JSON.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/JSONWriter.h"
#include "arccore/base/ArrayView.h"
#include "arccore/base/String.h"
#include "arccore/base/FatalErrorException.h"
#include <limits>

// Les deux lignes suivantes permettent d'utiliser des indices sur 64 bits
// au lieu de 32 par défaut.
#define RAPIDJSON_NO_SIZETYPEDEFINE
namespace rapidjson { typedef ::std::size_t SizeType; }

#define RAPIDJSON_HAS_STDSTRING 1
#include "arccore/common/internal/json/rapidjson/writer.h"
#include "arccore/common/internal/json/rapidjson/prettywriter.h"
#include "arccore/common/internal/json/rapidjson/document.h"
#include "arccore/common/internal/json/rapidjson/stringbuffer.h"

#include <sstream>
#include <iomanip>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class JSONWriter::Impl
{
 public:

  Impl(bool use_hex_float)
  : m_writer(m_buffer)
  , m_use_hex_float(use_hex_float)
  {
    m_writer.SetIndent(' ', 1);
    m_writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
    m_real_ostr.precision(std::numeric_limits<Real>::max_digits10);
  }

 public:

  void writeKey(StringView key)
  {
    Span<const Byte> bytes = key.bytes();
    Int64 size = bytes.size();
    if (size == 0)
      ARCCORE_FATAL("null size");
    // TODO: regarder s'il faut toujours mettre 'true' et donc faire une copie.
    m_writer.Key((const char*)bytes.data(), size, true);
  }
  void writeStringValue(StringView value)
  {
    Span<const Byte> bytes = value.bytes();
    Int64 size = bytes.size();
    if (size == 0) {
      m_writer.Null();
      return;
    }
    // TODO: regarder s'il faut toujours mettre 'true' et donc faire une copie.
    m_writer.String((const char*)bytes.data(), size, true);
  }
  void writeStringValue(const std::string& value)
  {
    m_writer.String(value);
  }
  void write(StringView key, const char* v);
  void write(StringView key, Real v)
  {
    writeKey(key);
    if (m_use_hex_float) {
      char buf[32];
      sprintf(buf, "%a", v);
      m_writer.String(buf);
    }
    else
      m_writer.Double(v);
  }
  void write(StringView key, Span<const Real> view)
  {
    writeKey(key);
    {
      m_real_ostr.str(std::string());
      // NOTE: avec le C++11, on peut utiliser std::hexfloat comme modificateur
      // pour écrire directement en hexadécimal flottant. Cependant ne fonctionne
      // qu'à partir de GCC 5.0 ou visual studio 2015.
      // ostr << std::hexfloat;
      char buf[32];
      for (Int64 i = 0, n = view.size(); i < n; ++i) {
        if (i != 0)
          m_real_ostr << ' ';
        if (m_use_hex_float) {
          sprintf(buf, "%a", view[i]);
          m_real_ostr << buf;
        }
        else
          m_real_ostr << view[i];
      }
      writeStringValue(m_real_ostr.str());
    }
  }

 public:

  rapidjson::StringBuffer m_buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> m_writer;
  bool m_use_hex_float;
  std::ostringstream m_real_ostr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONWriter::
JSONWriter(FormatFlags format_flags)
: m_p(nullptr)
{
  bool use_hex_float = (int)format_flags & (int)FormatFlags::HexFloat;
  m_p = new Impl(use_hex_float);
}

JSONWriter::
~JSONWriter()
{
  delete m_p;
}

void JSONWriter::
beginObject()
{
  m_p->m_writer.StartObject();
}
void JSONWriter::
endObject()
{
  m_p->m_writer.EndObject();
}
void JSONWriter::
writeKey(StringView key)
{
  m_p->writeKey(key);
}
void JSONWriter::
beginArray()
{
  m_p->m_writer.StartArray();
}
void JSONWriter::
endArray()
{
  m_p->m_writer.EndArray();
}

void JSONWriter::
writeValue(StringView str)
{
  m_p->writeStringValue(str);
}
void JSONWriter::
write(StringView key, bool v)
{
  m_p->writeKey(key);
  m_p->m_writer.Bool(v);
}
void JSONWriter::
_writeInt64(StringView key, Int64 v)
{
  m_p->writeKey(key);
  m_p->m_writer.Int64(v);
}
void JSONWriter::
_writeUInt64(StringView key, UInt64 v)
{
  m_p->writeKey(key);
  m_p->m_writer.Uint64(v);
}
void JSONWriter::
write(StringView key, Real v)
{
  m_p->write(key, v);
}
void JSONWriter::
write(StringView key, StringView str)
{
  m_p->writeKey(key);
  m_p->writeStringValue(str);
}
void JSONWriter::
write(StringView key, const char* v)
{
  m_p->writeKey(key);
  m_p->writeStringValue(StringView(v));
}
void JSONWriter::
write(StringView key, std::string_view v)
{
  m_p->writeKey(key);
  m_p->writeStringValue(v);
}
void JSONWriter::
writeIfNotNull(StringView key, const String& str)
{
  if (str.null())
    return;
  m_p->writeKey(key);
  m_p->writeStringValue(str);
}

void JSONWriter::
write(StringView key, Span<const Int32> view)
{
  m_p->writeKey(key);
  m_p->m_writer.StartArray();
  for (Int64 i = 0, n = view.size(); i < n; ++i)
    m_p->m_writer.Int(view[i]);
  m_p->m_writer.EndArray();
}

void JSONWriter::
write(StringView key, Span<const Int64> view)
{
  m_p->writeKey(key);
  m_p->m_writer.StartArray();
  for (Int64 i = 0, n = view.size(); i < n; ++i)
    m_p->m_writer.Int64(view[i]);
  m_p->m_writer.EndArray();
}

void JSONWriter::
write(StringView key, Span<const Real> view)
{
  m_p->write(key, view);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView JSONWriter::
getBuffer() const
{
  const Byte* buf_chars = reinterpret_cast<const Byte*>(m_p->m_buffer.GetString());
  Span<const Byte> bytes(buf_chars, m_p->m_buffer.GetSize());
  return StringView(bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
