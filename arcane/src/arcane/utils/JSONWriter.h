// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONWriter.h                                                (C) 2000-2023 */
/*                                                                           */
/* Ecrivain au format JSON.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_JSONWRITER_H
#define ARCANE_UTILS_JSONWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrivain au format JSON.
 *
 * \warning API Interne. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT JSONWriter
{
  class Impl;

 public:

  class Object
  {
   public:

    Object(JSONWriter& writer)
    : m_writer(writer)
    {
      m_writer.beginObject();
    }
    Object(JSONWriter& writer, StringView str)
    : m_writer(writer)
    {
      m_writer.writeKey(str);
      m_writer.beginObject();
    }
    ~Object() ARCANE_NOEXCEPT_FALSE
    {
      m_writer.endObject();
    }

   private:

    JSONWriter& m_writer;
  };
  class Array
  {
   public:

    Array(JSONWriter& writer, StringView name)
    : m_writer(writer)
    {
      m_writer.writeKey(name);
      m_writer.beginArray();
    }
    ~Array() ARCANE_NOEXCEPT_FALSE
    {
      m_writer.endArray();
    }

   private:

    JSONWriter& m_writer;
  };

 public:

  enum class FormatFlags
  {
    None = 0,
    // Indique qu'on sérialise les réels au format hexadécimal
    HexFloat = 1,

    Default = HexFloat
  };

 public:

  JSONWriter(FormatFlags format_flags = FormatFlags::Default);
  ~JSONWriter();

 public:

  void beginObject();
  void endObject();
  void beginArray();
  void endArray();
  void writeKey(StringView key);
  void writeValue(StringView str);
  void write(StringView key, const char* v);
  void write(StringView key, std::string_view v);
  void write(StringView key, bool v);
  void write(StringView key, Int64 v);
  void write(StringView key, UInt64 v);
  void write(StringView key, Real v);
  void write(StringView key, StringView str);
  void writeIfNotNull(StringView key, const String& str);
  void write(StringView key, Span<const Int32> view);
  void write(StringView key, Span<const Int64> view);
  void write(StringView key, Span<const Real> view);

 public:

  StringView getBuffer() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
