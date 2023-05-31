// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IHashDatabase.h                                             (C) 2000-2023 */
/*                                                                           */
/* Interface d'une base de données de hash.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_IHASHDATABASE_H
#define ARCANE_STD_INTERNAL_IHASHDATABASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Ref.h"

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HashDatabaseReadArgs
{
 public:

  HashDatabaseReadArgs() = default;

  HashDatabaseReadArgs(const String& hash_value, Span<std::byte> v)
  : m_hash_value(hash_value)
  , m_values(v)
  {}

 public:

  void setKey(const String& v) { m_key = v; }
  const String& key() const { return m_key; }

  void setValues(const Span<std::byte>& v) { m_values = v; }
  Span<std::byte> values() const { return m_values; }

  const String& hashValueAsString() const { return m_hash_value; }
  void setHashValueAsString(const String& v) { m_hash_value = v; }

 private:

  String m_key;
  String m_hash_value;
  Span<std::byte> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HashDatabaseWriteArgs
{
 public:

  HashDatabaseWriteArgs() = default;

  HashDatabaseWriteArgs(Span<const std::byte> v)
  : m_values(v)
  {}

 public:

  void setKey(const String& v) { m_key = v; }
  const String& key() const { return m_key; }

  void setValues(const Span<const std::byte>& v) { m_values = v; }
  Span<const std::byte> values() const { return m_values; }

 private:

  String m_key;
  Span<const std::byte> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HashDatabaseWriteResult
{
 public:

  const String& hashValueAsString() const { return m_hash_value; }
  void setHashValueAsString(const String& v) { m_hash_value = v; }

 private:

  String m_hash_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IHashDatabase
{
 public:

  virtual ~IHashDatabase() = default;

 public:

  virtual void writeValues(const HashDatabaseWriteArgs& args, HashDatabaseWriteResult& result) = 0;
  virtual void readValues(const HashDatabaseReadArgs& args) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IHashDatabase>
createFileHashDatabase(const String& directory);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
