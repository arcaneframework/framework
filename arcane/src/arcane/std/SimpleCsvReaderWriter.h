// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TODO */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_SIMPLECSVREADERWRITER_H
#define ARCANE_STD_SIMPLECSVREADERWRITER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableReaderWriter.h"

#include "arcane/Directory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleCsvReaderWriter
: public ISimpleTableReaderWriter
{
 public:
  SimpleCsvReaderWriter(SimpleTableInternal* simple_table_internal)
  : m_simple_table_internal(simple_table_internal)
  , m_separator(';')
  , m_precision_print(6)
  , m_is_fixed_print(true)
  {
  }

  SimpleCsvReaderWriter()
  : m_simple_table_internal(nullptr)
  , m_separator(';')
  , m_precision_print(6)
  , m_is_fixed_print(true)
  {
  }

  ~SimpleCsvReaderWriter() = default;

 public:
  bool writeTable(const Directory& dst, const String& file_name) override;
  bool readTable(const Directory& src, const String& file_name) override;
  void clearInternal() override;
  void print() override;

  Integer precision() override;
  void setPrecision(Integer precision) override;

  bool isFixed() override;
  void setFixed(bool fixed) override;

  String fileType() override { return m_output_file_type; };

  SimpleTableInternal* internal() override;
  void setInternal(SimpleTableInternal* simple_table_internal) override;

 protected:
  bool _openFile(std::ifstream& stream, Directory directory, const String& file);
  void _closeFile(std::ifstream& stream);
  void _print(std::ostream& stream);

 protected:
  SimpleTableInternal* m_simple_table_internal;

  char m_separator;

  Integer m_precision_print;
  bool m_is_fixed_print;

  const String m_output_file_type = "csv";
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
