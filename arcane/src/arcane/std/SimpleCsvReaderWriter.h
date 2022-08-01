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

#include "arcane/utils/UtilsTypes.h"
#include <arcane/ItemTypes.h>
#include "arcane/ArcaneTypes.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"
#include <arcane/Directory.h>
#include <arcane/utils/Iostream.h>
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"
#include "arcane/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleCsvReaderWriter
{
 public:
  SimpleCsvReaderWriter(IMesh* mesh)
  : m_mesh(mesh)
  , m_name_rows(0)
  , m_name_columns(0)
  , m_name_tab("")
  , m_separator(';')
  , m_precision_print(6)
  , m_is_fixed_print(true)
  {

  }

  ~SimpleCsvReaderWriter() = default;

 public:
  bool writeCsv(Directory dst, String file);
  bool readCsv(Directory src, String file);
  bool clearCsv();
  void printCsv(Integer only_proc = 0);
  
 protected:
  bool createDirectory(Directory& dir);
  bool isFileExist(Directory dir, String file);

 private:
  bool _openFile(std::ifstream& stream, Directory dir, String file);
  void _closeFile(std::ifstream& stream);
  void _print(std::ostream& stream);

 protected:
  IMesh* m_mesh;

  UniqueArray2<Real> m_values_csv;

  UniqueArray<String> m_name_rows;
  UniqueArray<String> m_name_columns;

  String m_name_tab;

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
