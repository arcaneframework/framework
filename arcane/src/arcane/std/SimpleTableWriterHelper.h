// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableWriterHelper.h                                   (C) 2000-2022 */
/*                                                                           */
/* Classe permettant d'écrire un SimpleTableInternal dans un fichier.        */
/* Simplifie l'utilisation de l'écrivain en gérant le multiprocessus et les  */
/* noms des fichiers/dossiers.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_SIMPLETABLEWRITERHELPER_H
#define ARCANE_STD_SIMPLETABLEWRITERHELPER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableInternalMng.h"
#include "arcane/ISimpleTableReaderWriter.h"
#include "arcane/ISimpleTableWriterHelper.h"

#include "arcane/Directory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SimpleTableWriterHelper
: public ISimpleTableWriterHelper
{
 public:
  SimpleTableWriterHelper(ISimpleTableReaderWriter* simple_table_reader_writer)
  : m_simple_table_internal(simple_table_reader_writer->internal())
  , m_simple_table_reader_writer(simple_table_reader_writer)
  , m_name_output_directory("")
  , m_name_table_without_computation("")
  , m_root()
  , m_name_table_computed(false)
  , m_name_table_once_process(false)
  {
  }

  SimpleTableWriterHelper()
  : m_simple_table_internal(nullptr)
  , m_simple_table_reader_writer(nullptr)
  , m_name_output_directory("")
  , m_name_table_without_computation("")
  , m_root()
  , m_name_table_computed(false)
  , m_name_table_once_process(false)
  {
  }

  virtual ~SimpleTableWriterHelper() = default;

 public:
  bool init(const Directory& root_directory, const String& table_name, const String& directory_name) override;

  void print(Integer rank) override;
  bool writeFile(Integer rank) override;
  bool writeFile(const Directory& root_directory, Integer rank) override;

  Integer precision() override;
  void setPrecision(Integer precision) override;

  bool isFixed() override;
  void setFixed(bool fixed) override;

  String outputDirectory() override;
  String outputDirectoryWithoutComputation() override;
  void setOutputDirectory(const String& directory) override;

  String tableName() override;
  String tableNameWithoutComputation() override;
  void setTableName(const String& name) override;

  String fileName() override;

  Directory outputPath() override;
  Directory rootPath() override;

  String fileType() override;

  bool isOneFileByRanksPermited() override;

  SimpleTableInternal* internal() override;

  ISimpleTableReaderWriter* readerWriter() override;
  void setReaderWriter(ISimpleTableReaderWriter* simple_table_reader_writer) override;

 protected:
  void _computeName();

 protected:
  SimpleTableInternal* m_simple_table_internal;
  ISimpleTableReaderWriter* m_simple_table_reader_writer;

  String m_name_output_directory;
  String m_name_table_without_computation;

  Directory m_root;

  bool m_name_table_computed;
  bool m_name_table_once_process;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
