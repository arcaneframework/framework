// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableWriterHelper.h                                   (C) 2000-2023 */
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
#include "arcane/utils/FatalErrorException.h"

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

  SimpleTableWriterHelper(const Ref<ISimpleTableReaderWriter>& simple_table_reader_writer)
  : m_simple_table_internal(simple_table_reader_writer->internal())
  , m_simple_table_reader_writer(simple_table_reader_writer)
  , m_name_output_directory("")
  , m_name_table_without_computation("")
  , m_root()
  , m_name_table_computed(false)
  , m_name_output_directory_computed(false)
  , m_name_table_one_file_by_ranks_permited(false)
  , m_name_output_directory_one_file_by_ranks_permited(false)
  {
    if (simple_table_reader_writer.isNull())
      ARCANE_FATAL("La réference passée en paramètre est Null.");
  }

  SimpleTableWriterHelper()
  : m_simple_table_internal()
  , m_simple_table_reader_writer()
  , m_name_output_directory("")
  , m_name_table_without_computation("")
  , m_root()
  , m_name_table_computed(false)
  , m_name_output_directory_computed(false)
  , m_name_table_one_file_by_ranks_permited(false)
  , m_name_output_directory_one_file_by_ranks_permited(false)
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

  bool isForcedToUseScientificNotation() override;
  void setForcedToUseScientificNotation(bool use_scientific) override;

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

  Ref<SimpleTableInternal> internal() override;

  Ref<ISimpleTableReaderWriter> readerWriter() override;
  void setReaderWriter(const Ref<ISimpleTableReaderWriter>& simple_table_reader_writer) override;

 protected:

  void _computeTableName();
  void _computeOutputDirectory();
  String _computeName(String name, bool& one_file_by_ranks_permited);

 protected:

  Ref<SimpleTableInternal> m_simple_table_internal;
  Ref<ISimpleTableReaderWriter> m_simple_table_reader_writer;

  String m_name_output_directory;
  String m_name_output_directory_without_computation;
  String m_name_table_without_computation;

  Directory m_root;

  bool m_name_table_computed;
  bool m_name_output_directory_computed;

  // Booleens permettant de savoir si le nom de fichier ou le nom de dossier
  // possède le symbole "@proc_id@" qui permet l'écriture de fichiers par
  // plusieurs processus.
  bool m_name_table_one_file_by_ranks_permited;
  bool m_name_output_directory_one_file_by_ranks_permited;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
