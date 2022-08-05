// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TODO                                   (C) 2000-2022 */
/*                                                                           */
/*    */
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
#include "arcane/utils/Array.h"
#include "arcane/utils/Array2.h"

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
  SimpleTableWriterHelper(ISimpleTableReaderWriter* strw)
  : m_sti(strw->internal())
  , m_strw(strw)
  , m_name_output_dir("")
  , m_name_tab_without_computation("")
  , m_root()
  , m_name_tab_computed(false)
  , m_name_tab_only_once(false)
  {
  }

  SimpleTableWriterHelper()
  : m_sti(nullptr)
  , m_strw(nullptr)
  , m_name_output_dir("")
  , m_name_tab_without_computation("")
  , m_root()
  , m_name_tab_computed(false)
  , m_name_tab_only_once(false)
  {
  }

  virtual ~SimpleTableWriterHelper() = default;

 public:
  bool init() override;
  bool init(const String& name_table) override;
  bool init(const String& name_table, const String& name_dir) override;

  void print(Integer only_proc) override;
  bool writeFile(Integer only_proc) override;
  bool writeFile(Directory root_dir, Integer only_proc) override;

  Integer precision() override;
  void setPrecision(Integer precision) override;

  bool fixed() override;
  void setFixed(bool fixed) override;

  String outputDir() override;
  String outputDirWithoutComputation() override;
  void setOutputDir(const String& dir) override;

  String tabName() override;
  String tabNameWithoutComputation() override;
  void setTabName(const String& name) override;

  String fileName() override;
  
  Directory outputPath() override;
  Directory rootPath() override;
  
  String typeFile() override;

  bool isOneFileByProcsPermited() override;

  SimpleTableInternal* internal() override;
  
  ISimpleTableReaderWriter* readerWriter() override;
  void setReaderWriter(ISimpleTableReaderWriter* strw) override;

 protected:
  void _computeName();

 protected:
  SimpleTableInternal* m_sti;
  ISimpleTableReaderWriter* m_strw;

  String m_name_output_dir;
  String m_name_tab_without_computation;

  Directory m_root;

  bool m_name_tab_computed;
  bool m_name_tab_only_once;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
