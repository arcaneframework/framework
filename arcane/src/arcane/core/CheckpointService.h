// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckpointService.h                                         (C) 2000-2025 */
/*                                                                           */
/* Service de protection/reprise.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CHECKPOINTSERVICE_H
#define ARCANE_CORE_CHECKPOINTSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"

#include "arcane/core/BasicService.h"
#include "arcane/core/ICheckpointReader.h"
#include "arcane/core/ICheckpointWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
class VersionInfo;
class ServiceBuildInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Service de protection/reprise.
 */
class ARCANE_CORE_EXPORT CheckpointService
: public BasicService
, public ICheckpointWriter
, public ICheckpointReader
{
 public:

  class Impl;

 public:

  explicit CheckpointService(const ServiceBuildInfo& sbi);

 public:

  void build() override {}

 public:

  void setFileName(const String& file_name) override { m_file_name = file_name; }
  String fileName() const override { return m_file_name; }
  void setCheckpointTimes(RealConstArrayView times) override;
  void setCurrentTimeAndIndex(Real current_time, Integer current_index) override;

  RealConstArrayView checkpointTimes() const override { return m_checkpoint_times; }
  //! Méta données pour le lecteur associé à cet écrivain
  String readerMetaData() const override { return m_reader_meta_data; }
  void setReaderMetaData(const String& s) override { m_reader_meta_data = s; }
  void setBaseDirectoryName(const String& dirname) override { m_base_directory_name = dirname; }
  String baseDirectoryName() const override { return m_base_directory_name; }

  Integer currentIndex() const { return m_current_index; }
  Real currentTime() const { return m_current_time; }

 private:

  String m_file_name;
  UniqueArray<Real> m_checkpoint_times;
  Real m_current_time = -1.0;
  Integer m_current_index = -1;
  String m_reader_meta_data;
  String m_base_directory_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

