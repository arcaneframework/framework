// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckpointService.h                                         (C) 2000-2018 */
/*                                                                           */
/* Service de protection/reprise.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CHECKPOINTSERVICE_H
#define ARCANE_CHECKPOINTSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"

#include "arcane/BasicService.h"
#include "arcane/ICheckpointReader.h"
#include "arcane/ICheckpointWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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

  CheckpointService(const ServiceBuildInfo& sbi);

 public:

  virtual void build() {}

 public:

  virtual void setFileName(const String& file_name){ m_file_name = file_name; }
  virtual String fileName() const { return m_file_name; }
  virtual void setCheckpointTimes(RealConstArrayView times);
  virtual void setCurrentTimeAndIndex(Real current_time,Integer current_index);

  RealConstArrayView checkpointTimes() const { return m_checkpoint_times; }
  Integer currentIndex() const { return m_current_index; }
  Real currentTime() const { return m_current_time; }
  //! Méta données pour le lecteur associé à cet écrivain
  virtual String readerMetaData() const { return m_reader_meta_data; }
  virtual void setReaderMetaData(const String& s) { m_reader_meta_data = s; }
  virtual void setBaseDirectoryName(const String& dirname)
    {
      m_base_directory_name = dirname;
    }
  virtual String baseDirectoryName() const
    {
      return m_base_directory_name;
    }

 public:

 private:

  String m_file_name;
  UniqueArray<Real> m_checkpoint_times;
  Real m_current_time;
  Integer m_current_index;
  String m_reader_meta_data;
  String m_base_directory_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

