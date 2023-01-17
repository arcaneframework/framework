// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckpointInfo.h                                            (C) 2000-2018 */
/*                                                                           */
/* Informations sur une protection.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CHECKPOINTINFO_H
#define ARCANE_CHECKPOINTINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ICheckpointReader2;
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une protection.
 */
class ARCANE_CORE_EXPORT CheckpointInfo
{
 public:

  CheckpointInfo();

 public:

  Int32 nbSubDomain() const { return m_nb_sub_domain; }
  void setNbSubDomain(Int32 v) { m_nb_sub_domain = v; }

  Int32 nbReplication() const { return m_nb_replication; }
  void setNbReplication(Int32 v) { m_nb_replication = v; }

  void setServiceName(const String& v) { m_service_name = v; }
  const String& serviceName() const { return m_service_name; }

  void setDirectory(const String& v) { m_directory = v; }
  const String& directory() const { return m_directory; }

  Int32 checkpointIndex() const { return m_checkpoint_index ; }
  void setCheckpointIndex(Int32 v) { m_checkpoint_index = v; }

  Real checkpointTime() const { return m_checkpoint_time; }
  void setCheckpointTime(Real v) { m_checkpoint_time = v; }

  void setReaderMetaData(const String& v) { m_reader_meta_data = v; }
  const String& readerMetaData() const { return m_reader_meta_data; }

  Int32 subDomainRank() const { return m_sub_domain_rank; }
  void setSubDomainRank(Int32 v) { m_sub_domain_rank = v; }

  Int32 replicationRank() const { return m_replication_rank; }
  void setReplicationRank(Int32 v) { m_replication_rank = v; }

  //! Indique s'il s'agit d'une reprise
  bool isRestart() const { return m_is_restart; }
  void setIsRestart(bool v) { m_is_restart = v; }

 private:

  Int32 m_nb_replication;
  Int32 m_nb_sub_domain;
  String m_service_name;
  String m_directory;
  Real m_checkpoint_time;
  Int32 m_checkpoint_index;
  String m_reader_meta_data;
  Int32 m_sub_domain_rank;
  Int32 m_replication_rank;
  bool m_is_restart;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de relecture d'une protection.
 */
class CheckpointReadInfo
{
 public:
  explicit CheckpointReadInfo(const CheckpointInfo& ci)
  : m_checkpoint_info(ci), m_reader(nullptr), m_parallel_mng(nullptr){}
 public:
  const CheckpointInfo& checkpointInfo() const { return  m_checkpoint_info; }
  ICheckpointReader2* reader() const { return m_reader; }
  void setReader(ICheckpointReader2* v) { m_reader = v; }
  IParallelMng* parallelMng() const { return m_parallel_mng; }
  void setParallelMng(IParallelMng* v) { m_parallel_mng = v; }
 private:
  CheckpointInfo m_checkpoint_info;
  ICheckpointReader2* m_reader;
  IParallelMng* m_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

