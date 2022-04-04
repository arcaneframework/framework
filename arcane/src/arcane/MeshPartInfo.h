// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartInfo.h                                              (C) 2000-2018 */
/*                                                                           */
/* Informations sur la partie d'un maillage.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHPARTINFO_H
#define ARCANE_MESHPARTINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations un maillage partitionné.
 */
class ARCANE_CORE_EXPORT MeshPartInfo
{
 public:
  MeshPartInfo(Int32 part_rank,Int32 nb_part,
               Int32 replication_rank,Int32 nb_replication)
  : m_part_rank(part_rank), m_nb_part(nb_part),
    m_replication_rank(replication_rank), m_nb_replication(nb_replication) {}
  MeshPartInfo()
  : m_part_rank(-1), m_nb_part(0),
    m_replication_rank(-1), m_nb_replication(0) {}
 public:
  Int32 partRank() const { return m_part_rank; }
  void setPartRank(Int32 v) { m_part_rank = v; }
  Int32 nbPart() const { return m_nb_part; }
  void setNbPart(Int32 v) { m_nb_part = v; }
  Int32 replicationRank() const { return m_replication_rank; }
  void setReplicationRank(Int32 v) { m_replication_rank = v; }
  Int32 nbReplication() const { return m_nb_replication; }
  void setNbReplication(Int32 v) { m_nb_replication = v; }
 private:
  Int32 m_part_rank;
  Int32 m_nb_part;
  Int32 m_replication_rank;
  Int32 m_nb_replication;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT
MeshPartInfo makeMeshPartInfoFromParallelMng(IParallelMng* pm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

