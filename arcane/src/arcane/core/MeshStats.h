﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshStats.h                                                 (C) 2000-2020 */
/*                                                                           */
/* Statistiques sur le maillage.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHSTATS_H
#define ARCANE_MESHSTATS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/IMeshStats.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IParallelMng;
class StringDictionary;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT MeshStats
: public TraceAccessor
, public IMeshStats
{
 public:

  MeshStats(ITraceMng* msg,IMesh* mesh,IParallelMng* pm);
  ~MeshStats();

 public:

  void dumpStats();
  
  void dumpGraphStats();
  
 private:

  IMesh* m_mesh;
  IParallelMng* m_parallel_mng;
  StringDictionary* m_dictionary;

 private:
  
  template<typename T>
  void _dumpStats();
  
  template<typename T>
  void _computeElementsOnGroup(Int64ArrayView nb_type,Int64ArrayView nb_kind, Integer istat);
  
  template<typename T>
  void _statLabel(String name);
  
  void _computeElementsOnGroup(Int64ArrayView nb_type,Int64ArrayView nb_kind,
                               ItemGroup group,Integer istat);
  
  void _printInfo(const String& name,Int64 nb_local,
                  Int64 nb_local_min,Integer min_rank,
                  Int64 nb_local_max,Integer max_rank,
                  Int64 nb_global,Integer nb_rank);
  void _computeNeighboorsComm();
  void _dumpLegacyConnectivityMemoryUsage();
  void _dumpIncrementalConnectivityMemoryUsage();
  void _dumpCommunicatingRanks();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

