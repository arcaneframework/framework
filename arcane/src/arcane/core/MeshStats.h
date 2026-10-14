// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshStats.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Statistics on the mesh.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHSTATS_H
#define ARCANE_CORE_MESHSTATS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IMeshStats.h"

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
/*!
 * \brief Display informations abour a mesh.
 *
 * An instance of this class is created using IMeshStats::create() method.
 */
class ARCANE_CORE_EXPORT MeshStats
: public TraceAccessor
, public IMeshStats
{
  friend IMeshStats;

 public:

  ARCANE_DEPRECATED_REASON("Y2026: Use MeshStats(ITraceMng*, IMesh*) instead")
  MeshStats(ITraceMng* msg, IMesh* mesh, IParallelMng* pm);

 private:

  MeshStats(ITraceMng* msg, IMesh* mesh);

 public:

  void dumpStats() override;

  void dumpGraphStats() override;

 private:

  IMesh* m_mesh = nullptr;
  IParallelMng* m_parallel_mng = nullptr;

 private:

  template <typename T>
  void _dumpStats();

  template <typename T>
  void _computeElementsOnGroup(Int64ArrayView nb_type, Int64ArrayView nb_kind, Integer istat);

  template <typename T>
  void _statLabel(const String& name);

  void _computeElementsOnGroup(Int64ArrayView nb_type, Int64ArrayView nb_kind,
                               ItemGroup group, Integer istat);

  void _printInfo(const String& name, Int64 nb_local,
                  Int64 nb_local_min, Integer min_rank,
                  Int64 nb_local_max, Integer max_rank,
                  Int64 nb_global, Integer nb_rank);
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
