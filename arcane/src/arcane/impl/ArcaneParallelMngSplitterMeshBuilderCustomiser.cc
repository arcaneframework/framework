// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartitionerTesterService.cc                             (C) 2000-2022 */
/*                                                                           */
/* Mesh partitioner tester.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMeshBuilderCustomiser.h"
#include "arcane/core/MeshBuildInfo.h"

#include "arcane/impl/ArcaneParallelMngSplitterMeshBuilderCustomiser_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneParallelMngSplitterMeshBuilderCustomiser
: public ArcaneArcaneParallelMngSplitterMeshBuilderCustomiserObject
{
 public:

  explicit ArcaneParallelMngSplitterMeshBuilderCustomiser(const ServiceBuildInfo& sbi)
  : ArcaneArcaneParallelMngSplitterMeshBuilderCustomiserObject(sbi)
  , m_default_parallel_mng(sbi.subDomain()->parallelMng())
  {
  }

  void applyCustomisation(MeshBuildInfo& build_info) override
  {
    Int32 divider = options()->subsetRatio();
    if (divider < 0)
      ARCANE_FATAL("subset-ratio '{0}' is strictly positive",divider);
    IParallelMng* pm = m_default_parallel_mng;
    Int32 my_rank = pm->commRank();
    Int32 nb_rank = pm->commSize();
    UniqueArray<int> kept_ranks;
    kept_ranks.reserve(nb_rank);
    for (Int32 i = 0; i < nb_rank; ++i) {
      if ((i % divider) == 0)
        kept_ranks.add(i);
    }
    info() << "Splitting IParallelMng my_rank=" << my_rank;
    info() << "RanksToKeep=" << kept_ranks;
    Ref<IParallelMng> new_pm = pm->createSubParallelMngRef(kept_ranks);
    if (new_pm.get())
      info() << "Rank=" << my_rank << " RankAfter=" << new_pm.get()->commRank();
    else
      info() << "Rank=" << my_rank << ": I am not in the new IParallelMng";
    build_info.addParallelMng(new_pm);
  }

 private:

  IParallelMng* m_default_parallel_mng = nullptr;
};

ARCANE_REGISTER_SERVICE_ARCANEPARALLELMNGSPLITTERMESHBUILDERCUSTOMISER(ArcaneParallelMngSplitterMeshBuilderCustomiser,
                                                                       ArcaneParallelMngSplitterMeshBuilderCustomiser);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
