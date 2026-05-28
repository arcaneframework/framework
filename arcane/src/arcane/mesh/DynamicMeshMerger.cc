// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshMerger.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Merging multiple meshes.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/DynamicMeshMerger.h"
#include "arcane/mesh/DynamicMesh.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/IParallelSuperMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IArcaneMain.h"
#include "arcane/core/IMainFactory.h"

#include "arcane/mesh/MeshExchanger.h"
#include "arcane/mesh/MeshExchangeMng.h"
#include "arcane/mesh/MeshExchanger.h"
#include "arcane/core/IMeshExchanger.h"
#include "arcane/core/IMeshExchangeMng.h"
#include "arcane/core/IItemFamilyExchanger.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/MeshVisitor.h"

#include "arccore/base/ReferenceCounter.h"

#include <thread>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMeshMergerHelper
: public TraceAccessor
{
 public:

  DynamicMeshMergerHelper(ITraceMng* tm, DynamicMesh* mesh,
                          IParallelMng* pm, Int32 local_rank)
  : TraceAccessor(tm)
  , m_mesh(mesh)
  , m_parallel_mng(pm)
  , m_local_rank(local_rank)
  {}

 public:

  void doMerge();

 private:

  DynamicMesh* m_mesh;
  IParallelMng* m_parallel_mng;
  Int32 m_local_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshMerger::
DynamicMeshMerger(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshMerger::
~DynamicMeshMerger()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  class LauncherThreadInfo
  {
   public:

    LauncherThreadInfo()
    : m_mesh(nullptr)
    , m_local_rank(-1)
    , m_has_error(false)
    {}

   public:

    DynamicMesh* m_mesh;
    Int32 m_local_rank;
    bool m_has_error;
    ReferenceCounter<ITraceMng> m_trace_mng;
    Ref<IParallelMng> m_parallel_mng;
  };

  void
  _doLaunch(LauncherThreadInfo* lti)
  {
    Int32 local_rank = lti->m_local_rank;
    IParallelMng* pm = lti->m_parallel_mng.get();
    ITraceMng* tm = lti->m_trace_mng.get();
    pm->barrier();
    tm->info() << "BEGIN LAUNCHER rank=" << local_rank << "!\n";
    pm->barrier();
    DynamicMeshMergerHelper helper(lti->m_trace_mng.get(), lti->m_mesh, pm, local_rank);
    helper.doMerge();
    tm->info() << "END EXCHANGE!\n";
    pm->barrier();
  }

  void
  _Launcher(LauncherThreadInfo* lti)
  {
    ITraceMng* tm = lti->m_trace_mng.get();
    try {
      _doLaunch(lti);
    }
    catch (const Exception& ex) {
      tm->info() << "FATAL: " << tm->traceId()
                 << " ex=" << ex
                 << " stack=" << ex.stackTrace();
      lti->m_has_error = true;
      throw;
    }
    catch (const std::exception& ex) {
      tm->info() << "FATAL: " << ex.what();
      lti->m_has_error = true;
      throw;
    }
    catch (...) {
      tm->info() << "UNKNOWN FATAL";
      lti->m_has_error = true;
      throw;
    }
  }
  class MergeMeshExchanger
  : public MeshExchanger
  {
   public:

    // TODO: implement own timeStats()
    MergeMeshExchanger(DynamicMesh* mesh, Int32 local_rank)
    : MeshExchanger(mesh, mesh->subDomain()->timeStats())
    , m_local_rank(local_rank)
    {}

   public:

    /*!
   * \brief Determines the entities to send.
   *
   * All meshes whose \a m_local_rank is different from
   * 0 send their entities to the rank 0 mesh.
   */
    bool computeExchangeInfos() override
    {
      mesh()->traceMng()->info() << "OVERRIDE COMPUTE EXCHANGE INFOS";
      // If I am not the local rank 0, I give all my entities.
      UniqueArray<std::set<Int32>> items_to_send;
      Int32 nb_rank = mesh()->parallelMng()->commSize();
      items_to_send.resize(nb_rank);

      for (IItemFamily* family : mesh()->itemFamilies()) {
        IItemFamilyExchanger* family_exchanger = this->findExchanger(family);
        if (m_local_rank != 0) {
          items_to_send[0].clear();
          ItemGroup all_items = family->allItems();
          // Change the owners to match the new decomposition.
          ENUMERATE_ITEM (iitem, all_items) {
            Item ii = *iitem;
            items_to_send[0].insert(iitem.itemLocalId());
            // The transferred entities will be deleted.
            // NOTE: it would be possible to make this optional if we
            // do not want to modify the mesh being merged.
            ii.mutableItemBase().addFlags(ItemFlags::II_NeedRemove);
          }
          family_exchanger->setExchangeItems(items_to_send);
        }
        family_exchanger->computeExchangeInfos();
      }

      _setNextPhase(ePhase::ProcessExchange);
      return false;
    }

   private:

    Int32 m_local_rank;
  };

  class MergerExchangeMng
  : public MeshExchangeMng
  {
   public:

    MergerExchangeMng(DynamicMesh* mesh, Int32 local_rank)
    : MeshExchangeMng(mesh)
    , m_mesh(mesh)
    , m_local_rank(local_rank)
    {}

   protected:

    IMeshExchanger* _createExchanger() override
    {
      m_mesh->traceMng()->info() << "CREATE MERGE MESH_EXCHANGER";
      MeshExchanger* ex = new MergeMeshExchanger(m_mesh, m_local_rank);
      ex->build();
      return ex;
    }

   private:

    DynamicMesh* m_mesh;
    Int32 m_local_rank;
  };
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshMergerHelper::
doMerge()
{
  // TODO: this routine is similar to DynamicMesh::_exchangeItemsNew()
  // and the two should be merged. This will be easier when the
  // old connectivities have disappeared.

  // Temporarily override the IParallelMng
  // TODO: Check that it is restored to the correct value in case
  // of an exception

  IParallelMng* old_parallel_mng = m_mesh->m_parallel_mng;
  m_mesh->m_parallel_mng = this->m_parallel_mng;

  // TODO: Check that everyone has the same families and in the same order.
  info() << "DOING MERGE pm_rank=" << m_mesh->parallelMng()->commRank()
         << " sd_part=" << m_mesh->meshPartInfo().partRank();

  // Cascade all meshes associated with this mesh
  typedef Collection<DynamicMesh*> DynamicMeshCollection;
  DynamicMeshCollection all_cascade_meshes = List<DynamicMesh*>();
  all_cascade_meshes.add(m_mesh);
  for (Integer i = 0; i < m_mesh->m_child_meshes.size(); ++i)
    all_cascade_meshes.add(m_mesh->m_child_meshes[i]);

  MergerExchangeMng exchange_mng(m_mesh, m_local_rank);
  IMeshExchanger* iexchanger = exchange_mng.beginExchange();
  IMeshExchanger* mesh_exchanger = iexchanger;

  // If there are no entities to exchange, stop the exchange immediately.
  if (mesh_exchanger->computeExchangeInfos()) {
    pwarning() << "No load balance is performed";
    exchange_mng.endExchange();
    m_mesh->m_parallel_mng = old_parallel_mng;
    return;
  }

  // Perform the info exchange
  mesh_exchanger->processExchange();

  // Delete entities that should no longer be in our sub-domain.
  mesh_exchanger->removeNeededItems();

  // Readjust the groups by deleting entities that are no longer in the mesh or by
  // invalidating calculated groups.
  // TODO: make a family method that does this.
  {
    auto action = [](ItemGroup& group) {
      if (group.internal()->hasComputeFunctor() || group.isLocalToSubDomain())
        group.invalidate();
      else
        group.internal()->removeSuppressedItems();
    };
    for (DynamicMesh* mesh : all_cascade_meshes) {
      meshvisitor::visitGroups(mesh, action);
    }
  }

  // Create the entities received from other sub-domains.
  mesh_exchanger->allocateReceivedItems();

  // Now we resume a standard endUpdate cycle
  // but interleaving the sub-mesh levels
  for (DynamicMesh* mesh : all_cascade_meshes)
    mesh->_internalEndUpdateInit(true);

  mesh_exchanger->updateItemGroups();

  // Recalculate synchronizers on groups.
  for (DynamicMesh* mesh : all_cascade_meshes)
    mesh->_computeGroupSynchronizeInfos();

  // Update the values of the variables of the received entities
  mesh_exchanger->updateVariables();

  // Finalize the modifications whose sorting and compaction
  for (DynamicMesh* mesh : all_cascade_meshes) {
    // Request info display for the current mesh
    bool print_info = (mesh == m_mesh);
    mesh->_internalEndUpdateFinal(print_info);
  }

  // Finalize the exchanges
  // For now, this is only useful for the TiedInterface but
  // this should be removed.
  mesh_exchanger->finalizeExchange();

  // TODO: guarantee this call in case of an exception.
  exchange_mng.endExchange();

  m_mesh->endUpdate();

  m_mesh->m_parallel_mng = old_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshMerger::
mergeMeshes(ConstArrayView<DynamicMesh*> meshes)
{
  UniqueArray<DynamicMesh*> all_meshes;
  // The first mesh in \a all_meshes is the one that will contain
  // the merge of the meshes in \a meshes at the end.
  all_meshes.add(m_mesh);
  for (auto mesh : meshes)
    all_meshes.add(mesh);

  // The merging algorithm uses the same mechanism as for entity exchanges
  // (MeshExchanger). To function, it is necessary to create
  // an IParallelMng per merged mesh. We therefore use a
  // shared memory IParallelMng, and launch a thread per
  // mesh. The rank 0 mesh will receive all entities.

  IArcaneMain* am = IArcaneMain::arcaneMain();
  Int32 nb_local_rank = all_meshes.size();
  // Search for the service used for parallelism
  String message_passing_service = "SharedMemoryParallelMngContainerFactory";
  ServiceBuilder<IParallelMngContainerFactory> sf(am->application());
  auto pbf = sf.createReference(message_passing_service, SB_AllowNull);
  if (!pbf)
    ARCANE_FATAL("Can not find service '{0}' implementing IParallelMngContainerFactory", message_passing_service);
  Parallel::Communicator comm = m_mesh->parallelMng()->communicator();
  Parallel::Communicator machine_comm = m_mesh->parallelMng()->machineCommunicator();
  Ref<IParallelMngContainer> parallel_builder(pbf->_createParallelMngBuilder(nb_local_rank, comm, machine_comm));

  IApplication* app = am->application();
  UniqueArray<LauncherThreadInfo> launch_infos(nb_local_rank);
  for (Integer i = 0; i < nb_local_rank; ++i) {
    LauncherThreadInfo& lti = launch_infos[i];

    Int32 local_rank = i;
    Int32 mesh_part_rank = all_meshes[i]->meshPartInfo().partRank();
    String file_suffix = String::format("mm_{0}", mesh_part_rank);
    lti.m_trace_mng = app->createAndInitializeTraceMng(m_mesh->traceMng(), file_suffix);
    ITraceMng* tm = lti.m_trace_mng.get();
    lti.m_parallel_mng = parallel_builder->_createParallelMng(local_rank, tm);
    tm->setTraceId(String::format("Exchanger part={0} pm_rank={1}", mesh_part_rank, local_rank));

    lti.m_mesh = all_meshes[i];
    lti.m_local_rank = local_rank;
  }

  UniqueArray<std::thread*> gths(nb_local_rank);
  for (Integer i = 0; i < nb_local_rank; ++i) {
    gths[i] = new std::thread(_Launcher, &launch_infos[i]);
  }
  bool has_error = false;
  for (Integer i = 0; i < nb_local_rank; ++i) {
    gths[i]->join();
    if (launch_infos[i].m_has_error)
      has_error = true;
    delete gths[i];
  }
  // TODO: propagate the exception via std::exception_ptr
  if (has_error)
    ARCANE_FATAL("Error during mesh merge");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
