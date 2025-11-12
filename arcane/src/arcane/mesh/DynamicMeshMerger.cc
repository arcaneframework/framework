// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshMerger.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Fusion de plusieurs maillages.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/DynamicMeshMerger.h"
#include "arcane/mesh/DynamicMesh.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/IParallelSuperMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IArcaneMain.h"
#include "arcane/IMainFactory.h"

#include "arcane/mesh/MeshExchanger.h"
#include "arcane/mesh/MeshExchangeMng.h"
#include "arcane/mesh/MeshExchanger.h"
#include "arcane/IMeshExchanger.h"
#include "arcane/IMeshExchangeMng.h"
#include "arcane/IItemFamilyExchanger.h"
#include "arcane/ItemPrinter.h"
#include "arcane/MeshVisitor.h"

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
  DynamicMeshMergerHelper(ITraceMng* tm,DynamicMesh* mesh,
                          IParallelMng* pm,Int32 local_rank)
  : TraceAccessor(tm), m_mesh(mesh), m_parallel_mng(pm), m_local_rank(local_rank) {}
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
  LauncherThreadInfo() : m_mesh(nullptr), m_local_rank(-1), m_has_error(false){}
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
  DynamicMeshMergerHelper helper(lti->m_trace_mng.get(),lti->m_mesh,pm,local_rank);
  helper.doMerge();
  tm->info() << "END EXCHANGE!\n";
  pm->barrier();
}

void
_Launcher(LauncherThreadInfo* lti)
{
  ITraceMng* tm = lti->m_trace_mng.get();
  try{
    _doLaunch(lti);
  }
  catch(const Exception& ex){
    tm->info() << "FATAL: " << tm->traceId()
               << " ex=" << ex
               << " stack=" << ex.stackTrace();
    lti->m_has_error = true;
    throw;
  }
  catch(const std::exception& ex){
    tm->info() << "FATAL: " << ex.what();
    lti->m_has_error = true;
    throw;
  }
  catch(...){
    tm->info() << "UNKNOWN FATAL";
    lti->m_has_error = true;
    throw;
  }
}
class MergeMeshExchanger
: public MeshExchanger
{
 public:
  // TODO: faire son propre timeStats()
  MergeMeshExchanger(DynamicMesh* mesh,Int32 local_rank)
  : MeshExchanger(mesh,mesh->subDomain()->timeStats()), m_local_rank(local_rank){}
 public:
  /*!
   * \brief Détermine les entités à envoyer.
   *
   * Tous les maillages dont \a m_local_rank est différent de
   * 0 envoient leur entités au maillage de rang 0.
   */
  bool computeExchangeInfos() override
  {
    mesh()->traceMng()->info() << "OVERRIDE COMPUTE EXCHANGE INFOS";
    // Si je ne suis pas le rang local 0, je donne toutes mes entités.
    UniqueArray<std::set<Int32>> items_to_send;
    Int32 nb_rank = mesh()->parallelMng()->commSize();
    items_to_send.resize(nb_rank);

    for( IItemFamily* family : mesh()->itemFamilies() ){
      IItemFamilyExchanger* family_exchanger = this->findExchanger(family);
      if (m_local_rank!=0){
        items_to_send[0].clear();
        ItemGroup all_items = family->allItems();
        // Change les propriétaires pour correspondre au nouveau découpage.
        ENUMERATE_ITEM(iitem,all_items){
          Item ii = *iitem;
          items_to_send[0].insert(iitem.itemLocalId());
          // Les entités transférées seront supprimées.
          // NOTE: il serait possible de rendre cela optionnel si on
          // ne souhaite pas modifier le maillage qui est fusionné.
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
  MergerExchangeMng(DynamicMesh* mesh,Int32 local_rank)
  : MeshExchangeMng(mesh), m_mesh(mesh), m_local_rank(local_rank){}
 protected:
  IMeshExchanger* _createExchanger() override
  {
    m_mesh->traceMng()->info() << "CREATE MERGE MESH_EXCHANGER";
    MeshExchanger* ex = new MergeMeshExchanger(m_mesh,m_local_rank);
    ex->build();
    return ex;
  }
 private:
  DynamicMesh* m_mesh;
  Int32 m_local_rank;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshMergerHelper::
doMerge()
{
  // TODO: cette routine est similaire à DynamicMesh::_exchangeItemsNew()
  // et il faudrait fusionner les deux. Cela sera plus facile lorsque les
  // anciennes connectivités auront disparues.

  // Surcharge temporairement le IParallelMng
  // TODO: Vérifier que c'est bien remis à la bonne valeur en sortie en cas
  // d'exception

  IParallelMng* old_parallel_mng = m_mesh->m_parallel_mng;
  m_mesh->m_parallel_mng = this->m_parallel_mng;

  // TODO: Vérifier que tout le monde a les mêmes familles et dans le même ordre.
  info() << "DOING MERGE pm_rank=" << m_mesh->parallelMng()->commRank()
         << " sd_part=" << m_mesh->meshPartInfo().partRank();

  // Cascade tous les maillages associés à ce maillage
  typedef Collection<DynamicMesh*> DynamicMeshCollection;
  DynamicMeshCollection all_cascade_meshes = List<DynamicMesh*>();
  all_cascade_meshes.add(m_mesh);
  for(Integer i=0;i<m_mesh->m_child_meshes.size();++i) 
    all_cascade_meshes.add(m_mesh->m_child_meshes[i]);

  MergerExchangeMng exchange_mng(m_mesh,m_local_rank);
  IMeshExchanger* iexchanger = exchange_mng.beginExchange();
  IMeshExchanger* mesh_exchanger = iexchanger;

  // S'il n'y a aucune entité à échanger, on arrête immédiatement l'échange.
  if (mesh_exchanger->computeExchangeInfos()){
    pwarning() << "No load balance is performed";
    exchange_mng.endExchange();
    m_mesh->m_parallel_mng = old_parallel_mng;
    return;
  }

  // Éffectue l'échange des infos
  mesh_exchanger->processExchange();

  // Supprime les entités qui ne doivent plus être dans notre sous-domaine.
  mesh_exchanger->removeNeededItems();

  // Réajuste les groupes en supprimant les entités qui ne sont plus dans le maillage ou en
  // invalidant les groupes calculés.
  // TODO: faire une méthode de la famille qui fait cela.
  {
    auto action = [](ItemGroup& group)
    {
      if (group.internal()->hasComputeFunctor() || group.isLocalToSubDomain())
        group.invalidate();
      else
        group.internal()->removeSuppressedItems();
    };
    for( DynamicMesh* mesh : all_cascade_meshes ){
      meshvisitor::visitGroups(mesh,action);
    }
  }

  // Créé les entités qu'on a recu des autres sous-domaines.
  mesh_exchanger->allocateReceivedItems();

  // On reprend maintenant un cycle standard de endUpdate
  // mais en entrelaçant les niveaux de sous-maillages
  for( DynamicMesh* mesh : all_cascade_meshes )
    mesh->_internalEndUpdateInit(true);

  mesh_exchanger->updateItemGroups();

  // Recalcule des synchroniseurs sur groupes.
  for( DynamicMesh* mesh : all_cascade_meshes )
    mesh->_computeGroupSynchronizeInfos();

  // Met à jour les valeurs des variables des entités receptionnées
  mesh_exchanger->updateVariables();

  // Finalise les modifications dont le triage et compactage
  for( DynamicMesh* mesh : all_cascade_meshes ){
    // Demande l'affichage des infos pour le maillage actuel
    bool print_info = (mesh==m_mesh);
    mesh->_internalEndUpdateFinal(print_info);
  }

  // Finalize les échanges
  // Pour l'instante cela n'est utile que pour les TiedInterface mais il
  // faudrait supprimer cela.
  mesh_exchanger->finalizeExchange();

  // TODO: garantir cet appel en cas d'exception.
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
  // Le premier maillage de \a all_meshes est celui qui contiendra
  // à la fin la fusion des maillages de \a meshes.
  all_meshes.add(m_mesh);
  for( auto mesh : meshes )
    all_meshes.add(mesh);

  // L'algorithme de fusion utilise le même mécanisme que pour les échanges
  // d'entités (MeshExchanger). Pour pouvoir fonctionner, il faut créer
  // un IParallelMng par maillage fusionné. On utilise donc un
  // IParallelMng en mémoire partagé, et on lance un thread par
  // maillage. Le maillage de rang 0 recevra toutes les entités.

  IArcaneMain* am = IArcaneMain::arcaneMain();
  Int32 nb_local_rank = all_meshes.size();
  // Recherche le service utilisé pour le parallélisme
  String message_passing_service = "SharedMemoryParallelMngContainerFactory";
  ServiceBuilder<IParallelMngContainerFactory> sf(am->application());
  auto pbf = sf.createReference(message_passing_service,SB_AllowNull);
  if (!pbf)
    ARCANE_FATAL("Can not find service '{0}' implementing IParallelMngContainerFactory",message_passing_service);
  Parallel::Communicator comm = m_mesh->parallelMng()->communicator();
  Parallel::Communicator machine_comm = m_mesh->parallelMng()->machineCommunicator();
  Ref<IParallelMngContainer> parallel_builder(pbf->_createParallelMngBuilder(nb_local_rank, comm, machine_comm));

  IApplication* app = am->application();
  UniqueArray<LauncherThreadInfo> launch_infos(nb_local_rank);
  for( Integer i=0; i<nb_local_rank; ++i ){
    LauncherThreadInfo& lti = launch_infos[i];

    Int32 local_rank = i;
    Int32 mesh_part_rank = all_meshes[i]->meshPartInfo().partRank();
    String file_suffix = String::format("mm_{0}",mesh_part_rank);
    lti.m_trace_mng = app->createAndInitializeTraceMng(m_mesh->traceMng(),file_suffix);
    ITraceMng* tm = lti.m_trace_mng.get();
    lti.m_parallel_mng = parallel_builder->_createParallelMng(local_rank,tm);
    tm->setTraceId(String::format("Exchanger part={0} pm_rank={1}",mesh_part_rank,local_rank));

    lti.m_mesh = all_meshes[i];
    lti.m_local_rank = local_rank;
  }

  UniqueArray<std::thread*> gths(nb_local_rank);
  for( Integer i=0; i<nb_local_rank; ++i ){
    gths[i] = new std::thread(_Launcher,&launch_infos[i]);
  }
  bool has_error = false;
  for( Integer i=0; i<nb_local_rank; ++i ){
    gths[i]->join();
    if (launch_infos[i].m_has_error)
      has_error = true;
    delete gths[i];
  }
  // TODO: propager l'exception via std::exception_ptr
  if (has_error)
    ARCANE_FATAL("Error during mesh merge");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
