// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshRefinement.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Manipulation d'un maillage AMR.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef AMRMAXCONSISTENCYITER
#define AMRMAXCONSISTENCYITER 10
#endif

// \brief classe de méthodes de raffinement des maillages déstructurés
//! AMR


#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/Item.h"

#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/SharedVariable.h"
#include "arcane/core/ItemRefinementPattern.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/ItemVector.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/ItemRefinement.h"
#include "arcane/mesh/MeshRefinement.h"
#include "arcane/mesh/ParallelAMRConsistency.h"
#include "arcane/mesh/FaceReorienter.h"
#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/EdgeFamily.h"

#include "arcane/core/materials/IMeshMaterialMng.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

namespace
{
void _setRefineFlags(Item v)
{
  Integer f = v.itemBase().flags();
  f &= ~ItemFlags::II_Coarsen;
  f |= ItemFlags::II_Refine;
  v.mutableItemBase().setFlags(f);
}
void _setCoarseFlags(Item v)
{
  Integer f = v.itemBase().flags();
  f &= ~ItemFlags::II_Refine;
  f |= ItemFlags::II_Coarsen;
  v.mutableItemBase().setFlags(f);
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ACTIVATE_PERF_COUNTER
const std::string MeshRefinement::PerfCounter::m_names[MeshRefinement::PerfCounter::NbCounters] =
  {
    "INIT",
    "CLEAR",
    "ENDUPDATE",
    "UPDATEMAP",
    "UPDATEMAP2",
    "CONSIST",
    "PCONSIST",
    "PCONSIST2",
    "PGCONSIST",
    "CONTRACT",
    "COARSEN",
    "REFINE",
    "INTERP",
    "PGHOST",
    "COMPACT"
} ;
#endif

// Mesh refinement methods
MeshRefinement::
MeshRefinement(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_face_family(&(mesh->trueFaceFamily()))
, m_node_finder(mesh)
, m_face_finder(mesh)
, m_coarsen_by_parents(false)
, m_max_level(-1)
, m_nb_cell_target(0)
, m_face_level_mismatch_limit(1)
, m_max_node_uid(NULL_ITEM_UNIQUE_ID)
, m_next_node_uid(NULL_ITEM_UNIQUE_ID)
, m_max_cell_uid(NULL_ITEM_UNIQUE_ID)
, m_next_cell_uid(NULL_ITEM_UNIQUE_ID)
, m_max_face_uid(NULL_ITEM_UNIQUE_ID)
, m_next_face_uid(NULL_ITEM_UNIQUE_ID)
, m_max_nb_hChildren(0)
, m_node_owner_memory(VariableBuildInfo(mesh,"NodeOwnerMemoryVar"))
{
  // \todo créer un builder
  m_item_refinement = new ItemRefinement(mesh);
  m_parallel_amr_consistency = new ParallelAMRConsistency(mesh);
  m_call_back_mng = new AMRCallBackMng();
  m_need_update = true ;

  ENUMERATE_NODE(inode,m_mesh->allNodes()) m_node_owner_memory[inode] = inode->owner();

#ifdef ACTIVATE_PERF_COUNTER
  m_perf_counter.init() ;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshRefinement::
~MeshRefinement()
{
  this->clear();
  delete m_item_refinement;
  delete m_parallel_amr_consistency;
  delete m_call_back_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
MeshRefinement::
clear()
{
  CHECKPERF( m_perf_counter.start(PerfCounter::CLEAR) )
  m_node_finder._clear();
  m_face_finder._clear();
  CHECKPERF( m_perf_counter.stop(PerfCounter::CLEAR) )
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
MeshRefinement::
init()
{
  // Recalcul le max uniqueId() des nodes/cells/faces.
  CHECKPERF( m_perf_counter.start(PerfCounter::INIT) )
  IParallelMng* pm = m_mesh->parallelMng();
  {
    Int64 max_node_uid = 0;
    ENUMERATE_NODE(inode,m_mesh->allNodes())
    {
      const Node& node = *inode;
      const Int64 uid = node.uniqueId();
      if (uid>max_node_uid)
      max_node_uid = uid;
    }

    if (pm->commSize() > 1)
      m_max_node_uid = pm->reduce(Parallel::ReduceMax, max_node_uid);
    else
      m_max_node_uid = max_node_uid;
    info() << "NODE_UID_INFO: MY_MAX_UID=" << max_node_uid << " GLOBAL=" << m_max_node_uid;
    m_next_node_uid = m_max_node_uid + 1 + m_mesh->parallelMng()->commRank();
  }
  ItemTypeMng* itm = m_mesh->itemTypeMng();

  {
    Int64 max_cell_uid = 0;
    Integer max_nb_hChildren = 0;
    ENUMERATE_CELL(icell,m_mesh->allCells())
    {
      const Cell& cell = *icell;
      const Int64 uid = cell.uniqueId();
      const Int32 nb_hChildren = itm->nbHChildrenByItemType(cell.type());
      if (uid>max_cell_uid)
      max_cell_uid = uid;
      if (nb_hChildren>max_nb_hChildren)
      max_nb_hChildren = nb_hChildren;

    }
    if (pm->commSize() > 1)
    {
      m_max_cell_uid = pm->reduce(Parallel::ReduceMax, max_cell_uid);
      m_max_nb_hChildren = pm->reduce(Parallel::ReduceMax, max_nb_hChildren);
    }
    else
    {
      m_max_cell_uid = max_cell_uid;
      m_max_nb_hChildren = max_nb_hChildren;
    }
    info() << "CELL_UID_INFO: MY_MAX_UID=" << max_cell_uid << " GLOBAL=" << m_max_cell_uid;
    m_next_cell_uid = m_max_cell_uid + 1 + pm->commRank() * m_max_nb_hChildren;
  }

  {
    Int64 max_face_uid = 0;
    ENUMERATE_FACE(iface,m_mesh->allFaces())
    {
      const Face& face = *iface;
      const Int64 uid = face.uniqueId();
      if (uid>max_face_uid)
      max_face_uid = uid;
    }

    if (pm->commSize() > 1)
      m_max_face_uid = pm->reduce(Parallel::ReduceMax, max_face_uid);
    else
      m_max_face_uid = max_face_uid;
    info() << "FACE_UID_INFO: MY_MAX_UID=" << max_face_uid << " GLOBAL=" << m_max_face_uid;
    m_next_face_uid = m_max_face_uid + 1 + pm->commRank();
  }

  m_parallel_amr_consistency->init() ;

  CHECKPERF( m_perf_counter.stop(PerfCounter::INIT) )
}


void MeshRefinement::
_updateMaxUid(ArrayView<ItemInternal*> cells)
{
  // Recalcul le max uniqueId() des nodes/cells/faces.
  CHECKPERF( m_perf_counter.start(PerfCounter::INIT) )
  IParallelMng* pm = m_mesh->parallelMng();
  ItemTypeMng* itm = m_mesh->itemTypeMng();
  {

    Int64 max_node_uid = m_max_node_uid;
    Int64 max_cell_uid = m_max_cell_uid;
    Integer max_nb_hChildren = m_max_nb_hChildren;
    Int64 max_face_uid = m_max_face_uid;

    typedef std::set<Int64> set_type ;
    typedef std::pair<set_type::iterator,bool> insert_return_type ;
    set_type node_list ;
    set_type face_list ;
    for(Integer icell=0;icell<cells.size();++icell){
      Cell cell = cells[icell];
      for (UInt32 i = 0, nc = cell.nbHChildren(); i < nc; i++){
        Cell child = cell.hChild(i);

        //UPDATE MAX CELL UID
        const Int64 cell_uid = child.uniqueId();
        const Int32 nb_hChildren = itm->nbHChildrenByItemType(child.type());
        if (cell_uid>max_cell_uid)
        max_cell_uid = cell_uid;
        if (nb_hChildren>max_nb_hChildren)
        max_nb_hChildren = nb_hChildren;

        //UPDATE MAX NODE UID
        for( Node inode : child.nodes() ){
          const Int64 uid = inode.uniqueId();
          insert_return_type value = node_list.insert(uid) ;
          if(value.second){
            if (uid>max_node_uid)
              max_node_uid = uid;
          }
        }


        //UPDATE MAX FACE UID
        for( Face iface : child.faces() ){
          const Int64 uid = iface.uniqueId();
          insert_return_type value = face_list.insert(uid) ;
          if(value.second){
            if (uid>max_face_uid)
              max_face_uid = uid;
          }
        }
      }
    }


    if (pm->commSize() > 1)
    {
      m_max_node_uid = pm->reduce(Parallel::ReduceMax, max_node_uid);
      m_max_cell_uid = pm->reduce(Parallel::ReduceMax, max_cell_uid);
      m_max_nb_hChildren = pm->reduce(Parallel::ReduceMax, max_nb_hChildren);
      m_max_face_uid = pm->reduce(Parallel::ReduceMax, max_face_uid);
    }
    else
    {
      m_max_node_uid = max_node_uid;
      m_max_cell_uid = max_cell_uid;
      m_max_nb_hChildren = max_nb_hChildren;
      m_max_face_uid = max_face_uid;
    }
    m_next_node_uid = m_max_node_uid + 1 + m_mesh->parallelMng()->commRank();
    m_next_cell_uid = m_max_cell_uid + 1 + pm->commRank() * m_max_nb_hChildren;
    m_next_face_uid = m_max_face_uid + 1 + pm->commRank();
  }

  CHECKPERF( m_perf_counter.stop(PerfCounter::INIT) )
}


void
MeshRefinement::initMeshContainingBox()
{
  m_mesh_containing_box.init(m_mesh) ;
  m_node_finder.setBox(&m_mesh_containing_box) ;
  m_face_finder.setBox(&m_mesh_containing_box) ;
}


void
MeshRefinement::update()
{
  init() ;
  initMeshContainingBox() ;
  m_item_refinement->initHMin() ;
  m_node_finder.init() ;
  //m_node_finder.check() ;
  m_face_finder.initFaceCenter() ;
  m_face_finder.init() ;
  //m_face_finder.check() ;
  m_parallel_amr_consistency->update() ;
  m_need_update = false ;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
flagCellToRefine(Int32ConstArrayView lids)
{
  CellLocalIdToCellConverter cells(m_mesh->cellFamily());
  for (Integer i = 0, is = lids.size(); i < is; i++) {
    Item item = cells[lids[i]];
    //ARCANE_ASSERT((item->type() ==IT_Hexaedron8),(""));
    item.mutableItemBase().addFlags(ItemFlags::II_Refine);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
flagCellToCoarsen(Int32ConstArrayView lids)
{
  ItemInfoListView cells(m_mesh->cellFamily());
  for (Integer i = 0, is = lids.size(); i < is; i++) {
    Item item = cells[lids[i]];
    //ARCANE_ASSERT((item->type() ==IT_Hexaedron8),(""));
    item.mutableItemBase().addFlags(ItemFlags::II_Coarsen);
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
MeshRefinement::
refineAndCoarsenItems(const bool maintain_level_one)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::INIT) )

  bool _maintain_level_one = maintain_level_one;

  // la règle de niveau-un est la seule condition implementée
  if (!maintain_level_one)
  {
    warning() << "Warning, level one rule is the only condition accepted for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  // Nous ne pouvons pas encore transformer un maillage de non-niveau-un en un maillage de niveau-un
  if (_maintain_level_one){
    ARCANE_ASSERT((_checkLevelOne(true)), ("checkLevelOne failed"));
  }

  // Nettoyage des flags de raffinement d'une étape précédente
  this->_cleanRefinementFlags();
   CHECKPERF( m_perf_counter.stop(PerfCounter::INIT) )

  // La consistence parallèle doit venir en premier, ou le déraffinement
  // le long des interfaces entre processeurs pourrait de temps en temps être
  // faussement empéché
  if (m_mesh->parallelMng()->isParallel())
    this->_makeFlagParallelConsistent();

  CHECKPERF( m_perf_counter.start(PerfCounter::CONSIST) )
  // Repete jusqu'au matching du changement de flags sur chaque processeur
  Integer iter = 0 ;
  do
  {
    // Repete jusqu'au matching des flags coarsen/refine localement
    bool satisfied = false;
    do
    {
      const bool coarsening_satisfied = this->_makeCoarseningCompatible(maintain_level_one);
      const bool refinement_satisfied = this->_makeRefinementCompatible(maintain_level_one);
      satisfied = (coarsening_satisfied && refinement_satisfied);
#ifdef ARCANE_DEBUG
      bool max_satisfied = satisfied,min_satisfied = satisfied;
      max_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMax,max_satisfied);
      min_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMin,min_satisfied);
      ARCANE_ASSERT ( (satisfied == max_satisfied), ("parallel max_satisfied failed"));
      ARCANE_ASSERT ( (satisfied == min_satisfied), ("parallel min_satisfied failed"));
#endif
    } while (!satisfied);
    ++iter ;
  } while (m_mesh->parallelMng()->isParallel() && !this->_makeFlagParallelConsistent() && iter<10 );
  if(iter==AMRMAXCONSISTENCYITER) fatal()<<" MAX CONSISTENCY ITER REACHED";
  CHECKPERF( m_perf_counter.stop(PerfCounter::CONSIST) )

  // D'abord déraffine les items flaggés.
  CHECKPERF( m_perf_counter.start(PerfCounter::COARSEN) )
  const bool coarsening_changed_mesh = this->_coarsenItems();
  CHECKPERF( m_perf_counter.stop(PerfCounter::COARSEN) )

  // Maintenant, raffine les items flaggés.  Ceci prendra
  // plus de mémoire, et peut être plus de ce qui est libre.
  Int64UniqueArray cells_to_refine;
  const bool refining_changed_mesh = this->_refineItems(cells_to_refine);

  // Finalement, préparation du nouveau maillage pour utilisation
  if (refining_changed_mesh || coarsening_changed_mesh) {
    bool do_compact  = m_mesh->properties()->getBool("compact");
    m_mesh->properties()->setBool("compact",true) ; // Forcing compaction prevents from bugs when using AMR

    // Raffinement
    CHECKPERF( m_perf_counter.start(PerfCounter::ENDUPDATE) )
    m_mesh->modifier()->endUpdate();
    m_mesh->properties()->setBool("compact",do_compact) ;
    CHECKPERF( m_perf_counter.stop(PerfCounter::ENDUPDATE) )

    // deraffinement
    //bool remove_ghost_children = false;
    if (coarsening_changed_mesh)
    {
      //remove_ghost_children=true;

      CHECKPERF( m_perf_counter.start(PerfCounter::CONTRACT) )
      this->_contract();
      CHECKPERF( m_perf_counter.stop(PerfCounter::CONTRACT) )

      CHECKPERF( m_perf_counter.start(PerfCounter::ENDUPDATE) )
      m_mesh->properties()->setBool("compact",true) ; // Forcing compaction prevents from bugs when using AMR (leads to problems whith dof)
      m_mesh->modifier()->endUpdate();
      m_mesh->properties()->setBool("compact",do_compact) ;
      CHECKPERF( m_perf_counter.stop(PerfCounter::ENDUPDATE) )

    }

    // callback pour transporter les variables sur le nouveau maillage
    CHECKPERF( m_perf_counter.start(PerfCounter::INTERP) )
    this->_interpolateData(cells_to_refine);
    CHECKPERF( m_perf_counter.stop(PerfCounter::INTERP) )
    //

    if (!coarsening_changed_mesh && m_mesh->parallelMng()->isParallel())
      this->_makeFlagParallelConsistent2();

    if (m_mesh->parallelMng()->isParallel())
    {
      CHECKPERF( m_perf_counter.start(PerfCounter::PGHOST) )
      m_mesh->modifier()->setDynamic(true);
      UniqueArray<Int64> ghost_cell_to_refine ;
      UniqueArray<Int64> ghost_cell_to_coarsen ;
      m_mesh->modifier()->updateGhostLayerFromParent(ghost_cell_to_refine,
                                                     ghost_cell_to_coarsen,
                                                     false);
      _update(ghost_cell_to_refine) ;
      CHECKPERF( m_perf_counter.stop(PerfCounter::PGHOST) )
      _checkOwner("refineAndCoarsenItems after ghost update");
    }

    /*
    if(do_compact)
    {
      CHECKPERF( m_perf_counter.start(PerfCounter::COMPACT) )
      m_mesh->properties()->setBool("compact",true) ;
      DynamicMesh* mesh = dynamic_cast<DynamicMesh*> (m_mesh);
      if(mesh)
        mesh->compact() ;
      CHECKPERF( m_perf_counter.stop(PerfCounter::COMPACT) )
    }*/

#ifdef ACTIVATE_PERF_COUNTER
    info()<<"MESH REFINEMENT PERF INFO" ;
    m_perf_counter.printInfo(info().file()) ;
    info()<<"NODE FINDER PERF INFO" ;
    m_node_finder.getPerfCounter().printInfo(info().file()) ;
    info()<<"FACE FINDER PERF INFO" ;
    m_face_finder.getPerfCounter().printInfo(info().file()) ;
    info()<<"PARALLEL AMR CONSISTENCY PERF INFO" ;
    m_parallel_amr_consistency->getPerfCounter().printInfo(info().file()) ;
#endif
    return true;
  }
  // Si il n'y avait aucun changement dans le maillage
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
MeshRefinement::
coarsenItems(const bool maintain_level_one)
{

  bool _maintain_level_one = maintain_level_one;

  // la rêgle de niveau-un est la seule condition implementée
  if (!maintain_level_one){
    warning() << "Warning, level one rule is the only condition accepted for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  // Nous ne pouvons pas encore transformer un maillage de non-niveau-un en un maillage de niveau-un
  if (_maintain_level_one){
    ARCANE_ASSERT((_checkLevelOne(true)), ("check_level_one failed"));
  }

  // Nettoyage des flags de raffinement de l'étape précédente
  this->_cleanRefinementFlags();

  // La consistence parallêle doit venir en premier, ou le déraffinement
  // le long des interfaces entre processeurs pourrait de temps en temps être
  // faussement empéché
  if (m_mesh->parallelMng()->isParallel())
    this->_makeFlagParallelConsistent();

  // Repete jusqu'au matching du changement de flags sur chaque processeur
  do
  {
    // Repete jusqu'au matching des flags localement.
    bool satisfied = false;
    do
    {
      const bool coarsening_satisfied = this->_makeCoarseningCompatible(maintain_level_one);
      satisfied = coarsening_satisfied;
#ifdef ARCANE_DEBUG
      bool max_satisfied = satisfied, min_satisfied = satisfied;
      max_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMax,max_satisfied);
      min_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMin,min_satisfied);
      ARCANE_ASSERT ( (satisfied == max_satisfied), ("parallel max_satisfied failed"));
      ARCANE_ASSERT ( (satisfied == min_satisfied), ("parallel min_satisfied failed"));
#endif
    } while (!satisfied);
  } while (m_mesh->parallelMng()->isParallel() && !this->_makeFlagParallelConsistent());

  // Déraffine les items flaggés.
  const bool mesh_changed = this->_coarsenItems();

  //if (_maintain_level_one)
  //ARCANE_ASSERT( (checkLevelOne(true)),("checkLevelOne failed"));
  //ARCANE_ASSERT( (this->makeCoarseningCompatible(maintain_level_one)), ("make_coarsening_comptaible failed"));

  // Finalement, préparation du nouveau maillage pour utilisation
  if (mesh_changed)
  {
    this->_contract();
    _checkOwner("coarsenItems");
    //
    //m_mesh->modifier()->setDynamic(true);
    //m_mesh->modifier()->updateGhostLayerFromParent(false);
  }

  return mesh_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
coarsenItemsV2(bool update_parent_flag)
{
  // Nettoyage des flags de raffinement de l'étape précédente
  this->_cleanRefinementFlags();

  UniqueArray<Int32> to_coarse;
  //UniqueArray<Int64> d_to_coarse_uid;

  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
    Cell cell = *icell;
    if (cell.mutableItemBase().flags() & ItemFlags::II_Coarsen) {
      // On ne peut pas dé-raffiner des mailles de niveau 0.
      if (cell.level() == 0) {
        ARCANE_FATAL("Cannot coarse level-0 cell");
      }
      Cell parent = cell.hParent();

      // TODO AH : Pour faire le dé-raffinement de plusieurs niveau en une fois,
      // le flag II_Inactive doit être retiré (pour la méthode FaceFamily::removeCellFromFace()).
      if (update_parent_flag) {
        parent.mutableItemBase().addFlags(ItemFlags::II_JustCoarsened);
        parent.mutableItemBase().removeFlags(ItemFlags::II_Inactive);
        parent.mutableItemBase().removeFlags(ItemFlags::II_CoarsenInactive);
      }

      // Pour une maille de niveau n-1, si une de ses mailles filles doit être dé-raffinée,
      // alors toutes ses mailles filles doivent être dé-raffinées.
      for (Integer i = 0; i < parent.nbHChildren(); ++i) {
        Cell child = parent.hChild(i);
        if (!(child.mutableItemBase().flags() & ItemFlags::II_Coarsen)) {
          ARCANE_FATAL("Parent cannot have children with coarse flag and children without coarse flag -- Parent uid: {0} -- Child uid: {1}", parent.uniqueId(), child.uniqueId());
        }
      }

      // Pour l'instant, il est impossible de dé-raffiner de plusieurs niveaux en une fois.
      // TODO AH : La méthode FaceReorienter::checkAndChangeOrientationAMR() va vérifier une
      // face sensée être supprimée, voir pourquoi.
      if (parent.mutableItemBase().flags() & ItemFlags::II_Coarsen) {
        ARCANE_FATAL("Cannot coarse parent and child in same time");
      }
      if (cell.nbHChildren() != 0) {
        ARCANE_FATAL("For now, cannot coarse cell with children");
        // for (Integer i = 0; i < cell.nbHChildren(); ++i) {
        //   Cell child = cell.hChild(i);
        //   if (!(child.mutableItemBase().flags() & ItemFlags::II_Coarsen)) {
        //     ARCANE_FATAL("Cannot coarse cell with non-coarsen children -- Parent uid: {0} -- Child uid: {1}", cell.uniqueId(), child.uniqueId());
        //   }
        // }
      }
      to_coarse.add(cell.localId());
      //d_to_coarse_uid.add(cell.uniqueId());
    }
  }

  if (m_mesh->parallelMng()->isParallel()) {
    bool has_ghost_layer = m_mesh->ghostLayerMng()->nbGhostLayer() != 0;

    ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
      Cell cell = *icell;
      if (cell.mutableItemBase().flags() & ItemFlags::II_Coarsen) {
        for (Face face : cell.faces()) {
          Cell other_cell = face.oppositeCell(cell);
          // debug() << "Check face uid : " << face.uniqueId();
          // Si la face est au bord, elle sera supprimée.
          if (other_cell.null()) { // && !has_ghost_layer) {
            continue;
            //needed_cell.add(face.uniqueId());
          }
          if (other_cell.level() != cell.level()) {
            //warning() << "Bad connectivity";
            continue;
          }
          // Si les deux mailles vont être supprimées, la face sera supprimée.
          if (other_cell.mutableItemBase().flags() & ItemFlags::II_Coarsen) {
            continue;
          }
          // Si la maille à côté est raffinée, on aura plus d'un niveau de décalage.
          if (other_cell.nbHChildren() != 0) { // && !(other_cell.mutableItemBase().flags() & ItemFlags::II_Coarsen)) { // Impossible de dé-raffiner plusieurs niveaux.
            ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell.uniqueId(), other_cell.uniqueId());
          }
          // Si la maille d'à côté n'est pas à nous, elle prend la propriété de la maille d'à côté.
          if (other_cell.owner() != cell.owner()) {
            // debug() << "Face uid : " << face.uniqueId()
            //         << " -- old owner: " << face.owner()
            //         << " -- new owner: " << other_cell.owner();
            face.mutableItemBase().setOwner(other_cell.owner(), cell.owner());
            // debug() << "Set owner face uid: " << face.uniqueId() << " -- New owner: " << other_cell.owner();
          }
        }
        for (Node node : cell.nodes()) {
          // debug() << "Check node uid : " << node.uniqueId();

          // Noeud sera supprimé ?
          {
            bool will_deleted = true;
            for (Cell cell2 : node.cells()) {
              if (!(cell2.mutableItemBase().flags() & ItemFlags::II_Coarsen)) {
                will_deleted = false;
                break;
              }
            }
            if (will_deleted) {
              continue;
            }
          }

          // Noeud devra changer de proprio ?
          {
            Integer node_owner = node.owner();
            Integer new_owner = -1;
            bool need_new_owner = true;
            for (Cell cell2 : node.cells()) {
              if (!(cell2.mutableItemBase().flags() & ItemFlags::II_Coarsen)) {
                if (cell2.owner() == node_owner) {
                  need_new_owner = false;
                  break;
                }
                new_owner = cell2.owner();
              }
            }
            if (!need_new_owner) {
              continue;
            }
            // debug() << "Node uid : " << node.uniqueId()
            //         << " -- old owner: " << node.owner()
            //         << " -- new owner: " << new_owner;
            node.mutableItemBase().setOwner(new_owner, cell.owner());
            // debug() << "Set owner node uid: " << node.uniqueId() << " -- New owner: " << new_owner;
          }
        }
      }
    }

    if (!has_ghost_layer) {
      // TODO
      ARCANE_NOT_YET_IMPLEMENTED("Support des maillages sans mailles fantômes à faire");
    }
  }

  //debug() << "Removed cells: " << d_to_coarse_uid;

  m_mesh->modifier()->removeCells(to_coarse);
  m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  m_mesh->faceFamily()->notifyItemsOwnerChanged();
  m_mesh->modifier()->endUpdate();
  m_mesh->cellFamily()->computeSynchronizeInfos();
  m_mesh->nodeFamily()->computeSynchronizeInfos();
  m_mesh->faceFamily()->computeSynchronizeInfos();
  m_mesh->modifier()->setDynamic(true);

  UniqueArray<Int64> ghost_cell_to_refine;
  UniqueArray<Int64> ghost_cell_to_coarsen;

  if (!update_parent_flag) {
    // Si les matériaux sont actifs, il faut forcer un recalcul des matériaux car les groupes
    // de mailles ont été modifiés et donc la liste des constituants aussi
    Materials::IMeshMaterialMng* mm = Materials::IMeshMaterialMng::getReference(m_mesh, false);
    if (mm)
      mm->forceRecompute();
  }

  m_mesh->modifier()->updateGhostLayerFromParent(ghost_cell_to_refine, ghost_cell_to_coarsen, true);

  return m_mesh->parallelMng()->reduce(Parallel::ReduceMax, (!to_coarse.empty()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
MeshRefinement::
refineItems(const bool maintain_level_one)
{

  bool _maintain_level_one = maintain_level_one;

  // la règle de niveau-un est la seule condition implementé
  if (!maintain_level_one)
  {
    warning() << "Warning, level one rule is the only condition accepted for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  if (_maintain_level_one){
    ARCANE_ASSERT((_checkLevelOne(true)), ("check_level_one failed"));
  }
  // Nettoyage des flags de raffinement de l'étape précédente
  this->_cleanRefinementFlags();

  // La consistence parallêle doit venir en premier, ou le déraffinement
  // le long des interfaces entre processeurs pourrait de temps en temps être
  // faussement empêché
  if (m_mesh->parallelMng()->isParallel())
    this->_makeFlagParallelConsistent();

  // Repete jusqu'au matching du changement de flags sur chaque processeur
  do
  {
    // Repete jusqu'au matching des flags localement.
    bool satisfied = false;
    do
    {
      const bool refinement_satisfied = this->_makeRefinementCompatible(maintain_level_one);
      satisfied = refinement_satisfied;
#ifdef ARCANE_DEBUG
      bool max_satisfied = satisfied,min_satisfied = satisfied;
      max_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMax,max_satisfied);
      min_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMin,min_satisfied);
      ARCANE_ASSERT ( (satisfied == max_satisfied), ("parallel max_satisfied failed"));
      ARCANE_ASSERT ( (satisfied == min_satisfied), ("parallel min_satisfied failed"));
#endif
    } while (!satisfied);
  } while (m_mesh->parallelMng()->isParallel() && !this->_makeFlagParallelConsistent());

  // Maintenant, raffine les items flaggés.  Ceci prendra
  // plus de mémoire, et peut être plus de ce qui est libre.
  Int64UniqueArray cells_to_refine;
  const bool mesh_changed = this->_refineItems(cells_to_refine);

  // Finalement, préparation du nouveau maillage pour utilisation
  if (mesh_changed){
    // mise a jour
    bool do_compact  = m_mesh->properties()->getBool("compact");
    m_mesh->properties()->setBool("compact",true) ; // Forcing compaction prevents from bugs when using AMR
    m_mesh->modifier()->endUpdate();
    m_mesh->properties()->setBool("compact",do_compact) ;

    // callback pour transporter les variables sur le nouveau maillage
    this->_interpolateData(cells_to_refine);

    // mise a jour des ghosts
    m_mesh->modifier()->setDynamic(true);
    UniqueArray<Int64> ghost_cell_to_refine ;
    UniqueArray<Int64> ghost_cell_to_coarsen ;
    m_mesh->modifier()->updateGhostLayerFromParent(ghost_cell_to_refine,ghost_cell_to_coarsen,false);
    _update(ghost_cell_to_refine) ;
  }

  //if (_maintain_level_one)
  //ARCANE_ASSERT( (checkLevelOne(true)), ("check_level_one failed"));
  //ARCANE_ASSERT( (this->makeRefinementCompatible(maintain_level_one)), ("make_refinment_compatible failed"));

  return mesh_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
uniformlyRefine(Integer n)
{
  // Raffine n fois
  // FIXME - ceci ne doit pas marcher si n>1 et le maillage
  // est déjà attaché au système d'équations à résoudre
  for (Integer rstep = 0; rstep < n; rstep++){
    // Nettoyage des flags de raffinement
    this->_cleanRefinementFlags();

    // itérer seulement sur les mailles actives
    // Flag tous les items actifs pour raffinement
    ENUMERATE_CELL(icell,m_mesh->ownActiveCells()){
      Cell cell = *icell;
      _setRefineFlags(cell);
    }
    // Raffine tous les items que nous avons flaggés.
    Int64UniqueArray cells_to_refine;
    this->_refineItems(cells_to_refine);
    warning() << "ATTENTION: No Data Projection with this method!";
  }

  bool do_compact  = m_mesh->properties()->getBool("compact");
  m_mesh->properties()->setBool("compact",true) ;// Forcing compaction prevents from bugs when using AMR
  m_mesh->modifier()->endUpdate();
  m_mesh->properties()->setBool("compact",do_compact) ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
MeshRefinement::
uniformlyCoarsen(Integer n)
{
  // Déraffine n fois
  for (Integer rstep = 0; rstep < n; rstep++){
    // Nettoyage des flags de raffinement
    this->_cleanRefinementFlags();

    // itérer seulement sur les mailles actives
    // Flag tous les items actifs pour déraffinement
    ENUMERATE_CELL(icell,m_mesh->ownActiveCells()){
      Cell cell = *icell;
      _setCoarseFlags(cell);
      if (cell.nbHParent() != 0){
        cell.hParent().mutableItemBase().addFlags(ItemFlags::II_CoarsenInactive);
      }
    }
    // Déraffine tous les items que nous venons de flagger.
    this->_coarsenItems();
    warning() << "ATTENTION: No Data Restriction with this method!";
  }

  // Finalement, préparation du nouveau maillage pour utilisation
  bool do_compact  = m_mesh->properties()->getBool("compact");
  m_mesh->properties()->setBool("compact",true) ;
  m_mesh->modifier()->endUpdate();
  m_mesh->properties()->setBool("compact",do_compact) ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64
MeshRefinement::
findOrAddNodeUid(const Real3& p, const Real& tol)
{
  //debug() << "addNode()";

  // Return the node if it already exists
  Int64 uid = m_node_finder.find(p, tol);
  if (uid != NULL_ITEM_ID)
  {
    //          debug() << "addNode() done";
    return uid;
  }
  // Add the node to the map.
  Int64 new_uid = m_next_node_uid;
  m_node_finder.insert(p, new_uid,tol);
  m_next_node_uid += m_mesh->parallelMng()->commSize() + 1;
  // Return the uid of the new node
  //  debug() << "addNode() done";
  return new_uid;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
Int64
MeshRefinement::
findOrAddFaceUid(const Real3& p, const Real& tol, bool& is_added)
{
  //debug() << "findOrAddFaceUid()";

  // Return the face if it already exists
  Int64 uid = m_face_finder.find(p, tol);
  if (uid != NULL_ITEM_ID)
  {
    //          debug() << "findOrAddFaceUid() done";
    is_added = false;
    return uid;
  }
  // Add the face to the map.
  is_added = true;
  Int64 new_uid = m_next_face_uid;
  m_face_finder.insert(p, new_uid,tol);
  m_next_face_uid += m_mesh->parallelMng()->commSize() + 1;
  // Return the uid of the new face
  //  debug() << "findOrAddFaceUid() done";
  return new_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MeshRefinement::
getFirstChildNewUid()
{
  Int64 new_uid = m_next_cell_uid;
  Int64 comm_size = m_mesh->parallelMng()->commSize();
  m_next_cell_uid += comm_size * m_max_nb_hChildren;
  return new_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*                        PRIVATE METHODS                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
MeshRefinement::
_updateLocalityMap()
{
  //jmg this->init(); // \todo pas necessaire de l'appeler a chaque m-a-j
  //m_node_finder.init();
  m_node_finder.check() ;
  //m_face_finder.init();
  m_face_finder.check() ;
  debug() << "[MeshRefinement::updateLocalityMap] done";
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
MeshRefinement::
_updateLocalityMap2()
{
  //this->init(); // \todo pas necessaire de l'appeler a chaque m-a-j
  //this->m_node_finder.init2();
  //m_node_finder.check2() ;
  //m_face_finder.init2();
  //m_face_finder.check2();
  //debug() << "[MeshRefinement::updateLocalityMap2] done";
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_checkLevelOne(bool arcane_assert_pass)
{
  bool failure = false;

  Integer sid = m_mesh->parallelMng()->commRank();
  // itérer seulement sur les mailles actives
  ENUMERATE_CELL(icell,m_mesh->allActiveCells())
  {
    Cell cell = *icell;
    for( Face face : cell.faces() ) {
      if (face.nbCell()!=2)
        continue;
      Cell back_cell = face.backCell();
      Cell front_cell = face.frontCell();

      // On choisit l'autre cellule du cote de la face
      Cell neighbor = (back_cell==cell)?front_cell:back_cell;
      if (neighbor.null() || !neighbor.isActive() || !(neighbor.owner()==sid))
        continue;
      //debug() << "#### " << ineighbor->uniqueId() << " " << ineighbor->level() << " " << cell.level();
      if ((neighbor.level() + 1 < cell.level())){
        failure = true;
        break;
      }
    }
  }

  // Si un processeur échoue, on échoue globalement
  failure = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, failure);

  if (failure){
    // Nous n'avons pas passé le test level-one, donc arcane_assert
    // en fonction du booléen d'entré.
    if (arcane_assert_pass)
      throw FatalErrorException(A_FUNCINFO,"checkLevelOne failed");
    return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_checkUnflagged(bool arcane_assert_pass)
{
  bool found_flag = false;

  // recherche pour les flags locaux
  // itérer seulement sur les mailles actives
  ENUMERATE_CELL(icell,m_mesh->ownActiveCells()){
    const Cell cell = *icell;
    const Integer f = cell.itemBase().flags();
    if ( (f & ItemFlags::II_Refine) | (f & ItemFlags::II_Coarsen))
    {
      found_flag = true;
      break;
    }
  }
  // Si nous trouvions un flag sur n'importe quel processeur, il compte
  found_flag = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, found_flag);
  if (found_flag){
    //nous n'avons pas passé le test "items are unflagged",
    //ainsi arcane_assert la non valeur de arcane_assert_pass
    if (arcane_assert_pass)
      throw FatalErrorException(A_FUNCINFO,"checkUnflagged failed");
    return false;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_makeFlagParallelConsistent()
{
  if (!m_mesh->parallelMng()->isParallel())
    return true;

  CHECKPERF( m_perf_counter.start(PerfCounter::PCONSIST) )
  debug() << "makeFlagsParallelConsistent() begin";
  bool parallel_consistent = true;
  VariableCellInteger flag_cells_consistent(VariableBuildInfo(m_mesh, "FlagCellsConsistent"));
  UniqueArray<Item> ghost_cells;
  ghost_cells.reserve(m_mesh->allCells().size()-m_mesh->ownCells().size()) ;
  ENUMERATE_CELL(icell,m_mesh->allCells()){
    Cell cell = *icell;
    if(cell.isOwn()) {
      Integer f = cell.itemBase().flags(); // TODO getAMRFlags()
      flag_cells_consistent[icell] = f;
    }
    else
      ghost_cells.add(cell);
  }
  flag_cells_consistent.synchronize();
  //ENUMERATE_CELL(icell,m_mesh->allCells())
  for(Integer icell=0, nb_cell=ghost_cells.size();icell<nb_cell;++icell) {
    Item iitem = ghost_cells[icell];
    Integer f = iitem.itemBase().flags();

    //if(iitem->owner() != sid)
    {
      // il est possible que les flags des ghosts soient (temporairement) plus
      // conservatifs que nos propres flags , comme quand un raffinement d'une
      // des mailles du processeur distant est dicté par un raffinement d'une de nos mailles
      const Integer g = flag_cells_consistent[Cell(iitem)];
      if((g & ItemFlags::II_Refine) && !(f & ItemFlags::II_Refine))
      {
        f |= ItemFlags::II_Refine;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_Coarsen) && !(f & ItemFlags::II_Coarsen))
      {
        f |= ItemFlags::II_Coarsen;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_JustCoarsened) && !(f & ItemFlags::II_JustCoarsened))
      {
        f |= ItemFlags::II_JustCoarsened;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_JustRefined) && !(f & ItemFlags::II_JustRefined))
      {
        f |= ItemFlags::II_JustRefined;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      /*
       else if ((g & ItemFlags::II_CoarsenInactive) && !(f & ItemFlags::II_CoarsenInactive)){
       f |= ItemFlags::II_CoarsenInactive;
       //f |= ItemFlags::II_Inactive;
       iitem->setFlags(f);
       parallel_consistent = false;
       }
       else if ((g & ItemFlags::II_DoNothing) && !(f & ItemFlags::II_DoNothing)){
       f |= ItemFlags::II_DoNothing;
       //f |= ItemFlags::II_Inactive;
       iitem->setFlags(f);
       parallel_consistent = false;
       }

       else if ((g & ItemFlags::II_Inactive) && !(f & ItemFlags::II_Inactive)){
       f |= ItemFlags::II_Inactive;
       //f |= ItemFlags::II_Inactive;
       iitem->setFlags(f);
       parallel_consistent = false;
       }*/
    }
  }
  // Si nous ne sommes pas consistent sur chaque processeur alors
  // nous ne le sommes pas globalement
  parallel_consistent = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, parallel_consistent);
  debug() << "makeFlagsParallelConsistent() end -- parallel_consistent : " << parallel_consistent;

  CHECKPERF( m_perf_counter.stop(PerfCounter::PCONSIST) )
  return parallel_consistent;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
bool
MeshRefinement::
_makeFlagParallelConsistent2()
{
  if (!m_mesh->parallelMng()->isParallel())
    return true;

  CHECKPERF( m_perf_counter.start(PerfCounter::PCONSIST2) )
  debug() << "makeFlagsParallelConsistent2() begin";
  bool parallel_consistent = true;
  VariableCellInteger flag_cells_consistent(VariableBuildInfo(m_mesh, "FlagCellsConsistent"));
  UniqueArray<Item> ghost_cells;
  ghost_cells.reserve(m_mesh->allCells().size()-m_mesh->ownCells().size()) ;
  ENUMERATE_CELL(icell,m_mesh->allCells())
  {
    Cell cell = *icell;
    if (cell.isOwn()) {
      Integer f = cell.itemBase().flags(); // TODO getAMRFlags()
      flag_cells_consistent[icell] = f;
    }
    else
      ghost_cells.add(cell);
  }
  flag_cells_consistent.synchronize();
  //ENUMERATE_CELL(icell,m_mesh->allCells())
  for(Integer icell=0, nb_cell=ghost_cells.size();icell<nb_cell;++icell) {
    //const Cell& cell = *icell;
    //ItemInternal * iitem = cell.internal();
    //Integer f = iitem->flags();
    Item iitem = ghost_cells[icell];
    Integer f = iitem.itemBase().flags();

    //if(iitem->owner() != sid)
    {
      Integer g = flag_cells_consistent[Cell(iitem)];
      if ((g & ItemFlags::II_JustCoarsened) && !(f & ItemFlags::II_JustCoarsened))
      {
        f |= ItemFlags::II_JustCoarsened;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_JustRefined) && !(f & ItemFlags::II_JustRefined))
      {
        f |= ItemFlags::II_JustRefined;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_Inactive) && !(f & ItemFlags::II_Inactive))
      {
        f |= ItemFlags::II_Inactive;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      /*else if ((f & ItemFlags::II_JustRefined) && !(g & ItemFlags::II_JustRefined)){
       f &= ~ItemFlags::II_JustRefined;
       iitem->setFlags(f);
       parallel_consistent = false;
       }*/
    }
  }
  // Si nous ne sommes pas consistent sur chaque processeur alors
  // nous ne le sommes pas globalement
  parallel_consistent = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, parallel_consistent);
  debug() << "makeFlagsParallelConsistent2() end";

  CHECKPERF( m_perf_counter.stop(PerfCounter::PCONSIST2) )
  return parallel_consistent;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
MeshRefinement::
_makeCoarseningCompatible(const bool maintain_level_one)
{

  debug() << "makeCoarseningCompatible() begin";

  bool _maintain_level_one = maintain_level_one;

  // la règle de niveau-un est la seule condition implementée
  if (!maintain_level_one){
    warning() << "Warning, level one rule is the only condition accepted for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  // à moins que nous rencontrions une situation spécifique, la règle niveau-un
  // sera satisfaite aprês avoir exécuté cette boucle juste une fois
  bool level_one_satisfied = true;

  // à moins que nous rencontrions une situation spéccifique, nous serons compatible
  // avec tous flags de raffinement choisis
  bool compatible_with_refinement = true;

  // Trouver le niveau maximum dans le maillage
  Integer max_level = 0;

  // d'abord nous regardons tous les items actifs de niveau 0.  Puisque ca n'a pas de sens de
  // les déraffiner nous devons donc supprimer leur flags de déraffinement si
  // ils sont déjà positionnés.
  // itérer seulement sur les mailles actives
  ENUMERATE_CELL(icell,m_mesh->allActiveCells()){
    const Cell cell = *icell;
    max_level = std::max(max_level, cell.level());

    Integer f = cell.itemBase().flags();
    if ((cell.level() == 0) && (f & ItemFlags::II_Coarsen)) {
      f &= ~ItemFlags::II_Coarsen;
      f |= ItemFlags::II_DoNothing;
      cell.mutableItemBase().setFlags(f);
    }
  }
  // Si il n'y a pas d'items à raffiner sur ce processeur alors
  // il n'y a pas de travail à faire pour nous
  if (max_level == 0){
    debug() << "makeCoarseningCompatible() done";

    // par contre il reste à vérifier avec les autres processeurs
    compatible_with_refinement = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, compatible_with_refinement);

    return compatible_with_refinement;
  }

  // Boucle sur tous les items actifs.  Si un item est marqué
  // pour déraffinement on check ses voisins.  Si un de ses voisins
  // est marqué pour raffinement et est de même niveau alors il y a un
  // conflit.  Par convention raffinement gagne, alors on démarque l'item pour
  // déraffinement.  Le niveau-un serait violé dans ce cas-ci ainsi nous devons réexécuter
  // la boucle.
  const Integer sid = m_mesh->parallelMng()->commRank();
  if (_maintain_level_one)
  {

    repeat: level_one_satisfied = true;

    do
    {
      level_one_satisfied = true;
      // itérer seulement sur les mailles actives
      ENUMERATE_CELL(icell,m_mesh->ownActiveCells()){
        Cell cell = *icell;
        //ItemInternal* iitem = cell.internal();
        bool my_flag_changed = false;
        Integer f = cell.itemBase().flags();
        if (f & ItemFlags::II_Coarsen){ // Si l'item est actif et le flag de déraffinement est placé
          const Int32 my_level = cell.level();
          for( Face face : cell.faces() ) {
            if (face.nbCell()!=2)
              continue;
            Cell back_cell = face.backCell();
            Cell front_cell = face.frontCell();

            // On choisit l'autre cellule du cote de la face
            Cell neighbor = (back_cell==cell)?front_cell:back_cell;
            //const ItemInternal* ineighbor = neighbor.internal();
            //if (ineighbor->owner() == sub_domain_id)   // J'ai un voisin ici

            {
              if (neighbor.isActive()) // et est actif
              {
                if ((neighbor.level() == my_level) &&
                    (neighbor.itemBase().flags() & ItemFlags::II_Refine)){ // le voisin est à mon niveau et veut être raffiné
                  f &= ~ItemFlags::II_Coarsen;
                  f |= ItemFlags::II_DoNothing;
                  cell.mutableItemBase().setFlags(f);
                  my_flag_changed = true;
                  break;
                }
              }
              else{
                // J'ai un voisin et n'est pas actif. Cela signifie qu'il a des enfants.
                // tandis qu'il peut être possible de me déraffiner si tous les enfants
                // de cet item veulent être déraffinés, il est impossible de savoir à ce stade.
                // On l'oublie pour le moment. Ceci peut être réalisé dans deux étapes.
                f &= ~ItemFlags::II_Coarsen;
                f |= ItemFlags::II_DoNothing;
                cell.mutableItemBase().setFlags(f);
                my_flag_changed = true;
                break;
              }
            }
          }
        }

        //si le flag de la cellule courante a changé, nous n'avons pas
        //satisfait la rêgle du niveau un.
        if (my_flag_changed)
          level_one_satisfied = false;

        //En plus, s'il a des voisins non-locaux, et
        //nous ne sommes pas en séquentiel, alors nous devons par la suite
        // retourner compatible_with_refinement= false, parce que
        //notre changement doit être propager aux processeurs voisins
        if (my_flag_changed && m_mesh->parallelMng()->isParallel())
          for( Face face : cell.faces() ){
            if (face.nbCell()!=2)
              continue;
            Cell back_cell = face.backCell();
            Cell front_cell = face.frontCell();

            // On choisit l'autre cellule du cote de la face
            Cell neighbor = (back_cell==cell)?front_cell:back_cell;
            //ItemInternal* ineighbor = neighbor.internal();
            if (neighbor.owner() != sid){ // J'ai un voisin ici
              compatible_with_refinement = false;
              break;
            }
            // TODO FIXME - pour les maillages non niveau-1 nous devons
            // tester tous les descendants
            if (neighbor.hasHChildren())
              for (Integer c=0; c != neighbor.nbHChildren(); ++c)
                if (neighbor.hChild(c).owner() != sid){
                  compatible_with_refinement = false;
                  break;
                }
          }

      }
    }
    while (!level_one_satisfied);

  } // end if (_maintain_level_one)

  //après, nous regardons tous les items ancêtres.
  //s'il y a un item parent avec tous ses enfants
  //voulant être déraffiné alors l'item est un candidat
  //pour le déraffinement.  Si tous les enfants ne
  //veulent pas être déraffiné alors tous ont besoin d'avoir leur
  // flag de déraffinement dégagés.
  for (int level=(max_level); level >= 0; level--){
    // itérer sur les mailles niveau par niveau
    ENUMERATE_CELL(icell,m_mesh->ownLevelCells(level)){
      const Cell cell = *icell;
      //ItemInternal* iitem = cell.internal();
      if(cell.isAncestor()){
        // à ce moment là l'item n'a pas été éliminé
        // en tant que candidat pour le déraffinement
        bool is_a_candidate = true;
        bool found_remote_child = false;

        for (Integer c=0; c<cell.nbHChildren(); c++){
          Cell child = cell.hChild(c);
          if (child.owner() != sid)
            found_remote_child = true;
          else if (!(child.itemBase().flags() & ItemFlags::II_Coarsen) || !child.isActive() )
            is_a_candidate = false;
        }

        if (!is_a_candidate && !found_remote_child){
          cell.mutableItemBase().addFlags(ItemFlags::II_Inactive);
          for (Integer c=0; c<cell.nbHChildren(); c++){
            Cell child = cell.hChild(c);
            if (child.owner() != sid)
              continue;
            if (child.itemBase().flags() & ItemFlags::II_Coarsen){
              level_one_satisfied = false;
              Int32 f = child.itemBase().flags();
              f &= ~ItemFlags::II_Coarsen;
              f |= ItemFlags::II_DoNothing;
              child.mutableItemBase().setFlags(f);
            }
          }
        }
      }
      }
  }
  if (!level_one_satisfied && _maintain_level_one) goto repeat;

  // Si tous les enfants d'un parent sont marqués pour déraffinement
  // Alors marque le parent à ce qu'il puisse tuer ses enfants.
  ENUMERATE_CELL(icell,m_mesh->ownCells()){
    const Cell cell = *icell;
    //ItemInternal* iitem = cell.internal();
    if(cell.isAncestor()){
      // Supposons que tous les enfants sont locaux et marqués pour
      // déraffinement et donc cherche pour une contradiction
      bool all_children_flagged_for_coarsening = true;
      bool found_remote_child = false;

      for (Integer c=0; c<cell.nbHChildren(); c++){
        Cell child = cell.hChild(c);
        if (child.owner() != sid)
          found_remote_child = true;
        else if (!(child.itemBase().flags() & ItemFlags::II_Coarsen))
          all_children_flagged_for_coarsening = false;
      }
      Integer f = cell.itemBase().flags();
      f &= ~ItemFlags::II_CoarsenInactive;
      if (!found_remote_child && all_children_flagged_for_coarsening)
      {
        f |= ItemFlags::II_CoarsenInactive;
        cell.mutableItemBase().setFlags(f);
      }
      else if (!found_remote_child)
      {
        f |= ItemFlags::II_Inactive;
        cell.mutableItemBase().setFlags(f);
      }
    }
  }

  debug() << "makeCoarseningCompatible() done";

  // Si nous sommes pas compatible sur un processeur, nous ne le sommes pas globalement
  compatible_with_refinement = m_mesh->parallelMng()->reduce(Parallel::ReduceMin,compatible_with_refinement);

  return compatible_with_refinement;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
MeshRefinement::
_makeRefinementCompatible(const bool maintain_level_one)
{

  debug() << "makeRefinementCompatible() begin";

  bool _maintain_level_one = maintain_level_one;

  // la règle de niveau-un est la seule condition implementée
  if (!maintain_level_one){
    warning() << "Warning, level one rule is the only condition accepted now for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  // à moins que nous rencontrions une situation spécifique, la règle niveau-un
  // sera satisfaite après avoir exécuté cette boucle juste une fois
  bool level_one_satisfied = true;

  // à moins que nous rencontrions une situation spécifique, nous serons compatible
  // avec tous flags de déraffinement choisis
  bool compatible_with_coarsening = true;

  // cette boucle impose la règle niveau-1.  Nous devrions seulement
  // l'exécuter si l'utilisateur veut en effet que la niveau-1 soit satisfaite !
  Integer sid = m_mesh->parallelMng()->commRank();
  if (_maintain_level_one){
    do {
      level_one_satisfied = true;
      // itérer seulement sur les mailles actives
      ENUMERATE_CELL(icell,m_mesh->allActiveCells()){
        const Cell cell = *icell;
        //ItemInternal* iitem = cell.internal();
        if (cell.itemBase().flags() & ItemFlags::II_Refine){ // Si l'item est actif et le flag de
          // raffinement est placé
          const Int32 my_level = cell.level();
          bool refinable = true;
          //check if refinable
          for( Face face : cell.faces() ){
            if (face.nbCell()!=2)
              continue;
            Cell back_cell = face.backCell();
            Cell front_cell = face.frontCell();

            // On choisit l'autre cellule du cote de la face
            Cell neighbor = (back_cell==cell)?front_cell:back_cell;
            //ItemInternal* ineighbor = neighbor.internal();
            //if (ineighbor->isActive() && ineighbor->owner() == sid)// J'ai un voisin ici et est actif
            if (neighbor.isActive() ){// J'ai un voisin ici et est actif
              // Cas 2: Le voisin est inférieur de un niveau que le mien.
              //         Le voisin doit être raffiné pour satisfaire
              //         la règle de niveau-1, indépendamment de s'il
              //         a été à l'origine marqué pour raffinement. S'il
              //         n'était pas flaggé déjà nous devons répéter
              //         ce processus.
              Integer f = neighbor.itemBase().flags();
              if ( ( (neighbor.level()+1) == my_level) &&
                  ( f & ItemFlags::II_UserMark1) ){
                refinable = false;
                Integer my_f = cell.itemBase().flags();
                my_f &= ~ItemFlags::II_Refine;
                cell.mutableItemBase().setFlags(my_f);
                break;
              }
            }
          }
          if(refinable)
            for( Face face : cell.faces() ){
              if (face.nbCell()!=2)
                continue;
              Cell back_cell = face.backCell();
              Cell front_cell = face.frontCell();

              // On choisit l'autre cellule du cote de la face
              Cell neighbor = (back_cell==cell)?front_cell:back_cell;
              //ItemInternal* ineighbor = neighbor.internal();
              if (neighbor.isActive() && neighbor.owner() == sid){ // J'ai un voisin ici et est actif

                // Cas 1:  Le voisin est au même niveau que moi.
                //        1a: Le voisin  sera raffiné           -> NO PROBLEM
                //        1b: Le voisin ne va pas être raffiné  -> NO PROBLEM
                //        1c: Le voisin veut être déjà raffiné     -> PROBLEM
                if (neighbor.level() == my_level){
                  Integer f = neighbor.itemBase().flags();
                  if (f & ItemFlags::II_Coarsen) {
                    f &= ~ItemFlags::II_Coarsen;
                    f |= ItemFlags::II_DoNothing;
                    neighbor.mutableItemBase().setFlags(f);
                    if (neighbor.nbHParent() != 0){
                      neighbor.hParent().mutableItemBase().addFlags(ItemFlags::II_Inactive);
                    }
                    compatible_with_coarsening = false;
                    level_one_satisfied = false;
                  }
                }

                // Cas 2: Le voisin est inférieur de un niveau que le mien.
                //         Le voisin doit être raffiné pour satisfaire
                //         la rêgle de niveau-1, indépendamment de s'il
                //         a été à l'origine marqué pour raffinement. S'il
                //         n'était pas flaggé déjà nous devons répéter
                //         ce processus.

                else if ((neighbor.level()+1) == my_level) {
                  Integer f = neighbor.itemBase().flags();
                  if (!(f & ItemFlags::II_Refine)) {
                    f &= ~ItemFlags::II_Coarsen;
                    f |= ItemFlags::II_Refine;
                    neighbor.mutableItemBase().setFlags(f);
                    if (neighbor.nbHParent() != 0){
                      neighbor.hParent().mutableItemBase().addFlags(ItemFlags::II_Inactive);
                    }
                    compatible_with_coarsening = false;
                    level_one_satisfied = false;
                  }
                }
#ifdef ARCANE_DEBUG
                // Contrôle. Nous ne devrions jamais entrer dans un
                // cas ou notre voisin est distancé de plus d'un niveau.

                else if ((neighbor.level()+1) < my_level)
                {
                  fatal() << "a neighbor is more than one level away";
                }

                // On note que la seule autre possibilité est que
                // le voisin ait déjà été raffiné, dans ce cas il n'est pas
                //actif et nous ne devrions jamais tomber ici.

                else
                {
                  fatal() << "serious problem: we should never get here";
                }
#endif
              }
            }
        }
      }
    }
    while (!level_one_satisfied);
  } // end if (_maintain_level_one)

  // Si nous sommes pas compatible sur un processeur, nous ne le sommes pas globalement
  compatible_with_coarsening = m_mesh->parallelMng()->reduce(Parallel::ReduceMin,compatible_with_coarsening);

  debug() << "makeRefinementCompatible() done";

  return compatible_with_coarsening;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
MeshRefinement::
_coarsenItems()
{
  debug() << "[MeshRefinement::_coarsenItems] begin"<<m_mesh->allNodes().size();
  // Flag indiquant si cet appel change réellement le maillage
  bool mesh_changed = false;

  // itérer sur toutes les mailles
  // Int32UniqueArray cell_to_detach;
  ENUMERATE_CELL(icell,m_mesh->ownCells()){
    Cell cell = *icell;
    Cell iitem = cell;
    // items actifs flaggés prour déraffinement ne seront
    // pas supprimés jusqu'à contraction via MeshRefinement::contract()

    if (cell.itemBase().flags() & ItemFlags::II_Coarsen){
      // Houups?  aucun item de niveau-0 ne doit être a la fois actif
      // et flaggé pour déraffinement.
      ARCANE_ASSERT ( (cell.level() != 0), ("no level-0 element should be active and flagged for coarsening"));

      // TODO Supprimer cet item de toute liste de voisinage
      // pointant vers lui.
      // FIXME à l'IFP, on utilise par défaut la macro REMOVE_UID_ON_DETACH suprimant le UID de CELL
      // dans la map des cell_uid donc on ne peut utiliser detachCell par défaut. En attendant
      // on utilise la méthode MeshRefinement::contract() après mise à jour des variables
      // cell_to_detach.add(iitem->localId());

      //cells_to_remove.add(cell);
      // TODO optimisation  des uids non utilisé.
      // m_unused_items.push_back (uid);

      // Ne pas détruire l'item jusqu'à MeshRefinement::contract()
      // m_mesh->modifier()->removeCells(iitem->localId());

      // Le maillage a certainement changé
      mesh_changed = true;
    }
    else if (cell.itemBase().flags() & ItemFlags::II_CoarsenInactive)
    {
      switch (cell.type())
      {
        case IT_Quad4:
          m_item_refinement->coarsenOneCell<IT_Quad4>(iitem, getRefinementPattern<IT_Quad4>());
          break;
        case IT_Tetraedron4:
          m_item_refinement->coarsenOneCell<IT_Tetraedron4>(iitem, getRefinementPattern<IT_Tetraedron4>());
          break;
        case IT_Pyramid5:
          m_item_refinement->coarsenOneCell<IT_Pyramid5>(iitem, getRefinementPattern<IT_Pyramid5>());
          break;
        case IT_Pentaedron6:
          m_item_refinement->coarsenOneCell<IT_Pentaedron6>(iitem, getRefinementPattern<IT_Pentaedron6>());
          break;
        case IT_Hexaedron8:
          m_item_refinement->coarsenOneCell<IT_Hexaedron8>(iitem, getRefinementPattern<IT_Hexaedron8>());
          break;
        case IT_HemiHexa7:
          m_item_refinement->coarsenOneCell<IT_HemiHexa7>(iitem, getRefinementPattern<IT_HemiHexa7>());
          break;
        case IT_HemiHexa6:
          m_item_refinement->coarsenOneCell<IT_HemiHexa6>(iitem, getRefinementPattern<IT_HemiHexa6>());
          break;
        case IT_HemiHexa5:
          m_item_refinement->coarsenOneCell<IT_HemiHexa5>(iitem, getRefinementPattern<IT_HemiHexa5>());
          break;
        case IT_AntiWedgeLeft6:
          m_item_refinement->coarsenOneCell<IT_AntiWedgeLeft6>(iitem, getRefinementPattern<IT_AntiWedgeLeft6>());
          break;
        case IT_AntiWedgeRight6:
          m_item_refinement->coarsenOneCell<IT_AntiWedgeRight6>(iitem, getRefinementPattern<IT_AntiWedgeRight6>());
          break;
        case IT_DiTetra5:
          m_item_refinement->coarsenOneCell<IT_DiTetra5>(iitem, getRefinementPattern<IT_DiTetra5>());
          break;
        default:
          ARCANE_FATAL("Not supported refinement Item Type type={0}",iitem.type());
      }
      ARCANE_ASSERT(cell.isActive(), ("cell_active failed"));

      // le maillage a certainement changé
      mesh_changed = true;
    }
  }
  // TODO
  // m_mesh->modifier->detachCells(cell_to_detach);

  // Si le maillage a changé sur un processeur, alors il a changé globalement
  mesh_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, mesh_changed);
  // Et peut être nous avons besoin de mettre à jour les entités refletant le changement
  //if (mesh_changed)
  // \todo compacte et update max_uids en parallel

  // si une maille est deraffinee ailleurs, les noeuds attaches a cette maille
  // doivent etre mis a jour. Cela est traite dans endUpdate()

  debug() << "[MeshRefinement::_coarsenItems()] done "<<m_mesh->allNodes().size();

  return mesh_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
MeshRefinement::
_refineItems(Int64Array& cell_to_refine_uids)
{
  // Mise à jour de m_node_finder, m_face_finder permettra au maillage
  // d'etre consistent globalement (uids consistency).
  debug() << "[MeshRefinement::_refineItems]"<<m_mesh->allNodes().size();
#ifdef ARCANE_DEBUG
  m_node_finder.check() ;
  m_face_finder.check() ;
#endif
  CHECKPERF( m_perf_counter.start(PerfCounter::REFINE) )
  m_face_finder.clearNewUids() ;
  // Iterer sur les items, compter les items
  // flaggés pour le raffinement.
  //Integer nb_cell_flagged = 0;
  UniqueArray<Cell> cell_to_refine_internals;
  ENUMERATE_CELL(icell,m_mesh->ownCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Refine) {
      cell_to_refine_uids.add(cell.uniqueId());
      cell_to_refine_internals.add(cell);
    }
  }
  debug() << "[MeshRefinement::_refineItems] " << cell_to_refine_uids.size() << " flagged cells for refinement";

  // Construire un vecteur local des items marqués
  // pour raffinement.
  /*
   local_copy_of_cells.reserve(nb_cell_flagged);

   ENUMERATE_CELL(icell,m_mesh->ownCells()){
   Cell cell = *icell;
   ItemInternal* iitem = cell.internal();
   if(iitem->flags() & ItemFlags::II_Refine)
   local_copy_of_cells.add(iitem);
   }
   */
  // Maintenant, itere sur les copies locales et raffine chaque item.
  const Int32 i_size = cell_to_refine_internals.size();
  for (Integer e = 0; e != i_size; ++e) {
    Cell iitem = cell_to_refine_internals[e];
    //debug()<<"\t[MeshRefinement::_refineItems] focus on cell "<<iitem->uniqueId();
    switch (iitem.type())
    {
      case IT_Quad4:
        m_item_refinement->refineOneCell<IT_Quad4>(iitem,*this);
        break;
      case IT_Tetraedron4:
        m_item_refinement->refineOneCell<IT_Tetraedron4>(iitem,*this);
        break;
      case IT_Pyramid5:
        m_item_refinement->refineOneCell<IT_Pyramid5>(iitem,*this);
        break;
      case IT_Pentaedron6:
        m_item_refinement->refineOneCell<IT_Pentaedron6>(iitem,*this);
        break;
      case IT_Hexaedron8:
        m_item_refinement->refineOneCell<IT_Hexaedron8>(iitem,*this);
        break;
      case IT_HemiHexa7:
        m_item_refinement->refineOneCell<IT_HemiHexa7>(iitem,*this);
        break;
      case IT_HemiHexa6:
        m_item_refinement->refineOneCell<IT_HemiHexa6>(iitem,*this);
        break;
      case IT_HemiHexa5:
        m_item_refinement->refineOneCell<IT_HemiHexa5>(iitem,*this);
        break;
      case IT_AntiWedgeLeft6:
        m_item_refinement->refineOneCell<IT_AntiWedgeLeft6>(iitem,*this);
        break;
      case IT_AntiWedgeRight6:
        m_item_refinement->refineOneCell<IT_AntiWedgeRight6>(iitem,*this);
        break;
      case IT_DiTetra5:
        m_item_refinement->refineOneCell<IT_DiTetra5>(iitem,*this);
        break;
      default:
        ARCANE_FATAL("Not supported refinement Item Type type={0}",iitem.type());
    }
  }

  // Le maillage change si des items sont raffinés
  bool mesh_changed = !(i_size == 0);

  // Si le maillage change sur un processeur, il change globalement
  mesh_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, mesh_changed);

  // Et nous avons besoin de mettre à jour le nombre des ids
  if (mesh_changed){
    for (Integer e = 0; e != i_size; ++e){
      Cell i_hParent_cell = cell_to_refine_internals[e];
      populateBackFrontCellsFromParentFaces(i_hParent_cell);
    }
  }
  CHECKPERF( m_perf_counter.stop(PerfCounter::REFINE) )
  if (mesh_changed && m_mesh->parallelMng()->isParallel())
  {
    CHECKPERF( m_perf_counter.start(PerfCounter::PGCONSIST) )
#ifdef ARCANE_DEBUG
    m_node_finder.check2() ;
    m_face_finder.check2() ;
#endif

    // Nodes and faces parallel consistency
    m_parallel_amr_consistency->makeNewItemsConsistent(m_node_finder, m_face_finder);
    CHECKPERF( m_perf_counter.stop(PerfCounter::PGCONSIST) )
  }

  debug() << "[MeshRefinement::_refineItems] done"<<m_mesh->allNodes().size();

  return mesh_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_cleanRefinementFlags()
{
  //Nettoyage des flags de raffinement d'une étape précédente
  ENUMERATE_CELL(icell,m_mesh->allCells()){
    Cell cell = *icell;
    auto mutable_cell = cell.mutableItemBase();
    Integer f = mutable_cell.flags();
    if (cell.isActive()){
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
    else{
      f |= ItemFlags::II_Inactive;
      mutable_cell.setFlags(f);
    }
    // Ceci pourrait être laissé de la derniè étape
    if (f & ItemFlags::II_JustRefined){
      f &= ~ItemFlags::II_JustRefined;
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
    if (f & ItemFlags::II_JustCoarsened){
      f &= ~ItemFlags::II_JustCoarsened;
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
    if (f & ItemFlags::II_JustAdded){
      f &= ~ItemFlags::II_JustAdded;
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
    if (f & ItemFlags::II_CoarsenInactive){
      f &= ~ItemFlags::II_CoarsenInactive;
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_contract()
{
  DynamicMesh* mesh = m_mesh;
  ItemInternalMap& cells_map = mesh->cellsMap();

  // Flag indicating if this call actually changes the mesh
  bool mesh_changed = false;

  if (arcaneIsDebug()){
    cells_map.eachItem([&](impl::ItemBase item) {
      if (item.isOwn())
        // une maille est soit active, subactive ou ancestor
        ARCANE_ASSERT((item.isActive() || item.isSubactive() || item.isAncestor()), (" "));
    });
  }

  //
  std::set < Int32 > cells_to_remove_set;
  UniqueArray<ItemInternal*> parent_cells;

  cells_map.eachItem([&](impl::ItemBase iitem) {
    if (!iitem.isOwn())
      return;

    // suppression des subactives
    if (iitem.isSubactive()) {
      // aucune maille de niveau 0 ne doit être subactive.
      ARCANE_ASSERT((iitem.nbHParent() != 0), (""));
      cells_to_remove_set.insert(iitem.localId());
      // informe le client du changement de maillage
      mesh_changed = true;
    }
    else{
      // Compression des mailles actives
      if (iitem.isActive()) {
        bool active_parent = false;
        for (Integer c = 0; c < iitem.nbHChildren(); c++) {
          impl::ItemBase ichild = iitem.hChildBase(c);
          if (!ichild.isSuppressed()) {
            //debug() << "[\tMeshRefinement::contract]PARENT UID=" << iitem->uniqueId() << " " << iitem->owner() << " "
            //    << iitem->level();
            cells_to_remove_set.insert(ichild.localId());
            //debug() << "[\tMeshRefinement::contract]CHILD UID=" << ichild->uniqueId() << " " << ichild->owner();
            active_parent = true;
          }
        }
        if (active_parent){
          parent_cells.add(iitem._itemInternal());
          ARCANE_ASSERT((iitem.flags() & ItemFlags::II_JustCoarsened), ("Incoherent JustCoarsened flag"));
        }
        // informe le client du changement de maillage
        mesh_changed = true;
      }
      else{
        ARCANE_ASSERT((iitem.isAncestor()), (""));
      }
    }
  });
  //
  UniqueArray<Int32> cell_lids(arcaneCheckArraySize(cells_to_remove_set.size()));
  std::copy(std::begin(cells_to_remove_set), std::end(cells_to_remove_set),std::begin(cell_lids));

  if (m_mesh->parallelMng()->isParallel()){
    this->_makeFlagParallelConsistent2();
    this->_removeGhostChildren();
    this->_updateItemOwner(cell_lids);
    m_mesh->parallelMng()->barrier();
  }
  if (cell_lids.size() > 0){
    this->_upscaleData(parent_cells);
    _invalidate(parent_cells);
    //_updateItemOwner(cells_local_id);
    m_mesh->modifier()->removeCells(cell_lids,false);
    const Integer ps = parent_cells.size();
    for (Integer i = 0; i < ps; i++)
      populateBackFrontCellsFromChildrenFaces(parent_cells[i]);
  }
  else
    mesh_changed = false;

  return mesh_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
registerCallBack(IAMRTransportFunctor* f)
{
  m_call_back_mng->registerCallBack(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
unRegisterCallBack(IAMRTransportFunctor* f)
{
  m_call_back_mng->unregisterCallBack(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_interpolateData(const Int64Array& cells_to_refine)
{
  const Int32 nb_cells = cells_to_refine.size();
  Int32UniqueArray lids(nb_cells);
  m_mesh->cellFamily()->itemsUniqueIdToLocalId(lids.view(), cells_to_refine.constView());
  CellInfoListView internals(m_mesh->cellFamily());

  UniqueArray<ItemInternal*> cells_to_refine_internals(nb_cells);
  for (Integer i = 0; i < nb_cells; i++) {
    cells_to_refine_internals[i] = ItemCompatibility::_itemInternal(internals[lids[i]]);
  }
  m_call_back_mng->callCallBacks(cells_to_refine_internals, Prolongation);
  _update(cells_to_refine_internals);
}

void MeshRefinement::
_update(ArrayView<Int64> cells_to_refine_uids)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::UPDATEMAP) )
  const Int32 nb_cells = cells_to_refine_uids.size();
  Int32UniqueArray lids(nb_cells);
  m_mesh->cellFamily()->itemsUniqueIdToLocalId(lids, cells_to_refine_uids);
  CellInfoListView internals(m_mesh->cellFamily());
  UniqueArray<ItemInternal*> cells_to_refine(nb_cells);
  for (Integer i = 0; i < nb_cells; i++) {
    cells_to_refine[i] = ItemCompatibility::_itemInternal(internals[lids[i]]);
  }
  m_node_finder.updateData(cells_to_refine);
  m_face_finder.updateData(cells_to_refine);
  _updateMaxUid(cells_to_refine);
  m_item_refinement->updateChildHMin(cells_to_refine) ;
  //m_face_finder.updateFaceCenter(cells_to_refine);
  CHECKPERF( m_perf_counter.stop(PerfCounter::UPDATEMAP) )
}

void MeshRefinement::
_update(ArrayView<ItemInternal*> cells_to_refine)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::UPDATEMAP) )
  m_node_finder.updateData(cells_to_refine) ;
  m_face_finder.updateData(cells_to_refine) ;
  _updateMaxUid(cells_to_refine) ;
  m_item_refinement->updateChildHMin(cells_to_refine) ;
  //m_face_finder.updateFaceCenter(cells_to_refine);
  CHECKPERF( m_perf_counter.stop(PerfCounter::UPDATEMAP) )
}

void MeshRefinement::
_invalidate(ArrayView<ItemInternal*> coarsen_cells)
{
  CHECKPERF( m_perf_counter.start(PerfCounter::CLEAR) )
  m_node_finder.clearData(coarsen_cells) ;
  m_face_finder.clearData(coarsen_cells) ;
  CHECKPERF( m_perf_counter.stop(PerfCounter::CLEAR) )
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_upscaleData(Array<ItemInternal*>& parent_cells)
{
  m_call_back_mng->callCallBacks(parent_cells,Restriction);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_updateItemOwner(Int32ArrayView cell_to_remove_lids)
{
  IItemFamily* cell_family = m_mesh->cellFamily();
  CellInfoListView cells_list(cell_family);

  VariableItemInt32& nodes_owner(m_mesh->nodeFamily()->itemsNewOwner());
  VariableItemInt32& faces_owner(m_mesh->faceFamily()->itemsNewOwner());

  const Integer sid = m_mesh->parallelMng()->commRank();

  bool node_owner_changed = false;
  bool face_owner_changed = false;

  std::map<Int32, bool> marker;

  for (Integer i = 0, is = cell_to_remove_lids.size(); i < is; i++){
    Cell item = cells_list[cell_to_remove_lids[i]];
    for (Node node : item.nodes()){

      if (marker.find(node.localId()) != marker.end())
        continue;
      else
        marker[node.localId()] = true;

      //debug() << "NODE " << FullItemPrinter(node);
      //const Int32 owner = node->owner();
      bool is_ok = false;
      Integer count = 0;
      const Integer node_cs = node.cells().size();
      for ( Cell cell : node.cells() ){
        if (cell_to_remove_lids.contains(cell.localId())){
          count++;
          if (count == node_cs)
            is_ok = true;
          continue;
        }
        // SDC : this condition contributes to desynchronize node owners...
//        if (cell->owner() == owner)
//        {
//          is_ok = true;
//          break;
//        }
      }
      if (!is_ok){
        Cell cell;
        for ( Cell cell2 : node.cells() ){
          if (cell_to_remove_lids.contains(cell2.localId()))
            continue;
          if (cell.null() || cell2.uniqueId() < cell.uniqueId())
            cell = cell2;
        }
        if (cell.null())
          ARCANE_FATAL("Inconsistent null cell owner reference");
        const Int32 new_owner = cell.owner();
        nodes_owner[node] = new_owner;
        node.mutableItemBase().setOwner(new_owner, sid);
        //debug() << " NODE CHANGED OWNER " << node->uniqueId();
        node_owner_changed = true;
      }
    }
    for ( Face face : item.faces() ){
      if (face.nbCell() != 2)
        continue;
      const Int32 owner = face.owner();
      bool is_ok = false;
      for (Cell cell : face.cells() ){
        if ((item.uniqueId() == cell.uniqueId()) || !(item.level() == cell.level()))
          continue;
        if (cell.owner() == owner){
          is_ok = true;
          break;
        }
      }
      if (!is_ok){
        for (Cell cell2 : face.cells()){
          if (item.uniqueId() == cell2.uniqueId())
            continue;
          faces_owner[face] = cell2.owner();
          face.mutableItemBase().setOwner(cell2.owner(), sid);
          //debug() << " FACE CHANGED OWNER " << FullItemPrinter(face);
          face_owner_changed = true;
        }
      }
    }
  }

  node_owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, node_owner_changed);
  if (node_owner_changed){
    // nodes_owner.synchronize(); // SDC Surtout pas la synchro est KO à ce moment là (fantômes non raffinés/déraffinés)
    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
    m_mesh->nodeFamily()->endUpdate();
  }
  face_owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, face_owner_changed);
  if (face_owner_changed){
    faces_owner.synchronize();
    m_mesh->faceFamily()->notifyItemsOwnerChanged();
    m_mesh->faceFamily()->endUpdate();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_updateItemOwner2()
{
  // il faut que tout sub-item est une cellule voisine de même propriétaire
  VariableItemInt32& nodes_owner(m_mesh->nodeFamily()->itemsNewOwner());

  NodeGroup own_nodes = m_mesh->ownNodes();
  bool owner_changed = false;
  ENUMERATE_NODE(inode,own_nodes){
    Node node = (*inode);
    Int32 owner = node.owner();
    bool is_ok = false;
    for( Cell cell : node.cells() ){
      if (cell.owner()==owner){
        is_ok = true;
        break;
      }
    }
    if (!is_ok){
      Cell cell;
      for( Cell cell2 : node.cells() ){
        if (cell.null() || cell2.uniqueId() < cell.uniqueId())
          cell = cell2;
      }
      ARCANE_ASSERT((!cell.null()),("Inconsistent null cell owner reference"));
      nodes_owner[node] = cell.owner();
      owner_changed =true;
    }
  }
  owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, owner_changed);
  if (owner_changed){
    nodes_owner.synchronize();
    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
    m_mesh->nodeFamily()->endUpdate();
  }

  // il faut que tout sub-item est une cellule voisine de même propriétaire
  VariableItemInt32& faces_owner(m_mesh->faceFamily()->itemsNewOwner());

  FaceGroup own_faces = m_mesh->ownFaces();
  owner_changed = false;
  ENUMERATE_FACE(iface,own_faces){
    Face face = (*iface);
    Int32 owner = face.owner();
    bool is_ok = false;
    for( Cell cell : face.cells() ){
      if (cell.owner()==owner){
        is_ok = true;
        break;
      }
    }
    if (!is_ok){
      if(face.nbCell() ==2)
        fatal() << "Face" << ItemPrinter(face) << " has a different owner with respect to Back/Front Cells";

      faces_owner[face] = face.boundaryCell().owner();
      owner_changed=true;
    }
  }
  owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, owner_changed);
  if (owner_changed){
    faces_owner.synchronize();
    m_mesh->faceFamily()->notifyItemsOwnerChanged();
    m_mesh->faceFamily()->endUpdate();
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_removeGhostChildren()
{
  const Integer sid = m_mesh->parallelMng()->commRank();
  DynamicMesh* mesh = m_mesh;
  ItemInternalMap& cells_map = mesh->cellsMap();

  // Suppression des mailles
  Int32UniqueArray cells_to_remove ;
  cells_to_remove.reserve(1000);
  UniqueArray<ItemInternal*> parent_cells;
  parent_cells.reserve(1000);

  cells_map.eachItem([&](impl::ItemBase cell) {
    if (cell.owner() == sid)
      return;

    if (cell.flags() & ItemFlags::II_JustCoarsened) {
      for (Integer c = 0, cs = cell.nbHChildren(); c < cs; c++) {
        cells_to_remove.add(cell.hChildBase(c).localId());
      }
      parent_cells.add(cell._itemInternal());
    }
  });

  _invalidate(parent_cells) ;

  // Avant suppression, mettre a jour les owner de Node/Face isoles
  _updateItemOwner(cells_to_remove);
  //info() << "Number of cells to remove: " << cells_to_remove.size();
  m_mesh->modifier()->removeCells(cells_to_remove,false);
  for (Integer i = 0, ps = parent_cells.size(); i < ps; i++)
    populateBackFrontCellsFromChildrenFaces(parent_cells[i]);

  return cells_to_remove.size() > 0 ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
populateBackFrontCellsFromParentFaces(Cell parent_cell)
{
  switch (parent_cell.type())
  {
    case IT_Quad4:
      _populateBackFrontCellsFromParentFaces<IT_Quad4>(parent_cell);
      break;
    case IT_Tetraedron4:
      _populateBackFrontCellsFromParentFaces<IT_Tetraedron4>(parent_cell);
      break;
    case IT_Pyramid5:
      _populateBackFrontCellsFromParentFaces<IT_Pyramid5>(parent_cell);
      break;
    case IT_Pentaedron6:
      _populateBackFrontCellsFromParentFaces<IT_Pentaedron6>(parent_cell);
      break;
    case IT_Hexaedron8:
      _populateBackFrontCellsFromParentFaces<IT_Hexaedron8>(parent_cell);
      break;
    case IT_HemiHexa7:
      _populateBackFrontCellsFromParentFaces<IT_HemiHexa7>(parent_cell);
      break;
    case IT_HemiHexa6:
      _populateBackFrontCellsFromParentFaces<IT_HemiHexa6>(parent_cell);
      break;
    case IT_HemiHexa5:
      _populateBackFrontCellsFromParentFaces<IT_HemiHexa5>(parent_cell);
      break;
    case IT_AntiWedgeLeft6:
      _populateBackFrontCellsFromParentFaces<IT_AntiWedgeLeft6>(parent_cell);
      break;
    case IT_AntiWedgeRight6:
      _populateBackFrontCellsFromParentFaces<IT_AntiWedgeRight6>(parent_cell);
      break;
    case IT_DiTetra5:
      _populateBackFrontCellsFromParentFaces<IT_DiTetra5>(parent_cell);
      break;
    default:
      ARCANE_FATAL("Not supported refinement Item Type type={0}",parent_cell.type());
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int typeID> void MeshRefinement::
_populateBackFrontCellsFromParentFaces(Cell parent_cell)
{
  Integer nb_children = parent_cell.nbHChildren();
  const ItemRefinementPatternT<typeID>& rp = getRefinementPattern<typeID>();
  for (Integer c = 0; c < nb_children; c++){
    Cell child = parent_cell.hChild(c);
    Integer nb_child_faces = child.nbFace();
    for (Integer fc = 0; fc < nb_child_faces; fc++){
      if (rp.face_mapping_topo(c, fc) == 0)
        continue;
      const Integer f = rp.face_mapping(c, fc);
      Face face = parent_cell.face(f);
      Integer nb_cell_face = face.nbCell();
      if (nb_cell_face == 1)
        continue;
      Face subface = child.face(fc);
      Integer nb_cell_subface = subface.nbCell();
      if (nb_cell_subface == 1){
        m_face_family->addBackFrontCellsFromParentFace(subface, face);
      }
      else{
        if (face.backCell().isOwn() != face.frontCell().isOwn()){
          m_face_family->replaceBackFrontCellsFromParentFace(child, subface, parent_cell, face);
        }
        else{
          if (!face.backCell().isOwn() && !face.frontCell().isOwn()){
            m_face_family->replaceBackFrontCellsFromParentFace(child, subface, parent_cell, face);
          }
        }
      }
      ARCANE_ASSERT((subface.backCell() != parent_cell && subface.frontCell() != parent_cell),
          ("back front cells error"));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
populateBackFrontCellsFromChildrenFaces(Cell parent_cell)
{
  ARCANE_ASSERT((parent_cell.isActive()), (""));
  Integer nb_faces = parent_cell.nbFace();
  for (Integer f = 0; f < nb_faces; f++){
    Face face = parent_cell.face(f);
    Integer nb_cell_face = face.nbCell();
    if (nb_cell_face == 1)
      continue;
    Cell neighbor_cell = (face.cell(0) == parent_cell) ? face.cell(1) : face.cell(0);
    if (neighbor_cell.isActive())
      continue;
    switch (neighbor_cell.type()){
      case IT_Quad4:
        _populateBackFrontCellsFromChildrenFaces<IT_Quad4>(face, parent_cell, neighbor_cell);
        break;
      case IT_Tetraedron4:
        _populateBackFrontCellsFromChildrenFaces<IT_Tetraedron4>(face, parent_cell, neighbor_cell);
        break;
      case IT_Pyramid5:
        _populateBackFrontCellsFromChildrenFaces<IT_Pyramid5>(face, parent_cell, neighbor_cell);
        break;
      case IT_Pentaedron6:
        _populateBackFrontCellsFromChildrenFaces<IT_Pentaedron6>(face, parent_cell, neighbor_cell);
        break;
      case IT_Hexaedron8:
        _populateBackFrontCellsFromChildrenFaces<IT_Hexaedron8>(face, parent_cell, neighbor_cell);
        break;
      case IT_HemiHexa7:
        _populateBackFrontCellsFromChildrenFaces<IT_HemiHexa7>(face, parent_cell, neighbor_cell);
        break;
      case IT_HemiHexa6:
        _populateBackFrontCellsFromChildrenFaces<IT_HemiHexa6>(face, parent_cell, neighbor_cell);
        break;
      case IT_HemiHexa5:
        _populateBackFrontCellsFromChildrenFaces<IT_HemiHexa5>(face, parent_cell, neighbor_cell);
        break;
      case IT_AntiWedgeLeft6:
        _populateBackFrontCellsFromChildrenFaces<IT_AntiWedgeLeft6>(face, parent_cell, neighbor_cell);
        break;
      case IT_AntiWedgeRight6:
        _populateBackFrontCellsFromChildrenFaces<IT_AntiWedgeRight6>(face, parent_cell, neighbor_cell);
        break;
      case IT_DiTetra5:
        _populateBackFrontCellsFromChildrenFaces<IT_DiTetra5>(face, parent_cell, neighbor_cell);
        break;
      default:
        ARCANE_FATAL("Not supported refinement Item Type type={0}",neighbor_cell.type());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int typeID> void MeshRefinement::
_populateBackFrontCellsFromChildrenFaces(Face face, Cell parent_cell,
                                         Cell neighbor_cell)
{
  const ItemRefinementPatternT<typeID>& rp = getRefinementPattern<typeID>();
  for(Integer f=0;f<neighbor_cell.nbFace();f++){
    if (neighbor_cell.face(f) == face){
      Integer nb_children = neighbor_cell.nbHChildren();
      for (Integer c = 0; c < nb_children; c++){
        Cell child = neighbor_cell.hChild(c);
        Integer nb_child_faces = child.nbFace();
        for (Integer fc = 0; fc < nb_child_faces; fc++){
          if (f == rp.face_mapping(c, fc) && (rp.face_mapping_topo(c, fc))){
            Face subface = child.face(fc);
            if (subface.itemBase().flags() & ItemFlags::II_HasBackCell){
              m_face_family->addFrontCellToFace(subface, parent_cell);
            }
            else if (subface.itemBase().flags() & ItemFlags::II_HasFrontCell){
              m_face_family->addBackCellToFace(subface, parent_cell);
            }
            ARCANE_ASSERT((subface.backCell() != subface.frontCell()), ("back front cells error"));
          }
        }
      }
      break;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_checkOwner(const String& msg)
{
  // This method has been introduced to patch node owner desynchronization occuring in IFPEN applications
  info() << "----CheckOwner in " << msg;
  VariableNodeInt32 syncvariable(VariableBuildInfo(m_mesh,"SyncVarNodeOwnerContract"));
  syncvariable.fill(-1);
  bool has_owner_changed = false;
  ENUMERATE_NODE(inode,m_mesh->ownNodes()) syncvariable[inode] = inode->owner();
  VariableNodeInt32 syncvariable_copy(VariableBuildInfo(m_mesh,"SyncVarNodeOwnerContractCopy"));
  syncvariable_copy.copy(syncvariable);
  syncvariable.synchronize();
  ItemVector desync_nodes(m_mesh->nodeFamily());
  ENUMERATE_NODE(inode,m_mesh->allNodes())
  {
    if (syncvariable[inode] == -1) {
        debug(Trace::Highest) << "----- Inconsistent owner (ghost everywhere) for node with uid : "
                              << inode->uniqueId().asInt64();
        desync_nodes.addItem(*inode);
        has_owner_changed = true;
    }
    if (inode->isOwn() && (syncvariable_copy[inode] != syncvariable[inode])) {
        debug(Trace::Highest) << "----- Inconsistent owner (own everywhere) for node with uid : "
                                      << inode->uniqueId().asInt64();
        desync_nodes.addItem(*inode);
        has_owner_changed = true;
    }
  }
  // Find a new owner different from the historical one stored in node_owner_memory
  // Indeed the desync occured while changing from this historical owner to avoid an isolated item
  // Going back to this owner would make this problem of isolated owner appear again
  // 1-Get the owners of all desync nodes on every process
  // 1.1 Synchronize desync_nodes : if a node is desynchronized in a domain, all the domains must do the correction
  Int64UniqueArray desync_node_uids(desync_nodes.size());
  ENUMERATE_NODE(inode, desync_nodes) {
    desync_node_uids[inode.index()] = inode->uniqueId().asInt64();
  }
  Int64UniqueArray desync_node_uids_gather;
  m_mesh->parallelMng()->allGatherVariable(desync_node_uids.view(),desync_node_uids_gather);
  Int32UniqueArray desync_node_lids_gather(desync_node_uids_gather.size());
  m_mesh->nodeFamily()->itemsUniqueIdToLocalId(desync_node_lids_gather,desync_node_uids_gather,false);
  for (auto lid : desync_node_lids_gather) {
    if (lid == NULL_ITEM_LOCAL_ID)
      continue;
    if (std::find(desync_nodes.viewAsArray().begin(), desync_nodes.viewAsArray().end(),lid) == desync_nodes.viewAsArray().end()){
      desync_nodes.add(lid);
    }
  }
  // 1.2 Exchange the owners of the desynchronized nodes
  // each process fill an array [node1_uid, node1_owner,...nodei_uid, nodei_owner,...]
  Int64UniqueArray desync_node_owners(2*desync_nodes.size());
  ENUMERATE_NODE(inode, desync_nodes) {
    desync_node_owners[2*inode.index()  ] = inode->uniqueId().asInt64();
    desync_node_owners[2*inode.index()+1] = inode->owner();
  }
  // 1.2 gather this array on every process
  Int64UniqueArray desync_node_owners_gather;
  m_mesh->parallelMng()->allGatherVariable(desync_node_owners.view(),desync_node_owners_gather);
  // 1.3 store the information in a map <uid, Array[owner] >
  std::map<Int64,Int32SharedArray> uid_owners_map;
  for (Integer node_index = 0; node_index+1 < desync_node_owners_gather.size();) {
    uid_owners_map[desync_node_owners_gather[node_index]].add((Int32)desync_node_owners_gather[node_index+1]);
    desync_node_uids_gather.add(desync_node_owners_gather[node_index]);
    node_index+=2;
  }
  // 2 choose the unique owner of the desynchronized nodes :
  // Choose the minimum owner (same choice on each proc) different from the historical owner
  Integer new_owner = m_mesh->parallelMng()->commSize()+1;
  ENUMERATE_NODE(inode, desync_nodes) {
    for (auto owner : uid_owners_map[inode->uniqueId().asInt64()]) {
      if (owner < new_owner && owner != m_node_owner_memory[inode]) new_owner = owner;
    }
    debug(Trace::Highest) << "------ Change owner for node " << inode->uniqueId() << " from " << inode->owner() << " to " << new_owner;
    inode->mutableItemBase().setOwner(new_owner, m_mesh->parallelMng()->commRank());
    new_owner  = m_mesh->parallelMng()->commSize()+1;
  }
  // Update family if owners have changed
  bool p_has_owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, has_owner_changed);
  if (p_has_owner_changed) {
      m_mesh->nodeFamily()->notifyItemsOwnerChanged();
      m_mesh->nodeFamily()->endUpdate();
      m_mesh->nodeFamily()->computeSynchronizeInfos();
  }
  // update node memory owner
  ENUMERATE_NODE(inode,m_mesh->allNodes()) m_node_owner_memory[inode] = inode->owner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const IMesh* MeshRefinement::
getMesh() const
{
  return m_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshRefinement::
getMesh()
{
  return m_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
