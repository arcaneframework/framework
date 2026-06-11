// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshRefinement.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Manipulation of an AMR mesh.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef AMRMAXCONSISTENCYITER
#define AMRMAXCONSISTENCYITER 10
#endif

// \brief class of methods for refining unstructured meshes
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

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ACTIVATE_PERF_COUNTER
const std::string MeshRefinement::PerfCounter::m_names[MeshRefinement::PerfCounter::NbCounters] = {
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
};
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
, m_node_owner_memory(VariableBuildInfo(mesh, "NodeOwnerMemoryVar"))
{
  // \todo create a builder
  m_item_refinement = new ItemRefinement(mesh);
  m_parallel_amr_consistency = new ParallelAMRConsistency(mesh);
  m_call_back_mng = new AMRCallBackMng();
  m_need_update = true;

  ENUMERATE_NODE (inode, m_mesh->allNodes())
    m_node_owner_memory[inode] = inode->owner();

#ifdef ACTIVATE_PERF_COUNTER
  m_perf_counter.init();
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

void MeshRefinement::
clear()
{
  CHECKPERF(m_perf_counter.start(PerfCounter::CLEAR))
  m_node_finder._clear();
  m_face_finder._clear();
  CHECKPERF(m_perf_counter.stop(PerfCounter::CLEAR))
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
init()
{
  // Recalculate the max uniqueId() of nodes/cells/faces.
  CHECKPERF(m_perf_counter.start(PerfCounter::INIT))
  IParallelMng* pm = m_mesh->parallelMng();
  {
    Int64 max_node_uid = 0;
    ENUMERATE_NODE (inode, m_mesh->allNodes()) {
      const Node& node = *inode;
      const Int64 uid = node.uniqueId();
      if (uid > max_node_uid)
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
    ENUMERATE_CELL (icell, m_mesh->allCells()) {
      const Cell& cell = *icell;
      const Int64 uid = cell.uniqueId();
      const Int32 nb_hChildren = itm->nbHChildrenByItemType(cell.type());
      if (uid > max_cell_uid)
        max_cell_uid = uid;
      if (nb_hChildren > max_nb_hChildren)
        max_nb_hChildren = nb_hChildren;
    }
    if (pm->commSize() > 1) {
      m_max_cell_uid = pm->reduce(Parallel::ReduceMax, max_cell_uid);
      m_max_nb_hChildren = pm->reduce(Parallel::ReduceMax, max_nb_hChildren);
    }
    else {
      m_max_cell_uid = max_cell_uid;
      m_max_nb_hChildren = max_nb_hChildren;
    }
    info() << "CELL_UID_INFO: MY_MAX_UID=" << max_cell_uid << " GLOBAL=" << m_max_cell_uid;
    m_next_cell_uid = m_max_cell_uid + 1 + pm->commRank() * m_max_nb_hChildren;
  }

  {
    Int64 max_face_uid = 0;
    ENUMERATE_FACE (iface, m_mesh->allFaces()) {
      const Face& face = *iface;
      const Int64 uid = face.uniqueId();
      if (uid > max_face_uid)
        max_face_uid = uid;
    }

    if (pm->commSize() > 1)
      m_max_face_uid = pm->reduce(Parallel::ReduceMax, max_face_uid);
    else
      m_max_face_uid = max_face_uid;
    info() << "FACE_UID_INFO: MY_MAX_UID=" << max_face_uid << " GLOBAL=" << m_max_face_uid;
    m_next_face_uid = m_max_face_uid + 1 + pm->commRank();
  }

  m_parallel_amr_consistency->init();

  CHECKPERF(m_perf_counter.stop(PerfCounter::INIT))
}

void MeshRefinement::
_updateMaxUid(ArrayView<ItemInternal*> cells)
{
  // Recalculate the max uniqueId() of nodes/cells/faces.
  CHECKPERF(m_perf_counter.start(PerfCounter::INIT))
  IParallelMng* pm = m_mesh->parallelMng();
  ItemTypeMng* itm = m_mesh->itemTypeMng();
  {

    Int64 max_node_uid = m_max_node_uid;
    Int64 max_cell_uid = m_max_cell_uid;
    Integer max_nb_hChildren = m_max_nb_hChildren;
    Int64 max_face_uid = m_max_face_uid;

    typedef std::set<Int64> set_type;
    typedef std::pair<set_type::iterator, bool> insert_return_type;
    set_type node_list;
    set_type face_list;
    for (Integer icell = 0; icell < cells.size(); ++icell) {
      Cell cell = cells[icell];
      for (UInt32 i = 0, nc = cell.nbHChildren(); i < nc; i++) {
        Cell child = cell.hChild(i);

        //UPDATE MAX CELL UID
        const Int64 cell_uid = child.uniqueId();
        const Int32 nb_hChildren = itm->nbHChildrenByItemType(child.type());
        if (cell_uid > max_cell_uid)
          max_cell_uid = cell_uid;
        if (nb_hChildren > max_nb_hChildren)
          max_nb_hChildren = nb_hChildren;

        //UPDATE MAX NODE UID
        for (Node inode : child.nodes()) {
          const Int64 uid = inode.uniqueId();
          insert_return_type value = node_list.insert(uid);
          if (value.second) {
            if (uid > max_node_uid)
              max_node_uid = uid;
          }
        }

        //UPDATE MAX FACE UID
        for (Face iface : child.faces()) {
          const Int64 uid = iface.uniqueId();
          insert_return_type value = face_list.insert(uid);
          if (value.second) {
            if (uid > max_face_uid)
              max_face_uid = uid;
          }
        }
      }
    }

    if (pm->commSize() > 1) {
      m_max_node_uid = pm->reduce(Parallel::ReduceMax, max_node_uid);
      m_max_cell_uid = pm->reduce(Parallel::ReduceMax, max_cell_uid);
      m_max_nb_hChildren = pm->reduce(Parallel::ReduceMax, max_nb_hChildren);
      m_max_face_uid = pm->reduce(Parallel::ReduceMax, max_face_uid);
    }
    else {
      m_max_node_uid = max_node_uid;
      m_max_cell_uid = max_cell_uid;
      m_max_nb_hChildren = max_nb_hChildren;
      m_max_face_uid = max_face_uid;
    }
    m_next_node_uid = m_max_node_uid + 1 + m_mesh->parallelMng()->commRank();
    m_next_cell_uid = m_max_cell_uid + 1 + pm->commRank() * m_max_nb_hChildren;
    m_next_face_uid = m_max_face_uid + 1 + pm->commRank();
  }

  CHECKPERF(m_perf_counter.stop(PerfCounter::INIT))
}

void MeshRefinement::initMeshContainingBox()
{
  m_mesh_containing_box.init(m_mesh);
  m_node_finder.setBox(&m_mesh_containing_box);
  m_face_finder.setBox(&m_mesh_containing_box);
}

void MeshRefinement::update()
{
  init();
  initMeshContainingBox();
  m_item_refinement->initHMin();
  m_node_finder.init();
  //m_node_finder.check() ;
  m_face_finder.initFaceCenter();
  m_face_finder.init();
  //m_face_finder.check() ;
  m_parallel_amr_consistency->update();
  m_need_update = false;
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

bool MeshRefinement::
refineAndCoarsenItems(const bool maintain_level_one)
{
  CHECKPERF(m_perf_counter.start(PerfCounter::INIT))

  bool _maintain_level_one = maintain_level_one;

  // the level-one rule is the only implemented condition
  if (!maintain_level_one) {
    warning() << "Warning, level one rule is the only condition accepted for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  // We cannot yet transform a non-level-one mesh into a level-one mesh
  if (_maintain_level_one) {
    ARCANE_ASSERT((_checkLevelOne(true)), ("checkLevelOne failed"));
  }

  // Clear refinement flags from a previous step
  this->_cleanRefinementFlags();
  CHECKPERF(m_perf_counter.stop(PerfCounter::INIT))

  // Parallel consistency must come first, otherwise
  // coarsening along interfaces between processors could occasionally be
  // falsely prevented
  if (m_mesh->parallelMng()->isParallel())
    this->_makeFlagParallelConsistent();

  CHECKPERF(m_perf_counter.start(PerfCounter::CONSIST))
  // Repeat until flag matching is achieved on each processor
  Integer iter = 0;
  do {
    // Repeat until coarsening/refinement flags are locally satisfied
    bool satisfied = false;
    do {
      const bool coarsening_satisfied = this->_makeCoarseningCompatible(maintain_level_one);
      const bool refinement_satisfied = this->_makeRefinementCompatible(maintain_level_one);
      satisfied = (coarsening_satisfied && refinement_satisfied);
#ifdef ARCANE_DEBUG
      bool max_satisfied = satisfied, min_satisfied = satisfied;
      max_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, max_satisfied);
      min_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, min_satisfied);
      ARCANE_ASSERT((satisfied == max_satisfied), ("parallel max_satisfied failed"));
      ARCANE_ASSERT((satisfied == min_satisfied), ("parallel min_satisfied failed"));
#endif
    } while (!satisfied);
    ++iter;
  } while (m_mesh->parallelMng()->isParallel() && !this->_makeFlagParallelConsistent() && iter < 10);
  if (iter == AMRMAXCONSISTENCYITER)
    fatal() << " MAX CONSISTENCY ITER REACHED";
  CHECKPERF(m_perf_counter.stop(PerfCounter::CONSIST))

  // First, coarsen the flagged items.
  CHECKPERF(m_perf_counter.start(PerfCounter::COARSEN))
  const bool coarsening_changed_mesh = this->_coarsenItems();
  CHECKPERF(m_perf_counter.stop(PerfCounter::COARSEN))

  // Now, refine the flagged items. This will take
  // more memory, and possibly more than is available.
  Int64UniqueArray cells_to_refine;
  const bool refining_changed_mesh = this->_refineItems(cells_to_refine);

  // Finally, preparing the new mesh for use
  if (refining_changed_mesh || coarsening_changed_mesh) {
    bool do_compact = m_mesh->properties()->getBool("compact");
    m_mesh->properties()->setBool("compact", true); // Forcing compaction prevents from bugs when using AMR

    // Refinement
    CHECKPERF(m_perf_counter.start(PerfCounter::ENDUPDATE))
    m_mesh->modifier()->endUpdate();
    m_mesh->properties()->setBool("compact", do_compact);
    CHECKPERF(m_perf_counter.stop(PerfCounter::ENDUPDATE))

    // Coarsening
    //bool remove_ghost_children = false;
    if (coarsening_changed_mesh) {
      //remove_ghost_children=true;

      CHECKPERF(m_perf_counter.start(PerfCounter::CONTRACT))
      this->_contract();
      CHECKPERF(m_perf_counter.stop(PerfCounter::CONTRACT))

      CHECKPERF(m_perf_counter.start(PerfCounter::ENDUPDATE))
      m_mesh->properties()->setBool("compact", true); // Forcing compaction prevents from bugs when using AMR (leads to problems whith dof)
      m_mesh->modifier()->endUpdate();
      m_mesh->properties()->setBool("compact", do_compact);
      CHECKPERF(m_perf_counter.stop(PerfCounter::ENDUPDATE))
    }

    // callback to transport variables onto the new mesh
    CHECKPERF(m_perf_counter.start(PerfCounter::INTERP))
    this->_interpolateData(cells_to_refine);
    CHECKPERF(m_perf_counter.stop(PerfCounter::INTERP))
    //

    if (!coarsening_changed_mesh && m_mesh->parallelMng()->isParallel())
      this->_makeFlagParallelConsistent2();

    if (m_mesh->parallelMng()->isParallel()) {
      CHECKPERF(m_perf_counter.start(PerfCounter::PGHOST))
      m_mesh->modifier()->setDynamic(true);
      UniqueArray<Int64> ghost_cell_to_refine;
      UniqueArray<Int64> ghost_cell_to_coarsen;
      m_mesh->modifier()->updateGhostLayerFromParent(ghost_cell_to_refine,
                                                     ghost_cell_to_coarsen,
                                                     false);
      _update(ghost_cell_to_refine);
      CHECKPERF(m_perf_counter.stop(PerfCounter::PGHOST))
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
    info() << "MESH REFINEMENT PERF INFO";
    m_perf_counter.printInfo(info().file());
    info() << "NODE FINDER PERF INFO";
    m_node_finder.getPerfCounter().printInfo(info().file());
    info() << "FACE FINDER PERF INFO";
    m_face_finder.getPerfCounter().printInfo(info().file());
    info() << "PARALLEL AMR CONSISTENCY PERF INFO";
    m_parallel_amr_consistency->getPerfCounter().printInfo(info().file());
#endif
    return true;
  }
  // If there were no changes in the mesh
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
coarsenItems(const bool maintain_level_one)
{

  bool _maintain_level_one = maintain_level_one;

  // the level-one rule is the only implemented condition
  if (!maintain_level_one) {
    warning() << "Warning, level one rule is the only condition accepted for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  // We cannot yet transform a non-level-one mesh into a level-one mesh
  if (_maintain_level_one) {
    ARCANE_ASSERT((_checkLevelOne(true)), ("check_level_one failed"));
  }

  // Cleaning up refinement flags from the previous step
  this->_cleanRefinementFlags();

  // Parallel consistency must come first, otherwise the coarsening
  // along interfaces between processors could occasionally be
  // falsely prevented
  if (m_mesh->parallelMng()->isParallel())
    this->_makeFlagParallelConsistent();

  // Repeat until the flag change matches on every processor
  do {
    // Repeat until the flags match locally.
    bool satisfied = false;
    do {
      const bool coarsening_satisfied = this->_makeCoarseningCompatible(maintain_level_one);
      satisfied = coarsening_satisfied;
#ifdef ARCANE_DEBUG
      bool max_satisfied = satisfied, min_satisfied = satisfied;
      max_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, max_satisfied);
      min_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, min_satisfied);
      ARCANE_ASSERT((satisfied == max_satisfied), ("parallel max_satisfied failed"));
      ARCANE_ASSERT((satisfied == min_satisfied), ("parallel min_satisfied failed"));
#endif
    } while (!satisfied);
  } while (m_mesh->parallelMng()->isParallel() && !this->_makeFlagParallelConsistent());

  // Coarsen the flagged items.
  const bool mesh_changed = this->_coarsenItems();

  //if (_maintain_level_one)
  //ARCANE_ASSERT( (checkLevelOne(true)),("checkLevelOne failed"));
  //ARCANE_ASSERT( (this->makeCoarseningCompatible(maintain_level_one)), ("make_coarsening_comptaible failed"));

  // Finally, preparing the new mesh for use
  if (mesh_changed) {
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
  // Cleaning up refinement flags from the previous step
  this->_cleanRefinementFlags();

  UniqueArray<Int32> to_coarse;
  //UniqueArray<Int64> d_to_coarse_uid;

  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
    Cell cell = *icell;
    if (cell.mutableItemBase().flags() & ItemFlags::II_Coarsen) {
      // We cannot coarsen level-0 cells.
      if (cell.level() == 0) {
        ARCANE_FATAL("Cannot coarse level-0 cell");
      }
      Cell parent = cell.hParent();

      // TODO AH: To perform multi-level coarsening at once,
      // the II_Inactive flag must be removed (for the FaceFamily::removeCellFromFace() method).
      if (update_parent_flag) {
        parent.mutableItemBase().addFlags(ItemFlags::II_JustCoarsened);
        parent.mutableItemBase().removeFlags(ItemFlags::II_Inactive);
        parent.mutableItemBase().removeFlags(ItemFlags::II_CoarsenInactive);
      }

      // For a level n-1 cell, if one of its child cells must be coarsened,
      // then all its child cells must be coarsened.
      for (Integer i = 0; i < parent.nbHChildren(); ++i) {
        Cell child = parent.hChild(i);
        if (!(child.mutableItemBase().flags() & ItemFlags::II_Coarsen)) {
          ARCANE_FATAL("Parent cannot have children with coarse flag and children without coarse flag -- Parent uid: {0} -- Child uid: {1}", parent.uniqueId(), child.uniqueId());
        }
      }

      // For now, it is impossible to coarsen multiple levels at once.
      // TODO AH: The FaceReorienter::checkAndChangeOrientationAMR() method will check a
      // face that should be deleted, see why.
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
          // If the face is on the boundary, it will be deleted.
          if (other_cell.null()) { // && !has_ghost_layer) {
            continue;
            //needed_cell.add(face.uniqueId());
          }
          if (other_cell.level() != cell.level()) {
            //warning() << "Bad connectivity";
            continue;
          }
          // If both cells are going to be deleted, the face will be deleted.
          if (other_cell.mutableItemBase().flags() & ItemFlags::II_Coarsen) {
            continue;
          }
          // If the adjacent cell is refined, we will have more than one level difference.
          if (other_cell.nbHChildren() != 0) { // && !(other_cell.mutableItemBase().flags() & ItemFlags::II_Coarsen)) { // Impossible to coarsen multiple levels.
            ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell.uniqueId(), other_cell.uniqueId());
          }
          // If the adjacent cell is not ours, it takes the ownership of the adjacent cell.
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

          // Will the node be deleted?
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

          // Will the node need to change owner?
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
    // If materials are active, material recalculation must be forced because the cell groups
    // have been modified and thus the list of constituents as well
    Materials::IMeshMaterialMng* mm = Materials::IMeshMaterialMng::getReference(m_mesh, false);
    if (mm)
      mm->forceRecompute();
  }

  m_mesh->modifier()->updateGhostLayerFromParent(ghost_cell_to_refine, ghost_cell_to_coarsen, true);

  return m_mesh->parallelMng()->reduce(Parallel::ReduceMax, (!to_coarse.empty()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
refineItems(const bool maintain_level_one)
{

  bool _maintain_level_one = maintain_level_one;

  // the level-one rule is the only implemented condition
  if (!maintain_level_one) {
    warning() << "Warning, level one rule is the only condition accepted for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  if (_maintain_level_one) {
    ARCANE_ASSERT((_checkLevelOne(true)), ("check_level_one failed"));
  }
  // Cleaning up refinement flags from the previous step
  this->_cleanRefinementFlags();

  // Parallel consistency must come first, otherwise the coarsening
  // along interfaces between processors could occasionally be
  // falsely prevented
  if (m_mesh->parallelMng()->isParallel())
    this->_makeFlagParallelConsistent();

  // Repeat until the flag change matches on every processor
  do {
    // Repeat until the flags match locally.
    bool satisfied = false;
    do {
      const bool refinement_satisfied = this->_makeRefinementCompatible(maintain_level_one);
      satisfied = refinement_satisfied;
#ifdef ARCANE_DEBUG
      bool max_satisfied = satisfied, min_satisfied = satisfied;
      max_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, max_satisfied);
      min_satisfied = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, min_satisfied);
      ARCANE_ASSERT((satisfied == max_satisfied), ("parallel max_satisfied failed"));
      ARCANE_ASSERT((satisfied == min_satisfied), ("parallel min_satisfied failed"));
#endif
    } while (!satisfied);
  } while (m_mesh->parallelMng()->isParallel() && !this->_makeFlagParallelConsistent());

  // Now, refine the flagged items. This will take
  // more memory, and possibly more than is available.
  Int64UniqueArray cells_to_refine;
  const bool mesh_changed = this->_refineItems(cells_to_refine);

  // Finally, preparing the new mesh for use
  if (mesh_changed) {
    // update
    bool do_compact = m_mesh->properties()->getBool("compact");
    m_mesh->properties()->setBool("compact", true); // Forcing compaction prevents from bugs when using AMR
    m_mesh->modifier()->endUpdate();
    m_mesh->properties()->setBool("compact", do_compact);

    // callback to transport variables onto the new mesh
    this->_interpolateData(cells_to_refine);

    // ghost update
    m_mesh->modifier()->setDynamic(true);
    UniqueArray<Int64> ghost_cell_to_refine;
    UniqueArray<Int64> ghost_cell_to_coarsen;
    m_mesh->modifier()->updateGhostLayerFromParent(ghost_cell_to_refine, ghost_cell_to_coarsen, false);
    _update(ghost_cell_to_refine);
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
  // Refine n times
  // FIXME - this should not work if n>1 and the mesh
  // is already attached to the system of equations to be solved
  for (Integer rstep = 0; rstep < n; rstep++) {
    // Cleaning up refinement flags
    this->_cleanRefinementFlags();

    // iterate only over active cells
    // Flag all active items for refinement
    ENUMERATE_CELL (icell, m_mesh->ownActiveCells()) {
      Cell cell = *icell;
      _setRefineFlags(cell);
    }
    // Refine all the items we have flagged.
    Int64UniqueArray cells_to_refine;
    this->_refineItems(cells_to_refine);
    warning() << "ATTENTION: No Data Projection with this method!";
  }

  bool do_compact = m_mesh->properties()->getBool("compact");
  m_mesh->properties()->setBool("compact", true); // Forcing compaction prevents from bugs when using AMR
  m_mesh->modifier()->endUpdate();
  m_mesh->properties()->setBool("compact", do_compact);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
uniformlyCoarsen(Integer n)
{
  // Coarsen n times
  for (Integer rstep = 0; rstep < n; rstep++) {
    // Cleaning refinement flags
    this->_cleanRefinementFlags();

    // Iterate only over active cells
    // Flag all active items for coarsening
    ENUMERATE_CELL (icell, m_mesh->ownActiveCells()) {
      Cell cell = *icell;
      _setCoarseFlags(cell);
      if (cell.nbHParent() != 0) {
        cell.hParent().mutableItemBase().addFlags(ItemFlags::II_CoarsenInactive);
      }
    }
    // Coarsen all items we just flagged.
    this->_coarsenItems();
    warning() << "ATTENTION: No Data Restriction with this method!";
  }

  // Finally, preparation of the new mesh for use
  bool do_compact = m_mesh->properties()->getBool("compact");
  m_mesh->properties()->setBool("compact", true);
  m_mesh->modifier()->endUpdate();
  m_mesh->properties()->setBool("compact", do_compact);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MeshRefinement::
findOrAddNodeUid(const Real3& p, const Real& tol)
{
  //debug() << "addNode()";

  // Return the node if it already exists
  Int64 uid = m_node_finder.find(p, tol);
  if (uid != NULL_ITEM_ID) {
    //          debug() << "addNode() done";
    return uid;
  }
  // Add the node to the map.
  Int64 new_uid = m_next_node_uid;
  m_node_finder.insert(p, new_uid, tol);
  m_next_node_uid += m_mesh->parallelMng()->commSize() + 1;
  // Return the uid of the new node
  //  debug() << "addNode() done";
  return new_uid;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
Int64 MeshRefinement::
findOrAddFaceUid(const Real3& p, const Real& tol, bool& is_added)
{
  //debug() << "findOrAddFaceUid()";

  // Return the face if it already exists
  Int64 uid = m_face_finder.find(p, tol);
  if (uid != NULL_ITEM_ID) {
    //          debug() << "findOrAddFaceUid() done";
    is_added = false;
    return uid;
  }
  // Add the face to the map.
  is_added = true;
  Int64 new_uid = m_next_face_uid;
  m_face_finder.insert(p, new_uid, tol);
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

void MeshRefinement::
_updateLocalityMap()
{
  //jmg this->init(); // \todo not necessary to call on every update
  //m_node_finder.init();
  m_node_finder.check();
  //m_face_finder.init();
  m_face_finder.check();
  debug() << "[MeshRefinement::updateLocalityMap] done";
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_updateLocalityMap2()
{
  //this->init(); // \todo not necessary to call on every update
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
  // Iterate only over active cells
  ENUMERATE_CELL (icell, m_mesh->allActiveCells()) {
    Cell cell = *icell;
    for (Face face : cell.faces()) {
      if (face.nbCell() != 2)
        continue;
      Cell back_cell = face.backCell();
      Cell front_cell = face.frontCell();

      // We choose the other cell on the face side
      Cell neighbor = (back_cell == cell) ? front_cell : back_cell;
      if (neighbor.null() || !neighbor.isActive() || !(neighbor.owner() == sid))
        continue;
      //debug() << "#### " << ineighbor->uniqueId() << " " << ineighbor->level() << " " << cell.level();
      if ((neighbor.level() + 1 < cell.level())) {
        failure = true;
        break;
      }
    }
  }

  // If one processor fails, we fail globally
  failure = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, failure);

  if (failure) {
    // We did not pass the level-one test, so arcane_assert
    // based on the input boolean.
    if (arcane_assert_pass)
      throw FatalErrorException(A_FUNCINFO, "checkLevelOne failed");
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

  // Search for local flags
  // Iterate only over active cells
  ENUMERATE_CELL (icell, m_mesh->ownActiveCells()) {
    const Cell cell = *icell;
    const Integer f = cell.itemBase().flags();
    if ((f & ItemFlags::II_Refine) | (f & ItemFlags::II_Coarsen)) {
      found_flag = true;
      break;
    }
  }
  // If we find a flag on any processor, it counts
  found_flag = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, found_flag);
  if (found_flag) {
    // We did not pass the "items are unflagged" test,
    // thus arcane_assert the non-value of arcane_assert_pass
    if (arcane_assert_pass)
      throw FatalErrorException(A_FUNCINFO, "checkUnflagged failed");
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

  CHECKPERF(m_perf_counter.start(PerfCounter::PCONSIST))
  debug() << "makeFlagsParallelConsistent() begin";
  bool parallel_consistent = true;
  VariableCellInteger flag_cells_consistent(VariableBuildInfo(m_mesh, "FlagCellsConsistent"));
  UniqueArray<Item> ghost_cells;
  ghost_cells.reserve(m_mesh->allCells().size() - m_mesh->ownCells().size());
  ENUMERATE_CELL (icell, m_mesh->allCells()) {
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
  for (Integer icell = 0, nb_cell = ghost_cells.size(); icell < nb_cell; ++icell) {
    Item iitem = ghost_cells[icell];
    Integer f = iitem.itemBase().flags();

    //if(iitem->owner() != sid)
    {
      // it is possible that the ghost flags are (temporarily) more
      // conservative than our own flags, such as when a refinement of one of our
      // cells on the remote processor is dictated by a refinement of one of our cells
      const Integer g = flag_cells_consistent[Cell(iitem)];
      if ((g & ItemFlags::II_Refine) && !(f & ItemFlags::II_Refine)) {
        f |= ItemFlags::II_Refine;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_Coarsen) && !(f & ItemFlags::II_Coarsen)) {
        f |= ItemFlags::II_Coarsen;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_JustCoarsened) && !(f & ItemFlags::II_JustCoarsened)) {
        f |= ItemFlags::II_JustCoarsened;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_JustRefined) && !(f & ItemFlags::II_JustRefined)) {
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
  // If we are not consistent on every processor then
  // we are not globally consistent
  parallel_consistent = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, parallel_consistent);
  debug() << "makeFlagsParallelConsistent() end -- parallel_consistent : " << parallel_consistent;

  CHECKPERF(m_perf_counter.stop(PerfCounter::PCONSIST))
  return parallel_consistent;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
bool MeshRefinement::
_makeFlagParallelConsistent2()
{
  if (!m_mesh->parallelMng()->isParallel())
    return true;

  CHECKPERF(m_perf_counter.start(PerfCounter::PCONSIST2))
  debug() << "makeFlagsParallelConsistent2() begin";
  bool parallel_consistent = true;
  VariableCellInteger flag_cells_consistent(VariableBuildInfo(m_mesh, "FlagCellsConsistent"));
  UniqueArray<Item> ghost_cells;
  ghost_cells.reserve(m_mesh->allCells().size() - m_mesh->ownCells().size());
  ENUMERATE_CELL (icell, m_mesh->allCells()) {
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
  for (Integer icell = 0, nb_cell = ghost_cells.size(); icell < nb_cell; ++icell) {
    //const Cell& cell = *icell;
    //ItemInternal * iitem = cell.internal();
    //Integer f = iitem->flags();
    Item iitem = ghost_cells[icell];
    Integer f = iitem.itemBase().flags();

    //if(iitem->owner() != sid)
    {
      Integer g = flag_cells_consistent[Cell(iitem)];
      if ((g & ItemFlags::II_JustCoarsened) && !(f & ItemFlags::II_JustCoarsened)) {
        f |= ItemFlags::II_JustCoarsened;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_JustRefined) && !(f & ItemFlags::II_JustRefined)) {
        f |= ItemFlags::II_JustRefined;
        iitem.mutableItemBase().setFlags(f);
        parallel_consistent = false;
      }
      else if ((g & ItemFlags::II_Inactive) && !(f & ItemFlags::II_Inactive)) {
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
  // If we are not consistent on every processor then
  // we are not globally consistent
  parallel_consistent = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, parallel_consistent);
  debug() << "makeFlagsParallelConsistent2() end";

  CHECKPERF(m_perf_counter.stop(PerfCounter::PCONSIST2))
  return parallel_consistent;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_makeCoarseningCompatible(const bool maintain_level_one)
{

  debug() << "makeCoarseningCompatible() begin";

  bool _maintain_level_one = maintain_level_one;

  // the level-one rule is the only implemented condition
  if (!maintain_level_one) {
    warning() << "Warning, level one rule is the only condition accepted for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  // unless we encounter a specific situation, the level-one rule
  // will be satisfied after executing this loop just once
  bool level_one_satisfied = true;

  // unless we encounter a specific situation, we will be compatible
  // with all chosen refinement flags
  bool compatible_with_refinement = true;

  // Find the maximum level in the mesh
  Integer max_level = 0;

  // first we look at all active level 0 items. Since it makes no sense to
  // coarsen them, we must therefore remove their coarsening flags if
  // they are already positioned.
  // Iterate only over active cells
  ENUMERATE_CELL (icell, m_mesh->allActiveCells()) {
    const Cell cell = *icell;
    max_level = std::max(max_level, cell.level());

    Integer f = cell.itemBase().flags();
    if ((cell.level() == 0) && (f & ItemFlags::II_Coarsen)) {
      f &= ~ItemFlags::II_Coarsen;
      f |= ItemFlags::II_DoNothing;
      cell.mutableItemBase().setFlags(f);
    }
  }
  // If there are no items to refine on this processor then
  // there is no work for us
  if (max_level == 0) {
    debug() << "makeCoarseningCompatible() done";

    // however, it remains to check with the other processors
    compatible_with_refinement = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, compatible_with_refinement);

    return compatible_with_refinement;
  }
  // Loop over all active items. If an item is marked
  // for coarsening, we check its neighbors. If one of its neighbors
  // is marked for refinement and is at the same level, then there is a
  // conflict. By convention, refinement wins, so we unmark the item for
  // coarsening. Level-one would be violated in this case, so we must re-execute
  // the loop.
  const Integer sid = m_mesh->parallelMng()->commRank();
  if (_maintain_level_one) {

  repeat:
    level_one_satisfied = true;

    do {
      level_one_satisfied = true;
      // iterate only over active cells
      ENUMERATE_CELL (icell, m_mesh->ownActiveCells()) {
        Cell cell = *icell;
        //ItemInternal* iitem = cell.internal();
        bool my_flag_changed = false;
        Integer f = cell.itemBase().flags();
        if (f & ItemFlags::II_Coarsen) { // If the item is active and the coarsening flag is set
          const Int32 my_level = cell.level();
          for (Face face : cell.faces()) {
            if (face.nbCell() != 2)
              continue;
            Cell back_cell = face.backCell();
            Cell front_cell = face.frontCell();

            // We choose the other cell on the side of the face
            Cell neighbor = (back_cell == cell) ? front_cell : back_cell;
            //const ItemInternal* ineighbor = neighbor.internal();
            //if (ineighbor->owner() == sub_domain_id)   // I have a neighbor here

            {
              if (neighbor.isActive()) // and is active
              {
                if ((neighbor.level() == my_level) &&
                    (neighbor.itemBase().flags() & ItemFlags::II_Refine)) { // the neighbor is at my level and wants to be refined
                  f &= ~ItemFlags::II_Coarsen;
                  f |= ItemFlags::II_DoNothing;
                  cell.mutableItemBase().setFlags(f);
                  my_flag_changed = true;
                  break;
                }
              }
              else {
                // I have a neighbor and it is not active. This means it has children.
                // while it may be possible to coarsen me if all children
                // of this item want to be coarsened, it is impossible to know at this stage.
                // We forget it for now. This can be achieved in two steps.
                f &= ~ItemFlags::II_Coarsen;
                f |= ItemFlags::II_DoNothing;
                cell.mutableItemBase().setFlags(f);
                my_flag_changed = true;
                break;
              }
            }
          }
        }

        // if the flag of the current cell has changed, we have not
        // satisfied the level one rule.
        if (my_flag_changed)
          level_one_satisfied = false;

        // Furthermore, if it has non-local neighbors, and
        // we are not in sequential mode, then we must subsequently
        // return compatible_with_refinement= false, because
        // our change must be propagated to neighboring processors
        if (my_flag_changed && m_mesh->parallelMng()->isParallel())
          for (Face face : cell.faces()) {
            if (face.nbCell() != 2)
              continue;
            Cell back_cell = face.backCell();
            Cell front_cell = face.frontCell();

            // We choose the other cell on the side of the face
            Cell neighbor = (back_cell == cell) ? front_cell : back_cell;
            //ItemInternal* ineighbor = neighbor.internal();
            if (neighbor.owner() != sid) { // I have a neighbor here
              compatible_with_refinement = false;
              break;
            }
            // TODO FIXME - for non level-1 meshes we must
            // test all descendants
            if (neighbor.hasHChildren())
              for (Integer c = 0; c != neighbor.nbHChildren(); ++c)
                if (neighbor.hChild(c).owner() != sid) {
                  compatible_with_refinement = false;
                  break;
                }
          }
      }
    } while (!level_one_satisfied);

  } // end if (_maintain_level_one)

  // afterwards, we look at all ancestor items.
  // if there is a parent item with all its children
  // wanting to be coarsened, then the item is a candidate
  // for coarsening. If all children do not
  // want to be coarsened, then all need to have their
  // coarsening flag cleared.
  for (int level = (max_level); level >= 0; level--) {
    // iterate over cells level by level
    ENUMERATE_CELL (icell, m_mesh->ownLevelCells(level)) {
      const Cell cell = *icell;
      //ItemInternal* iitem = cell.internal();
      if (cell.isAncestor()) {
        // at this point the item has not been eliminated
        // as a candidate for coarsening
        bool is_a_candidate = true;
        bool found_remote_child = false;

        for (Integer c = 0; c < cell.nbHChildren(); c++) {
          Cell child = cell.hChild(c);
          if (child.owner() != sid)
            found_remote_child = true;
          else if (!(child.itemBase().flags() & ItemFlags::II_Coarsen) || !child.isActive())
            is_a_candidate = false;
        }

        if (!is_a_candidate && !found_remote_child) {
          cell.mutableItemBase().addFlags(ItemFlags::II_Inactive);
          for (Integer c = 0; c < cell.nbHChildren(); c++) {
            Cell child = cell.hChild(c);
            if (child.owner() != sid)
              continue;
            if (child.itemBase().flags() & ItemFlags::II_Coarsen) {
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
  if (!level_one_satisfied && _maintain_level_one)
    goto repeat;

  // If all children of a parent are marked for coarsening
  // Then mark the parent so it can kill its children.
  ENUMERATE_CELL (icell, m_mesh->ownCells()) {
    const Cell cell = *icell;
    //ItemInternal* iitem = cell.internal();
    if (cell.isAncestor()) {
      // Assume that all children are local and marked for
      // coarsening and thus look for a contradiction
      bool all_children_flagged_for_coarsening = true;
      bool found_remote_child = false;

      for (Integer c = 0; c < cell.nbHChildren(); c++) {
        Cell child = cell.hChild(c);
        if (child.owner() != sid)
          found_remote_child = true;
        else if (!(child.itemBase().flags() & ItemFlags::II_Coarsen))
          all_children_flagged_for_coarsening = false;
      }
      Integer f = cell.itemBase().flags();
      f &= ~ItemFlags::II_CoarsenInactive;
      if (!found_remote_child && all_children_flagged_for_coarsening) {
        f |= ItemFlags::II_CoarsenInactive;
        cell.mutableItemBase().setFlags(f);
      }
      else if (!found_remote_child) {
        f |= ItemFlags::II_Inactive;
        cell.mutableItemBase().setFlags(f);
      }
    }
  }

  debug() << "makeCoarseningCompatible() done";

  // If we are not compatible on one processor, we are not compatible globally
  compatible_with_refinement = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, compatible_with_refinement);

  return compatible_with_refinement;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_makeRefinementCompatible(const bool maintain_level_one)
{

  debug() << "makeRefinementCompatible() begin";

  bool _maintain_level_one = maintain_level_one;

  // the level-one rule is the only condition implemented
  if (!maintain_level_one) {
    warning() << "Warning, level one rule is the only condition accepted now for AMR!";
  }
  else
    _maintain_level_one = m_face_level_mismatch_limit;

  // unless we encounter a specific situation, the level-one rule
  // will be satisfied after running this loop just once
  bool level_one_satisfied = true;

  // unless we encounter a specific situation, we will be compatible
  // with all chosen coarsening flags
  bool compatible_with_coarsening = true;

  // this loop enforces the level-1 rule. We should only
  // execute it if the user actually wants level-1 to be satisfied!
  Integer sid = m_mesh->parallelMng()->commRank();
  if (_maintain_level_one) {
    do {
      level_one_satisfied = true;
      // iterate only over active cells
      ENUMERATE_CELL (icell, m_mesh->allActiveCells()) {
        const Cell cell = *icell;
        //ItemInternal* iitem = cell.internal();
        if (cell.itemBase().flags() & ItemFlags::II_Refine) { // If the item is active and the refinement flag is set
          const Int32 my_level = cell.level();
          bool refinable = true;
          //check if refinable
          for (Face face : cell.faces()) {
            if (face.nbCell() != 2)
              continue;
            Cell back_cell = face.backCell();
            Cell front_cell = face.frontCell();

            // We choose the other cell on the side of the face
            Cell neighbor = (back_cell == cell) ? front_cell : back_cell;
            //ItemInternal* ineighbor = neighbor.internal();
            //if (ineighbor->isActive() && ineighbor->owner() == sid)// I have a neighbor here and it is active
            if (neighbor.isActive()) { // I have a neighbor here and it is active
              // Case 2: The neighbor is one level below mine.
              // The neighbor must be refined to satisfy
              // the level-1 rule, regardless of whether it
              // was originally marked for refinement. If it
              // was not already flagged, we must repeat
              // this process.
              Integer f = neighbor.itemBase().flags();
              if (((neighbor.level() + 1) == my_level) &&
                  (f & ItemFlags::II_UserMark1)) {
                refinable = false;
                Integer my_f = cell.itemBase().flags();
                my_f &= ~ItemFlags::II_Refine;
                cell.mutableItemBase().setFlags(my_f);
                break;
              }
            }
          }
          if (refinable)
            for (Face face : cell.faces()) {
              if (face.nbCell() != 2)
                continue;
              Cell back_cell = face.backCell();
              Cell front_cell = face.frontCell();

              // We choose the other cell on the side of the face
              Cell neighbor = (back_cell == cell) ? front_cell : back_cell;
              //ItemInternal* ineighbor = neighbor.internal();
              if (neighbor.isActive() && neighbor.owner() == sid) { // I have a neighbor here and it is active

                // Case 1: The neighbor is at the same level as me.
                // 1a: The neighbor will be refined -> NO PROBLEM
                // 1b: The neighbor will not be refined -> NO PROBLEM
                // 1c: The neighbor already wants to be refined -> PROBLEM
                if (neighbor.level() == my_level) {
                  Integer f = neighbor.itemBase().flags();
                  if (f & ItemFlags::II_Coarsen) {
                    f &= ~ItemFlags::II_Coarsen;
                    f |= ItemFlags::II_DoNothing;
                    neighbor.mutableItemBase().setFlags(f);
                    if (neighbor.nbHParent() != 0) {
                      neighbor.hParent().mutableItemBase().addFlags(ItemFlags::II_Inactive);
                    }
                    compatible_with_coarsening = false;
                    level_one_satisfied = false;
                  }
                }

                // Case 2: The neighbor is one level below mine.
                // The neighbor must be refined to satisfy
                // the level-1 rule, regardless of whether it
                // was originally marked for refinement. If it
                // was not already flagged, we must repeat
                // this process.

                else if ((neighbor.level() + 1) == my_level) {
                  Integer f = neighbor.itemBase().flags();
                  if (!(f & ItemFlags::II_Refine)) {
                    f &= ~ItemFlags::II_Coarsen;
                    f |= ItemFlags::II_Refine;
                    neighbor.mutableItemBase().setFlags(f);
                    if (neighbor.nbHParent() != 0) {
                      neighbor.hParent().mutableItemBase().addFlags(ItemFlags::II_Inactive);
                    }
                    compatible_with_coarsening = false;
                    level_one_satisfied = false;
                  }
                }
#ifdef ARCANE_DEBUG
                // Check. We should never enter a
                // case where our neighbor is more than one level away.

                else if ((neighbor.level() + 1) < my_level) {
                  fatal() << "a neighbor is more than one level away";
                }

                // Note that the only other possibility is that
                // the neighbor has already been refined, in this case it is not
                // active and we should never fall here.

                else {
                  fatal() << "serious problem: we should never get here";
                }
#endif
              }
            }
        }
      }
    } while (!level_one_satisfied);
  } // end if (_maintain_level_one)

  // If we are not compatible on one processor, we are not compatible globally
  compatible_with_coarsening = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, compatible_with_coarsening);

  debug() << "makeRefinementCompatible() done";

  return compatible_with_coarsening;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_coarsenItems()
{
  debug() << "[MeshRefinement::_coarsenItems] begin" << m_mesh->allNodes().size();
  // Flag indicating if this call actually changes the mesh
  bool mesh_changed = false;

  // iterate over all cells
  // Int32UniqueArray cell_to_detach;
  ENUMERATE_CELL (icell, m_mesh->ownCells()) {
    Cell cell = *icell;
    Cell iitem = cell;
    // active items flagged for coarsening will not be
    // removed until contraction via MeshRefinement::contract()

    if (cell.itemBase().flags() & ItemFlags::II_Coarsen) {
      // Whoops? no level-0 item should be both active
      // and flagged for coarsening.
      ARCANE_ASSERT((cell.level() != 0), ("no level-0 element should be active and flagged for coarsening"));

      // TODO Remove this item from any neighborhood list
      // pointing to it.
      // FIXME at IFP, we use the REMOVE_UID_ON_DETACH macro by default which deletes the CELL UID
      // in the cell_uid map, so we cannot use detachCell by default. For now
      // we use the MeshRefinement::contract() method after updating the variables
      // cell_to_detach.add(iitem->localId());

      //cells_to_remove.add(cell);
      // TODO optimization of unused uids.
      // m_unused_items.push_back (uid);

      // Do not destroy the item until MeshRefinement::contract()
      // m_mesh->modifier()->removeCells(iitem->localId());

      // The mesh has certainly changed
      mesh_changed = true;
    }
    else if (cell.itemBase().flags() & ItemFlags::II_CoarsenInactive) {
      switch (cell.type()) {
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
        ARCANE_FATAL("Not supported refinement Item Type type={0}", iitem.type());
      }
      ARCANE_ASSERT(cell.isActive(), ("cell_active failed"));

      // The mesh has certainly changed
      mesh_changed = true;
    }
  }
  // TODO
  // m_mesh->modifier->detachCells(cell_to_detach);

  // If the mesh changed on one processor, then it changed globally
  mesh_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, mesh_changed);
  // And maybe we need to update the entities reflecting the change
  //if (mesh_changed)
  // \todo compact and update max_uids in parallel

  // if a cell is derefined elsewhere, the nodes attached to this cell
  // must be updated. This is handled in endUpdate()

  debug() << "[MeshRefinement::_coarsenItems()] done " << m_mesh->allNodes().size();

  return mesh_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshRefinement::
_refineItems(Int64Array& cell_to_refine_uids)
{
  // Updating m_node_finder, m_face_finder will allow the mesh
  // to be globally consistent (uids consistency).
  debug() << "[MeshRefinement::_refineItems]" << m_mesh->allNodes().size();
#ifdef ARCANE_DEBUG
  m_node_finder.check();
  m_face_finder.check();
#endif
  CHECKPERF(m_perf_counter.start(PerfCounter::REFINE))
  m_face_finder.clearNewUids();
  // Iterate over the items, count the items
  // flagged for refinement.
  //Integer nb_cell_flagged = 0;
  UniqueArray<Cell> cell_to_refine_internals;
  ENUMERATE_CELL (icell, m_mesh->ownCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Refine) {
      cell_to_refine_uids.add(cell.uniqueId());
      cell_to_refine_internals.add(cell);
    }
  }
  debug() << "[MeshRefinement::_refineItems] " << cell_to_refine_uids.size() << " flagged cells for refinement";

  // Build a local vector of the marked items
  // for refinement.
  /*
   local_copy_of_cells.reserve(nb_cell_flagged);

   ENUMERATE_CELL(icell,m_mesh->ownCells()){
   Cell cell = *icell;
   ItemInternal* iitem = cell.internal();
   if(iitem->flags() & ItemFlags::II_Refine)
   local_copy_of_cells.add(iitem);
   }
   */
  // Now, iterate over the local copies and refine each item.
  const Int32 i_size = cell_to_refine_internals.size();
  for (Integer e = 0; e != i_size; ++e) {
    Cell iitem = cell_to_refine_internals[e];
    //debug()<<"\t[MeshRefinement::_refineItems] focus on cell "<<iitem->uniqueId();
    switch (iitem.type()) {
    case IT_Quad4:
      m_item_refinement->refineOneCell<IT_Quad4>(iitem, *this);
      break;
    case IT_Tetraedron4:
      m_item_refinement->refineOneCell<IT_Tetraedron4>(iitem, *this);
      break;
    case IT_Pyramid5:
      m_item_refinement->refineOneCell<IT_Pyramid5>(iitem, *this);
      break;
    case IT_Pentaedron6:
      m_item_refinement->refineOneCell<IT_Pentaedron6>(iitem, *this);
      break;
    case IT_Hexaedron8:
      m_item_refinement->refineOneCell<IT_Hexaedron8>(iitem, *this);
      break;
    case IT_HemiHexa7:
      m_item_refinement->refineOneCell<IT_HemiHexa7>(iitem, *this);
      break;
    case IT_HemiHexa6:
      m_item_refinement->refineOneCell<IT_HemiHexa6>(iitem, *this);
      break;
    case IT_HemiHexa5:
      m_item_refinement->refineOneCell<IT_HemiHexa5>(iitem, *this);
      break;
    case IT_AntiWedgeLeft6:
      m_item_refinement->refineOneCell<IT_AntiWedgeLeft6>(iitem, *this);
      break;
    case IT_AntiWedgeRight6:
      m_item_refinement->refineOneCell<IT_AntiWedgeRight6>(iitem, *this);
      break;
    case IT_DiTetra5:
      m_item_refinement->refineOneCell<IT_DiTetra5>(iitem, *this);
      break;
    default:
      ARCANE_FATAL("Not supported refinement Item Type type={0}", iitem.type());
    }
  }

  // The mesh changes if items are refined
  bool mesh_changed = !(i_size == 0);

  // If the mesh changes on one processor, it changes globally
  mesh_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, mesh_changed);

  // And we need to update the number of ids
  if (mesh_changed) {
    for (Integer e = 0; e != i_size; ++e) {
      Cell i_hParent_cell = cell_to_refine_internals[e];
      populateBackFrontCellsFromParentFaces(i_hParent_cell);
    }
  }
  CHECKPERF(m_perf_counter.stop(PerfCounter::REFINE))
  if (mesh_changed && m_mesh->parallelMng()->isParallel()) {
    CHECKPERF(m_perf_counter.start(PerfCounter::PGCONSIST))
#ifdef ARCANE_DEBUG
    m_node_finder.check2();
    m_face_finder.check2();
#endif

    // Nodes and faces parallel consistency
    m_parallel_amr_consistency->makeNewItemsConsistent(m_node_finder, m_face_finder);
    CHECKPERF(m_perf_counter.stop(PerfCounter::PGCONSIST))
  }

  debug() << "[MeshRefinement::_refineItems] done" << m_mesh->allNodes().size();

  return mesh_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_cleanRefinementFlags()
{
  // Cleanup of refinement flags from a previous step
  ENUMERATE_CELL (icell, m_mesh->allCells()) {
    Cell cell = *icell;
    auto mutable_cell = cell.mutableItemBase();
    Integer f = mutable_cell.flags();
    if (cell.isActive()) {
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
    else {
      f |= ItemFlags::II_Inactive;
      mutable_cell.setFlags(f);
    }
    // This could be left from the last step
    if (f & ItemFlags::II_JustRefined) {
      f &= ~ItemFlags::II_JustRefined;
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
    if (f & ItemFlags::II_JustCoarsened) {
      f &= ~ItemFlags::II_JustCoarsened;
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
    if (f & ItemFlags::II_JustAdded) {
      f &= ~ItemFlags::II_JustAdded;
      f |= ItemFlags::II_DoNothing;
      mutable_cell.setFlags(f);
    }
    if (f & ItemFlags::II_CoarsenInactive) {
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

  if (arcaneIsDebug()) {
    cells_map.eachItem([&](impl::ItemBase item) {
      if (item.isOwn())
        // a cell is either active, subactive, or ancestor
        ARCANE_ASSERT((item.isActive() || item.isSubactive() || item.isAncestor()), (" "));
    });
  }

  //
  std::set<Int32> cells_to_remove_set;
  UniqueArray<ItemInternal*> parent_cells;

  cells_map.eachItem([&](impl::ItemBase iitem) {
    if (!iitem.isOwn())
      return;

    // suppression of subactive cells
    if (iitem.isSubactive()) {
      // no level 0 cell should be subactive.
      ARCANE_ASSERT((iitem.nbHParent() != 0), (""));
      cells_to_remove_set.insert(iitem.localId());
      // inform the client of the mesh change
      mesh_changed = true;
    }
    else {
      // Compression of active cells
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
        if (active_parent) {
          parent_cells.add(iitem._itemInternal());
          ARCANE_ASSERT((iitem.flags() & ItemFlags::II_JustCoarsened), ("Incoherent JustCoarsened flag"));
        }
        // inform the client of the mesh change
        mesh_changed = true;
      }
      else {
        ARCANE_ASSERT((iitem.isAncestor()), (""));
      }
    }
  });
  //
  UniqueArray<Int32> cell_lids(arcaneCheckArraySize(cells_to_remove_set.size()));
  std::copy(std::begin(cells_to_remove_set), std::end(cells_to_remove_set), std::begin(cell_lids));

  if (m_mesh->parallelMng()->isParallel()) {
    this->_makeFlagParallelConsistent2();
    this->_removeGhostChildren();
    this->_updateItemOwner(cell_lids);
    m_mesh->parallelMng()->barrier();
  }
  if (cell_lids.size() > 0) {
    this->_upscaleData(parent_cells);
    _invalidate(parent_cells);
    //_updateItemOwner(cells_local_id);
    m_mesh->modifier()->removeCells(cell_lids, false);
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
  CHECKPERF(m_perf_counter.start(PerfCounter::UPDATEMAP))
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
  m_item_refinement->updateChildHMin(cells_to_refine);
  //m_face_finder.updateFaceCenter(cells_to_refine);
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPDATEMAP))
}

void MeshRefinement::
_update(ArrayView<ItemInternal*> cells_to_refine)
{
  CHECKPERF(m_perf_counter.start(PerfCounter::UPDATEMAP))
  m_node_finder.updateData(cells_to_refine);
  m_face_finder.updateData(cells_to_refine);
  _updateMaxUid(cells_to_refine);
  m_item_refinement->updateChildHMin(cells_to_refine);
  //m_face_finder.updateFaceCenter(cells_to_refine);
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPDATEMAP))
}

void MeshRefinement::
_invalidate(ArrayView<ItemInternal*> coarsen_cells)
{
  CHECKPERF(m_perf_counter.start(PerfCounter::CLEAR))
  m_node_finder.clearData(coarsen_cells);
  m_face_finder.clearData(coarsen_cells);
  CHECKPERF(m_perf_counter.stop(PerfCounter::CLEAR))
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
_upscaleData(Array<ItemInternal*>& parent_cells)
{
  m_call_back_mng->callCallBacks(parent_cells, Restriction);
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

  for (Integer i = 0, is = cell_to_remove_lids.size(); i < is; i++) {
    Cell item = cells_list[cell_to_remove_lids[i]];
    for (Node node : item.nodes()) {

      if (marker.find(node.localId()) != marker.end())
        continue;
      else
        marker[node.localId()] = true;

      //debug() << "NODE " << FullItemPrinter(node);
      //const Int32 owner = node->owner();
      bool is_ok = false;
      Integer count = 0;
      const Integer node_cs = node.cells().size();
      for (Cell cell : node.cells()) {
        if (cell_to_remove_lids.contains(cell.localId())) {
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
      if (!is_ok) {
        Cell cell;
        for (Cell cell2 : node.cells()) {
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
    for (Face face : item.faces()) {
      if (face.nbCell() != 2)
        continue;
      const Int32 owner = face.owner();
      bool is_ok = false;
      for (Cell cell : face.cells()) {
        if ((item.uniqueId() == cell.uniqueId()) || !(item.level() == cell.level()))
          continue;
        if (cell.owner() == owner) {
          is_ok = true;
          break;
        }
      }
      if (!is_ok) {
        for (Cell cell2 : face.cells()) {
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
  if (node_owner_changed) {
    // nodes_owner.synchronize(); // SDC Especially not the sync is KO at this moment (unrefined/de-refined ghosts)
    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
    m_mesh->nodeFamily()->endUpdate();
  }
  face_owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, face_owner_changed);
  if (face_owner_changed) {
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
  // It is necessary that every sub-item is a neighboring cell with the same owner
  VariableItemInt32& nodes_owner(m_mesh->nodeFamily()->itemsNewOwner());

  NodeGroup own_nodes = m_mesh->ownNodes();
  bool owner_changed = false;
  ENUMERATE_NODE (inode, own_nodes) {
    Node node = (*inode);
    Int32 owner = node.owner();
    bool is_ok = false;
    for (Cell cell : node.cells()) {
      if (cell.owner() == owner) {
        is_ok = true;
        break;
      }
    }
    if (!is_ok) {
      Cell cell;
      for (Cell cell2 : node.cells()) {
        if (cell.null() || cell2.uniqueId() < cell.uniqueId())
          cell = cell2;
      }
      ARCANE_ASSERT((!cell.null()), ("Inconsistent null cell owner reference"));
      nodes_owner[node] = cell.owner();
      owner_changed = true;
    }
  }
  owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, owner_changed);
  if (owner_changed) {
    nodes_owner.synchronize();
    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
    m_mesh->nodeFamily()->endUpdate();
  }

  // It is necessary that every sub-item is a neighboring cell with the same owner
  VariableItemInt32& faces_owner(m_mesh->faceFamily()->itemsNewOwner());

  FaceGroup own_faces = m_mesh->ownFaces();
  owner_changed = false;
  ENUMERATE_FACE (iface, own_faces) {
    Face face = (*iface);
    Int32 owner = face.owner();
    bool is_ok = false;
    for (Cell cell : face.cells()) {
      if (cell.owner() == owner) {
        is_ok = true;
        break;
      }
    }
    if (!is_ok) {
      if (face.nbCell() == 2)
        fatal() << "Face" << ItemPrinter(face) << " has a different owner with respect to Back/Front Cells";

      faces_owner[face] = face.boundaryCell().owner();
      owner_changed = true;
    }
  }
  owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, owner_changed);
  if (owner_changed) {
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

  // Removal of cells
  Int32UniqueArray cells_to_remove;
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

  _invalidate(parent_cells);

  // Before removal, update the owners of isolated Nodes/Faces
  _updateItemOwner(cells_to_remove);
  //info() << "Number of cells to remove: " << cells_to_remove.size();
  m_mesh->modifier()->removeCells(cells_to_remove, false);
  for (Integer i = 0, ps = parent_cells.size(); i < ps; i++)
    populateBackFrontCellsFromChildrenFaces(parent_cells[i]);

  return cells_to_remove.size() > 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshRefinement::
populateBackFrontCellsFromParentFaces(Cell parent_cell)
{
  switch (parent_cell.type()) {
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
    ARCANE_FATAL("Not supported refinement Item Type type={0}", parent_cell.type());
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int typeID> void MeshRefinement::
_populateBackFrontCellsFromParentFaces(Cell parent_cell)
{
  Integer nb_children = parent_cell.nbHChildren();
  const ItemRefinementPatternT<typeID>& rp = getRefinementPattern<typeID>();
  for (Integer c = 0; c < nb_children; c++) {
    Cell child = parent_cell.hChild(c);
    Integer nb_child_faces = child.nbFace();
    for (Integer fc = 0; fc < nb_child_faces; fc++) {
      if (rp.face_mapping_topo(c, fc) == 0)
        continue;
      const Integer f = rp.face_mapping(c, fc);
      Face face = parent_cell.face(f);
      Integer nb_cell_face = face.nbCell();
      if (nb_cell_face == 1)
        continue;
      Face subface = child.face(fc);
      Integer nb_cell_subface = subface.nbCell();
      if (nb_cell_subface == 1) {
        m_face_family->addBackFrontCellsFromParentFace(subface, face);
      }
      else {
        if (face.backCell().isOwn() != face.frontCell().isOwn()) {
          m_face_family->replaceBackFrontCellsFromParentFace(child, subface, parent_cell, face);
        }
        else {
          if (!face.backCell().isOwn() && !face.frontCell().isOwn()) {
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
  for (Integer f = 0; f < nb_faces; f++) {
    Face face = parent_cell.face(f);
    Integer nb_cell_face = face.nbCell();
    if (nb_cell_face == 1)
      continue;
    Cell neighbor_cell = (face.cell(0) == parent_cell) ? face.cell(1) : face.cell(0);
    if (neighbor_cell.isActive())
      continue;
    switch (neighbor_cell.type()) {
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
      ARCANE_FATAL("Not supported refinement Item Type type={0}", neighbor_cell.type());
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
  for (Integer f = 0; f < neighbor_cell.nbFace(); f++) {
    if (neighbor_cell.face(f) == face) {
      Integer nb_children = neighbor_cell.nbHChildren();
      for (Integer c = 0; c < nb_children; c++) {
        Cell child = neighbor_cell.hChild(c);
        Integer nb_child_faces = child.nbFace();
        for (Integer fc = 0; fc < nb_child_faces; fc++) {
          if (f == rp.face_mapping(c, fc) && (rp.face_mapping_topo(c, fc))) {
            Face subface = child.face(fc);
            if (subface.itemBase().flags() & ItemFlags::II_HasBackCell) {
              m_face_family->addFrontCellToFace(subface, parent_cell);
            }
            else if (subface.itemBase().flags() & ItemFlags::II_HasFrontCell) {
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
  VariableNodeInt32 syncvariable(VariableBuildInfo(m_mesh, "SyncVarNodeOwnerContract"));
  syncvariable.fill(-1);
  bool has_owner_changed = false;
  ENUMERATE_NODE (inode, m_mesh->ownNodes())
    syncvariable[inode] = inode->owner();
  VariableNodeInt32 syncvariable_copy(VariableBuildInfo(m_mesh, "SyncVarNodeOwnerContractCopy"));
  syncvariable_copy.copy(syncvariable);
  syncvariable.synchronize();
  ItemVector desync_nodes(m_mesh->nodeFamily());
  ENUMERATE_NODE (inode, m_mesh->allNodes()) {
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
  ENUMERATE_NODE (inode, desync_nodes) {
    desync_node_uids[inode.index()] = inode->uniqueId().asInt64();
  }
  Int64UniqueArray desync_node_uids_gather;
  m_mesh->parallelMng()->allGatherVariable(desync_node_uids.view(), desync_node_uids_gather);
  Int32UniqueArray desync_node_lids_gather(desync_node_uids_gather.size());
  m_mesh->nodeFamily()->itemsUniqueIdToLocalId(desync_node_lids_gather, desync_node_uids_gather, false);
  for (auto lid : desync_node_lids_gather) {
    if (lid == NULL_ITEM_LOCAL_ID)
      continue;
    if (std::find(desync_nodes.viewAsArray().begin(), desync_nodes.viewAsArray().end(), lid) == desync_nodes.viewAsArray().end()) {
      desync_nodes.add(lid);
    }
  }
  // 1.2 Exchange the owners of the desynchronized nodes
  // each process fill an array [node1_uid, node1_owner,...nodei_uid, nodei_owner,...]
  Int64UniqueArray desync_node_owners(2 * desync_nodes.size());
  ENUMERATE_NODE (inode, desync_nodes) {
    desync_node_owners[2 * inode.index()] = inode->uniqueId().asInt64();
    desync_node_owners[2 * inode.index() + 1] = inode->owner();
  }
  // 1.2 gather this array on every process
  Int64UniqueArray desync_node_owners_gather;
  m_mesh->parallelMng()->allGatherVariable(desync_node_owners.view(), desync_node_owners_gather);
  // 1.3 store the information in a map <uid, Array[owner] >
  std::map<Int64, Int32SharedArray> uid_owners_map;
  for (Integer node_index = 0; node_index + 1 < desync_node_owners_gather.size();) {
    uid_owners_map[desync_node_owners_gather[node_index]].add((Int32)desync_node_owners_gather[node_index + 1]);
    desync_node_uids_gather.add(desync_node_owners_gather[node_index]);
    node_index += 2;
  }
  // 2 choose the unique owner of the desynchronized nodes :
  // Choose the minimum owner (same choice on each proc) different from the historical owner
  Integer new_owner = m_mesh->parallelMng()->commSize() + 1;
  ENUMERATE_NODE (inode, desync_nodes) {
    for (auto owner : uid_owners_map[inode->uniqueId().asInt64()]) {
      if (owner < new_owner && owner != m_node_owner_memory[inode])
        new_owner = owner;
    }
    debug(Trace::Highest) << "------ Change owner for node " << inode->uniqueId() << " from " << inode->owner() << " to " << new_owner;
    inode->mutableItemBase().setOwner(new_owner, m_mesh->parallelMng()->commRank());
    new_owner = m_mesh->parallelMng()->commSize() + 1;
  }
  // Update family if owners have changed
  bool p_has_owner_changed = m_mesh->parallelMng()->reduce(Parallel::ReduceMax, has_owner_changed);
  if (p_has_owner_changed) {
    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
    m_mesh->nodeFamily()->endUpdate();
    m_mesh->nodeFamily()->computeSynchronizeInfos();
  }
  // update node memory owner
  ENUMERATE_NODE (inode, m_mesh->allNodes())
    m_node_owner_memory[inode] = inode->owner();
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
