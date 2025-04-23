// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartitionerBase.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un partitionneur de maillage                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/MultiArray2.h"

#define INSURE_CONSTRAINTS

#include "arcane/core/ServiceBuildInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/IMeshPartitionConstraintMng.h"
#include "arcane/core/ILoadBalanceMng.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/internal/ILoadBalanceMngInternal.h"

#include "arcane/std/MeshPartitionerBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshPartitionerBase::
MeshPartitionerBase(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_sub_domain(sbi.subDomain())
, m_mesh(sbi.mesh())
, m_cell_family(sbi.mesh()->cellFamily())
, m_lbMng(sbi.subDomain()->loadBalanceMng())
, m_lb_mng_internal(sbi.subDomain()->loadBalanceMng()->_internalApi())
{
  IParallelMng* pm = m_mesh->parallelMng();
  m_pm_sub = pm;
  m_is_non_manifold_mesh = m_mesh->meshKind().isNonManifold();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshPartitionerBase::
~MeshPartitionerBase()
{
  freeConstraints();
  delete m_unique_id_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* MeshPartitionerBase::
getCommunicator() const
{
  return m_pm_sub->getMPICommunicator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Parallel::Communicator MeshPartitionerBase::
communicator() const
{
  return m_pm_sub->communicator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
changeOwnersFromCells()
{
  m_mesh->utilities()->changeOwnersFromCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
initConstraints(bool uidref)
{
  m_mesh_dimension = m_mesh->dimension();

  _initArrayCellsWithConstraints();

  _initFilterLidCells();

  if (uidref)
    _initUidRef();

  _initLid2LidCompacted();

  _initNbCellsWithConstraints();

  m_lb_mng_internal->initAccess(m_mesh);

  info() << "Weight (" << subDomain()->commonVariables().globalIteration()
         << "): " << m_lb_mng_internal->nbCriteria(m_mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
freeConstraints()
{
  m_lb_mng_internal->endAccess();
  _clearCellWgt();
  m_cells_with_constraints.clear();
  m_cells_with_weak_constraints.clear();
  m_nb_cells_with_constraints = 0;
  m_filter_lid_cells.clear();
  m_local_id_2_local_id_compacted.clear();
  delete m_unique_id_reference;
  m_unique_id_reference = nullptr;
  m_check.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshPartitionerBase::
_createConstraintsLists(Int64MultiArray2& tied_uids)
{
  int allLocal = 0;

  // It is important to define constraints only once !
  m_cells_with_constraints.clear();

  IItemFamily* cellFamily = m_mesh->itemFamily(IK_Cell);

//   info()<<"tied_uids.dim1Size() = "<<tied_uids.dim1Size();
  for( Integer i=0, n=tied_uids.dim1Size(); i<n; ++i ){
    // la liste d'uniqueId pour une contrainte
    Int64ConstArrayView uids(tied_uids[i]);

    // cette même liste en localId, sachant que certaines mailles ne sont pas locales
    Int32UniqueArray lids(uids.size());
    cellFamily->itemsUniqueIdToLocalId(lids,uids,false);

    // la liste locale en localId (sans les id des mailles sur d'autres procs)
    Int32UniqueArray lids_loc;
    lids_loc.reserve(lids.size());
    for( Integer j=0, js=lids.size(); j<js; ++j ){
      Int32 lid = lids[j];
      if (lid!=NULL_ITEM_LOCAL_ID)
        lids_loc.add(lid);
    }

    // le tableau avec les mailles de cette contrainte
    ItemVectorView items_view = cellFamily->view(lids_loc);
    SharedArray<Cell> cells;
    for ( Integer j=0, js=items_view.size(); j<js; j++)
      if (items_view[j].isOwn())
        cells.add(items_view[j].toCell());

    // Les elements ne sont pas a cheval avec ce processeur
    allLocal += (((cells.size() == 0)||
                  (cells.size() == uids.size()))?0:1);

    // on ajoute ce tableau à la liste
    if (!cells.empty())
      m_cells_with_constraints.add(cells);
  }
  IParallelMng* pm = m_mesh->parallelMng();

  // Return reduction because we need all subdomains to be correct
  int sum = pm->reduce(Parallel::ReduceSum,allLocal);
  return (sum == 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
_initArrayCellsWithConstraints()
{
  // It is important to define constraints only once !
  m_cells_with_constraints.clear();
  m_cells_with_weak_constraints.clear();

  if (!m_mesh->partitionConstraintMng())
    return;

  // c'est ici qu'il faut récupérer les listes de listes de mailles
  // avec comme contrainte de ne pas être séparé lors du repartitionnement

  // un tableau 2D avec les uniqueId des mailles
  Int64MultiArray2 tied_uids;

  // Compute tied_uids first because we cannot use this after redistribution
  // Nota: It is correct to do so because ConstraintList is global, so not changed
  //       by the redistribution.
  m_mesh->partitionConstraintMng()->computeConstraintList(tied_uids);
  // Be sure that constraints are local !
#ifdef INSURE_CONSTRAINTS
  if (!_createConstraintsLists(tied_uids)) {
    if (m_is_non_manifold_mesh)
      ARCANE_FATAL("Constraints are not supported for non manifold mesh");
    // Only appends for the first iteration because constraints are not set before !
    VariableItemInt32& cells_new_owner(m_mesh->cellFamily()->itemsNewOwner());
    ENUMERATE_CELL(icell,m_mesh->cellFamily()->allItems()){
      cells_new_owner[icell] = (*icell).owner();
    }
    m_mesh->modifier()->setDynamic(true);
    m_mesh->partitionConstraintMng()->computeAndApplyConstraints();
    m_mesh->utilities()->changeOwnersFromCells();
    m_mesh->toPrimaryMesh()->exchangeItems();
#endif // INSURE_CONSTRAINTS
  if (!_createConstraintsLists(tied_uids))
    throw FatalErrorException(A_FUNCINFO, "Issue with constraints !");
#ifdef INSURE_CONSTRAINTS
  }
  m_mesh->partitionConstraintMng()->computeWeakConstraintList(tied_uids);

  for(Integer i=0;i<tied_uids.dim1Size();++i)
  {
    std::pair<Int64, Int64> ids(tied_uids[i][0],tied_uids[i][1]);
    m_cells_with_weak_constraints.insert(ids);
  }
#endif //INSURE_CONSTRAINTS

}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
_initFilterLidCells()
{
  CellGroup all_cells = m_mesh->allCells();

  // Mise en place d'un filtre sur les localId avec comme marque un eMarkCellWithConstraint
  m_filter_lid_cells.resize(m_mesh->cellFamily()->maxLocalId());
  m_filter_lid_cells.fill(eCellGhost);

  ENUMERATE_CELL(icell, m_mesh->ownCells()){
    m_filter_lid_cells[icell->localId()] = eCellClassical;
  }

  for (Integer i=0; i<m_cells_with_constraints.size(); ++i){
    Array<Cell> & listCell = m_cells_with_constraints[i];
    m_filter_lid_cells[listCell[0].localId()] = eCellReference;
    for (Integer j=1; j<listCell.size(); ++j) {
#if 0
      if (m_filter_lid_cells[listCell[j].localId()] != eCellClassical)
        info() << "Pb in constraint " << i << " with cell[" << j
               <<"] = " << listCell[j].uniqueId();
#endif
      m_filter_lid_cells[listCell[j].localId()] = eCellGrouped;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
_checkCreateVar()
{
  if (!m_unique_id_reference)
    m_unique_id_reference = new VariableCellInt64(VariableBuildInfo(m_mesh, "CellUniqueIdRef", IVariable::PNoDump));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
_initUidRef()
{
  _checkCreateVar();
  VariableCellInt64 uids_ref(*m_unique_id_reference);
  // Mise en place d'un tableau d'indirection entre les cell et un uniqueId de référence
  // permet de connaitre l'uid de la première maille de chacune des contraintes
  // y compris pour les mailles fantômes
  ENUMERATE_CELL(icell, m_mesh->allCells()){
    uids_ref[icell] = icell->uniqueId();
  }
  for (Integer i=0; i<m_cells_with_constraints.size(); ++i){
    Array<Cell> & listCell = m_cells_with_constraints[i];
    Int64 id_ref = listCell[0].uniqueId();
    for (Integer j=1; j<listCell.size(); ++j)
      uids_ref[listCell[j]] = id_ref;
  }
  uids_ref.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
_initUidRef(VariableCellInteger& cell_renum_uid)
{
  _checkCreateVar();
  VariableCellInt64 uids_ref(*m_unique_id_reference);

  // Mise en place d'un tableau d'indirection entre les cell et un uniqueId de référence
  // permet de connaitre l'uid de la première maille de chacune des contraintes
  // y compris pour les mailles fantômes
  ENUMERATE_CELL (icell, m_mesh->allCells()) {
    uids_ref[icell] = cell_renum_uid[icell];
  }
  for (Integer i=0; i<m_cells_with_constraints.size(); ++i){
    Array<Cell> & listCell = m_cells_with_constraints[i];
    Int64 id_ref = cell_renum_uid[listCell[0]];
    for (Integer j=1; j<listCell.size(); ++j)
      uids_ref[listCell[j]] = id_ref;
  }
  uids_ref.synchronize();

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerBase::
_initLid2LidCompacted()
{
  // Construction du tableau d'indirection entre la numérotation locale pour toutes les mailles
  // et une numérotation sans les mailles fantômes ni les mailles regroupées
  Integer index = 0;
  m_local_id_2_local_id_compacted.resize(m_mesh->cellFamily()->maxLocalId());
  m_check.resize(m_mesh->cellFamily()->maxLocalId());
  m_check.fill(-1);
    
  ENUMERATE_CELL(icell, m_mesh->allCells()){
    switch (m_filter_lid_cells[icell.itemLocalId()]){
    case eCellClassical:
    case eCellReference:
      m_local_id_2_local_id_compacted[icell->localId()] = index++;
      ///info()<<"m_local_id_2_local_id_compacted["<<icell->localId()<<"(gid:"<<icell->uniqueId()<<")] = "<<index-1;
      break;
    case eCellGrouped:
    case eCellGhost:
      m_local_id_2_local_id_compacted[icell->localId()] = -1;
      break;
    default:
      throw FatalErrorException(A_FUNCINFO,"Invalid filter value");
    }
  }
  //  info()<<"m_local_id_2_local_id_compacted de 0 à "<<index-1;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void MeshPartitionerBase::
_initNbCellsWithConstraints()
{
  // Calcul le nombre de mailles internes en tenant compte des regroupement suivant les contraintes
  m_nb_cells_with_constraints = m_mesh->ownCells().size();
  for (Integer i=0; i<m_cells_with_constraints.size(); ++i){
    m_nb_cells_with_constraints -= (m_cells_with_constraints[i].size()-1);
  }

#if 0
  for (Integer i=0; i<m_cells_with_constraints.size(); ++i){
    Array<Cell> & listCell = m_cells_with_constraints[i];
    for (Integer j = 1 ; j < listCell.size() ; ++j) {
      if (m_filter_lid_cells[listCell[j].localId()] != eCellGrouped) {
        info() << "Pb in group " << i << " " << listCell[j].localId() << "is not grouped";
      }
    }
    info() << "Group of size " << i << " : " << listCell.size();
  }
#endif

  info() <<"allCells().size() = "<<m_mesh->allCells().size();
  info() <<"ownCells().size() = "<<m_mesh->ownCells().size();
  info() <<"m_nb_cells_with_constraints = "<<m_nb_cells_with_constraints;
}


/*---------------------------------------------------------------------------*/
Int32 MeshPartitionerBase::
nbOwnCellsWithConstraints() const
{
  return m_nb_cells_with_constraints;
}
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
Integer MeshPartitionerBase::
nbNeighbourCellsWithConstraints(Cell cell)
{
  Integer nbNeighbors = 0;

  if (m_filter_lid_cells[cell.localId()] == eCellClassical
      || m_filter_lid_cells[cell.localId()] == eCellReference) {
    Int64UniqueArray neighbors;
    neighbors.resize(0);
    getNeighbourCellsUidWithConstraints(cell, neighbors);
    nbNeighbors = neighbors.size();
  }
  else
    nbNeighbors = -1;
  return (nbNeighbors);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//< Add a ngb cell if not already present.
Real MeshPartitionerBase::
_addNgb(const Cell& cell, const Face& face,
        Int64Array& neighbourcells, Array<bool>& contrib,
        HashTableMapT<Int64,Int32>& map,
        Array<float> *ptrcommWeights, Int32 offset,
        HashTableMapT<Int32,Int32>& lids, bool special)
{
  ARCANE_UNUSED(contrib);
  ARCANE_UNUSED(lids);

  Int64 uid = (*m_unique_id_reference)[cell];
  bool toAdd = false;
  Int32 myoffset = neighbourcells.size();
  Real hg_contrib = 0;
  const VariableItemReal& commCost = m_lb_mng_internal->commCost(m_mesh);
  const float face_comm_cost = static_cast<float>(commCost[face]);
  // Maille traditionnelle, on peut ajouter
  if ((!special) &&(m_filter_lid_cells[cell.localId()] == eCellClassical))
    toAdd = true;
  else {
    HashTableMapT<Int64,Int32>::Data* ptr;
    ptr = map.lookupAdd(uid, myoffset, toAdd);
    if (!toAdd && ptrcommWeights) {
      myoffset = ptr->value();
      (*ptrcommWeights)[offset + myoffset] += face_comm_cost;
    }
  }
  if (toAdd) {
    neighbourcells.add(uid);
    if (ptrcommWeights){
      (*ptrcommWeights).add(face_comm_cost);
    }
  }

  // TODO make hg_contrib work again.
  //contrib[myoffset] = true;

  // Count cell contrib only once, even if it appears several times
  // if (ptrcommWeights) {
  //   lids.lookupAdd(cell.localId(), myoffset, toAdd);
  //   if (toAdd) {
  //     hg_contrib = m_criteria.getResidentMemory(cell);
  //     (*ptrcommWeights)[offset + myoffset] += hg_contrib;
  //   }
  // }

  return (hg_contrib);
}


/*---------------------------------------------------------------------------*/
Real MeshPartitionerBase::
getNeighbourCellsUidWithConstraints(Cell cell, Int64Array& neighbourcells,
                                    Array<float> *ptrcommWeights,
                                    bool no_cell_contrib)
{
  ARCANE_UNUSED(no_cell_contrib);

  Int32 offset = 0;
  Real hg_contrib = 0;

  if ((m_filter_lid_cells[cell.localId()] != eCellClassical)
      &&(m_filter_lid_cells[cell.localId()] != eCellReference))
    return 0.0;

  if (ptrcommWeights)
    offset = (*ptrcommWeights).size();

  neighbourcells.resize(0);

#ifdef MY_DEBUG
  VariableCellInt64 uids_ref(*m_unique_id_reference);
  Int64 uid = uids_ref[cell];
#endif /* MY_DEBUG */

  Integer index = -1;
  Integer nbFaces = cell.nbFace();

  // First compute max degree
  if (m_filter_lid_cells[cell.localId()] == eCellReference) {
    for (index=0; index<m_cells_with_constraints.size() && m_cells_with_constraints[index][0] != cell; ++index){
    }
    if (index==m_cells_with_constraints.size())
      throw FatalErrorException(A_FUNCINFO,"Unable to find cell");

    Array<Cell>& listCell = m_cells_with_constraints[index];
    nbFaces = 0;
    // Activate constraint, but not the reference !
    for (Integer j=1; j<listCell.size(); ++j) {
      m_filter_lid_cells[listCell[j].localId()] = eCellInAConstraint;
      nbFaces += listCell[j].nbFace();
    }
  }

  HashTableMapT<Int64,Int32> difficultNgb(nbFaces,true);
  HashTableMapT<Int32,Int32> lids(nbFaces,true);
  UniqueArray<bool>contrib(nbFaces);
  // Array<Real>memUsed(nbFaces); // (HP): bug sur cette structure dans la suite du code
  contrib.fill(false);

  if (m_filter_lid_cells[cell.localId()] == eCellClassical){
    bool use_face = true;
    if (m_is_non_manifold_mesh) {
      // En cas de maillage non manifold, si la maille est
      // de dimension 2 pour un maillage de dimension 3, alors
      // elle contient des arêtes au lieu des faces.
      // On utilise donc les arêtes pour déterminer les voisines.
      // Dans ce cas, on ne prend en compte que les voisines
      // qui sont aussi de dimension 2.
      Int32 dim = cell.typeInfo()->dimension();
      if (dim == 2 && m_mesh_dimension == 3) {
        use_face = false;
        for (Edge sub_edge : cell.edges()) {
          // on ne prend que les arêtes ayant une maille voisine
          if (sub_edge.nbCell() >= 2) {
            for (Cell sub_cell : sub_edge.cells()) {
              if (sub_cell != cell && sub_cell.typeInfo()->dimension() == 2) {
                hg_contrib += 1.0;
                neighbourcells.add((*m_unique_id_reference)[sub_cell]);
                // TODO: regarder la valeur qu'il faut ajouter pour les communications
                if (ptrcommWeights)
                  (*ptrcommWeights).add(1.0f);
              }
            }
          }
        }
      }
    }
    if (use_face) {
      for (Integer z = 0; z < cell.nbFace(); ++z) {
        Face face = cell.face(z);
        // on ne prend que les faces ayant une maille voisine
        if (face.nbCell() == 2) {
          // recherche de la maille externe
          Cell opposite_cell = (face.cell(0) != cell ? face.cell(0) : face.cell(1));
          hg_contrib += _addNgb(opposite_cell, face, neighbourcells, contrib, difficultNgb,
                                ptrcommWeights, offset, lids);
        }
      }
    }
    // //Now add cell contribution to edge weight
    // if ((ptrcommWeights) &&(!noCellContrib)) {
    //   float mymem = m_criteria.getOverallMemory(cell);
    //   for (Integer j = 0 ; j < neighbourcells.size() ;++j) {
    //     (*ptrcommWeights)[offset+j] += mymem;
    //   }
    // }
  }
  else { //if (m_filter_lid_cells[cell.localId()] == eCellReference){
    Array<Cell>& listCell = m_cells_with_constraints[index];
    m_filter_lid_cells[listCell[0].localId()] = eCellInAConstraint;
    // memUsed.fill(0);
    for (Integer j=0; j<listCell.size(); ++j){
      contrib.fill(false);
      // hg_contrib += m_criteria.getOverallMemory(listCell[j]);
      // memUsed[j] = hg_contrib;
      for( Integer z=0; z<listCell[j].nbFace(); ++z ){
        const Face& face = listCell[j].face(z);
        // on ne prend que les faces ayant une maille non marquée à eCellInAConstraint et avec une maille voisine
        if ((face.nbCell()==2)
            && (m_filter_lid_cells[face.cell(0).localId()] != eCellInAConstraint
                 || m_filter_lid_cells[face.cell(1).localId()] != eCellInAConstraint)){
          // recherche de la maille externe
          const Cell& opposite_cell = (m_filter_lid_cells[face.cell(0).localId()] != eCellInAConstraint?
                                       face.cell(0):face.cell(1));
          hg_contrib += _addNgb(opposite_cell, face,  neighbourcells, contrib, difficultNgb,
                  ptrcommWeights, offset, lids, true);
        }
      }
      // //Now add cell contribution to edge weight
      // if (ptrcommWeights && !noCellContrib) {
      //   for (Integer c = 0 ; c < neighbourcells.size() ;++c) {
      //     if (contrib[c])
      //       (*ptrcommWeights)[offset+c] += memUsed[j];
      //   }
      // }
    }
    m_filter_lid_cells[listCell[0].localId()] = eCellReference;
    for (Integer j=1; j<listCell.size(); ++j)
      m_filter_lid_cells[listCell[j].localId()] = eCellGrouped;

  } // end if eCellReference

  return (hg_contrib);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void MeshPartitionerBase::
getNeighbourNodesUidWithConstraints(Cell cell, Int64UniqueArray neighbournodes)
{
  neighbournodes.resize(cell.nbNode());

  for( Integer z=0; z<cell.nbNode(); ++z ){
    neighbournodes[z] = cell.node(z).uniqueId();
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
Int32 MeshPartitionerBase::
localIdWithConstraints(Cell cell)
{
  return m_local_id_2_local_id_compacted[cell.localId()];
}
/*---------------------------------------------------------------------------*/
Int32 MeshPartitionerBase::
localIdWithConstraints(Int32 cell_lid)
{
  //info()<<"localIdWithConstraints("<<cell_lid<<") => "<<m_local_id_2_local_id_compacted[cell_lid];
  return m_local_id_2_local_id_compacted[cell_lid];
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void MeshPartitionerBase::
invertArrayLid2LidCompacted()
{
  //info()<<"MeshPartitionerBase::invertArrayLid2LidCompacted()";
  Integer index = 0;
  for (Integer i=0; i<m_mesh->allCells().size(); i++){
    if (m_local_id_2_local_id_compacted[i] != -1)
      m_local_id_2_local_id_compacted[index++] = i;
  }
  for (;index<m_mesh->allCells().size(); index++)
    m_local_id_2_local_id_compacted[index] = -2;
}
/*---------------------------------------------------------------------------*/

SharedArray<float> MeshPartitionerBase::
cellsSizeWithConstraints()
{
  VariableCellReal mWgt = m_lb_mng_internal->massResWeight(m_mesh);
  return _cellsProjectWeights(mWgt);
}

SharedArray<float> MeshPartitionerBase::
cellsWeightsWithConstraints(Int32 max_nb_weight, bool ask_lb_cells)
{
  ARCANE_UNUSED(ask_lb_cells);

  Int32 nb_weight = max_nb_weight;

  Int32 nb_criteria = m_lb_mng_internal->nbCriteria(m_mesh);

  if (max_nb_weight <= 0 || max_nb_weight > nb_criteria)
    nb_weight = nb_criteria;

  info() <<  "Number of weights " << nb_weight << " / " << nb_criteria;

  VariableCellArrayReal mWgt = m_lb_mng_internal->mCriteriaWeight(m_mesh);
  return _cellsProjectWeights(mWgt, nb_weight);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<float> MeshPartitionerBase::
_cellsProjectWeights(VariableCellArrayReal& cellWgtIn, Int32 nbWgt) const
{
  SharedArray<float> cellWgtOut(nbOwnCellsWithConstraints()*nbWgt);
  if (nbWgt > cellWgtIn.arraySize()) {
    ARCANE_FATAL("Asked for too many weights n={0} array_size={1}",nbWgt,cellWgtIn.arraySize());
  }

  ENUMERATE_CELL(icell, m_mesh->ownCells()){
    if(m_filter_lid_cells[icell->localId()]==eCellClassical)
      for ( int i = 0 ; i < nbWgt ; ++i){
        float v = static_cast<float>(cellWgtIn[icell][i]);
        cellWgtOut[m_local_id_2_local_id_compacted[icell->localId()]*nbWgt+i] = v;
      }
  }
  RealUniqueArray w(nbWgt);
  for( auto& ptr : m_cells_with_constraints ){
    w.fill(0);
    for( const auto& cell : ptr ){
      for (int i = 0 ; i <nbWgt ; ++i)
        w[i] += cellWgtIn[cell][i];
    }
    for (int i=0 ; i<nbWgt ; ++i)
      cellWgtOut[m_local_id_2_local_id_compacted[ptr[0].localId()]*nbWgt+i] = (float)(w[i]);
  }

  return cellWgtOut;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
SharedArray<float> MeshPartitionerBase::
_cellsProjectWeights(VariableCellReal& cellWgtIn) const
{
  SharedArray<float> cellWgtOut(nbOwnCellsWithConstraints());

  ENUMERATE_CELL(icell, m_mesh->ownCells()){
    if(m_filter_lid_cells[icell->localId()]==eCellClassical)
      cellWgtOut[m_local_id_2_local_id_compacted[icell->localId()]]
      = (float)cellWgtIn[icell];
  }
  for( auto& ptr : m_cells_with_constraints ){
    Real w = 0;
    for( Cell cell : ptr ){
      w += cellWgtIn[cell];
    }
    cellWgtOut[m_local_id_2_local_id_compacted[ptr[0].localId()]]
    = (float)w;
  }

  return cellWgtOut;
}

/*---------------------------------------------------------------------------*/
bool MeshPartitionerBase::
cellUsedWithConstraints(Cell cell)
{
  eMarkCellWithConstraint marque = m_filter_lid_cells[cell.localId()];
//   info()<<"cellUsedWithConstraints("<<cell<<") => "<<(marque == eCellClassical || marque == eCellReference);
  return (marque == eCellClassical || marque == eCellReference);
}

bool MeshPartitionerBase::
cellUsedWithWeakConstraints(std::pair<Int64,Int64>& paired_item)
{
  std::pair<Int64,Int64> other_pair(paired_item.second, paired_item.first);
  return ((m_cells_with_weak_constraints.find(paired_item)!=m_cells_with_weak_constraints.end()) || m_cells_with_weak_constraints.find(other_pair)!=m_cells_with_weak_constraints.end());
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void MeshPartitionerBase::
changeCellOwner(Item cell, VariableItemInt32& cells_new_owner, Int32 new_owner)
{
  // TODO à optimiser pour le cas où il y aurait plein de petites contraintes

  //info()<<"changeCellOwner "<<cell<<", new_owner = "<<new_owner;
  cells_new_owner[cell] = new_owner;

  if (m_filter_lid_cells[cell.localId()] == eCellReference){
    Integer index = -1;
    for (index=0; index<m_cells_with_constraints.size() && m_cells_with_constraints[index][0] != cell; ++index){
    }
    if (index==m_cells_with_constraints.size())
      throw FatalErrorException("MeshPartitionerBase::changeCellOwner(): unable to find cell");

    Array<Cell>& listCell = m_cells_with_constraints[index];
    //info()<<"  changement en plus pour listCell: "<<listCell;
    for (Integer i=1; i<listCell.size(); i++)
      cells_new_owner[listCell[i]] = new_owner;
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Fonctions utiles pour l'ancienne interface de partitionnement.

void
MeshPartitionerBase::setCellsWeight(ArrayView<float> weights,Integer nb_weight)
{
  m_lb_mng_internal->reset(m_mesh);
  _clearCellWgt();

  for (int i = 0 ; i <nb_weight ; ++i) {
    StringBuilder varName("LB_wgt_");
    varName += (i+1);

    VariableCellReal myvar(VariableBuildInfo(m_mesh, varName.toString(),
                                             IVariable::PNoDump|IVariable::PTemporary|IVariable::PExecutionDepend));
    ENUMERATE_CELL (icell, m_mesh->ownCells()) {
      (myvar)[icell] = weights[icell->localId()*nb_weight+i];
    }
    m_lb_mng_internal->addCriterion(myvar, m_mesh);
  }

  m_lb_mng_internal->initAccess(m_mesh);
  m_lb_mng_internal->setMassAsCriterion(m_mesh, false);
  m_lb_mng_internal->setNbCellsAsCriterion(m_mesh, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MeshPartitionerBase::
nbCellWeight() const
{
  return math::max(m_lb_mng_internal->nbCriteria(m_mesh), 1);
}

ArrayView<float> MeshPartitionerBase::
cellsWeight() const
{
  ARCANE_FATAL("NotImplemented");
}

void MeshPartitionerBase::
_clearCellWgt()
{
  //m_cell_wgt.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Fonction auxiliaire pour dumper le graphe.
template <class ArrayType> Parallel::Request
centralizePartInfo(String filename, IParallelMng *pm,
                   UniqueArray<ArrayType> data, String header, int step=1 )
{
  Parallel::Request req;
  UniqueArray<Integer> sizes(pm->commSize());
  UniqueArray<Integer> mysize(1);
  mysize[0] = data.size();

  pm->gather(mysize, sizes, pm->masterIORank());

  req = pm->send(data, pm->masterIORank(),false);


  if (pm->isMasterIO()) {
    ofstream ofile;

    ofile.open(filename.localstr());
    if (!header.null()) {
      ofile << header;
      Int64 sum = 0;
      for (ConstIterT<UniqueArray<Integer> > iter(sizes) ; iter() ; ++iter)
        sum += *iter;
      ofile << sum << std::endl;
    }

    for (int rank = 0 ; rank < pm->commSize() ; ++rank) {
      UniqueArray<ArrayType> otherdata(sizes[rank]);
      pm->recv(otherdata, rank, true);
      for ( ConstIterT<ArrayView<ArrayType> > myiter(otherdata) ; myiter() ; ) {
        for (int j = 0 ; (j < step) && myiter() ; ++j, ++myiter)
          ofile << *myiter << " ";
        ofile << std::endl ;
      }
    }
    ofile.close();
  }

  return req;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
MeshPartitionerBase::dumpObject(String filebase)
{
  int i = 0;
  IParallelMng* pm = m_mesh->parallelMng();
  String header;

  // ---
  // Send vertex uid first
  Int64UniqueArray uid(m_nb_cells_with_constraints);
  uid.fill(-1);
  VariableCellInt64 uids_ref(*m_unique_id_reference);
  i=0;
  ENUMERATE_CELL(icell, m_mesh->ownCells()) {
    if ((m_filter_lid_cells[icell->localId()]!=eCellClassical)
        &&(m_filter_lid_cells[icell->localId()]!=eCellReference))
      continue;

    uid[i++] = uids_ref[icell];
  }


  UniqueArray<Parallel::Request> reqs;
  Parallel::Request req;
  req = centralizePartInfo<Int64>(filebase+".uid", pm, uid, header);
  reqs.add(req);

  UniqueArray<float> vwgt = cellsWeightsWithConstraints(0);
  // Hack: use Real to avoid bugs in pm->recv ...
  // TODO: Fix this.
  UniqueArray<Real> rvwgt(vwgt.size());
  IterT<UniqueArray<Real> > myiterr(rvwgt);
  for (ConstIterT<UniqueArray<float> > myiterf(vwgt)  ;
       myiterf() ; ++myiterf, ++myiterr)
    (*myiterr) = (Real) (*myiterf);

  req = centralizePartInfo<Real>(filebase+".vwgt", pm, rvwgt, header, nbCellWeight());
  reqs.add(req);

  // Send vertex coords
  VariableNodeReal3& coords(mesh()->nodesCoordinates());
  UniqueArray<Real3> my_coords(m_nb_cells_with_constraints);
  i=0;
  ENUMERATE_CELL(icell, m_mesh->ownCells()) {
    if ((m_filter_lid_cells[icell->localId()]!=eCellClassical)
        &&(m_filter_lid_cells[icell->localId()]!=eCellReference))
      continue;

    // on calcul un barycentre
    for( Integer z=0, zs = (*icell).nbNode(); z<zs; ++z ){
      const Node& node = (*icell).node(z);
      my_coords[i] += coords[node];
    }
    my_coords[i] /= Convert::toDouble((*icell).nbNode());
    i++;
  }
  req = centralizePartInfo<Real3>(filebase+".xyz", pm, my_coords, header);
  reqs.add(req);


  // Send relationships to master
  UniqueArray<Real3> nnz;
  ENUMERATE_CELL(icell, m_mesh->ownCells()) {
    Int64UniqueArray neighbourcells;
    UniqueArray<float> commWeights;

    if ((m_filter_lid_cells[icell->localId()]!=eCellClassical)
        &&(m_filter_lid_cells[icell->localId()]!=eCellReference))
      continue;
    getNeighbourCellsUidWithConstraints(*icell, neighbourcells, &commWeights);
    Int64 my_uid = uids_ref[icell];
    for (Integer j = 0 ; j < neighbourcells.size() ; ++j) {
      if (neighbourcells[j] > my_uid)
        continue;
      Real3 tmp(static_cast<Real>(my_uid+1), static_cast<Real>(neighbourcells[j]+1), commWeights[j]);
      nnz.add(tmp);
    }
  }
  Integer nbvertices;
  nbvertices = pm->reduce(Parallel::ReduceSum, m_nb_cells_with_constraints);
  StringBuilder myheader = "%%MatrixMarket matrix coordinate real symmetric\n";
  myheader += nbvertices;
  myheader += " ";
  myheader += nbvertices;
  myheader += " ";
  req = centralizePartInfo<Real3>(filebase+".mtx", pm, nnz, myheader.toString());
  reqs.add(req);

  pm->waitAllRequests(reqs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
