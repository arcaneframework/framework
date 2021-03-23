// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemRefinement.cc                                           (C) 2000-2017 */
/*                                                                           */
/* liste de méthodes de manipulation d'un maillage AMR.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/IParallelMng.h"

#include "arcane/IMesh.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IItemFamily.h"
#include "arcane/Item.h"
#include "arcane/ItemRefinementPattern.h"
#include "arcane/VariableTypes.h"
#include "arcane/GeometricUtilities.h"
#include "arcane/ItemPrinter.h"
#include "arcane/SharedVariable.h"
#include "arcane/ItemVector.h"

#include <vector>
#include "arcane/mesh/ItemRefinement.h"
#include "arcane/mesh/MeshRefinement.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// \brief only compile these functions if the user requests AMR support
//! AMR

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Real ItemRefinement::TOLERENCE = 10.0e-6;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Mesh refinement methods
ItemRefinement::
ItemRefinement (IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_cell_hmin(VariableBuildInfo(mesh,"AMRCellHMin"))
, m_orig_nodes_coords(mesh->nodesCoordinates())
, m_refine_factor(2)
, m_nb_cell_to_add(0)
, m_nb_face_to_add(0)
, m_nb_node_to_add(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
ItemRefinement::
~ItemRefinement()
{
	//delete m_irp;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 ItemRefinement::
faceCenter(ItemInternal* iface,SharedVariableNodeReal3& nodes_coords) const
{
  Face face(iface);
  Real3 pfc = Real3::null();
  for( Node node : face.nodes() ){
    pfc += nodes_coords[node];
  }
  pfc /= static_cast<Real> (face->nbNode());
  return pfc ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemRefinement::
initHMin()
{
  ENUMERATE_CELL(item,m_mesh->allCells()){
    Real h_min=1.e30;
    for (Integer i=0; i<item->nbNode(); i++)
      for (Integer j=i+1; j<item->nbNode(); j++){
        Real3 diff = (m_orig_nodes_coords[item->node(i)] - m_orig_nodes_coords[item->node(j)]) ;
        h_min = std::min(h_min,diff.abs());
      }
    m_cell_hmin[item] =  h_min;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemRefinement::
updateChildHMin(ArrayView<ItemInternal*> refine_cells)
{
  for (Integer k=0; k<refine_cells.size(); k++){
    Cell parent = refine_cells[k];
    for (UInt32 i = 0, nc = parent.nbHChildren(); i < nc; i++){
      Cell item = parent.hChild(i) ;
      Real h_min=1.e30;
      for (Integer i=0; i<item.nbNode(); i++)
        for (Integer j=i+1; j<item.nbNode(); j++){
          Real3 diff = (m_orig_nodes_coords[item.node(i)] - m_orig_nodes_coords[item.node(j)]) ;
          h_min = std::min(h_min,diff.abs());
        }
      m_cell_hmin[item] =  h_min;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real ItemRefinement::
hmin(Cell item) const
{
  Real h_min=1.e30;
  for (Integer i=0; i<item.nbNode(); i++)
    for (Integer j=i+1; j<item.nbNode(); j++){
      Real3 diff = (m_orig_nodes_coords[item.node(i)] - m_orig_nodes_coords[item.node(j)]) ;
      h_min = std::min(h_min,diff.abs());
    }

  return h_min;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int typeID> void ItemRefinement::
refineOneCell(Cell item, MeshRefinement& mesh_refinement)
{
  ItemTypeMng* itm = ItemTypeMng::singleton();
  const Int32 nb_hChildren = itm->nbHChildrenByItemType(item.type());
  const Int32 nb_nodes = item.nbNode();
  ARCANE_ASSERT((item.internal()->flags() & ItemInternal::II_Refine), ("Item is not flagged for refinement"));
  ARCANE_ASSERT((item.isActive()), ("Refine non-active item is forbidden!"));

  debug(Trace::High) << "[refineOneCell] nb_hChildren=" << nb_hChildren;
  debug(Trace::High) << "[refineOneCell] nb_nodes=" << nb_nodes;
  //!

  bool has_hChildren = item.hasHChildren();
  debug(Trace::Highest) << "[refineOneCell] has_hChildren=" << has_hChildren;

  m_nb_cell_to_add = 0;
  m_nb_face_to_add = 0;
  m_nb_node_to_add = 0;

  if (!has_hChildren){
    // Creation des enfants
    computeHChildren<typeID>(item, mesh_refinement);
    debug()<<"[ItemRefinement::refineOneCell] "<<m_nb_cell_to_add<<" new cells, "
           <<m_nb_node_to_add<<" new nodes & "<<m_nb_face_to_add<<" faces";

    // Créé les noeuds et positionne leur coordonnées
    {
      m_nodes_lid.resize(m_nb_node_to_add);
      m_mesh->modifier()->addNodes(m_nodes_unique_id, m_nodes_lid);
      m_mesh->nodeFamily()->endUpdate();
      ItemInternalList nodes(m_mesh->nodeFamily()->itemsInternal());
      for (Integer i = 0; i < m_nb_node_to_add; ++i) {
        m_orig_nodes_coords[nodes[m_nodes_lid[i]]] = m_nodes_to_create_coords[i];
      }
    }

    // Créé les faces
    {
      m_faces_lid.resize(m_nb_face_to_add);
      m_mesh->modifier()->addFaces(m_nb_face_to_add, m_faces_infos, m_faces_lid);
    }

    // Créé les mailles
    {
      m_cells_lid.resize(m_nb_cell_to_add);
      m_mesh->modifier()->addHChildrenCells(item.internal(), m_nb_cell_to_add, m_cells_infos, m_cells_lid);
      //! \todo vérfier l'ordre des enfants après leurs création
      ItemInternalList cells(m_mesh->cellFamily()->itemsInternal());
      for (Integer i = 0; i < m_nb_cell_to_add; ++i){
        ItemInternal* child = cells[m_cells_lid[i]];
        Integer cf = child->flags();
        cf |= ItemInternal::II_JustAdded;
        child->setFlags(cf);
      }
    }
  }
  else{
    for (Integer c = 0; c < nb_hChildren; c++){
      Cell child = item.hChild(c);
      //debug() << "[refineOneCell] child #"<<child->localId();
      ARCANE_ASSERT((child.isSubactive()), ("child must be a sub active item!"));
      Integer f = child.internal()->flags();
      f |= ItemInternal::II_JustAdded;
      f &= ~ItemInternal::II_Inactive;
      child.internal()->setFlags(f);
    }
  }

  // Maintenant, Unset le flag de raffinement de l'item
  //debug(Trace::High) << "[refineOneCell] et on flush le flag";
  Integer f = item.internal()->flags();
  f &= ~ItemInternal::II_Refine;
  f |= ItemInternal::II_Inactive;
  f |= ItemInternal::II_JustRefined;
  item.internal()->setFlags(f);
#if defined(ARCANE_DEBUG_ASSERT)
  for (Integer c = 0; c < nb_hChildren; c++){
    Cell hParent = item.hChild(c).hParent();
    //debug() << "[refineOneCell] child #"<<c;
    ARCANE_ASSERT((hParent == item), ("parent-child relationship is not consistent"));
    ARCANE_ASSERT((item.hChild(c).isActive()), ("children must be active"));
  }
  ARCANE_ASSERT((item.isAncestor()), ("current item must be an ancestor!"));
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//!
template <int typeID> void ItemRefinement::
coarsenOneCell(Cell item, const ItemRefinementPatternT<typeID>& rp)
{
  //! HexEmbeddingMatrix hex_em;
  //! refine(hex_em,item)

  ARCANE_ASSERT ( (item->internal()->flags() & ItemInternal::II_CoarsenInactive), ("Item is not for coarsening!"));
  ARCANE_ASSERT ( (!item->isActive()), ("Item is active!"));
  //debug(Trace::High) << "[coarsenOneCell] "<<item_internal->uniqueId();
  // ATT: Nous ne supprimons pas les enfants jusqu'à contraction via MeshRefinement::contract()

  // re-calcul des noeuds hanging
  IParallelMng* pm = m_mesh->parallelMng();
  const Integer sid = pm->commRank();
  computeOrigNodesCoords<typeID>(item,rp,sid);

  for (Integer c=0; c<item.nbHChildren(); c++){
    Cell mychild = item.hChild(c);
    //debug(Trace::High) << "\t[coarsenOneCell] child #"<<c << ' ' << mychild->uniqueId() << " " << mychild->owner();

    if (mychild->owner() != sid)
      continue;
    Integer f = mychild.internal()->flags();
    ARCANE_ASSERT ((f & ItemInternal::II_Coarsen),("Item is not flagged for coarsening"));
    f &= ~ItemInternal::II_Coarsen;
    f |= ItemInternal::II_Inactive;
    //      f |= ItemInternal::II_NeedRemove; // TODO activer le flag de suppression
    mychild.internal()->setFlags(f);
  }
  Integer f = item.internal()->flags();
  f &= ~ItemInternal::II_Inactive; // TODO verify if this condition is needed
  f &= ~ItemInternal::II_CoarsenInactive;
  f |= ItemInternal::II_JustCoarsened;
  //debug(Trace::High) << "[coarsenOneCell] item_internal->flags()="<<f;
  item.internal()->setFlags(f);

  ARCANE_ASSERT ( (item.isActive()), ("item must be active!"));
  //debug() << "[coarsenOneCell] done";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int typeID> void ItemRefinement::
computeHChildren(Cell item, MeshRefinement& mesh_refinement)
{
  debug(Trace::High) << "[refineOneCell] Computing new coordinates";

  const ItemRefinementPatternT<typeID>& rp = mesh_refinement.getRefinementPattern<typeID>();
  const Int32 nb_hChildren = rp.getNbHChildren();

  m_p.resize(nb_hChildren);
  m_nodes_uid.resize(nb_hChildren);

  const Int32 nb_nodes = item.nbNode();
  m_coord.resize(nb_nodes);
  for (Integer i = 0; i < nb_nodes; ++i)
    m_coord[i] = m_orig_nodes_coords[item.node(i)];

  const Integer head_size = 2;
  m_cells_infos.clear();
  m_cells_infos.reserve(nb_hChildren * (head_size + nb_nodes));
  debug(Trace::High) << "[refineOneCell] cells_infos reserved size is " << (nb_hChildren * (head_size + nb_nodes));

  Int64 first_cell_uid = mesh_refinement.getFirstChildNewUid();
  debug(Trace::Highest) << "[refineOneCell] first_cell_uid=" << first_cell_uid;

  m_faces_infos.clear();
  m_faces_infos.reserve(m_nb_face_to_add * ((m_mesh->dimension() == 3) ? 6 : (m_mesh->dimension() == 2) ? 4 : -1));

  const Real tol = m_cell_hmin[item] * TOLERENCE;

  typedef std::set<Int64> NodesSet;
  NodesSet nodes_set;
  Integer nb_cnodes_max_total = 0;

  ItemTypeMng* itm = ItemTypeMng::singleton();
  for (Integer c = 0; c < nb_hChildren; c++){
    const Integer c_type_id = rp.hChildrenTypeId(c);
    ItemTypeInfo* c_type = itm->typeFromId(c_type_id);

    const Integer nb_cnodes = c_type->nbLocalNode();
    nb_cnodes_max_total += nb_cnodes;
    m_p[c].resize(nb_cnodes);
    m_nodes_uid[c].resize(nb_cnodes);

    for (Integer nc = 0; nc < nb_cnodes; nc++){
      // initialisation
      m_p[c][nc] = Real3::null();
      m_nodes_uid[c][nc] = NULL_ITEM_ID;

      for (Integer n = 0; n < nb_nodes; n++){
        // La valeur à partir de la matrice de raffinement
        const Real em_val = rp.refine_matrix(c, nc, n);

        if (em_val != 0.){
          m_p[c][nc] += m_coord[n] * em_val;

          //nous avons pu trouver le noeud, dans ce cas nous
          //n'aurons pas besoin de le chercher plus tard ni de le creer.
          if (em_val == 1.){
            m_nodes_uid[c][nc] = item.node(n)->uniqueId().asInt64();
            nodes_set.insert(m_nodes_uid[c][nc]);
          }
        }
      }

      // assignation des noeuds aux enfants
      if (m_nodes_uid[c][nc] == NULL_ITEM_ID){
        m_nodes_uid[c][nc] = mesh_refinement.findOrAddNodeUid(m_p[c][nc], tol);
        debug(Trace::Highest) << "\t[refineOneCell] assigning node " << nc << " to l'uid:" << m_nodes_uid[c][nc];
      }
      m_nb_node_to_add = m_nb_node_to_add + 1;
    }

    // Création des mailles

    // Infos pour la création des mailles
    // par maille: 1 pour son unique id,
    //             1 pour son type,
    //             nb_nodes pour les uid de ses noeuds

    Int64 cell_unique_id = first_cell_uid + c;
    debug(Trace::Highest) << "[refineOneCell] CELL TYPE:" << c_type_id << ", uid=" << cell_unique_id;

    m_cells_infos.add(c_type_id);
    m_cells_infos.add(cell_unique_id);
    for (Integer nc = 0; nc < nb_cnodes; nc++)
      m_cells_infos.add(m_nodes_uid[c][nc]);
    m_nb_cell_to_add = m_nb_cell_to_add + 1;

    // Création des faces

    // Infos pour la création des faces
    // par face: 1 pour son unique id,
    //           1 pour son type,
    //           nb_nodes pour ses noeuds

    const Integer nb_cface = c_type->nbLocalFace();
    debug(Trace::High) << "[refineOneCell] nb faces à créer:" << nb_cface;

    for (Integer f = 0; f < nb_cface; f++){
      const Integer nb_node_face = c_type->localFace(f).nbNode();
      m_face.resize(nb_node_face);
      Real3 pfc = Real3::null();
      for (Integer nc = 0; nc < nb_node_face; nc++){
        const Integer node_face_rank = c_type->localFace(f).node(nc);
        m_face[nc] = m_nodes_uid[c][node_face_rank];
        pfc += m_p[c][node_face_rank];
      }
      pfc /= static_cast<Real>(nb_node_face);

      bool is_added = false;
      Int64 new_face_uid = mesh_refinement.findOrAddFaceUid(pfc, tol, is_added);
      debug(Trace::Highest) << "\t[refineOneCell] FACE TYPE:" << c_type->localFace(f).typeId();
      if (is_added) {
        m_faces_infos.add(c_type->localFace(f).typeId());
        m_faces_infos.add(new_face_uid);
        for (Integer nc = 0; nc < nb_node_face; nc++)
          m_faces_infos.add(m_face[nc]);
        m_nb_face_to_add = m_nb_face_to_add + 1;
      }
    }
  }

  m_nodes_to_create_coords.clear();
  m_nodes_to_create_coords.reserve(nb_cnodes_max_total);
  m_nodes_unique_id.clear();
  m_nodes_unique_id.reserve(nb_cnodes_max_total);

  Integer node_local_id = 0;
  debug(Trace::High) << "[refineOneCell] Create nodes and set their coordinates";
  for (Integer c = 0; c < nb_hChildren; c++){
    const Integer c_type_id = rp.hChildrenTypeId(c);
    ItemTypeInfo* c_type = itm->typeFromId(c_type_id);

    const Int32 nb_cnodes = c_type->nbLocalNode();
    for (Integer nc = 0; nc < nb_cnodes; nc++){
      const Int64 uid = m_nodes_uid[c][nc];
      if (nodes_set.find(uid) != nodes_set.end())
        continue;
      else{
        nodes_set.insert(uid);
        m_nodes_to_create_coords.add(m_p[c][nc]);
        m_nodes_unique_id.add(uid);
        ++node_local_id;
      }
    }
  }
  m_nb_node_to_add = node_local_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int typeID> void ItemRefinement::
computeOrigNodesCoords(Cell item, const ItemRefinementPatternT<typeID>& rp, const Integer sid)
{
  const Integer nb_nodes = item.nbNode();
  m_coord.resize(nb_nodes);
  for (Integer i = 0; i < nb_nodes; ++i)
    m_coord[i] = m_orig_nodes_coords[item.node(i)];

  for (Integer c = 0; c < item.nbHChildren(); c++){
    //debug() << "[coarsenOneCell] child #"<<c;
    Cell mychild = item.hChild(c);
    if (mychild.owner() != sid)
      continue;
    for (Integer nc = 0; nc < mychild.nbNode(); nc++){
      //debug() << "\t[coarsenOneCell] node #"<<nc;
      Real3 new_pos = Real3::null();
      bool calculated_new_pos = false;

      for (Integer n = 0; n < nb_nodes; n++){
        //debug() << "\t\t[coarsenOneCell] pos #"<<n;
        // La valeur à partir de la matrice de raffinement
        const Real em_val = rp.refine_matrix(c, nc, n);

        // La position du noeud est quelque part entre les sommets existants
        if ((em_val != 0.) && (em_val != 1.)){
          new_pos += em_val * m_coord[n];
          calculated_new_pos = true;
        }
      }

      if (calculated_new_pos)
        //Déplacement du noeud existant de nouveau dans sa position d'origine
        m_orig_nodes_coords[mychild.node(nc)] = new_pos;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE(typeID) \
template void ItemRefinement::refineOneCell<typeID>(Cell item_internal, MeshRefinement& mesh_refinement);\
template void ItemRefinement::coarsenOneCell<typeID>(Cell item_internal, const ItemRefinementPatternT<typeID>& rp);\
template void ItemRefinement::computeHChildren<typeID>(Cell item, MeshRefinement& mesh_refinement); \
template void ItemRefinement::computeOrigNodesCoords<typeID>(Cell item, const ItemRefinementPatternT<typeID>& rp, const Integer sid)

ARCANE_INSTANTIATE(IT_Quad4);
ARCANE_INSTANTIATE(IT_Tetraedron4);
ARCANE_INSTANTIATE(IT_Pyramid5);
ARCANE_INSTANTIATE(IT_Pentaedron6);
ARCANE_INSTANTIATE(IT_Hexaedron8);
ARCANE_INSTANTIATE(IT_HemiHexa7);
ARCANE_INSTANTIATE(IT_HemiHexa6);
ARCANE_INSTANTIATE(IT_HemiHexa5);
ARCANE_INSTANTIATE(IT_AntiWedgeLeft6);
ARCANE_INSTANTIATE(IT_AntiWedgeRight6);
ARCANE_INSTANTIATE(IT_DiTetra5);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

