// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConnectivityNewWithDependenciesTypes.h                      (C) 2000-2022 */
/*                                                                           */
/* Types utilisés dans le mode connectivités avec dépendances des familles   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_CONNECTIVITYNEWWITHDEPENDENCIESTYPES_H_
#define ARCANE_MESH_CONNECTIVITYNEWWITHDEPENDENCIESTYPES_H_
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/IMesh.h"
#include "arcane/ItemInternal.h"
#include "arcane/ItemTypeInfo.h"
#include "arcane/MeshUtils.h"
#include "arcane/ConnectivityItemVector.h"
#include "arcane/IItemFamilyNetwork.h"
#include "arcane/ItemPrinter.h"

#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/CompactIncrementalItemConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef IncrementalItemConnectivity ConnectivityType; // standard connectivity type

template <class SourceFamily, class TargetFamily>
class ARCANE_MESH_EXPORT CustomConnectivity
{
public:
  typedef ConnectivityType type;
};

template <class SourceFamily>
class LegacyConnectivityTraitsT
{};

template <>
class LegacyConnectivityTraitsT<NodeFamily>
{
public:
  typedef NodeInternalConnectivityIndex type;
};

template <>
class LegacyConnectivityTraitsT<FaceFamily>
{
public:
  typedef FaceInternalConnectivityIndex type;
};

template <>
class LegacyConnectivityTraitsT<EdgeFamily>
{
public:
  typedef EdgeInternalConnectivityIndex type;
};

template <>
class LegacyConnectivityTraitsT<CellFamily>
{
public:
  typedef CellInternalConnectivityIndex type;
};


template <class SourceFamily, class TargetFamily>
class ARCANE_MESH_EXPORT LegacyConnectivity
{
public:
  typedef typename LegacyConnectivityTraitsT<TargetFamily>::type type;
};


static String connectivityName(IItemFamily* source_family, IItemFamily* target_family)
{
  return String::concat(source_family->name(),target_family->name());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mutualisation pour les connectuvutés Face-Maille (Legacy ou New)
 */
 class FaceToCellConnectivity
 {
 public:
   FaceToCellConnectivity(IItemFamily* source_family,IItemFamily* target_family)
   : m_face_family(nullptr)
   , m_cell_family(nullptr) {
     if (source_family->itemKind() != IK_Face || target_family->itemKind()!= IK_Cell)
       throw FatalErrorException("FaceToCellIncrementalConnectivity must be created with face family as source and cell family as target. Exiting.");
     m_face_family = dynamic_cast<FaceFamily*>(source_family);
     m_cell_family = dynamic_cast<CellFamily*>(target_family);
     // todo supprimer le cast, le type concret n'est plus nécessaire
   }

 protected:
   bool isFrontCell(ItemLocalId source_item,ItemLocalId target_item)
   {
  ItemInternal* face = m_face_family->itemsInternal()[source_item.localId()];
  ItemInternal* cell = m_cell_family->itemsInternal()[target_item.localId()];
  // Find if cell is front or back for given face
  // get face and cell nodes
  IMesh* mesh = m_face_family->mesh();
  ConnectivityItemVector face_nodes_connectivity(mesh->itemFamilyNetwork()->getConnectivity(m_face_family,mesh->nodeFamily(), connectivityName(m_face_family,mesh->nodeFamily())));
  ItemVectorView face_nodes = face_nodes_connectivity.connectedItems(Face(face));
  ConnectivityItemVector cell_nodes_connectivity(mesh->itemFamilyNetwork()->getConnectivity(m_cell_family,mesh->nodeFamily(), connectivityName(m_cell_family,mesh->nodeFamily())));
  ItemVectorView cell_nodes = cell_nodes_connectivity.connectedItems(Cell(cell));
  ItemTypeInfo* cell_type_info = cell->typeInfo();
  // Find which local_face
  Integer face_index = -1;
  Int64UniqueArray face_node_uids;
  for (const auto& node : face_nodes)
    face_node_uids.add(node.uniqueId().asInt64());
  std::set<Int64> face_nodes_set(face_node_uids.begin(),face_node_uids.end());
  Int64UniqueArray local_face_node_uids;
  for (Integer local_face_index = 0; local_face_index < cell_type_info->nbLocalFace() && face_index == -1; ++local_face_index)
  {
    local_face_node_uids.clear();
    ItemTypeInfo::LocalFace local_face = cell_type_info->localFace(local_face_index);
    if (local_face.nbNode() != face_nodes.size()) continue;
    for (Integer local_node_index = 0; local_node_index < local_face.nbNode() ; ++local_node_index) local_face_node_uids.add(cell_nodes[local_face.node(local_node_index)].uniqueId().asInt64());
    std::set<Int64> local_face_nodes_set(local_face_node_uids.begin(),local_face_node_uids.end());
    if (local_face_nodes_set == face_nodes_set) face_index = local_face_index; // we found the face in the cell
  }
  if (face_index == -1) throw FatalErrorException(String::format("Face {0} does not belong to Cell {1}. Cannot connect. Exiting."));
  if (mesh->dimension() == 1) // 1d mesh : back cell is local face 0, front is local face 1
    return (face_index == 1);
  else
    {
      UniqueArray<Int64> ordered_face_node_ids(face_node_uids.size());
      if (mesh_utils::reorderNodesOfFace(local_face_node_uids,ordered_face_node_ids)) return true;
      else return false;
    }
}

void _checkValidSourceTargetItems(ItemInternal* source,ItemInternal* target)
{
#ifdef ARCANE_CHECK
  arcaneThrowIfNull(source,"source","Invalid null source item");
  arcaneThrowIfNull(target,"target","Invalid null target item");
#else
  ARCANE_UNUSED(source);
  ARCANE_UNUSED(target);
#endif
}

 protected:
     FaceFamily* m_face_family;
     CellFamily* m_cell_family;

 };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Surcharge du type IncrementalItemConnectivity pour gérer les connectivités Face -> BackCell et FrontCell.
 */
class ARCANE_MESH_EXPORT FaceToCellIncrementalItemConnectivity
: public FaceToCellConnectivity
, public IncrementalItemConnectivity
{
public:
  FaceToCellIncrementalItemConnectivity(IItemFamily* source_family,IItemFamily* target_family,const String& aname)
  : FaceToCellConnectivity(source_family,target_family)
  , IncrementalItemConnectivity(source_family,target_family,aname){}

  virtual ~FaceToCellIncrementalItemConnectivity() {}

//    const String& name() const override {return m_item_connectivity.name();}
//    ConstArrayView<IItemFamily*> families() const override {return m_item_connectivity.families();}
//    IItemFamily* sourceFamily() const override {return m_item_connectivity.sourceFamily();}
//    IItemFamily* targetFamily() const override {return m_item_connectivity.targetFamily();}

  void addConnectedItem(ItemLocalId source_item,ItemLocalId target_item) override
  {

    if (isFrontCell(source_item,target_item))
      _addFrontCellToFace(m_face_family->itemsInternal()[source_item.localId()],m_cell_family->itemsInternal()[target_item.localId()]);
    else
      _addBackCellToFace(m_face_family->itemsInternal()[source_item.localId()],m_cell_family->itemsInternal()[target_item.localId()]);
  }

public :

  void removeConnectedItem(ItemLocalId source_item,ItemLocalId target_item) override
  {
    _removeConnectedItem(m_face_family->itemsInternal()[source_item.localId()],target_item);
  }


private:


  void _addFrontCellToFace(ItemInternal* face, ItemInternal* cell){
    _checkValidSourceTargetItems(face,cell);

    Integer nb_cell = nbConnectedItem(ItemLocalId(face));

    // SDP: les tests suivants sont imcompatibles avec le raffinement
    // par couches
    // SDC: dans l'etat le check ne peut etre fait car les flags sont positionnes par la connectivite legacy.
    //      On peut donc avoir une front_cell qui n'est pas dans cette connectivite. C'est donc legal et de plus
    //      le check va planter puisque connectedItemLocalId(ItemLocalId(face),1) va retourner -1...
    //      Dans le nouveau mode les check devront etre faits via des proprietes.
    const bool check_orientation = false; // SDC cf. ci-dessus
    if (check_orientation){
      if (face->flags() & ItemFlags::II_HasFrontCell){
        ItemInternal* current_cell = m_cell_family->itemsInternal()[connectedItemLocalId(ItemLocalId(face),1)]; // FrontCell is the second connected cell
        ARCANE_FATAL("Face already having a front cell."
            " This is most probably due to the fact that the face"
            " is connected to a reverse cell with a negative volume."
            " Face={0}. new_cell={1} current_cell={2}",face->uniqueId().asInt64(),
            cell->uniqueId().asInt64(),current_cell->uniqueId().asInt64()); // FullItemPrinter cannot be used here, the connectivities are not entirely set
      }
    }

    if (nb_cell>=2)
      ARCANE_FATAL("face '{0}' already has two cells",face->uniqueId().asInt64());// FullItemPrinter cannot be used here, the connectivities are not entirely set

    // Si on a déjà une maille, il s'agit de la back cell.
    Int32 iback_cell_lid = (nb_cell==1) ? face->cellId(0) : NULL_ITEM_LOCAL_ID;
    ItemLocalId back_cell_lid(iback_cell_lid);
    ItemLocalId front_cell_lid(cell->localId());
    _setBackAndFrontCells(face,back_cell_lid,front_cell_lid);


  }

  void _addBackCellToFace(ItemInternal* face, ItemInternal* cell){
    _checkValidSourceTargetItems(face,cell);

    Integer nb_cell = nbConnectedItem(ItemLocalId(face));

    // SDP: les tests suivants sont imcompatibles avec le raffinement
    // par couches
    // SDC: dans l'etat le check ne peut etre fait car les flags sont positionnes par la connectivite legacy.
    //      On peut donc avoir une back_cell qui n'est pas dans cette connectivite. C'est donc legal et de plus
    //      le check va planter puisque connectedItemLocalId(ItemLocalId(face),0) va retourner -1.
    //      Dans le nouveau mode les check devront etre faits via des proprietes.
    const bool check_orientation = false; // SDC cf. ci-dessus
    if (check_orientation){
      if (face->flags() & ItemFlags::II_HasBackCell){
        ItemInternal* current_cell = m_cell_family->itemsInternal()[connectedItemLocalId(ItemLocalId(face),0)]; // BackCell is the first connected cell
        ARCANE_FATAL("Face already having a back cell."
            " This is most probably due to the fact that the face"
            " is connected to a reverse cell with a negative volume."
            " Face={0}. new_cell={1} current_cell={2}",face->uniqueId().asInt64(),
            cell->uniqueId().asInt64(),current_cell->uniqueId().asInt64()); // FullItemPrinter cannot be used here, the connectivities are not entirely set
      }
    }

    if (nb_cell>=2)
      ARCANE_FATAL("face '{0}' already has two cells",face->uniqueId().asInt64());// FullItemPrinter cannot be used here, the connectivities are not entirely set
    // Si on a déjà une maille, il s'agit de la front cell.
    Int32 ifront_cell_lid = (nb_cell==1) ? face->cellId(0) : NULL_ITEM_LOCAL_ID;

    ItemLocalId back_cell_lid(cell->localId());
    ItemLocalId front_cell_lid(ifront_cell_lid);
    _setBackAndFrontCells(face,back_cell_lid,front_cell_lid);

  }

private:

  void _setBackAndFrontCells(ItemInternal* face,ItemLocalId back_cell_lid,ItemLocalId front_cell_lid){
    ItemLocalId face_lid(face->localId());
    // Supprime toutes les mailles connectées => la méthode est mutualisée pour les ajouts ou suppressions
    // TODO: optimiser en ne supprimant pas s'il n'y a pas besoin pour éviter
    // des réallocations.
    removeConnectedItems(face_lid);
    Int32 mod_flags = 0;
    if (front_cell_lid==NULL_ITEM_LOCAL_ID){
      if (back_cell_lid!=NULL_ITEM_LOCAL_ID){
        // Reste uniquement la back_cell
        IncrementalItemConnectivity::addConnectedItem(face_lid,back_cell_lid); // add the class name for the case of Face to Cell connectivity. The class is overridden to handle family dependencies but the base method must be called here.
        // add flags
        mod_flags = (ItemFlags::II_Boundary | ItemFlags::II_HasBackCell | ItemFlags::II_BackCellIsFirst);
      }
      // Ici reste aucune maille mais comme on a tout supprimé il n'y a rien
      // à faire
    }
    else if (back_cell_lid==NULL_ITEM_LOCAL_ID){
      // Reste uniquement la front cell
      IncrementalItemConnectivity::addConnectedItem(face_lid,front_cell_lid);
      // add flags
      mod_flags = (ItemFlags::II_Boundary | ItemFlags::II_HasFrontCell | ItemFlags::II_FrontCellIsFirst);
    }
    else{
      // Il y a deux mailles connectées. La back_cell est toujours la première.
      IncrementalItemConnectivity::addConnectedItem(face_lid,back_cell_lid);
      IncrementalItemConnectivity::addConnectedItem(face_lid,front_cell_lid);
      // add flags
      mod_flags = (ItemFlags::II_HasFrontCell | ItemFlags::II_HasBackCell | ItemFlags::II_BackCellIsFirst);
    }
    Int32 face_flags = face->flags();
    face_flags &= ~ItemFlags::II_InterfaceFlags;
    face_flags |= mod_flags;
    face->setFlags(face_flags);
  }

  void _removeConnectedItem(ItemInternal* face,ItemLocalId cell_to_remove_lid)
  {
    // Code taken from FaceFamily::removeCellFromFace where new and legacy are separated.
    // Here only for new connectivity.
    // This duplication allows to reach the specificity of Cell/Face connectivity through the unique interface removeConnectedItem (instead of removeCellFromFace)
    _checkValidSourceTargetItems(face,m_cell_family->itemsInternal()[cell_to_remove_lid]);

    Integer nb_cell = nbConnectedItem(ItemLocalId(face));

#ifdef ARCANE_CHECK
    if (face->isSuppressed())
      ARCANE_FATAL("Can not remove cell from destroyed face={0}",ItemPrinter(face));
    if (nb_cell==0)
      ARCANE_FATAL("Can not remove cell lid={0} from face uid={1} with no cell connected",
                   cell_to_remove_lid, face->uniqueId());
#endif /* ARCANE_CHECK */

    Integer nb_cell_after = nb_cell-1;
    const Int32 null_cell_lid = NULL_ITEM_LOCAL_ID;

    //! AMR : todo later (go back to FaceFamily::removeCellFromFace)

    // OFF AMR
    if (nb_cell_after!=0){
      Int32 cell0 = face->cellId(0);
      Int32 cell1 = face->cellId(1);
      // On avait obligatoirement deux mailles connectées avant,
      // donc la back_cell est la maille 0, la front cell la maille 1
      if (cell0==cell_to_remove_lid){
        // Reste la front cell
        _setBackAndFrontCells(face,ItemLocalId(null_cell_lid),ItemLocalId(cell1));
      }
      else{
        // Reste la back cell
        _setBackAndFrontCells(face,ItemLocalId(cell0),ItemLocalId(null_cell_lid));
      }
    }
    else{
      _setBackAndFrontCells(face,ItemLocalId(null_cell_lid),ItemLocalId(null_cell_lid));
    }
  }
};

template<>
class ARCANE_MESH_EXPORT CustomConnectivity<FaceFamily,CellFamily>
{
public:
  typedef FaceToCellIncrementalItemConnectivity type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#endif /* ARCANE_CONNECTIVITYNEWWITHDEPENDENCIESTYPES_H_ */
