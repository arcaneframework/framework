// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodeFamily.h                                                (C) 2000-2021 */
/*                                                                           */
/* Famille de noeuds.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_NODEFAMILY_H
#define ARCANE_MESH_NODEFAMILY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ItemFamily.h"
#include "arcane/IItemFamilyModifier.h"
#include "arcane/mesh/MeshInfos.h"
#include "arcane/mesh/ItemInternalConnectivityIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Famille de noeuds.
 */
class ARCANE_MESH_EXPORT NodeFamily
: public ItemFamily
, public IItemFamilyModifier
{
  class ItemCompare2;
  class ItemCompare3;
  class TopologyModifier;

  typedef ItemConnectivitySelectorT<EdgeInternalConnectivityIndex,IncrementalItemConnectivity> EdgeConnectivity;
  typedef ItemConnectivitySelectorT<FaceInternalConnectivityIndex,IncrementalItemConnectivity> FaceConnectivity;
  typedef ItemConnectivitySelectorT<CellInternalConnectivityIndex,IncrementalItemConnectivity> CellConnectivity;

 public:

  NodeFamily(IMesh* mesh,const String& name);
  virtual ~NodeFamily(); //<! Libère les ressources

 public:

  void build() override;
  void computeSynchronizeInfos() override;
  void endAllocate() override;

 public:

  void preAllocate(Integer nb_item);

 public:

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  /*!
   * \brief Alloue une face de numéro unique \a uid et de type \a type.
   *
   * Cette version est faite pour être appelée dans un bloc générique ignorant le type
   * de l'item. La mise à jour du nombre d'item du maillage est donc fait dans cette méthode,
   * et non dans le bloc appelant.
   */

  // IItemFamilyModifier interface
  ItemInternal* allocOne(Int64 uid,ItemTypeInfo* type, MeshInfos& mesh_info) override
  {
    ARCANE_ASSERT((type->typeId() == IT_Vertex),("Creating node with a type not equal to IT_Vertex"));
    ARCANE_UNUSED(type);
    ++mesh_info.nbNode();
    return allocOne(uid);
  }

  ItemInternal* allocOne(Int64 uid)
  {
    ItemInternal* item = _allocOne(uid);
    m_item_internal_list->nodes = _itemsInternal();
    _allocateInfos(item,uid,m_node_type);
    return item;
  }

  // IItemFamilyModifier interface
  ItemInternal* findOrAllocOne(Int64 uid,ItemTypeInfo* type,MeshInfos& mesh_info, bool& is_alloc) override
  {
    ARCANE_ASSERT((type->typeId() == IT_Vertex),("Creating node with a type not equal to IT_Vertex"));
    ARCANE_UNUSED(type);
    auto node = findOrAllocOne(uid,is_alloc);
    if (is_alloc) ++mesh_info.nbNode();
    return node;
  }

  ItemInternal* findOrAllocOne(Int64 uid,bool& is_alloc)
  {
    ItemInternal* item = _findOrAllocOne(uid,is_alloc);
    if (is_alloc){
      m_item_internal_list->nodes = _itemsInternal();
      _allocateInfos(item,uid,m_node_type);
    }
    return item;
  }

  // IItemFamilyModifier interface
  IItemFamily* family() override {return this;}

  void replaceEdge(ItemLocalId node,Integer index,ItemLocalId edge);
  void replaceFace(ItemLocalId node,Integer index,ItemLocalId face);
  void replaceCell(ItemLocalId node,Integer index,ItemLocalId cell);

  void addCellToNode(ItemInternal* node,ItemInternal* new_cell);
  void addFaceToNode(ItemInternal* node,ItemInternal* new_face);
  void addEdgeToNode(ItemInternal* node,ItemInternal* new_edge);
  ARCANE_DEPRECATED_260 void removeEdgeFromNode(ItemInternal* node,ItemInternal* edge_to_remove);
  void removeEdgeFromNode(ItemLocalId node,ItemLocalId edge_to_remove);
  ARCANE_DEPRECATED_260 void removeFaceFromNode(ItemInternal* node,ItemInternal* face_to_remove);
  void removeFaceFromNode(ItemLocalId node,ItemLocalId face_to_remove);
  //! \deprecated Utiliser la version removeCellFromNode(ItemInternal*,ItemLocalId)
  ARCANE_DEPRECATED_240 void removeCellFromNode(ItemInternal* node,ItemInternal* cell_to_remove,bool no_destroy=false);
  void removeCellFromNode(ItemInternal* node,ItemLocalId cell_to_remove_lid);
  //! Supprime le noeud siln'est plus connecté
  void removeNodeIfNotConnected(ItemInternal* node);

  VariableNodeReal3& nodesCoordinates()
  {
    ARCANE_ASSERT((m_nodes_coords),("NodeFamily::nodesCoordinates is available only on primary meshes"));
    return *m_nodes_coords;
  }

  //! Définit la connectivité active pour le maillage associé
  /*! Ceci conditionne les connectivités à la charge de cette famille */
  void setConnectivity(const Integer c);

  void sortInternalReferences();

  void notifyItemsUniqueIdChanged() override;

 private:
  
  ItemTypeInfo* m_node_type = nullptr; //!< Instance contenant le type des noeuds
  Integer m_edge_prealloc;
  Integer m_face_prealloc;
  Integer m_cell_prealloc;
  Integer m_mesh_connectivity;
  bool m_no_face_connectivity;
  VariableNodeReal3* m_nodes_coords = nullptr;
  EdgeConnectivity* m_edge_connectivity = nullptr;
  FaceConnectivity* m_face_connectivity = nullptr;
  CellConnectivity* m_cell_connectivity = nullptr;
  FaceFamily* m_face_family = nullptr;
  inline void _removeNode(ItemInternal* node);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

