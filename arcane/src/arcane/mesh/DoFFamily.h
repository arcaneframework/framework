// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoFFamily.h                                             (C) 2000-2014     */
/*                                                                           */
/* Famille de degres de liberte                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DOFFAMILY_H_ 
#define ARCANE_DOFFAMILY_H_ 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/ItemTypes.h"
#include "arcane/IMesh.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"

#include "arcane/mesh/DynamicMeshIncrementalBuilder.h" // can be removed, used only for ENUMERATE_ITEM_INTERNAL_MAP_DATA macro...


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DoFUids
{
  /*!
   * Utilitaire d'obtention d'identifiant unique. Version triviale pour experimentation.
   */
public:
  // item with one dof
  static Int64 uid(Int64 connected_item_uid) {return connected_item_uid;} // Trivial.

  // item with several dof
  static Int64 uid(Int64 max_dof_family_uid,
                   Int64 max_connected_item_family_uid,
                   Int64 connected_item_uid,
                   Int32 dof_index_in_item)
    { return connected_item_uid+(max_connected_item_family_uid+1)*dof_index_in_item + max_dof_family_uid+1;} // very temporary solution...

  // utilities
  static Int64 getMaxItemUid(IItemFamily* family)
  {
    Int64 max_uid = 0;
    // This method can be used within IItemFamily::endUpdate when new items have been created but the groups are not yet updated.
    // Therefore we use internal map enumeration instead of group enumeration
    ItemFamily* item_family = static_cast<ItemFamily*>(family);
    ENUMERATE_ITEM_INTERNAL_MAP_DATA(item,item_family->itemsMap())
    {
      if (max_uid < item->value()->uniqueId().asInt64()) max_uid = item->value()->uniqueId().asInt64();
    }
    Int64 pmax_uid = family->mesh()->parallelMng()->reduce(Parallel::ReduceMax,max_uid);
    return pmax_uid;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT DoFFamily : public ItemFamily
{

  /** Constructeur de la classe  */
private:

  //! La famille ne peut pas etre cree directement, il faut utiliser le DoFManager
  DoFFamily(IMesh* mesh, const String& name);

public:
  /** Destructeur de la classe */
  virtual ~DoFFamily() {}

public:

  //! En entree les uids des dofs et on recupere leurs lids
  DoFVectorView addDoFs(Int64ConstArrayView dof_uids, Int32ArrayView dof_lids);

  //! L'ajout de fantomes doit etre suivi d'un appel de computeSynchronizeInfos
  DoFVectorView addGhostDoFs(Int64ConstArrayView dof_uids, Int32ArrayView dof_lids, Int32ConstArrayView owners);

  //! Operation collective
  void computeSynchronizeInfos();

  DoFGroup allDoFs() {return allItems();}
  DoFGroup ownDoFs() {return allItems().own();}
  DoFGroup ghostDoFs() {return allItems().ghost();}

  void removeDoFs(Int32ConstArrayView items_local_id);

private:

  void build(); //! Construction de l'item Family. C'est le DoFManager qui en a la responsabilite.
  void addItems(Int64ConstArrayView unique_ids, Int32ArrayView items);
  void addGhostItems(Int64ConstArrayView unique_ids, Int32ArrayView items, Int32ConstArrayView owners);
  void removeItems(Int32ConstArrayView local_ids,bool keep_ghost =false) {internalRemoveItems(local_ids,keep_ghost);};
  void internalRemoveItems(Int32ConstArrayView local_ids,bool keep_ghost=false);
//  void compactItems(bool do_sort) {m_need_prepare_dump = false;} //! Surcharge ItemFamily::compactItems car pas de compactage pour l'instant dans les DoFs.


  // FOR DEBUG
  void _printInfos(Integer nb_added);

private:

  void preAllocate(Integer nb_item);
  ItemInternal* _allocDoF(const Int64 uid);
  ItemInternal* _allocDoFGhost(const Int64 uid, const Int32 owner);

  ItemSharedInfo* m_shared_info;

  friend class DynamicMesh;

};

ARCANE_MESH_END_NAMESPACE

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* DOFFAMILY_H_ */
