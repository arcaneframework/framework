// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartitionConstraintMng.cc                               (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire de contraintes de partitionnement de maillage.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/Deleter.h"

#include "arcane/IMesh.h"
#include "arcane/IParallelMng.h"
#include "arcane/IItemFamily.h"
#include "arcane/Item.h"
#include "arcane/VariableTypes.h"
#include "arcane/IMeshPartitionConstraint.h"

#include "arcane/mesh/MeshPartitionConstraintMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Helper pour la gestion des contrainte.
 *
 * Il faut d'abord appeler merge(), puis changeOwners()
 * ou computeList().
 */
class MeshPartitionConstraintMng::Helper
: public TraceAccessor
{
 private:
  typedef  HashTableMapT<Int64,Int64>::Data ItemFirstUidData;
 public:
  Helper(IMesh* mesh,bool is_debug)
  : TraceAccessor(mesh->traceMng()),
    m_mesh(mesh), m_is_debug(is_debug), m_true_first_uid(5000,true),
    m_uids_owner(5000,true)
  {
  }
 public:
  void merge(Int64Array& linked_items,Int32Array& linked_owners);
  void changeOwners();
  void computeList(MultiArray2<Int64> & tied_uids);
 private:
  IMesh* m_mesh;
  bool m_is_debug;
  HashTableMapT<Int64,Int64> m_true_first_uid;
  HashTableMapT<Int64,Int32> m_uids_owner;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshPartitionConstraintMng::
MeshPartitionConstraintMng(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_is_debug(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshPartitionConstraintMng::
~MeshPartitionConstraintMng()
{
  m_constraints.each(Deleter());
  m_weak_constraints.each(Deleter());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::
addConstraint(IMeshPartitionConstraint* constraint)
{
  m_constraints.add(constraint);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::
removeConstraint(IMeshPartitionConstraint* constraint)
{
  m_constraints.remove(constraint);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::
computeAndApplyConstraints()
{
  Helper h(m_mesh,m_is_debug);
  _computeAndApplyConstraints(h);
  h.changeOwners();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::
computeConstraintList(Int64MultiArray2 & tied_uids)
{
  Helper h(m_mesh,m_is_debug);
  _computeAndApplyConstraints(h);
  h.computeList(tied_uids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::
_computeAndApplyConstraints(Helper& h)
{
  Int64UniqueArray linked_cells;
  Int32UniqueArray linked_owners;
  for( List<IMeshPartitionConstraint*>::Enumerator i(m_constraints); ++i; ){
    IMeshPartitionConstraint* c = *i;
    c->addLinkedCells(linked_cells,linked_owners);
  }
  h.merge(linked_cells,linked_owners);
}

void MeshPartitionConstraintMng::
_computeAndApplyWeakConstraints(Helper& h)
{
  Int64UniqueArray linked_cells;
  Int32UniqueArray linked_owners;
  for( List<IMeshPartitionConstraint*>::Enumerator i(m_weak_constraints); ++i; ){
    IMeshPartitionConstraint* c = *i;
    c->addLinkedCells(linked_cells,linked_owners);
  }
  h.merge(linked_cells,linked_owners);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::
addWeakConstraint(IMeshPartitionConstraint* constraint)
{
  m_weak_constraints.add(constraint);
}

void MeshPartitionConstraintMng::
removeWeakConstraint(IMeshPartitionConstraint* constraint)
{
  m_weak_constraints.remove(constraint);
}

void MeshPartitionConstraintMng::
computeAndApplyWeakConstraints()
{
  Helper h(m_mesh,m_is_debug);
  _computeAndApplyWeakConstraints(h);
  h.changeOwners();
}

void MeshPartitionConstraintMng::
computeWeakConstraintList(Int64MultiArray2 & tied_uids)
{
  Helper h(m_mesh,m_is_debug);
  _computeAndApplyWeakConstraints(h);
  h.computeList(tied_uids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::Helper::
merge(Int64Array& linked_items,Int32Array& linked_owners)
{
  //TODO: ne pas faire un allGather mais plutot dispatcher la liste
  // sur plusieurs processeurs, chaque processeur faisant
  // le tri sur ses entités locales et ensuite faire
  // plusieurs itérations pour fusionner.
  IParallelMng* pm = m_mesh->parallelMng();
  Int64UniqueArray global_linked_items;
  pm->allGatherVariable(linked_items.view(),global_linked_items);
  Int32UniqueArray global_linked_owners;
  pm->allGatherVariable(linked_owners.view(),global_linked_owners);
  Integer nb_global = global_linked_items.size() / 2;
  info() << "NB_GLOBAL_LINKED_CELL=" << nb_global;
  //ItemLinkedList linked_list;
  m_true_first_uid.clear();
  m_uids_owner.clear();
  Integer nb_changed = 0;
  Integer nb_iter = 0;
  if (m_is_debug){
    for( Integer i=0; i<nb_global; ++i ){
      Int64 first_uid = global_linked_items[(i*2)];
      Int64 second_uid = global_linked_items[(i*2)+1];
      Int32 first_owner = global_linked_owners[i];
      info() << "LinkedItem: first_uid=" << first_uid
             << " second_uid=" << second_uid
             << " first_owner=" << first_owner;
    }
  }
  do {
    nb_changed = 0;
    ++nb_iter;
    for( Integer i=0; i<nb_global; ++i ){
      Int64 first_uid = global_linked_items[(i*2)];
      Int64 second_uid = global_linked_items[(i*2)+1];
      if (nb_iter==1){
        Int32 first_owner = global_linked_owners[i];
        m_uids_owner.add(first_uid,first_owner);
      }
      // IL FAUT TOUJOURS first_uid < second_uid
      ItemFirstUidData* first_data = m_true_first_uid.lookup(first_uid);
      ItemFirstUidData* second_data = m_true_first_uid.lookup(second_uid);
      Int64 first_data_uid = first_uid;
      Int64 second_data_uid = second_uid;
      if (first_data)
        first_data_uid = first_data->value();
      if (second_data)
        second_data_uid = second_data->value();
      Int64 smallest_uid = math::min(first_data_uid,second_data_uid);
      if (second_data){
        if (second_data_uid>smallest_uid){
          if (m_is_debug)
            info() << "Changed second current=" << second_data->value() << " to " << smallest_uid
                   << " (first=" << first_uid << ",second=" << second_uid << ")";
          second_data->setValue(smallest_uid);
          ++nb_changed;
        }
      }
      else{
        if (m_is_debug)
          info() << "Changed second add current=" << second_uid << " to " << smallest_uid
                 << " (first=" << first_uid << ",second=" << second_uid << ")";
        // Attention, cela peut invalider first_data et second_data
        m_true_first_uid.add(second_uid,smallest_uid);
        ++nb_changed;
      }

      if (first_data){
        if (first_data_uid>smallest_uid){
          if (m_is_debug)
            info() << "Changed first current=" << first_data->value() << " to " << smallest_uid
                   << " (first=" << first_uid << ",second=" << second_uid << ")";
          first_data->setValue(smallest_uid);
          ++nb_changed;
        }
      }
      else{
        if (m_is_debug)
          info() << "Changed first add current=" << first_uid << " to " << smallest_uid
                 << " (first=" << first_uid << ",second=" << second_uid << ")";
        // Attention, cela peut invalider first_data et second_data
        m_true_first_uid.add(first_uid,smallest_uid);
        ++nb_changed;
      }
      //linked_list.add(first_uid,second_uid);
    }
    debug(Trace::High) << "NB_CHANGED=" << nb_changed << " iter=" << nb_iter;
  } while (nb_changed!=0 && nb_iter<100);
  if (nb_iter>=100)
    fatal() << "Too many iterations";
  // Ajoute tous les uids qui ne sont pas encore dans la table de hashage.
  // Leur propriétaire est celui directement spécifié dans linked_cells
  for( Integer i=0; i<nb_global; ++i ){
    Int64 first_uid = global_linked_items[(i*2)];
    Int64 second_uid = global_linked_items[(i*2)+1];
    ItemFirstUidData* first_data = m_true_first_uid.lookup(first_uid);
    if (!first_data)
      m_true_first_uid.add(first_uid,first_uid);
    ItemFirstUidData* second_data = m_true_first_uid.lookup(second_uid);
    if (!second_data)
      m_true_first_uid.add(second_uid,first_uid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::Helper::
changeOwners()
{
  // Positionne les propriétaires pour chaque maille
  // de notre sous-domaine.
  Int64UniqueArray cells_uid;
  Int32UniqueArray cells_owner;

  for( auto i : m_true_first_uid.buckets() )
    for( ItemFirstUidData* nbid = i; nbid; nbid = nbid->next() ){
      cells_uid.add(nbid->key());
      cells_owner.add(m_uids_owner[nbid->value()]);
    }
  Integer nb_cell = cells_uid.size();
  Int32UniqueArray cells_local_id(nb_cell);
  IItemFamily* cell_family = m_mesh->cellFamily();
  cell_family->itemsUniqueIdToLocalId(cells_local_id,cells_uid,false);
  VariableItemInt32& cells_new_owner = cell_family->itemsNewOwner();
  CellInfoListView cells_internal(cell_family);
  for( Integer i=0; i<nb_cell; ++i ){
    Int32 lid = cells_local_id[i];
    if (lid!=NULL_ITEM_LOCAL_ID){
      Cell cell = cells_internal[lid];
      if (m_is_debug)
        info() << "Change cell owner uid=" << cell.uniqueId() << " old=" << cell.owner()
               << " new=" << cells_owner[i];
      cells_new_owner[cell] = cells_owner[i];
    }
  }
  // Pas besoin de synchroniser car le maillage s'en occupe, ni de mettre
  // à jour les autres types d'éléments.
  cells_new_owner.synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionConstraintMng::Helper::
computeList(MultiArray2<Int64> & tied_uids)
{
  IntegerUniqueArray nb_tieds;
  nb_tieds.reserve(1000);
  HashTableMapT<Int64,Integer> uids_tied_index(5000,true);

  for( auto i : m_true_first_uid.buckets() )
    for( ItemFirstUidData* nbid = i; nbid; nbid = nbid->next() ){
      Int64 tied_uid = nbid->value();
      HashTableMapT<Int64,Integer>::Data* d = uids_tied_index.lookup(tied_uid);
      if (d)
        ++nb_tieds[d->value()];
      else{
        Integer index = nb_tieds.size();
        uids_tied_index.add(tied_uid,index);
        nb_tieds.add(1);
      }
    }

  info() << "NB_TIED=" << nb_tieds.size();
  tied_uids.resize(nb_tieds);
  nb_tieds.fill(0);

  for( auto i : m_true_first_uid.buckets() )
    for( ItemFirstUidData* nbid = i; nbid; nbid = nbid->next() ){
      Int64 tied_uid = nbid->value();
      Integer index = uids_tied_index[tied_uid];
      Integer index2 = nb_tieds[index];
      ++nb_tieds[index];
      tied_uids.setAt(index,index2,nbid->key());
    }
#ifdef ARCANE_DEBUG
  for( Integer i=0, n=tied_uids.dim1Size(); i<n; ++i ){
    Int64ConstArrayView uids(tied_uids[i]);
    debug(Trace::Highest) << " N=" << uids.size();
    for( Integer j=0, js=uids.size(); j<js; ++j ){
      debug(Trace::Highest) << " " << uids[j];
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
