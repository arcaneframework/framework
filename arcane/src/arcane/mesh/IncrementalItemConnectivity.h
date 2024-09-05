// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IncrementalItemConnectivity.h                               (C) 2000-2024 */
/*                                                                           */
/* Connectivité incrémentale des entités.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_INCREMENTALITEMCONNECTIVITY_H
#define ARCANE_MESH_INCREMENTALITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IIncrementalItemConnectivity.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IncrementalItemConnectivityContainer;
class IndexedItemConnectivityAccessor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe abstraite de gestion des connectivités.
 *
 * Cette classe gère les informations communes à tous les types de
 * connectivité comme son nom, les familles sources et cible, ...
 */
class ARCANE_MESH_EXPORT AbstractIncrementalItemConnectivity
: public TraceAccessor
, public ReferenceCounterImpl
, public IIncrementalItemConnectivity
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  AbstractIncrementalItemConnectivity(IItemFamily* source_family,
                                      IItemFamily* target_family,
                                      const String& connectivity_name);

 public:

  String name() const final { return m_name; }

 public:

  ConstArrayView<IItemFamily*> families() const override { return m_families.constView();}
  IItemFamily* sourceFamily() const override { return m_source_family;}
  IItemFamily* targetFamily() const override { return m_target_family;}

  Ref<IIncrementalItemSourceConnectivity> toSourceReference() override;
  Ref<IIncrementalItemTargetConnectivity> toTargetReference() override;

 protected:

  ConstArrayView<IItemFamily*> _families() const { return m_families.constView();}
  IItemFamily* _sourceFamily() const { return m_source_family;}
  IItemFamily* _targetFamily() const { return m_target_family;}

 private:

  IItemFamily* m_source_family;
  IItemFamily* m_target_family;
  SharedArray<IItemFamily*> m_families;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base pour les connectivités incrémentales item->item[].
 */
class ARCANE_MESH_EXPORT IncrementalItemConnectivityBase
: public AbstractIncrementalItemConnectivity
{
 public:

  template <class SourceFamily, class TargetFamily, class LegacyType, class CustomType>
  friend class NewWithLegacyConnectivity;

 public:

  IncrementalItemConnectivityBase(IItemFamily* source_family,IItemFamily* target_family,
                                  const String& aname);
  ~IncrementalItemConnectivityBase() override;

 public:

  // On interdit la copie à cause de \a m_p
  IncrementalItemConnectivityBase(const IncrementalItemConnectivityBase&) = delete;
  IncrementalItemConnectivityBase(IncrementalItemConnectivityBase&&) = delete;
  IncrementalItemConnectivityBase& operator=(const IncrementalItemConnectivityBase&) = delete;
  IncrementalItemConnectivityBase& operator=(IncrementalItemConnectivityBase&&) = delete;

 public:

  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override;
  void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) override;
  Integer nbConnectedItem(ItemLocalId lid) const final
  {
    return m_connectivity_nb_item[lid];
  }
  Int32 connectedItemLocalId(ItemLocalId lid,Integer index) const final
  {
    return m_connectivity_list[ m_connectivity_index[lid] + index ];
  }

  IndexedItemConnectivityViewBase connectivityView() const;
  IndexedItemConnectivityAccessor connectivityAccessor() const;
  ItemConnectivityContainerView connectivityContainerView() const;

  Int32 maxNbConnectedItem() const override;

  void reserveMemoryForNbSourceItems(Int32 n, bool pre_alloc_connectivity) override;

 public:

  Int32ConstArrayView _connectedItemsLocalId(ItemLocalId lid) const
  {
    Int32 nb = m_connectivity_nb_item[lid];
    Int32 index = m_connectivity_index[lid];
    return { nb, &m_connectivity_list[index] };
  }
  
  // TODO: voir si on garde cette méthode. A utiliser le moins possible.
  Int32ArrayView _connectedItemsLocalId(ItemLocalId lid)
  {
     Int32 nb = m_connectivity_nb_item[lid];
     Int32 index = m_connectivity_index[lid];
     return { nb, &m_connectivity_list[index] };
  }

 public:

  Int32ArrayView connectivityIndex() { return m_connectivity_index; }
  Int32ArrayView connectivityList() { return m_connectivity_list; }

  void setItemConnectivityList(ItemInternalConnectivityList* ilist,Int32 index);
  void dumpInfos();

 protected:

  void _initializeStorage(ConnectivityItemVector* civ) override
  {
    ARCANE_UNUSED(civ);
  }
  ItemVectorView _connectedItems(ItemLocalId item,ConnectivityItemVector& con_items) const final;

 protected:

  bool m_is_empty = true;
  Int32ArrayView m_connectivity_nb_item;
  Int32ArrayView m_connectivity_index;
  Int32ArrayView m_connectivity_list;
  IncrementalItemConnectivityContainer* m_p = nullptr;
  ItemInternalConnectivityList* m_item_connectivity_list = nullptr;
  Integer m_item_connectivity_index;

 protected:

  void _notifyConnectivityListChanged();
  void _notifyConnectivityIndexChanged();
  void _notifyConnectivityNbItemChanged();
  void _notifyConnectivityNbItemChangedFromObservable();
  void _computeMaxNbConnectedItem();
  void _setNewMaxNbConnectedItems(Int32 new_max);
  void _setMaxNbConnectedItemsInConnectivityList();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Connectivité incrémentale item->item[]
 */
class ARCANE_MESH_EXPORT IncrementalItemConnectivity
: public IncrementalItemConnectivityBase
{
 private:

  //! Pour accès à _internalNotifySourceItemsAdded().
  friend class IndexedIncrementalItemConnectivityMng;

 public:

  IncrementalItemConnectivity(IItemFamily* source_family,IItemFamily* target_family,
                              const String& aname);
  ~IncrementalItemConnectivity() override;

 public:

  void addConnectedItems(ItemLocalId source_item,Integer nb_item);
  void removeConnectedItems(ItemLocalId source_item) override;
  void addConnectedItem(ItemLocalId source_item,ItemLocalId target_local_id) override;
  void removeConnectedItem(ItemLocalId source_item,ItemLocalId target_local_id) override;
  void replaceConnectedItem(ItemLocalId source_item,Integer index,ItemLocalId target_local_id) override;
  void replaceConnectedItems(ItemLocalId source_item,Int32ConstArrayView target_local_ids) override;
  bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const override;
  void notifySourceItemAdded(ItemLocalId item) override;
  void notifyReadFromDump() override;

 private:

  void _internalNotifySourceItemsAdded(ConstArrayView<Int32> local_ids) override;

 public:

  Integer preAllocatedSize() const final { return m_pre_allocated_size; }
  void setPreAllocatedSize(Integer value) final;

  void dumpStats(std::ostream& out) const override;

  void compactConnectivityList();

 private:

  Int64 m_nb_add     = 0;
  Int64 m_nb_remove  = 0;
  Int64 m_nb_memcopy = 0;
  Integer m_pre_allocated_size = 0;

 private:

  inline void _increaseIndexList(Int32 lid,Integer size,Int32 target_lid);
  inline Integer _increaseConnectivityList(Int32 new_lid);
  inline Integer _increaseConnectivityList(Int32 new_lid,Integer nb_value);
  inline Integer _computeAllocSize(Integer nb_item);
  void _checkAddNullItem();
  void _resetConnectivityList();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Connectivité incrémentale item->item.
 *
 * Il s'agit d'une spécialisation de IncrementalItemConnectivity
 * pour le cas où il n'y a qu'une entité connectée.
 *
 * Dans ce cas simple, on a toujours pour une entité de localId() \a lid:
 * - m_connectivity_index[lid] = lid
 * - m_connectivity_nb_item[lid] = 1 (ou 0 si pas encore d'entité ajoutée)
 * - m_connectivity_list[lid] = localId() de l'entité connectée.
 */
class ARCANE_MESH_EXPORT OneItemIncrementalItemConnectivity
: public IncrementalItemConnectivityBase
{
 private:
 public:

  OneItemIncrementalItemConnectivity(IItemFamily* source_family,IItemFamily* target_family,
                              const String& aname);
  ~OneItemIncrementalItemConnectivity() override;

 public:

  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override;
  void removeConnectedItems(ItemLocalId source_item) override;
  void addConnectedItem(ItemLocalId source_item,ItemLocalId target_local_id) override;
  void removeConnectedItem(ItemLocalId source_item,ItemLocalId target_local_id) override;
  void replaceConnectedItem(ItemLocalId source_item,Integer index,ItemLocalId target_local_id) override;
  void replaceConnectedItems(ItemLocalId source_item,Int32ConstArrayView target_local_ids) override;
  bool hasConnectedItem(ItemLocalId source_item,ItemLocalId targer_local_id) const override;
  void notifySourceItemAdded(ItemLocalId item) override;
  void notifyReadFromDump() override;
  Integer preAllocatedSize() const override final { return 1; }
  void setPreAllocatedSize([[maybe_unused]] Int32 value) override final {}

 public:

  void dumpStats(std::ostream& out) const override;

  void compactConnectivityList();

 private:

  inline void _checkResizeConnectivityList();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
