// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivitySelector.h                                  (C) 2000-2017 */
/*                                                                           */
/* Sélection entre les connectivités historiques et à la demande.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMCONNECTIVITYSELECTOR_H
#define ARCANE_MESH_ITEMCONNECTIVITYSELECTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/ItemInternal.h"

#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"

#include "arcane/IItemFamilyNetwork.h"

#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class IItemFamily;
class IIncrementalItemConnectivity;
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sélection entre les connectivités historiques et à la demande.
 */
class ARCANE_MESH_EXPORT ItemConnectivitySelector
: public TraceAccessor
{
 public:

  ItemConnectivitySelector(ItemFamily* source_family,IItemFamily* target_family,
                           const String& connectivity_name,Integer connectivity_index);
  virtual ~ItemConnectivitySelector()
  {
  }
 public:

  virtual void build();
  virtual IIncrementalItemConnectivity* legacyConnectivity() const =0;
  virtual IIncrementalItemConnectivity* customConnectivity() const =0;
  virtual void updateItemConnectivityList(Int32ConstArrayView items_data_view) const =0;
  virtual void checkValidConnectivityList() const =0;
  virtual void compactConnectivities() =0;

 public:

  void setPreAllocatedSize(Integer size);
  Integer preAllocatedSize() const { return m_pre_allocated_size; }
  Int32 itemConnectivityIndex() const { return m_item_connectivity_index; }

 protected:

  virtual void _createLegacyConnectivity(const String& name) =0;
  virtual void _createCustomConnectivity(const String& name) =0;
  virtual void _buildCustomConnectivity() =0;

 protected:

  ItemFamily* m_source_family;
  IItemFamily* m_target_family;
  String m_connectivity_name;
  Integer m_pre_allocated_size;
  // Numéro dans ItemInternalConnectivityList. (-1) si aucun.
  Int32 m_item_connectivity_index;
  ItemInternalConnectivityList* m_item_connectivity_list;
  // Vrai si on accède aux connectivités des entités via les nouvelles connectivités.
  bool m_use_custom_connectivity_accessor;
  bool m_is_built;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sélection entre les connectivités historiques et à la demande.
 */
template<typename LegacyType,typename CustomType>
class ARCANE_MESH_EXPORT ItemConnectivitySelectorT
: public ItemConnectivitySelector
{
 public:

  ItemConnectivitySelectorT(ItemFamily* source_family,IItemFamily* target_family,
                            const String& connectivity_name)
  : ItemConnectivitySelector(source_family,target_family,connectivity_name,LegacyType::connectivityIndex())
  , m_legacy_connectivity(nullptr), m_custom_connectivity(nullptr)
  {
  }
  ~ItemConnectivitySelectorT()
  {
    // NOTE: les connectivités sont détuites par les familles.
  }

 public:

  IIncrementalItemConnectivity* legacyConnectivity() const override
  {
    return trueLegacyConnectivity();
  }

  IIncrementalItemConnectivity* customConnectivity() const override
  {
    return trueCustomConnectivity();
  }

  void updateItemConnectivityList(Int32ConstArrayView items_data_view) const override
  {
    if (m_item_connectivity_index<0)
      return;
    auto x = m_item_connectivity_list;
    if (m_use_custom_connectivity_accessor){
      // C'est maitenant automatiquement mis a jour par customConnectivity().
    }
    else
      x->setConnectivityList(m_item_connectivity_index,items_data_view);
  }

  void checkValidConnectivityList() const override
  {
    if (m_item_connectivity_index<0)
      return;
    auto x = m_item_connectivity_list;
    if (m_use_custom_connectivity_accessor){
      auto current_list = x->connectivityList(m_item_connectivity_index);
      auto ref_list = m_custom_connectivity->connectivityList();
      auto current_indexes = x->connectivityIndex(m_item_connectivity_index);
      auto ref_indexes = m_custom_connectivity->connectivityIndex();
      auto current_list_ptr = current_list.unguardedBasePointer();
      auto ref_list_ptr = ref_list.unguardedBasePointer();
      if (current_list_ptr!=ref_list_ptr)
        ARCANE_FATAL("Bad list base pointer current={0} ref={1}",current_list_ptr,ref_list_ptr);
      if (current_list.size()!=ref_list.size())
        ARCANE_FATAL("Bad list size current={0} ref={1}",current_list.size(),ref_list.size());
      auto current_indexes_ptr = current_indexes.unguardedBasePointer();
      auto ref_indexes_ptr = ref_indexes.unguardedBasePointer();
      if (current_indexes_ptr!=ref_indexes_ptr)
        ARCANE_FATAL("Bad indexes base pointer current={0} ref={1}",current_indexes_ptr,ref_indexes_ptr);
      if (current_indexes.size()!=ref_indexes.size())
        ARCANE_FATAL("Bad indexes size current={0} ref={1}",current_indexes.size(),ref_indexes.size());
    }
    else{
      // TODO
      //x->m_list[m_item_connectivity_index] = items_data_view;
    }
  }

  void compactConnectivities() override
  {
    if (m_custom_connectivity)
      m_custom_connectivity->compactConnectivityList();
  }

 public:

  void addConnectedItem(ItemLocalId item_lid,ItemLocalId sub_item_lid)
  {
    if (m_legacy_connectivity)
      m_legacy_connectivity->addConnectedItem(item_lid,sub_item_lid);

    if (m_custom_connectivity)
      m_custom_connectivity->addConnectedItem(item_lid,sub_item_lid);
  }

  void removeConnectedItem(ItemLocalId item_lid,ItemLocalId sub_item_lid)
  {
    if (m_legacy_connectivity)
      m_legacy_connectivity->removeConnectedItem(item_lid,sub_item_lid);

    if (m_custom_connectivity)
      m_custom_connectivity->removeConnectedItem(item_lid,sub_item_lid);
  }

  void removeConnectedItems(ItemLocalId item_lid)
  {
    if (m_legacy_connectivity)
      m_legacy_connectivity->removeConnectedItems(item_lid);

    if (m_custom_connectivity)
      m_custom_connectivity->removeConnectedItems(item_lid);
  }

  void replaceItems(ItemLocalId item_lid,Int32ConstArrayView sub_item_lids)
  {
    if (m_legacy_connectivity)
      m_legacy_connectivity->replaceConnectedItems(item_lid,sub_item_lids);

    if (m_custom_connectivity)
      m_custom_connectivity->replaceConnectedItems(item_lid,sub_item_lids);
  }

  void replaceItem(ItemLocalId item_lid,Integer index,ItemLocalId sub_item_lid)
  {
    if (m_legacy_connectivity)
      m_legacy_connectivity->replaceConnectedItem(item_lid,index,sub_item_lid);

    if (m_custom_connectivity)
      m_custom_connectivity->replaceConnectedItem(item_lid,index,sub_item_lid);
  }

  bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const {
    if (m_legacy_connectivity)
      return m_legacy_connectivity->hasConnectedItem(source_item,target_local_id);

    if (m_custom_connectivity)
      return m_custom_connectivity->hasConnectedItem(source_item,target_local_id);

    return false;
  }

  LegacyType* trueLegacyConnectivity() const { return m_legacy_connectivity; }
  CustomType* trueCustomConnectivity() const { return m_custom_connectivity; }

 protected:

  void _createLegacyConnectivity(const String& name) override
  {
    m_legacy_connectivity = new LegacyType(m_source_family,m_target_family,name);
  }

  void _createCustomConnectivity(const String& name) override
  {
    m_custom_connectivity = new CustomType(m_source_family,m_target_family,name);
  }

  // used only with family dependencies. The concrete type of families are needed only for FaceToCellConnectivity
  // where the CompactIncrementalItemConnectivity has been overloaded (to handle back/front cell connectivity)
  template <class SourceFamily, class TargetFamily>
  void _createLegacyConnectivity(const String& name)
  {
    m_legacy_connectivity = new typename LegacyConnectivity<SourceFamily,TargetFamily>::type(m_source_family,m_target_family,name);
  }

  // used only with family dependencies. The concrete type of families are needed only for FaceToCellConnectivity
  // where the IncrementalItemConnectivity has been overloaded (to handle back/front cell connectivity)
  template <class SourceFamily, class TargetFamily>
  void _createCustomConnectivity(const String& name)
    {
      m_custom_connectivity = new typename CustomConnectivity<SourceFamily,TargetFamily>::type(m_source_family,m_target_family,name);
    }

  void _buildCustomConnectivity() override
  {
    // Indique à l'IncrementalItemConnectivity qu'elle doit mettre à jour
    // les accesseurs de ItemInternal.
    if (m_use_custom_connectivity_accessor && m_custom_connectivity)
      m_custom_connectivity->setItemConnectivityList(m_item_connectivity_list,
                                                     m_item_connectivity_index);
  }

 public:
  // Build called when using family dependencies
  template<class SourceFamily, class TargetFamily>
  void build()
  {
    if (m_is_built)
      return;
    InternalConnectivityPolicy policy = m_source_family->mesh()->_connectivityPolicy();
    bool alloc_custom = (policy!=InternalConnectivityPolicy::Legacy);

    m_use_custom_connectivity_accessor = InternalConnectivityInfo::useNewConnectivityAccessor(policy);
    bool use_legacy_connectivity = (policy!=InternalConnectivityPolicy::NewOnly);
    if (use_legacy_connectivity)
      _createLegacyConnectivity<SourceFamily,TargetFamily>(m_connectivity_name+"Compact");
    if (alloc_custom){
      _createCustomConnectivity<SourceFamily,TargetFamily>(m_connectivity_name);
      info() << "Family: " << m_source_family->fullName()
             << " create new connectivity: " << m_connectivity_name;
    }

    _buildCustomConnectivity();
    m_is_built = true;
  }

 private:

  LegacyType* m_legacy_connectivity;
  CustomType* m_custom_connectivity;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
