// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivitySelector.h                                  (C) 2000-2021 */
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

namespace Arcane
{
class IItemFamily;
class IIncrementalItemConnectivity;
}

namespace Arcane::mesh
{

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
  ARCCORE_DEPRECATED_2021("This method always return 'nullptr'")
  virtual IIncrementalItemConnectivity* legacyConnectivity() const { return nullptr; }
  virtual IIncrementalItemConnectivity* customConnectivity() const =0;
  ARCCORE_DEPRECATED_2021("This method doesn't do anything")
  virtual void updateItemConnectivityList(Int32ConstArrayView) const {}
  virtual void checkValidConnectivityList() const =0;
  virtual void compactConnectivities() =0;

 public:

  void setPreAllocatedSize(Integer size);
  Integer preAllocatedSize() const { return m_pre_allocated_size; }
  Int32 itemConnectivityIndex() const { return m_item_connectivity_index; }

 protected:

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
  bool m_is_built;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sélection entre les connectivités historiques et à la demande.
 */
template<typename ConnectivityIndexType,typename CustomType>
class ARCANE_MESH_EXPORT ItemConnectivitySelectorT
: public ItemConnectivitySelector
{
 public:

  ItemConnectivitySelectorT(ItemFamily* source_family,IItemFamily* target_family,
                            const String& connectivity_name)
  : ItemConnectivitySelector(source_family,target_family,connectivity_name,ConnectivityIndexType::connectivityIndex())
  , m_custom_connectivity(nullptr)
  {
  }
  ~ItemConnectivitySelectorT()
  {
    // NOTE: les connectivités sont détuites par les familles.
  }

 public:

  IIncrementalItemConnectivity* customConnectivity() const override
  {
    return trueCustomConnectivity();
  }

  void checkValidConnectivityList() const override
  {
    if (m_item_connectivity_index<0)
      return;
    auto x = m_item_connectivity_list;
    auto current_list = x->connectivityList(m_item_connectivity_index);
    auto ref_list = m_custom_connectivity->connectivityList();
    auto current_indexes = x->connectivityIndex(m_item_connectivity_index);
    auto ref_indexes = m_custom_connectivity->connectivityIndex();
    auto* current_list_ptr = current_list.data();
    auto* ref_list_ptr = ref_list.data();
    if (current_list_ptr!=ref_list_ptr)
      ARCANE_FATAL("Bad list base pointer current={0} ref={1}",current_list_ptr,ref_list_ptr);
    if (current_list.size()!=ref_list.size())
      ARCANE_FATAL("Bad list size current={0} ref={1}",current_list.size(),ref_list.size());
    auto* current_indexes_ptr = current_indexes.data();
    auto* ref_indexes_ptr = ref_indexes.data();
    if (current_indexes_ptr!=ref_indexes_ptr)
      ARCANE_FATAL("Bad indexes base pointer current={0} ref={1}",current_indexes_ptr,ref_indexes_ptr);
    if (current_indexes.size()!=ref_indexes.size())
      ARCANE_FATAL("Bad indexes size current={0} ref={1}",current_indexes.size(),ref_indexes.size());
  }

  void compactConnectivities() override
  {
    if (m_custom_connectivity)
      m_custom_connectivity->compactConnectivityList();
  }

 public:

  void addConnectedItem(ItemLocalId item_lid,ItemLocalId sub_item_lid)
  {
    if (m_custom_connectivity)
      m_custom_connectivity->addConnectedItem(item_lid,sub_item_lid);
  }

  void removeConnectedItem(ItemLocalId item_lid,ItemLocalId sub_item_lid)
  {
    if (m_custom_connectivity)
      m_custom_connectivity->removeConnectedItem(item_lid,sub_item_lid);
  }

  void removeConnectedItems(ItemLocalId item_lid)
  {
    if (m_custom_connectivity)
      m_custom_connectivity->removeConnectedItems(item_lid);
  }

  void replaceItems(ItemLocalId item_lid,Int32ConstArrayView sub_item_lids)
  {
    if (m_custom_connectivity)
      m_custom_connectivity->replaceConnectedItems(item_lid,sub_item_lids);
  }

  void replaceItem(ItemLocalId item_lid,Integer index,ItemLocalId sub_item_lid)
  {
    if (m_custom_connectivity)
      m_custom_connectivity->replaceConnectedItem(item_lid,index,sub_item_lid);
  }

  bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const
  {
    if (m_custom_connectivity)
      return m_custom_connectivity->hasConnectedItem(source_item,target_local_id);

    return false;
  }

  CustomType* trueCustomConnectivity() const { return m_custom_connectivity; }

 protected:

  void _createCustomConnectivity(const String& name) override
  {
    m_custom_connectivity = new CustomType(m_source_family,m_target_family,name);
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
    if (m_custom_connectivity)
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

    _createCustomConnectivity<SourceFamily,TargetFamily>(m_connectivity_name);
    info() << "Family: " << m_source_family->fullName()
           << " create new connectivity: " << m_connectivity_name;

    _buildCustomConnectivity();
    m_is_built = true;
  }

 private:

  CustomType* m_custom_connectivity;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
