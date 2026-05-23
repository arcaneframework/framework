// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemData.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Class gathering item data : ids and connectivities                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMDATA_H_
#define ARCANE_ITEMDATA_H_
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <stdexcept>
#include <map>

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/mesh/MeshGlobal.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IItemFamily.h"
#include "arcane/IItemFamilyModifier.h"
#include "arcane/ISerializer.h"
#include "arcane/IMesh.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ItemData
{
public:

  /*!
   * \brief Data structure necessary for generic item addition.
   *
   * and where ItemData aggregates the item id/connectivity information. The item_infos array (ItemData::itemInfos()) follows this structure:
   * item_infos[0]   = nb_connected_families // Only constitutive (owning) connections.
   * item_infos[i]   = first_item_type
   * item_infos[i+1] = first_item_uid
   * item_infos[i+2] = first_connected_family_id
   * item_infos[i+3] = nb_connected_items_in_first_family
   * item_infos[i+4...i+n] = first_family connected items uids
   * item_infos[i+n+1] = second_connected_family_id
   * item_infos[i+n+1...i+m] = second_family connected items uids
   * item_infos[i+m+1] = second_item_type
   * item_infos[i+m+2] = second_item_uid
   * ...same as first item
   */

  /** Build empty data */
  ItemData() : m_nb_items(0), m_item_family(nullptr), m_item_family_modifier(nullptr), m_subdomain_id(-1){}

  //! Copy constructor.
  ItemData(const ItemData& rhs) = default;

  /*! The item_lids argument is an output argument. It must be sized to nb_items.
   * It is filled with the lids of the items created when ItemData is used for item addition.
   * This constructor is used when these lids are already in an external array that needs to be filled.
   */
  ItemData(Integer nb_items, Integer info_size, Int32ArrayView item_lids, IItemFamily* item_family,
           IItemFamilyModifier* item_family_modifier, Int32 subdomain_id)
    : m_nb_items(nb_items)
    , m_item_infos(info_size)
    , m_item_lids(item_lids)
    , m_item_family(item_family)
    , m_item_family_modifier(item_family_modifier)
    , m_subdomain_id(subdomain_id)
  {
    _ownerDefaultInit();
  }

  /*! Here, item_lids are not provided and are therefore created internally.
   *
   */
  ItemData(Integer nb_items, Integer info_size, IItemFamily* item_family,
           IItemFamilyModifier* item_family_modifier, Int32 subdomain_id)
    : m_nb_items(nb_items)
    , m_item_infos(info_size)
    , _internal_item_lids(nb_items)
    , m_item_lids(_internal_item_lids)
    , m_item_family(item_family)
    , m_item_family_modifier(item_family_modifier)
    , m_subdomain_id(subdomain_id)
  {
    _ownerDefaultInit();
  }

  /** Class destructor */
  virtual ~ItemData() {}

public:

  Integer nbItems() const {return m_nb_items;}
  Int64Array& itemInfos() {return m_item_infos;} // Need to return Array& since size is not always known at construction
  Int64ConstArrayView itemInfos() const {return m_item_infos;}
  Int32ArrayView itemLids() {return m_item_lids;}
  Int32ArrayView itemOwners() { return m_item_owners;}
  Int32ConstArrayView itemOwners() const { return m_item_owners;}
  IItemFamily* itemFamily() {return m_item_family;}
  IItemFamily const* itemFamily() const {return m_item_family;}
  IItemFamilyModifier* itemFamilyModifier() {return m_item_family_modifier;}
  Integer subDomainId() const {return m_subdomain_id;}

  void serialize(ISerializer* buffer); // Fill the buffer from the data
  void deserialize(ISerializer* buffer, IMesh* mesh); // Fill the buffer from the data : using an internal lids array
  void deserialize(ISerializer* buffer, IMesh* mesh, Int32Array& item_lids); // Fill the data from the buffer using external lids array. item_lids must live as long as ItemData does...
  void clear(); // Clear all internal data

 private:

  void _deserialize(ISerializer* buffer, IMesh* mesh);
  void _ownerDefaultInit() { m_item_owners.resize(m_nb_items); m_item_owners.fill(m_subdomain_id);}

  Integer m_nb_items;
  // Todo optimization use std::reference_wrapper to avoid copy ids (int64 & int32)
  // => in this case the second constructor won't be possible ?...(resize not possible with reference wrapper since not default constructible)
  Int64SharedArray m_item_infos;
  Int32UniqueArray _internal_item_lids; // m_item_lids points on it if the view is not given in the constructor
  Int32ArrayView m_item_lids;
  IItemFamily* m_item_family;
  IItemFamilyModifier* m_item_family_modifier;
  Integer m_subdomain_id;
  Int32UniqueArray m_item_owners;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ItemDataList
{
public:

  /*!
   * \brief Collection of data structures necessary for generic item addition (ItemData).
   *
   * The \a ItemDataList object is a map <family_index,ItemData> where family_index is taken equal to the family's item_kind.
   *
   * */

  /** Class constructor */
  ItemDataList() {}

  /** Class destructor */
  virtual ~ItemDataList() {}

public:

  ItemData& itemData(const Integer family_id)
  {
    auto inserter = m_item_infos_list.insert(std::make_pair(family_id,ItemData()));
    ARCANE_ASSERT((inserter.second),(String::format("Cannot insert twice ItemData for family with id {0} in ItemInfosList",family_id).localstr()));
    return inserter.first->second;
  }

  ItemData& itemData(const Integer family_id,
                       Integer nb_items,
                       Integer info_size,
                       Int32ArrayView item_lids,
                       IItemFamily* item_family,
                       IItemFamilyModifier* family_modifier,
                       const Integer& subdomain_id)
  {
    auto inserter = m_item_infos_list.insert(std::make_pair(family_id,ItemData(nb_items,info_size,item_lids,item_family,family_modifier,subdomain_id)));
    ARCANE_ASSERT((inserter.second),(String::format("Cannot insert twice ItemData for family with id {0} in ItemInfosList",family_id).localstr()));
    return inserter.first->second;
  }

  ItemData& operator[] (const Integer family_id){
    return m_item_infos_list[family_id];
  }

  const ItemData& operator[] (const Integer family_id) const {
    try {
      return m_item_infos_list.at(family_id);
    } catch (const std::out_of_range&) {
      ARCANE_FATAL("Cannot return family with id {0}, not inserted in current ItemDataList",family_id);
    }
  }

  Integer size() const { return arcaneCheckArraySize(m_item_infos_list.size()); }

  bool contains(const Integer family_id) {
    return m_item_infos_list.find(family_id) != m_item_infos_list.end();
  }

  void clear(const Integer family_id) {
    m_item_infos_list.erase(family_id);
  }

private:
  std::map<Integer, ItemData> m_item_infos_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ITEMDATA_H_ */
