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
   * \brief Structure de données nécessaire à l'ajout générique d'item.
   *
   * et où ItemData aggrège les informations id/connectivités des items Le tableau item_infos (ItemData::itemInfos()) à la structure suivante :
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
   * ...idem first item
   */

  /** Build empty data */
  ItemData() : m_nb_items(0), m_item_family(nullptr), m_item_family_modifier(nullptr), m_subdomain_id(-1){}

  //! Constructeur de recopie.
  ItemData(const ItemData& rhs) = default;

  /*! L'argument item_lids est un argument de sortie. Il doit être taillé à nb_items.
   * Il est rempli avec les lids des items créés lorsque ItemData est utilisée pour de l'ajout d'item.
   * Ce constructeur est utilisé lorsque ces lids sont déjà dans un tableau externe qui doit être rempli.
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

  /*! Ici on ne fournit pas les item_lids qui sont donc créés en internes.
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

  /** Destructeur de la classe */
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
   * \brief Collection de structure de données nécessaire à l'ajout générique d'item (ItemData).
   *
   * L'objet \a ItemDataList est une map <family_index,ItemData> où family_index est pris égal à l'item_kind de la famille.
   *
   * */

  /** Constructeur de la classe */
  ItemDataList() {}

  /** Destructeur de la classe */
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
