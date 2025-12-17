// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Family                                          (C) 2000-2025             */
/*                                                                           */
/* Family of mesh items                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NEO_FAMILY_H
#define NEO_FAMILY_H

#include <memory>
#include <map>

#include "neo/Utils.h"
#include "neo/Properties.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Neo
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FamilyMap;

class Family
{
 public:
  ItemKind m_ik;
  std::string m_name;
  std::string m_prop_lid_name;
  std::map<std::string, Property> m_properties;
  mutable ItemRange m_all;

  friend class FamilyMap;

private:
  Family(Family const&) = default;
  Family& operator=(Family const&) = default;

public:
  Family(ItemKind ik, const std::string& name)
  : m_ik(ik)
  , m_name(name)
  , m_prop_lid_name(name) {
    m_prop_lid_name.append("_lids");
    m_properties[lidPropName()] = ItemLidsProperty{ lidPropName() };
  }

  constexpr std::string const& name() const noexcept { return m_name; }
  constexpr ItemKind const& itemKind() const noexcept { return m_ik; }

  std::vector<Neo::utils::Int32> itemUniqueIdsToLocalids(std::vector<Neo::utils::Int64> const& item_uids) const {
    return _lidProp().operator[](item_uids);
  }
  void itemUniqueIdsToLocalids(std::vector<Neo::utils::Int32>& item_lids, std::vector<Neo::utils::Int64> const& item_uids) const {
    assert(("In itemUniqueIdsToLocalIds, lids and uids sizes differ.", item_lids.size() == item_uids.size()));
    _lidProp()._getLidsFromUids(item_lids, item_uids);
  }

  template <typename T>
  void addScalarProperty(std::string const& name, T init_value = T{}) {
    auto [iter, is_inserted] = m_properties.insert(std::make_pair(name, ScalarPropertyT<T>{ name, init_value }));
    if (is_inserted)
      Neo::print() << "= Add scalar property " << name << " in Family " << m_name
                   << std::endl;
  }

  template <typename T>
  void addArrayProperty(std::string const& name) {
    auto [iter, is_inserted] = m_properties.insert(std::make_pair(name, ArrayPropertyT<T>{ name }));
    if (is_inserted)
      Neo::print() << "= Add scalar property " << name << " in Family " << m_name
                   << std::endl;
  }

  template <typename T>
  void addMeshScalarProperty(std::string const& name) {
    auto [iter, is_inserted] = m_properties.insert(std::make_pair(name, MeshScalarPropertyT<T>{ name }));
    if (is_inserted)
      Neo::print() << "= Add property " << name << " in Family " << m_name
                   << std::endl;
  }

  template <typename T>
  void addMeshArrayProperty(std::string const& name) {
    auto [iter, is_inserted] = m_properties.insert(std::make_pair(name, MeshArrayPropertyT<T>{ name }));
    if (is_inserted)
      Neo::print() << "= Add array property " << name << " in Family " << m_name
                   << std::endl;
  }

  void removeProperty(std::string const& name) {
    m_properties.erase(name);
  }

  void removeProperties() {
    m_properties.clear();
  }

  Property& getProperty(std::string const& name) {
    auto found_property = m_properties.find(name);
    if (found_property == m_properties.end())
      throw std::invalid_argument("Cannot find Property " + name);
    return found_property->second;
  }

  Property const& getProperty(std::string const& name) const {
    auto found_property = m_properties.find(name);
    if (found_property == m_properties.end())
      throw std::invalid_argument("Cannot find Property " + name);
    return found_property->second;
  }

  bool hasProperty(std::string const& name) const noexcept {
    if (m_properties.find(name) == m_properties.end())
      return false;
    else
      return true;
  }

  bool hasProperty() const noexcept {
    return (m_properties.size() > 1); // a family always has a lid_property
  }

  template <typename PropertyType>
  PropertyType& getConcreteProperty(const std::string& name) {
    return std::get<PropertyType>(getProperty(name));
  }

  template <typename PropertyType>
  PropertyType const& getConcreteProperty(const std::string& name) const {
    return std::get<PropertyType>(getProperty(name));
  }

  std::string const& lidPropName() { return m_prop_lid_name; }

  std::size_t nbElements() const {
    return _lidProp().size();
  }

  ItemRange& all() const {
    auto& lid_prop = _lidProp();
    //      if (m_all.isEmpty() || lid_prop.hasChanged()) // todo hasChanged ??? possible ? voir le cycle de modif des propriétés
    m_all = lid_prop.values();
    return m_all;
  }

  ItemLidsProperty& _lidProp() {
    auto prop_iterator = m_properties.find(m_prop_lid_name);
    return std::get<ItemLidsProperty>(prop_iterator->second);
  }

  ItemLidsProperty const& _lidProp() const {
    auto prop_iterator = m_properties.find(m_prop_lid_name);
    return std::get<ItemLidsProperty>(prop_iterator->second);
  }

  friend inline bool operator==(Neo::Family const& family1, Neo::Family const& family2) {
    return (family1.itemKind() == family2.itemKind() && family1.name() == family2.name());
  }

  friend inline bool operator!=(Neo::Family const& family1, Neo::Family const& family2) {
    return !(family1 == family2);
  }

 private:
  Property& _getProperty(const std::string& name) {
    auto found_property = m_properties.find(name);
    if (found_property == m_properties.end())
      throw std::invalid_argument("Cannot find Property " + name);
    return found_property->second;
  }
};

//----------------------------------------------------------------------------/

class FamilyMap
{
 private:
  std::map<std::pair<ItemKind, std::string>, std::unique_ptr<Family>> m_families;
  void _copyMap(FamilyMap const& family_map) {
    for (auto const& family_info : family_map.m_families) {
      auto const& [item_kind, name] = family_info.first;
      push_back(item_kind, name);
    }
  }

 public:
  FamilyMap() = default;
  FamilyMap(FamilyMap const& family_map) {
    _copyMap(family_map);
  }
  FamilyMap(FamilyMap&& family_map) {
    m_families = std::move(family_map.m_families);
  }
  FamilyMap& operator=(FamilyMap const& family_map) {
    _copyMap(family_map);
    return *this;
  }
  FamilyMap& operator=(FamilyMap&& family_map) {
    m_families = std::move(family_map.m_families);
    return *this;
  }
  Family& operator()(ItemKind const& ik, std::string const& name) const noexcept(ndebug) {
    auto found_family = m_families.find(std::make_pair(ik, name));
    assert(("Cannot find Family ", found_family != m_families.end()));
    return *(found_family->second.get());
  }
  Family& push_back(ItemKind const& ik, std::string const& name) {
    return *(m_families.emplace(std::make_pair(ik, name), std::make_unique<Family>(ik, name)).first->second.get());
  }

  auto begin() noexcept { return m_families.begin(); }
  auto begin() const noexcept { return m_families.begin(); }
  auto end() noexcept { return m_families.end(); }
  auto end() const noexcept { return m_families.end(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Neo

#endif //NEO_FAMILY_H
