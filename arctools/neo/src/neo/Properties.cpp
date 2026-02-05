// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Properties                                      (C) 2000-2026             */
/*                                                                           */
/* Classes and tools for Property                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "neo/Properties.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Neo::ItemRange Neo::ItemLidsProperty::append(std::vector<Neo::utils::Int64> const& uids) {
  ItemLocalIds item_local_ids{};
  // handle mutliple insertion
  auto empty_lid_size = m_empty_lids.size();
  auto min_size = std::min(empty_lid_size, uids.size());
  auto& non_contiguous_lids = item_local_ids.m_non_contiguous_lids;
  non_contiguous_lids.resize(min_size);
  auto used_empty_lid_count = 0;
  for (auto i = 0; i < (int)min_size; ++i) {
    const auto [inserted, do_insert] = m_uid2lid.insert({ uids[i], m_empty_lids[empty_lid_size - 1 - used_empty_lid_count] });
    non_contiguous_lids[i] = inserted->second;
    if (do_insert)
      ++used_empty_lid_count;
  }
  // Use remaining empty lids if needed (if insertion of existing items)
  auto remaining_empty_lids = empty_lid_size - used_empty_lid_count;
  auto remaining_items = uids.size() - min_size;
  auto remaining_lids_to_take = std::min(remaining_empty_lids, remaining_items);
  non_contiguous_lids.resize(min_size + remaining_lids_to_take);
  for (auto i = 0; i < (int)remaining_lids_to_take; ++i) {
    const auto [inserted, do_insert] = m_uid2lid.insert({ uids[min_size + i], m_empty_lids[empty_lid_size - 1 - used_empty_lid_count] });
    non_contiguous_lids[min_size + i] = inserted->second;
    if (do_insert)
      ++used_empty_lid_count;
  }
  m_empty_lids.resize(empty_lid_size - used_empty_lid_count);
  min_size += remaining_lids_to_take;
  using item_index_and_lid = std::pair<int, Neo::utils::Int32>;
  std::vector<item_index_and_lid> existing_items;
  existing_items.reserve(uids.size() - min_size);
  auto first_contiguous_id = m_last_id + 1;
  item_local_ids.m_first_contiguous_lid = first_contiguous_id;
  for (auto i = min_size; i < uids.size(); ++i) {
    const auto [inserted, do_insert] = m_uid2lid.insert({ uids[i], ++m_last_id });
    if (!do_insert) {
      existing_items.push_back({ i - min_size, inserted->second });
      --m_last_id;
    }
    ++item_local_ids.m_nb_contiguous_lids;
  }
  // if an existing item is inserted, cannot use contiguous indexes, otherwise the range
  // will not handle the items in their insertion order, all lids must be in non_contiguous_indexes
  if (!existing_items.empty()) {
    std::vector<Neo::utils::Int32> non_contiguous_from_contigous_lids(
    item_local_ids.m_nb_contiguous_lids);
    std::iota(non_contiguous_from_contigous_lids.begin(), non_contiguous_from_contigous_lids.end(), first_contiguous_id);
    for (const auto& [item_index, item_lid] : existing_items) {
      non_contiguous_from_contigous_lids[item_index] = item_lid;
      std::for_each(non_contiguous_from_contigous_lids.begin() + item_index + 1, non_contiguous_from_contigous_lids.end(), [](auto& current_lid) { return --current_lid; });
    }
    item_local_ids.m_nb_contiguous_lids = 0;
    item_local_ids.m_non_contiguous_lids.insert(
    item_local_ids.m_non_contiguous_lids.end(),
    non_contiguous_from_contigous_lids.begin(),
    non_contiguous_from_contigous_lids.end());
  }
  return ItemRange{ std::move(item_local_ids) };
}

/*---------------------------------------------------------------------------*/

Neo::ItemRange Neo::ItemLidsProperty::remove(std::vector<utils::Int64> const& uids) noexcept {
  ItemLocalIds item_local_ids{};
  item_local_ids.m_non_contiguous_lids.resize(uids.size());
  auto empty_lids_size = m_empty_lids.size();
  m_empty_lids.resize(empty_lids_size + uids.size());
  auto counter = 0;
  auto empty_lids_index = empty_lids_size;
  for (auto uid : uids) {
    // remove from map
    // add in range and in empty_lids
    auto uid_lid_ite = m_uid2lid.find(uid);
    auto lid = utils::NULL_ITEM_LID;
    if (uid_lid_ite != m_uid2lid.end()) {
      lid = uid_lid_ite->second;
      m_uid2lid.erase(uid_lid_ite);
    } // uid_lid_ite is now invalid
    if (lid != utils::NULL_ITEM_LID)
      m_empty_lids[empty_lids_index++] = lid;
    item_local_ids.m_non_contiguous_lids[counter++] = lid;
  }
  return ItemRange{ std::move(item_local_ids) };
}

/*---------------------------------------------------------------------------*/

std::size_t Neo::ItemLidsProperty::size() const {
  return m_last_id + 1 - m_empty_lids.size();
}

/*---------------------------------------------------------------------------*/

Neo::ItemRange Neo::ItemLidsProperty::values() const {
  // TODO...; + il faut mettre en cache (dans la famille ?). ? de la mise à jour (la Propriété peut dire si la range est à jour)
  // 2 stratégies : on crée l'étendue continue avant ou après les non contigus...
  // (on estime que l'on décime les id les plus élevés ou les plus faibles), avoir le choix (avec un paramètre par défaut)
  if (size() == 0)
    return ItemRange{};
  ItemLocalIds item_local_ids{};
  if (m_empty_lids.empty()) { // range contiguous
    item_local_ids = ItemLocalIds{ {}, 0, m_last_id + 1 };
  }
  else { // range discontiguous
    std::vector<Neo::utils::Int32> lids(m_last_id + 1);
    std::iota(lids.begin(), lids.end(), 0);
    std::for_each(m_empty_lids.begin(), m_empty_lids.end(),
                  [&lids](auto const& empty_lid) {
                    lids[empty_lid] = Neo::utils::NULL_ITEM_LID;
                  });
    auto& active_lids = item_local_ids.m_non_contiguous_lids;
    active_lids.resize(lids.size() - m_empty_lids.size());
    std::copy_if(lids.begin(), lids.end(), active_lids.begin(),
                 [](auto const& lid_source) {
                   return lid_source != Neo::utils::NULL_ITEM_LID;
                 });
  }
  return ItemRange{ std::move(item_local_ids) };
}

/*---------------------------------------------------------------------------*/

void Neo::ItemLidsProperty::debugPrint(int rank) const {
  if constexpr (ndebug)
    return;
  Neo::NeoOutputStream oss{traceLevel(),rank};
  oss << "= Print property " << m_name << ", size = " << size() << Neo::endline;
  for (auto uid : m_uid2lid) {
    if (uid.second != Neo::utils::NULL_ITEM_LID)
      oss << " uid to lid  " << uid.first << " : " << uid.second << "\n";
  }
  oss << "available lids : " << m_available_lids;
  oss<< Neo::endline;
}

/*---------------------------------------------------------------------------*/

Neo::utils::Int32 Neo::ItemLidsProperty::_getLidFromUid(utils::Int64 const uid) const {
  auto iterator = m_uid2lid.find(uid);
  if (iterator == m_uid2lid.end())
    return utils::NULL_ITEM_LID;
  else
    return iterator->second;
}

/*---------------------------------------------------------------------------*/

void Neo::ItemLidsProperty::_getLidsFromUids(std::vector<utils::Int32>& lids, std::vector<utils::Int64> const& uids) const {
  std::transform(uids.begin(), uids.end(), std::back_inserter(lids), [this](auto const& uid) { return this->_getLidFromUid(uid); });
}

/*---------------------------------------------------------------------------*/

std::vector<Neo::utils::Int32> Neo::ItemLidsProperty::operator[](std::vector<utils::Int64> const& uids) const {
  std::vector<utils::Int32> lids;
  _getLidsFromUids(lids, uids);
  return lids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
