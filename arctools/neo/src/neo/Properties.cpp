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
  item_local_ids.m_non_contiguous_lids.reserve(uids.size());
  item_local_ids.m_nb_contiguous_lids = 0;
  // These commented lines should be used when a mode mixing contiguous and not contiguous items will work
  //std::vector<Neo::utils::Int32> lids(uids.size());
  //_getLidsFromUids(lids,uids);
  //bool has_existing_items = (std::count(lids.begin(), lids.end(), utils::NULL_ITEM_LID) != std::size_t(lids.size()));
  bool use_contiguous_indexes = false; // temporary to make things work. Todo : refactor ItemLocalIds (change order between non contiguous and contiguous ?)
  auto nb_available_lids = m_available_lids.size();
  auto lid_index = 0;
  bool is_first_contiguous_lid = true;
  for (auto uid : uids) {
    auto lid = _getLidFromUid(uid);
    if (lid != utils::NULL_ITEM_LID) {// existing item
        item_local_ids.m_non_contiguous_lids.push_back(lid);
      use_contiguous_indexes = false;
    }
    else {
      if (nb_available_lids > 0) {
        auto available_lid = m_available_lids[nb_available_lids - 1];
        m_uid2lid.emplace(uids[lid_index], available_lid);
        --nb_available_lids;
        m_available_lids.pop_back();
        item_local_ids.m_non_contiguous_lids.push_back(available_lid);
        use_contiguous_indexes = false;
      }
      else {
        auto new_lid = ++m_last_id;
        m_uid2lid.emplace(uids[lid_index], new_lid);
        if (use_contiguous_indexes) {
          item_local_ids.m_nb_contiguous_lids++;
          if (is_first_contiguous_lid) {
            item_local_ids.m_first_contiguous_lid = new_lid;
            is_first_contiguous_lid = false;
          }
        }
        else {
          item_local_ids.m_non_contiguous_lids.push_back(new_lid);
        }
      }
    }
    ++lid_index;
  }
  return ItemRange{ std::move(item_local_ids) };
}

/*---------------------------------------------------------------------------*/

Neo::ItemRange Neo::ItemLidsProperty::remove(std::vector<utils::Int64> const& uids) noexcept {
  ItemLocalIds item_local_ids{};
  item_local_ids.m_non_contiguous_lids.resize(uids.size());
  auto empty_lids_size = m_available_lids.size();
  m_available_lids.resize(empty_lids_size + uids.size());
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
      m_available_lids[empty_lids_index++] = lid;
    item_local_ids.m_non_contiguous_lids[counter++] = lid;
  }
  return ItemRange{ std::move(item_local_ids) };
}

/*---------------------------------------------------------------------------*/

std::size_t Neo::ItemLidsProperty::size() const {
  return m_uid2lid.size();
}

/*---------------------------------------------------------------------------*/

Neo::ItemRange Neo::ItemLidsProperty::values() const {
  // TODO...; + il faut mettre en cache (dans la famille ?). ? de la mise à jour (la Propriété peut dire si la range est à jour)
  // 2 stratégies : on crée l'étendue continue avant ou après les non contigus...
  // (on estime que l'on décime les id les plus élevés ou les plus faibles), avoir le choix (avec un paramètre par défaut)
  if (size() == 0)
    return ItemRange{};
  ItemLocalIds item_local_ids{};
  if (m_available_lids.empty()) { // range contiguous
    item_local_ids = ItemLocalIds{ {}, 0, m_last_id + 1 };
  }
  else { // range discontiguous
    std::vector<Neo::utils::Int32> lids(m_last_id + 1);
    std::iota(lids.begin(), lids.end(), 0);
    std::for_each(m_available_lids.begin(), m_available_lids.end(),
                  [&lids](auto const& empty_lid) {
                    lids[empty_lid] = Neo::utils::NULL_ITEM_LID;
                  });
    auto& active_lids = item_local_ids.m_non_contiguous_lids;
    active_lids.resize(lids.size() - m_available_lids.size());
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
  // todo check size
  std::transform(uids.begin(), uids.end(), lids.begin(), [this](auto const& uid) { return this->_getLidFromUid(uid); });
}

/*---------------------------------------------------------------------------*/

std::vector<Neo::utils::Int32> Neo::ItemLidsProperty::operator[](std::vector<utils::Int64> const& uids) const {
  std::vector<utils::Int32> lids(uids.size());
  _getLidsFromUids(lids, uids);
  return lids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
