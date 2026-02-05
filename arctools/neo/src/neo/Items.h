// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Items                                     (C) 2000-2026                   */
/*                                                                           */
/* Tooling to manipulate Mesh Items                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NEO_ITEMS_H
#define NEO_ITEMS_H

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <string>
#include <variant>

#include "neo/Utils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Neo
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum class ItemKind
{
  IK_None,
  IK_Node,
  IK_Edge,
  IK_Face,
  IK_Cell,
  IK_Dof
};

/*---------------------------------------------------------------------------*/

namespace utils
{ // not defined in Utils.h since needs Neo::ItemKind

  inline std::string itemKindName(ItemKind item_kind) {
    switch (item_kind) {
    case ItemKind::IK_Node:
      return "IK_Node";
      break;
    case ItemKind::IK_Edge:
      return "IK_Edge";
      break;
    case ItemKind::IK_Face:
      return "IK_Face";
      break;
    case ItemKind::IK_Cell:
      return "IK_Cell";
      break;
    case ItemKind::IK_Dof:
      return "IK_Dof";
      break;
    case ItemKind::IK_None:
      return "IK_None";
      break;
    }
  }
} // namespace utils

/*---------------------------------------------------------------------------*/

struct ItemLocalIds
{
  std::vector<utils::Int32> m_non_contiguous_lids = {};
  utils::Int32 m_first_contiguous_lid = 0;
  int m_nb_contiguous_lids = 0;

  int size() const { return m_non_contiguous_lids.size() + m_nb_contiguous_lids; }

  int operator()(int index) const {
    if (index >= size())
      return size();
    if (index < 0)
      return -1;
    auto item_lid = 0;
    (index >= m_non_contiguous_lids.size() || m_non_contiguous_lids.size() == 0) ? item_lid = m_first_contiguous_lid + (index - m_non_contiguous_lids.size()) : // work on fluency
    item_lid = m_non_contiguous_lids[index];
    return item_lid;
  }

  std::vector<utils::Int32> itemArray() const {
    std::vector<utils::Int32> item_array(m_non_contiguous_lids.size() +
                                         m_nb_contiguous_lids);
    std::copy(m_non_contiguous_lids.begin(), m_non_contiguous_lids.end(), item_array.begin());
    std::iota(item_array.begin() + m_non_contiguous_lids.size(), item_array.end(), m_first_contiguous_lid);
    return item_array;
  }

  utils::Int32 maxLocalId() const {
    auto max_contiguous_lid = m_first_contiguous_lid + m_nb_contiguous_lids - 1;
    if (m_non_contiguous_lids.empty())
      return max_contiguous_lid;
    else {
      auto max_non_contiguous_lid = *std::max_element(m_non_contiguous_lids.begin(),
                                                      m_non_contiguous_lids.end());
      return std::max(max_contiguous_lid, max_non_contiguous_lid);
    }
  }

  static std::vector<utils::Int32> getIndexes(std::vector<utils::Int32> const& item_lids) {
    std::vector<utils::Int32> indexes{};
    std::copy_if(item_lids.begin(), item_lids.end(), std::back_inserter(indexes), [](auto const& lid) { return lid != utils::NULL_ITEM_LID; });
    return indexes;
  }

  void clear() {
    m_non_contiguous_lids.clear();
    m_first_contiguous_lid = 0;
    m_nb_contiguous_lids = 0;
  }
};

/*---------------------------------------------------------------------------*/

struct ItemIterator
{
  using iterator_category = std::forward_iterator_tag;
  using value_type = int;
  using difference_type = int;
  using pointer = int*;
  using reference = int;
  explicit ItemIterator(ItemLocalIds item_indexes, int index)
  : m_index(index)
  , m_item_indexes(item_indexes) {}
  ItemIterator& operator++() {
    ++m_index;
    return *this;
  } // todo (handle traversal order...)
  ItemIterator operator++(int) {
    auto retval = *this;
    ++(*this);
    return retval;
  } // todo (handle traversal order...)
  int operator*() const { return m_item_indexes(m_index); }
  bool operator==(const ItemIterator& item_iterator) { return m_index == item_iterator.m_index; }
  bool operator!=(const ItemIterator& item_iterator) { return !(*this == item_iterator); }
  int m_index;
  ItemLocalIds m_item_indexes;
};

/*---------------------------------------------------------------------------*/

struct ItemRange
{
  ItemLocalIds m_item_lids;
  using size_type = int;

  std::vector<utils::Int32> localIds() const { return m_item_lids.itemArray(); }
  bool isContiguous() const { return m_item_lids.m_non_contiguous_lids.empty(); };
  ItemIterator begin() const { return ItemIterator{ m_item_lids, 0 }; }
  ItemIterator end() const { return ItemIterator{ m_item_lids, m_item_lids.size() }; } // todo : consider reverse range : constructeur (ItemLocalIds, traversal_order=forward) enum à faire
  size_type size() const { return (size_type)m_item_lids.size(); }
  bool isEmpty() const { return size() == 0; }
  utils::Int32 maxLocalId() const noexcept { return m_item_lids.maxLocalId(); }
  void clear() noexcept { m_item_lids.clear(); }

  friend inline NeoOutputStream& operator<<(NeoOutputStream& os, const Neo::ItemRange& item_range) {
    os << "Item Range : lids ";
    for (auto lid : item_range.m_item_lids.m_non_contiguous_lids) {
      os << lid;
      os << " ";
    }
    auto last_contiguous_lid = item_range.m_item_lids.m_first_contiguous_lid + item_range.m_item_lids.m_nb_contiguous_lids;
    for (auto i = item_range.m_item_lids.m_first_contiguous_lid; i < last_contiguous_lid; ++i) {
      os << i;
      os << " ";
    }
    os << Neo::endline;
    return os;
  }
};
/*---------------------------------------------------------------------------*/

} // namespace Neo

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Neo
{
namespace utils
{ // not defined in Utils.h since needs Neo::ItemRange
  inline Int32 maxItem(ItemRange const& item_range) {
    if (item_range.isEmpty())
      return utils::NULL_ITEM_LID;
    return *std::max_element(item_range.begin(), item_range.end());
  }

  inline Int32 minItem(ItemRange const& item_range) {
    if (item_range.isEmpty())
      return utils::NULL_ITEM_LID;
    return *std::min_element(item_range.begin(), item_range.end());
  }
} // namespace utils

/*---------------------------------------------------------------------------*/

namespace MeshKernel{
  class AlgorithmPropertyGraph;
}

class EndOfMeshUpdate
{
  friend class MeshKernel::AlgorithmPropertyGraph;

 private:
  EndOfMeshUpdate() = default;
};

/*---------------------------------------------------------------------------*/

struct FutureItemRange
{

  ItemRange new_items;
  bool is_data_released = false;

  FutureItemRange() = default;
  virtual ~FutureItemRange() = default;

  FutureItemRange(FutureItemRange const&) = default;
  FutureItemRange& operator=(FutureItemRange const&) = default;

  FutureItemRange(FutureItemRange&&) = default;
  FutureItemRange& operator=(FutureItemRange&&) = default;

  operator ItemRange&() {
    return _toItemRange();
  }

  virtual ItemRange get(EndOfMeshUpdate const&) {
    if (is_data_released)
      throw std::runtime_error(
      "Impossible to call FutureItemRange.get(), data already released.");
    is_data_released = true;
    return std::move(new_items);
  }

  ItemRange::size_type size() const {
    if (is_data_released)
      throw std::runtime_error(
      "Impossible to call FutureItemRange.size(), data already released.");
    return new_items.size();
  }

 private:
  virtual ItemRange& _toItemRange() {
    return new_items;
  }
};

/*---------------------------------------------------------------------------*/

struct FilteredFutureItemRange : public FutureItemRange
{

  FutureItemRange const& m_future_range;
  std::vector<int> m_filter;
  bool is_data_filtered = false;

  FilteredFutureItemRange() = delete;

  FilteredFutureItemRange(FutureItemRange const& future_item_range_ref,
                          std::vector<int> filter)
  : m_future_range(future_item_range_ref)
  , m_filter(std::move(filter)) {}

  virtual ~FilteredFutureItemRange() = default;

  FilteredFutureItemRange(FilteredFutureItemRange const&) = default;
  FilteredFutureItemRange& operator=(FilteredFutureItemRange const&) = default;

  FilteredFutureItemRange(FilteredFutureItemRange&&) = default;
  FilteredFutureItemRange& operator=(FilteredFutureItemRange&&) = default;

  operator ItemRange&() {
    return _toItemRange();
  }

  ItemRange get(EndOfMeshUpdate const& end_update) override {
    _toItemRange(); // filter data if not yet done
    return FutureItemRange::get(end_update); // move ItemRange. Class no longer usable
  }

 private:
  ItemRange& _toItemRange() override {
    if (!is_data_filtered) {
      std::vector<Neo::utils::Int32> filtered_lids(m_filter.size());
      auto local_ids = m_future_range.new_items.localIds();
      std::transform(m_filter.begin(), m_filter.end(), filtered_lids.begin(),
                     [&local_ids](auto index) {
                       return local_ids[index];
                     });
      new_items = ItemRange{ { std::move(filtered_lids) } };
      is_data_filtered = true;
    }
    return new_items;
  }
};

/*---------------------------------------------------------------------------*/

inline Neo::FutureItemRange make_future_range() {
  return FutureItemRange{};
}

inline Neo::FilteredFutureItemRange make_future_range(FutureItemRange& future_item_range,
                                                      std::vector<int> filter) {
  return FilteredFutureItemRange{ future_item_range, std::move(filter) };
}

/*---------------------------------------------------------------------------*/

template <typename ItemRangeT>
struct ItemRangeRefTypeT
{
  using type = ItemRange&;
};

template <>
struct ItemRangeRefTypeT<ItemRange const>
{
  using type = ItemRange const&;
};

template <typename ItemRangeT>
struct ItemRangeWrapper
{
  ItemRangeT& m_item_range;
  using ItemRangeRefType = typename ItemRangeRefTypeT<ItemRangeT>::type;

  ItemRangeRefType get() const noexcept {
    return m_item_range;
  }
};

//----------------------------------------------------------------------------/

/*!
 * Create a FilteredFutureItemRange filtering an FutureItemRange. The filter is here computed.
 * @tparam item_id Maybe local_id (Neo::utils::Int32) or unique_id (Neo::utils::Int64)
 * @param future_item_range The future range filtered
 * @param future_item_range_ids The set of ids in the future range
 * @param ids_subset The subset of ids kept from the future range
 * @return The FilteredFutureItemRange
 */
template <typename item_id>
inline Neo::FilteredFutureItemRange make_future_range(FutureItemRange& future_item_range,
                                                      std::vector<item_id> future_item_range_ids,
                                                      std::vector<item_id> ids_subset) {
  std::vector<int> filter(ids_subset.size());
  std::unordered_map<item_id, int> uid_index_map;
  auto index = 0;
  for (auto uid : future_item_range_ids) {
    uid_index_map.insert({ uid, index++ });
  }
  auto error = 0;
  auto i = 0;
  for (auto uid : ids_subset) {
    auto iterator = uid_index_map.find(uid);
    if (iterator == uid_index_map.end())
      ++error;
    else
      filter[i++] = (*iterator).second;
  }
  if (error > 0)
    throw std::runtime_error("in make_future_range, ids_subset contains element not present in future_item_range_ids");

  return FilteredFutureItemRange{ future_item_range, std::move(filter) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Neo

#endif //NEO_ITEMS_H
