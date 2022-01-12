//
// Created by dechaiss on 1/22/20.
//

#ifndef SRC_NEO_H
#define SRC_NEO_H

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <variant>
#include <vector>
#include <stdexcept>
#include <utility>

#include "Utils.h"

/*-------------------------
 * sdc - (C)-2019 -
 * NEtwork Oriented kernel
 * POC version 0.0
 * true version could be derived
 * from ItemFamilyNetwork
 *--------------------------
 */

namespace Neo {

enum class ItemKind {
  IK_None, IK_Node, IK_Edge, IK_Face, IK_Cell, IK_Dof
};

namespace utils {

inline std::string itemKindName(ItemKind item_kind){
  switch (item_kind) {
  case ItemKind::IK_Node :
    return "IK_Node";
    break;
  case ItemKind::IK_Edge :
    return "IK_Edge";
    break;
  case ItemKind::IK_Face :
    return "IK_Face";
    break;
  case ItemKind::IK_Cell :
    return "IK_Cell";
    break;
  case ItemKind::IK_Dof :
    return "IK_Dof";
    break;
  case ItemKind::IK_None :
    return "IK_None";
    break;
  }
}
}

struct ItemLocalId {};
struct ItemUniqueId {};

// todo: check if used ??
using DataType = std::variant<utils::Int32, utils::Int64, utils::Real3>;// ajouter des types dans la def de famille si necessaire
using DataIndex = std::variant<int,ItemUniqueId>;

struct ItemLocalIds {
  std::vector<utils::Int32 > m_non_contiguous_lids = {};
  utils::Int32 m_first_contiguous_lid = 0;
  int m_nb_contiguous_lids = 0;

  int size()  const {return (int)m_non_contiguous_lids.size()+ m_nb_contiguous_lids;}

  int operator() (int index) const{
    if (index >= int(size())) return  size();
    if (index < 0) return -1;
    auto item_lid = 0;
    (index >= (int) m_non_contiguous_lids.size() || m_non_contiguous_lids.size()==0) ?
        item_lid = m_first_contiguous_lid + (index  - (int)m_non_contiguous_lids.size()) : // work on fluency
        item_lid = m_non_contiguous_lids[index];
    return item_lid;
  }

  std::vector<utils::Int32> itemArray() const {
    std::vector<utils::Int32> item_array(m_non_contiguous_lids.size()+
                                              m_nb_contiguous_lids);
    std::copy(m_non_contiguous_lids.begin(), m_non_contiguous_lids.end(),item_array.begin());
    std::iota(item_array.begin() + m_non_contiguous_lids.size(),item_array.end(), m_first_contiguous_lid);
    return item_array;
  }

  utils::Int32 maxLocalId() const {
    auto max_contiguous_lid = m_first_contiguous_lid+m_nb_contiguous_lids-1;
    if (m_non_contiguous_lids.empty()) return max_contiguous_lid;
    else {
      auto max_non_contiguous_lid = *std::max_element(m_non_contiguous_lids.begin(),
                                                      m_non_contiguous_lids.end());
      return std::max(max_contiguous_lid,max_non_contiguous_lid);
    }
  }

  static std::vector<utils::Int32 > getIndexes(std::vector<utils::Int32> const& item_lids){
    std::vector<utils::Int32> indexes{};
    std::copy_if(item_lids.begin(),item_lids.end(),std::back_inserter(indexes),[](auto const& lid) { return lid != utils::NULL_ITEM_LID; });
    return indexes;
  }
};

struct ItemIterator {
  using iterator_category = std::input_iterator_tag;
  using value_type = int;
  using difference_type = int;
  using pointer = int*;
  using reference = int;
  explicit ItemIterator(ItemLocalIds item_indexes, int index) : m_index(index), m_item_indexes(item_indexes){}
  ItemIterator& operator++() {++m_index;return *this;} // todo (handle traversal order...)
  ItemIterator operator++(int) {auto retval = *this; ++(*this); return retval;} // todo (handle traversal order...)
  int operator*() const {return m_item_indexes(m_index);}
  bool operator==(const ItemIterator& item_iterator) {return m_index == item_iterator.m_index;}
  bool operator!=(const ItemIterator& item_iterator) {return !(*this == item_iterator);}
  int m_index;
  ItemLocalIds m_item_indexes;
};
struct ItemRange {
  std::vector<utils::Int32> localIds() const {return m_item_lids.itemArray();}
  ItemLocalIds m_item_lids;

  bool isContiguous() const {return m_item_lids.m_non_contiguous_lids.empty();};
  ItemIterator begin() const {return ItemIterator{m_item_lids,0};}
  ItemIterator end() const {return ItemIterator{m_item_lids,int(m_item_lids.size())};} // todo : consider reverse range : constructeur (ItemLocalIds, traversal_order=forward) enum à faire
  std::size_t size() const {return m_item_lids.size();}
  bool isEmpty() const  {return size() == 0;}
  utils::Int32 maxLocalId() const noexcept { return m_item_lids.maxLocalId();}

};
}// end namespace Neo

inline
std::ostream &operator<<(std::ostream &os, const Neo::ItemRange &item_range){
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
  os << std::endl;
  return os;
}

namespace Neo{
namespace utils {
inline
Int32 maxItem(ItemRange const &item_range) {
  if (item_range.isEmpty())
    return utils::NULL_ITEM_LID;
  return *std::max_element(item_range.begin(), item_range.end());
}

inline
Int32 minItem(ItemRange const &item_range) {
  if (item_range.isEmpty())
    return utils::NULL_ITEM_LID;
  return *std::min_element(item_range.begin(), item_range.end());
}
}

template <typename DataType>
class PropertyView
{
public:
  std::vector<int> const m_indexes;
  Neo::utils::ArrayView<DataType> m_data_view;
  DataType& operator[] (int index) {
    assert(("Error, exceeds property view size",index < m_indexes.size()));
    return m_data_view[m_indexes[index]];}
};

template <typename DataType>
class PropertyConstView
{
public:
  std::vector<int> const m_indexes;
  Neo::utils::ConstArrayView<DataType> m_data_view;
  DataType const& operator[] (int index) const{
    assert(("Error, exceeds property view size",index < m_indexes.size()));
    return m_data_view[m_indexes[index]];}
};

class PropertyBase{
public:
  std::string m_name;
};

template <typename DataType>
class PropertyT : public PropertyBase  {
public:
  std::vector<DataType> m_data;

  void init(const ItemRange& item_range, const DataType& value){
    if (isInitializableFrom(item_range))
      init(item_range, std::vector<DataType>(item_range.size(), value));
    else
      append(item_range,std::vector<DataType>(item_range.size(), value));
  }

  bool isInitializableFrom(const ItemRange& item_range) const {return item_range.isContiguous() && (*item_range.begin() ==0) && m_data.empty() ;}

  // The difference between init and append is done to handle values copy or move
  void init(const ItemRange& item_range, std::vector<DataType> values){
    // data must be empty
    assert(item_range.isContiguous() && (*item_range.begin() ==0) && m_data.empty()); // todo comprehensive test (message for user)
    m_data = std::move(values);
  }

  void append(const ItemRange& item_range, const std::vector<DataType>& values, DataType default_value=DataType{}) {
    if (item_range.size()==0) return;
    assert(("item_range and values sizes differ",item_range.size() == values.size()));
    auto max_item = utils::maxItem(item_range);
    if (max_item > m_data.size()) m_data.resize(max_item+1,default_value);
    std::size_t counter{0};
    for (auto item : item_range) {
      m_data[item] = values[counter++];
    }
  }

  DataType & operator[] (utils::Int32 item) {
    assert(("Input item lid > max local id, In PropertyT[]",item < m_data.size()));
    return m_data[item]; }
  DataType const& operator[] (utils::Int32 item) const {
    assert(("Input item lid > max local id, In PropertyT[]",item < m_data.size()));
    return m_data[item]; }
  std::vector<DataType> operator[] (std::vector<utils::Int32>const& items) const {
    return _arrayAccessor(items);
  }
  std::vector<DataType> operator[] (utils::ConstArrayView<utils::Int32>const& items) const {
    return _arrayAccessor(items);
  }

  template <typename ArrayType>
  std::vector<DataType> _arrayAccessor (ArrayType items) const {
    // check bounds
    assert(("Max input item lid > max local id, In PropertyT[]",
               *(std::max_element(items.begin(),items.end()))< (int)m_data.size()));

    std::vector<DataType> values;
    values.reserve(items.size());
    std::transform(items.begin(),items.end(),std::back_inserter(values),
                   [this](auto item){return m_data[item];});
    return values;
  }

  void debugPrint() const {
    std::cout << "= Print property " << m_name << " =" << std::endl;
    for (auto &val : m_data) {
      std::cout << "\"" << val << "\" ";
    }
    std::cout << std::endl;
  }

  utils::ArrayView<DataType> values() {return Neo::utils::ArrayView<DataType>{m_data.size(), m_data.data()};}

  std::size_t size() const {return m_data.size();}

  PropertyView<DataType> view() {
    std::vector<int> indexes(m_data.size()); std::iota(indexes.begin(),indexes.end(),0);
    return PropertyView<DataType>{std::move(indexes),Neo::utils::ArrayView<DataType>{m_data.size(),m_data.data()}};}

  PropertyView<DataType> view(ItemRange const& item_range) {
    std::vector<int> indexes; indexes.reserve(item_range.size());
    for (auto item : item_range) indexes.push_back(item);
    return PropertyView<DataType>{std::move(indexes),Neo::utils::ArrayView<DataType>{m_data.size(),m_data.data()}};}

  PropertyConstView<DataType> constView() const {
    std::vector<int> indexes(m_data.size()); std::iota(indexes.begin(),indexes.end(),0);
    return PropertyConstView<DataType>{std::move(indexes),Neo::utils::ConstArrayView<DataType>{m_data.size(),m_data.data()}};}

  PropertyConstView<DataType> constView(ItemRange const& item_range) const {
    std::vector<int> indexes; indexes.reserve(item_range.size());
    for (auto item : item_range) indexes.push_back(item);
    return PropertyConstView<DataType>{std::move(indexes),Neo::utils::ConstArrayView<DataType>{m_data.size(),m_data.data()}};}

};

template <typename DataType>
class ArrayProperty : public PropertyBase {

public:
  std::vector<DataType> m_data;
  std::vector<int> m_offsets;
  std::vector<int> m_indexes;
  int m_size;

public:
  void resize(std::vector<int> sizes){ // only 2 moves if a rvalue is passed. One copy + one move if lvalue
    m_offsets = std::move(sizes);
    _updateIndexes();
  }
  bool isInitializableFrom(const ItemRange& item_range){return item_range.isContiguous() && (*item_range.begin() ==0) && m_data.empty() ;}
  void init(const ItemRange& item_range, std::vector<DataType> values){
    assert(isInitializableFrom(item_range));
    m_data = std::move(values);
  }
  void append(ItemRange const& item_range, std::vector<DataType> const& values, std::vector<int> const& nb_values_per_item){
    if (item_range.size()==0) return;
    // todo: see how to handle new element add or remove impact on property (size/values)
    assert(item_range.size()==nb_values_per_item.size());
    assert(("connected items array size and nb_values_per_item size are not compatible",
    values.size()==std::accumulate(nb_values_per_item.begin(),nb_values_per_item.end(),0)));
    if (utils::minItem(item_range) >= m_offsets.size()) _appendByBackInsertion(item_range,values,nb_values_per_item); // only new items
    else _appendByReconstruction(item_range,values,nb_values_per_item); // includes existing items
  }

  void _appendByReconstruction(ItemRange const& item_range, std::vector<DataType> const& values, std::vector<int> const& nb_connected_item_per_item){
    std::cout << "Append in ArrayProperty by reconstruction" << std::endl;
    // Compute new offsets
    std::vector<int> new_offsets(m_offsets);
    if (utils::maxItem(item_range) >= new_offsets.size()) new_offsets.resize(utils::maxItem(item_range)+1);// todo ajouter ArrayProperty::resize(maxlid)
    auto index = 0;
    for (auto item : item_range) {
      new_offsets[item] = nb_connected_item_per_item[index++];
    }
    // Compute new indexes
    std::vector<int> new_indexes;
    _computeIndexesFromOffsets(new_indexes, new_offsets);
    // Compute new values
    auto new_data_size = _computeSizeFromOffsets(new_offsets);
    std::vector<DataType> new_data(new_data_size);
    // copy new_values
    auto global_index = 0;
    std::vector<bool> marked_items(new_offsets.size(),false);
    for (auto item : item_range) {
      marked_items[item] = true;
      auto item_index = new_indexes[item];
      for (auto connected_item_index = item_index; connected_item_index < item_index + new_offsets[item]; ++connected_item_index) {
        new_data[connected_item_index] = values[global_index++];
      }
    }
    // copy old values
    ItemRange old_values_range{ItemLocalIds{{},0,(int)m_offsets.size()}};
    for (auto item : old_values_range) {
      if (!marked_items[item]) {
        auto connected_items = (*this)[item];
        std::copy(connected_items.begin(),connected_items.end(),&new_data[new_indexes[item]]);
      }
    }
    m_offsets = std::move(new_offsets);
    m_indexes = std::move(new_indexes);
    m_data    = std::move(new_data);
    m_size = new_data_size;
  }

  void _appendByBackInsertion(ItemRange const& item_range, std::vector<DataType> const& values, std::vector<int> const& nb_connected_item_per_item){
    if (item_range.isContiguous()) {
      std::cout << "Append in ArrayProperty by back insertion, contiguous range" << std::endl;
      auto max_existing_lid = m_offsets.size()-1;
      auto min_new_lid = utils::minItem(item_range);
      if (min_new_lid > max_existing_lid+1) {
        m_offsets.resize(min_new_lid,0);
      }
      std::copy(nb_connected_item_per_item.begin(),
                nb_connected_item_per_item.end(),
                std::back_inserter(m_offsets));
      std::copy(values.begin(), values.end(), std::back_inserter(m_data));
      _updateIndexes();
    }
    else {
      std::cout << "Append in ArrayProperty by back insertion, non contiguous range" << std::endl;
      m_offsets.resize(utils::maxItem(item_range) + 1);
      auto index = 0;
      for (auto item : item_range) m_offsets[item] = nb_connected_item_per_item[index++];
      m_data.resize(m_data.size()+values.size(),DataType());
      _updateIndexes();
      index = 0;
      for (auto item : item_range) {
        auto connected_items = (*this)[item];
        for (auto& connected_item : connected_items) {
          connected_item = values[index++];
        }
      }
    }
  }

  utils::ArrayView<DataType> operator[](const utils::Int32 item) {
    assert(("item local id must be >0 in ArrayProperty::[item_lid]]",item >= 0));
    return utils::ArrayView<DataType>{m_offsets[item],&m_data[m_indexes[item]]};
  }

  utils::ConstArrayView<DataType> operator[](const utils::Int32 item) const {
    assert(("item local id must be >0 in ArrayProperty::[item_lid]]",item >= 0));
    return utils::ConstArrayView<DataType>{m_offsets[item],&m_data[m_indexes[item]]};
  }

  void debugPrint() const {
    std::cout << "= Print array property " << m_name << " =" << std::endl;
    for (auto &val : m_data) {
      std::cout << "\"" << val << "\" ";
    }
    std::cout << std::endl;
    Neo::utils::printContainer(m_offsets, "Offsets ");
    Neo::utils::printContainer(m_indexes, "Indexes");
  }

   int size() const {
    return m_size;
  }

private:

  void _updateIndexes(){
    _computeIndexesFromOffsets(m_indexes, m_offsets);
    m_size = _computeSizeFromOffsets(m_offsets);
  }

  void _computeIndexesFromOffsets(std::vector<int>& new_indexes, std::vector<int> const& new_offsets){
    new_indexes.resize(new_offsets.size());
    auto i = 0, offset_sum = 0;
    for (auto &index : new_indexes) {
      index = offset_sum;
      offset_sum += new_offsets[i++];
    }
    // todo use algo version instead with more recent compilers (gcc >=9, clang >=5)
    //std::exclusive_scan(new_offsets.begin(),new_offsets.end(),new_indexes.begin(),0);
  }

  int _computeSizeFromOffsets(std::vector<int> const& new_offsets) {
    return std::accumulate(new_offsets.begin(), new_offsets.end(), 0);
  }
};

// special case of local ids property
class ItemLidsProperty : public PropertyBase {
public:
  explicit ItemLidsProperty(std::string const& name) : PropertyBase{name}{};

  ItemRange append(std::vector<Neo::utils::Int64> const& uids) {
    ItemLocalIds item_local_ids{};
    // handle mutliple insertion
    auto min_size = std::min(m_empty_lids.size(),uids.size());
    auto empty_lid_size = m_empty_lids.size();
    auto& non_contiguous_lids = item_local_ids.m_non_contiguous_lids;
    non_contiguous_lids.resize(min_size);
    auto used_empty_lid_count = 0;
    for(auto i = 0; i < (int) min_size;++i){
      const auto [inserted, do_insert] = m_uid2lid.insert({uids[i],m_empty_lids[empty_lid_size-1-used_empty_lid_count]});
      non_contiguous_lids[i]= inserted->second;
      if (do_insert) ++used_empty_lid_count;
    }
    m_empty_lids.resize(empty_lid_size-used_empty_lid_count);
    using item_index_and_lid = std::pair<int,Neo::utils::Int32> ;
    std::vector<item_index_and_lid> existing_items;
    existing_items.reserve(uids.size() - min_size);
    auto first_contiguous_id = m_last_id+1;
    item_local_ids.m_first_contiguous_lid = first_contiguous_id;
    for (auto i = min_size; i < uids.size();++i){
      const auto [inserted, do_insert] = m_uid2lid.insert({uids[i],++m_last_id});
      if (!do_insert) {
        existing_items.push_back({i-min_size,inserted->second});
        --m_last_id;
    }
      ++item_local_ids.m_nb_contiguous_lids;
    }
    // if an existing item is inserted, cannot use contiguous indexes, otherwise the range
    // will not handle the items in their insertion order, all lids must be in non_contiguous_indexes
    if (! existing_items.empty()) {
      std::vector<Neo::utils::Int32> non_contiguous_from_contigous_lids(
          item_local_ids.m_nb_contiguous_lids);
      std::iota(non_contiguous_from_contigous_lids.begin(),non_contiguous_from_contigous_lids.end(),first_contiguous_id);
      for (const auto [item_index,item_lid] : existing_items){
        non_contiguous_from_contigous_lids[item_index] = item_lid;
        std::for_each(non_contiguous_from_contigous_lids.begin()+item_index+1,non_contiguous_from_contigous_lids.end(),[](auto& current_lid){return --current_lid;});
      }
      item_local_ids.m_nb_contiguous_lids = 0;
      item_local_ids.m_non_contiguous_lids.insert(
          item_local_ids.m_non_contiguous_lids.end(),
              non_contiguous_from_contigous_lids.begin(),
              non_contiguous_from_contigous_lids.end());
    }
    return ItemRange{std::move(item_local_ids)};
  }

  /*! Remove item with uids \a uids.
   *
   * @param uids :unique ids of item to remove. If a non existing uid is given,
   * the code won't throw. The return range will contain a NULL_ITEM_LID for this
   * unexisting uid.
   * @return Returns a range  containing the local ids of removed items. They can
   * be used to update properties. If a local id is NULL_ITEM_LID, it means the
   * corresponding uids was not existing.
   *
   */
  ItemRange remove(std::vector<utils::Int64> const& uids) noexcept {
    ItemLocalIds item_local_ids{};
    item_local_ids.m_non_contiguous_lids.resize(uids.size());
    auto empty_lids_size = m_empty_lids.size();
    m_empty_lids.resize( empty_lids_size + uids.size());
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
      if (lid != utils::NULL_ITEM_LID) m_empty_lids[empty_lids_index++] = lid;
      item_local_ids.m_non_contiguous_lids[counter++] = lid;
    }
    return ItemRange{std::move(item_local_ids)};
  }

  std::size_t size() const {
    return m_last_id+1-m_empty_lids.size();
  }

  /*! Access to the item_lids stored in the property through an ItemRange object.
   *
   * @return  the ItemRange containing the lids of the property.
   */
  ItemRange values() const {
    // TODO...; + il faut mettre en cache (dans la famille ?). ? de la mise à jour (la Propriété peut dire si la range est à jour)
    // 2 stratégies : on crée l'étendue continue avant ou après les non contigus...
    // (on estime que l'on décime les id les plus élevés ou les plus faibles), avoir le choix (avec un paramètre par défaut)
    ItemLocalIds item_local_ids{};
    if (m_empty_lids.empty()) { // range contiguous
      item_local_ids = ItemLocalIds{{},0,m_last_id+1};
    }
    else {// range discontiguous
      std::vector<Neo::utils::Int32> lids(m_last_id + 1);
      std::iota(lids.begin(), lids.end(), 0);
      std::for_each(m_empty_lids.begin(), m_empty_lids.end(),
                    [&lids](auto const &emty_lid) {
                      lids[emty_lid] = Neo::utils::NULL_ITEM_LID;
                    });
      auto& active_lids = item_local_ids.m_non_contiguous_lids;
      active_lids.resize(lids.size() - m_empty_lids.size());
      std::copy_if(lids.begin(), lids.end(), active_lids.begin(),
                   [](auto const &lid_source) {
                     return lid_source != Neo::utils::NULL_ITEM_LID;
                   });
    }
    return ItemRange{std::move(item_local_ids)};
  }

  void debugPrint() const {
    std::cout << "= Print property " << m_name << " =" << std::endl;
    for (auto uid : m_uid2lid){
      if (uid.second != Neo::utils::NULL_ITEM_LID)
        std::cout << " uid to lid  " << uid.first << " : " << uid.second;
    }
    std::cout << std::endl;
  }

  utils::Int32 _getLidFromUid(utils::Int64 const uid) const {
    auto iterator = m_uid2lid.find(uid);
    if (iterator == m_uid2lid.end()) return utils::NULL_ITEM_LID;
    else return iterator->second;

  }
  void _getLidsFromUids(std::vector<utils::Int32>& lids, std::vector<utils::Int64> const& uids) const {
    std::transform(uids.begin(),uids.end(),std::back_inserter(lids),[this](auto const& uid){return this->_getLidFromUid(uid);});
  }
  std::vector<utils::Int32> operator[](std::vector<utils::Int64> const& uids) const {
    std::vector<utils::Int32> lids;
    _getLidsFromUids(lids,uids);
    return lids;
  }

private:
  std::vector<Neo::utils::Int32> m_empty_lids;
  std::map<Neo::utils::Int64, Neo::utils::Int32 > m_uid2lid; // todo at least unordered_map
  int m_last_id = -1;

};

using Property = std::variant<
    PropertyT<utils::Int32>,
    //PropertyT<int>, // int and Int32 are same types
    PropertyT<utils::Real3>,
    PropertyT<utils::Int64>,
    ItemLidsProperty,
    //ArrayProperty<int>, // int and Int32 are same types
    ArrayProperty<utils::Int32>>;

namespace tye {
template <typename... T> struct VisitorOverload : public T... {
  using T::operator()...;
};

template <typename Func, typename Variant>
void apply(Func &func, Variant &arg) {
#ifdef _MSC_VER
  auto default_func = [](auto arg) {
#else
  auto default_func = []([[maybe_unused]] auto arg) {
#endif
    std::cout << "Wrong Property Type" << std::endl;
  }; // todo: prevent this behavior (statically ?)
  std::visit(VisitorOverload{default_func, func}, arg);
}

template <typename Func, typename Variant>
void apply(Func &func, Variant& arg1, Variant& arg2) {
  std::visit([&arg2, &func](auto& concrete_arg1) {
    std::visit([&concrete_arg1, &func](auto& concrete_arg2){
#ifdef _MSC_VER // incomplete maybe_unused support on visual (v 19.29)
        auto functor = VisitorOverload{[](const auto&  arg1,const auto& arg2) {std::cout << "Wrong one." << std::endl;},func}; // todo: prevent this behavior (statically ?)
#else
        auto functor = VisitorOverload{[]([[maybe_unused]] const auto&  arg1,[[maybe_unused]] const auto& arg2) {std::cout << "Wrong one." << std::endl;},func}; // todo: prevent this behavior (statically ?)
#endif
      functor(concrete_arg1,concrete_arg2);// arg1 & arg2 are variants, concrete_arg* are concrete arguments
    },arg2);
  },arg1);
}

template <typename Func, typename Variant>
void apply(Func& func, Variant& arg1, Variant& arg2, Variant& arg3) {
  std::visit([&arg2, &arg3, &func](auto& concrete_arg1) {
    std::visit([&concrete_arg1, &arg3, &func](auto &concrete_arg2) {
      std::visit([&concrete_arg1, &concrete_arg2, &func](auto &concrete_arg3) {
#ifdef _MSC_VER
        auto functor = VisitorOverload{[](const auto &arg1,const auto &arg2,const auto &arg3) {std::cout << "Wrong one." << std::endl;},func}; // todo: prevent this behavior (statically ?)
#else
        auto functor = VisitorOverload{[]([[maybe_unused]] const auto &arg1,[[maybe_unused]] const auto &arg2, [[maybe_unused]]const auto &arg3) {std::cout << "Wrong one." << std::endl;},func}; // todo: prevent this behavior (statically ?)
#endif
        functor(concrete_arg1, concrete_arg2,concrete_arg3); // arg1 & arg2 are variants, concrete_arg* are concrete arguments
      }, arg3);
    }, arg2);
  }, arg1);
}

// template deduction guides
template <typename...T> VisitorOverload(T...) -> VisitorOverload<T...>;

}// todo move in TypeEngine (proposal change namespace to tye..)


class Family {
public:

  ItemKind m_ik;
  std::string m_name;
  std::string m_prop_lid_name;
  std::map<std::string, Property> m_properties;
  ItemRange m_all;

  Family(ItemKind ik, std::string name) : m_ik(ik), m_name(std::move(name)), m_prop_lid_name(name) {
    m_prop_lid_name.append("_lids");
    m_properties[lidPropName()] = ItemLidsProperty{lidPropName()};
  }

  constexpr std::string const& name() const noexcept {return m_name;}
  constexpr ItemKind const& itemKind() const noexcept {return m_ik;}

  std::vector<Neo::utils::Int32> itemUniqueIdsToLocalids(std::vector<Neo::utils::Int64> const& item_uids) const {
    return _lidProp().operator[](item_uids);
  }
  void itemUniqueIdsToLocalids(std::vector<Neo::utils::Int32> & item_lids, std::vector<Neo::utils::Int64> const& item_uids) const {
    assert(("In itemUniqueIdsToLocalIds, lids and uids sizes differ.",item_lids.size() == item_uids.size()));
    _lidProp()._getLidsFromUids(item_lids,item_uids);
  }

  template<typename T>
  void addProperty(std::string const& name){
    auto [iter,is_inserted] = m_properties.insert(std::make_pair(name,PropertyT<T>{name}));
    if (is_inserted) std::cout << "Add property " << name << " in Family " << m_name
                << std::endl;
  }

  Property& getProperty(const std::string& name) {
    auto found_property = m_properties.find(name);
    if (found_property == m_properties.end()) throw std::invalid_argument("Cannot find Property "+name);
    return found_property->second;
  }

  Property const& getProperty(const std::string& name) const {
    auto found_property = m_properties.find(name);
    if (found_property == m_properties.end()) throw std::invalid_argument("Cannot find Property "+name);
    return found_property->second;
  }

  template <typename PropertyType>
  PropertyType& getConcreteProperty(const std::string& name) {
    return std::get<PropertyType>(getProperty(name));
  }

  template <typename PropertyType>
  PropertyType const& getConcreteProperty(const std::string& name) const {
    return std::get<PropertyType>(getProperty(name));
  }

  template <typename T>
  void addArrayProperty(std::string const& name){
    auto [iter, is_inserted] = m_properties.insert(std::make_pair(name, ArrayProperty<T>{name}));
    if (is_inserted) std::cout << "Add array property " << name << " in Family " << m_name
                << std::endl;
  }

  std::string const&  lidPropName()
  { return m_prop_lid_name;}

  std::size_t nbElements() const {
    return _lidProp().size();
  }

  ItemRange& all() {
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



private :
  Property& _getProperty(const std::string& name) {
    auto found_property = m_properties.find(name);
    if (found_property == m_properties.end()) throw std::invalid_argument("Cannot find Property "+name);
    return found_property->second;
  }
};

class FamilyMap {
public:
  Family& operator()(ItemKind const & ik,std::string const& name) const noexcept (ndebug)
  {
    auto found_family = m_families.find(std::make_pair(ik,name));
    assert(("Cannot find Family ",found_family!= m_families.end()));
    return *(found_family->second.get());
  }
  Family& push_back(ItemKind const & ik,std::string const& name)
  {
    return *(m_families.emplace(std::make_pair(ik, name), std::make_unique<Family>(Family(ik,name))).first->second.get());
  }

  auto begin() noexcept {return m_families.begin();}
  auto begin() const noexcept {return m_families.begin();}
  auto end() noexcept { return m_families.end();}
  auto end() const noexcept {return m_families.end();}

private:
  std::map<std::pair<ItemKind,std::string>, std::unique_ptr<Family>> m_families;

};

struct PropertyHolder{
  auto& operator() () {
    return m_family.getProperty(m_name);
  }

  std::string uniqueName() const noexcept {
    return m_name + "_" + m_family.name();
  }
  Family& m_family;
  std::string m_name;

};
struct InProperty : public PropertyHolder{};

struct OutProperty : public PropertyHolder{};

struct IAlgorithm {
  virtual void operator() () = 0;
};


template <typename Algorithm>
struct AlgoHandler : public IAlgorithm {
  AlgoHandler(InProperty&& in_prop, OutProperty&& out_prop, Algorithm&& algo)
      : m_in_property(std::move(in_prop))
      , m_out_property(std::move(out_prop))
      , m_algo(std::forward<Algorithm>(algo)){}
  InProperty m_in_property;
  OutProperty m_out_property;
  Algorithm m_algo;
  void operator() () override {
    tye::apply(m_algo,m_in_property(),m_out_property());
  }
};

template <typename Algorithm>
struct DualInAlgoHandler : public IAlgorithm {
  DualInAlgoHandler(InProperty&& in_prop1, InProperty&& in_prop2, OutProperty&& out_prop, Algorithm&& algo)
      : m_in_property1(std::move(in_prop1))
      , m_in_property2(std::move(in_prop2))
      , m_out_property(std::move(out_prop))
      , m_algo(std::forward<Algorithm>(algo)){}
  InProperty m_in_property1;
  InProperty m_in_property2;
  OutProperty m_out_property;
  Algorithm m_algo;
  void operator() () override {
    tye::apply(m_algo,m_in_property1(),m_in_property2(),m_out_property());
  }
};

template <typename Algorithm>
struct NoDepsAlgoHandler : public IAlgorithm {
  NoDepsAlgoHandler(OutProperty&& out_prop, Algorithm&& algo)
      : m_out_property(std::move(out_prop))
      , m_algo(std::forward<Algorithm>(algo)){}
  OutProperty m_out_property;
  Algorithm m_algo;
  void operator() () override {
    tye::apply(m_algo,m_out_property());
  }
};

template <typename Algorithm>
struct NoDepsDualOutAlgoHandler : public IAlgorithm {
  NoDepsDualOutAlgoHandler(OutProperty&& out_prop1, OutProperty&& out_prop2, Algorithm&& algo)
      : m_out_property1(std::move(out_prop1))
      , m_out_property2(std::move(out_prop2))
      , m_algo(std::forward<Algorithm>(algo)){}
  OutProperty m_out_property1;
  OutProperty m_out_property2;
  Algorithm m_algo;
  void operator() () override {
    tye::apply(m_algo,m_out_property1(),m_out_property2());
  }
};

class MeshBase;

class EndOfMeshUpdate {
  friend class MeshBase;
private:
  EndOfMeshUpdate() = default;
};

struct FutureItemRange {

  ItemRange new_items;
  bool is_data_released = false;

  FutureItemRange() = default;
  virtual ~FutureItemRange() = default;

  FutureItemRange(FutureItemRange&) = default;
  FutureItemRange& operator=(FutureItemRange&) = default;

  FutureItemRange(FutureItemRange&&) = default;
  FutureItemRange& operator=(FutureItemRange&&) = default;

  virtual ItemRange get(EndOfMeshUpdate const &) {
    if (is_data_released)
      throw std::runtime_error(
          "Impossible to call FutureItemRange.get(), data already released.");
    is_data_released = true;
    return std::move(new_items);
  }

  virtual ItemRange& __internal__() {
    return new_items;
  }

};

struct FilteredFutureItemRange : public FutureItemRange {

  FutureItemRange const& m_future_range;
  std::vector<int> m_filter;
  bool is_data_filtered = false;

  FilteredFutureItemRange() = delete;

  FilteredFutureItemRange(FutureItemRange& future_item_range_ref,
                          std::vector<int> filter)
    : m_future_range(future_item_range_ref)
    , m_filter(std::move(filter)){}

  virtual ~FilteredFutureItemRange() = default;

  FilteredFutureItemRange(FilteredFutureItemRange&) = default;
  FilteredFutureItemRange& operator=(FilteredFutureItemRange&) = default;

  FilteredFutureItemRange(FilteredFutureItemRange&&) = default;
  FilteredFutureItemRange& operator=(FilteredFutureItemRange&&) = default;

  ItemRange get(EndOfMeshUpdate const & end_update) override {
    if (!is_data_filtered) throw std::runtime_error("Impossible to call FilteredFutureItemRange.get(), data not yet filtered. Call __internal()__ first.");
    return FutureItemRange::get(end_update);
  }

  ItemRange& __internal__() override {
    if (!is_data_filtered) {
      std::vector<Neo::utils::Int32> filtered_lids(m_filter.size());
      auto local_ids = m_future_range.new_items.localIds();
      std::transform(m_filter.begin(),m_filter.end(),filtered_lids.begin(),
                     [&local_ids](auto index){
                       return local_ids[index];
                     });
      new_items = ItemRange{{std::move(filtered_lids)}};
      is_data_filtered = true;
    }
    return new_items;
  }
};

inline Neo::FutureItemRange make_future_range(){
  return FutureItemRange{};
}

inline Neo::FilteredFutureItemRange make_future_range(FutureItemRange& future_item_range,
                                               std::vector<int> filter) {
  return FilteredFutureItemRange{future_item_range,std::move(filter)};
}

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
                                                      std::vector<item_id> ids_subset){
    std::vector<int> filter(ids_subset.size());
    std::map<item_id, int> uid_index_map;
    auto index = 0;
    for (auto uid : future_item_range_ids) {
      uid_index_map.insert({uid,index++});
    }
    auto error = 0;
    auto i = 0;
    for (auto uid : ids_subset) {
      auto iterator = uid_index_map.find(uid);
      if (iterator == uid_index_map.end()) ++error;
      else filter[i++] = (*iterator).second;
    }
    if (error > 0 ) throw std::runtime_error("in make_future_range, ids_subset contains element not present in future_item_range_ids");

  return FilteredFutureItemRange{future_item_range,std::move(filter)};

}

class MeshBase {
 public:
  std::string m_name;
  FamilyMap m_families;
  std::list<std::shared_ptr<IAlgorithm>> m_algos;
  int m_dimension = 3;
  using AlgoPtr = std::shared_ptr<IAlgorithm>;
  using ProducingAlgoArray = std::vector<AlgoPtr>;
  using ConsumingAlgoArray = std::vector<AlgoPtr>;
  using PropertyRef = std::reference_wrapper<const PropertyHolder>;
  std::map<std::string,std::pair<ProducingAlgoArray,ConsumingAlgoArray>> m_property_algorithms;
//  SGraph::DirectedAcyclicGraph<AlgoPtr,PropertyHolder> m_dag;
  enum class AlgorithmExecutionOrder {FIFO, LIFO, DAG};

public:
  Family& addFamily(ItemKind ik, std::string&& name) {
    std::cout << "Add Family " << name << " in mesh " << m_name << std::endl;
    return m_families.push_back(ik, name);
  }

  Family& getFamily(ItemKind ik, std::string const& name) const noexcept(ndebug) { return m_families.operator()(ik,name);}

  [[nodiscard]] int nbItems(ItemKind ik) const {
    return std::accumulate(m_families.begin(),m_families.end(),0,
                           [ik](auto const& nb_item, auto const& family_map_element){
                             return (family_map_element.first.first == ik) ? nb_item + family_map_element.second->nbElements() : nb_item;
    });
  }

  template <typename Algorithm>
  void addAlgorithm(InProperty&& in_property, OutProperty&& out_property, Algorithm algo){// problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
    //?? ajout dans le graphe. recuperer les prop...à partir nom et kind…
    // mock the graph : play the action in the given order...
    auto algo_hander = std::make_shared<AlgoHandler<decltype(algo)>>(std::move(in_property),std::move(out_property),std::forward<Algorithm>(algo));
    m_algos.push_back(algo_hander);
    _addProducingAlgo(algo_hander->m_out_property, algo_hander);
    _addConsumingAlgo(algo_hander->m_in_property,algo_hander);
  }

  template <typename Algorithm>
  void addAlgorithm(OutProperty&& out_property, Algorithm algo) { // problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
    auto algo_handler = std::make_shared<NoDepsAlgoHandler<decltype(algo)>>(std::move(out_property),std::forward<Algorithm>(algo));
    m_algos.push_back(algo_handler);
    _addProducingAlgo(algo_handler->m_out_property, algo_handler);
  }

  template <typename Algorithm>
  void addAlgorithm(InProperty&& in_property1, InProperty&& in_property2, OutProperty&& out_property, Algorithm algo){// problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
    auto algo_handler = std::make_shared<DualInAlgoHandler<decltype(algo)>>(
    std::move(in_property1),
    std::move(in_property2),
    std::move(out_property),
    std::forward<Algorithm>(algo));
    m_algos.push_back(algo_handler);
    _addProducingAlgo(algo_handler->m_out_property, algo_handler);
    _addConsumingAlgo(algo_handler->m_in_property1, algo_handler);
    _addConsumingAlgo(algo_handler->m_in_property2, algo_handler);
  }

  template <typename Algorithm>
  void addAlgorithm(OutProperty&& out_property1, OutProperty&& out_property2, Algorithm algo) {// problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
    auto algo_handler = std::make_shared<NoDepsDualOutAlgoHandler<decltype(algo)>>(std::move(out_property1), std::move(out_property2), std::forward<Algorithm>(algo));
    m_algos.push_back(algo_handler);
    _addProducingAlgo(algo_handler->m_out_property1,algo_handler);
    _addProducingAlgo(algo_handler->m_out_property2,algo_handler);
  }

  EndOfMeshUpdate applyAlgorithms() {
    std::cout << "apply added algorithms" << std::endl;
    std::for_each(m_algos.begin(),m_algos.end(),[](auto& algo){(*algo.get())();});
    m_algos.clear();
    return EndOfMeshUpdate{};
  }



  void _addProducingAlgo(OutProperty const& out_property, std::shared_ptr<IAlgorithm> algo){
    // add algo as one of producing algo of out_property
    auto& [producing_algo_array, consuming_algo_array] = m_property_algorithms[out_property.uniqueName()];
    producing_algo_array.push_back(algo);
  }

  void _addConsumingAlgo(InProperty const& in_property, std::shared_ptr<IAlgorithm> algo){
    // add algo as one of the consuming algos of out_property
    auto& [producing_algo_array, consuming_algo_array] = m_property_algorithms[in_property.uniqueName()];
    consuming_algo_array.push_back(algo);
  }

};

} // end namespace Neo


#endif // SRC_NEO_H
