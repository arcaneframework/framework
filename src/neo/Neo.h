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

#include "gtest/gtest.h"

#include "neo/Utils.h"

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

struct ItemIndexes { // todo change type, theses are local ids cf. Issue#7
  std::vector<std::size_t> m_non_contiguous_indexes = {};
  std::size_t m_first_contiguous_index = 0;
  std::size_t m_nb_contiguous_indexes = 0;
  std::size_t size()  const {return m_non_contiguous_indexes.size()+m_nb_contiguous_indexes;}
  int operator() (int index) const{
    if (index >= int(size())) return  size();
    if (index < 0) return -1;
    auto item_lid = 0;
    (index >= m_non_contiguous_indexes.size() || m_non_contiguous_indexes.size()==0) ?
        item_lid = m_first_contiguous_index + (index  - m_non_contiguous_indexes.size()) : // work on fluency
        item_lid = m_non_contiguous_indexes[index];
    return item_lid;
  }

  static std::vector<std::size_t> getIndexes(std::vector<Neo::utils::Int32> const& item_lids){
    std::vector<std::size_t> indexes{};
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
  explicit ItemIterator(ItemIndexes item_indexes, int index) : m_index(index), m_item_indexes(item_indexes){}
  ItemIterator& operator++() {++m_index;return *this;} // todo (handle traversal order...)
  ItemIterator operator++(int) {auto retval = *this; ++(*this); return retval;} // todo (handle traversal order...)
  int operator*() const {return m_item_indexes(m_index);}
  bool operator==(const ItemIterator& item_iterator) {return m_index == item_iterator.m_index;}
  bool operator!=(const ItemIterator& item_iterator) {return !(*this == item_iterator);}
  int m_index;
  ItemIndexes m_item_indexes;
};
struct ItemRange {
  bool isContiguous() const {return m_indexes.m_non_contiguous_indexes.empty();};
  ItemIterator begin() const {return ItemIterator{m_indexes,0};}
  ItemIterator end() const {return ItemIterator{m_indexes,int(m_indexes.size())};} // todo : consider reverse range : constructeur (ItemIndexes, traversal_order=forward) enum à faire
  std::size_t size() const {return m_indexes.size();}
  bool isEmpty() const  {return size() == 0;}
  ItemIndexes m_indexes;
};
}// end namespace Neo

inline
std::ostream &operator<<(std::ostream &os, const Neo::ItemRange &item_range){
  os << "Item Range : lids ";
  for (auto lid : item_range.m_indexes.m_non_contiguous_indexes) {
    os << lid;
    os << " ";
  }
  auto last_contiguous_index = item_range.m_indexes.m_first_contiguous_index + item_range.m_indexes.m_nb_contiguous_indexes;
  for (auto i = item_range.m_indexes.m_first_contiguous_index; i < last_contiguous_index; ++i) {
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

  void append(const ItemRange& item_range, const std::vector<DataType>& values) {
    if (item_range.size()==0) return;
    assert(("item_range and values sizes differ",item_range.size() == values.size()));
    auto max_item = utils::maxItem(item_range);
    if (max_item > m_data.size()) m_data.resize(max_item+1);
    std::size_t counter{0};
    for (auto item : item_range) {
      m_data[item] = values[counter++];
    }
  }

  DataType & operator[] (Neo::utils::Int32 item) { return m_data[item]; }
  DataType const& operator[] (Neo::utils::Int32 item) const { return m_data[item]; }

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

  PropertyConstView<DataType> constView() {
    std::vector<int> indexes(m_data.size()); std::iota(indexes.begin(),indexes.end(),0);
    return PropertyConstView<DataType>{std::move(indexes),Neo::utils::ConstArrayView<DataType>{m_data.size(),m_data.data()}};}
  PropertyConstView<DataType> constView(ItemRange const& item_range) {
    std::vector<int> indexes; indexes.reserve(item_range.size());
    for (auto item : item_range) indexes.push_back(item);
    return PropertyConstView<DataType>{std::move(indexes),Neo::utils::ConstArrayView<DataType>{m_data.size(),m_data.data()}};}

};

template <typename DataType>
class ArrayProperty : public PropertyBase {
public:
  void resize(std::vector<std::size_t> sizes){ // only 2 moves if a rvalue is passed. One copy + one move if lvalue
    m_offsets = std::move(sizes);
  }
  bool isInitializableFrom(const ItemRange& item_range){return item_range.isContiguous() && (*item_range.begin() ==0) && m_data.empty() ;}
  void init(const ItemRange& item_range, std::vector<DataType> values){
    assert(isInitializableFrom(item_range));
    m_data = std::move(values);
  }
  void append(ItemRange const& item_range, std::vector<DataType> const& values, std::vector<std::size_t> const& nb_values_per_item){
    if (item_range.size()==0) return;
    // todo: see how to handle new element add or remove impact on property (size/values)
    assert(item_range.size()==nb_values_per_item.size());
    assert(values.size()==std::accumulate(nb_values_per_item.begin(),nb_values_per_item.end(),0));
    if (utils::minItem(item_range) >= m_offsets.size()) _appendByBackInsertion(item_range,values,nb_values_per_item); // only new items
    else _appendByReconstruction(item_range,values,nb_values_per_item); // includes existing items
  }

  void _appendByReconstruction(ItemRange const& item_range, std::vector<DataType> const& values, std::vector<std::size_t> const& nb_connected_item_per_item){
    std::cout << "Append in ArrayProperty by reconstruction" << std::endl;
    // Compute new offsets
    std::vector<std::size_t> new_offsets(m_offsets);
    if (utils::maxItem(item_range) >= new_offsets.size()) new_offsets.resize(utils::maxItem(item_range)+1);// todo ajouter ArrayProperty::resize(maxlid)
    auto index = 0;
    for (auto item : item_range) {
      new_offsets[item] = nb_connected_item_per_item[index++];
    }
    // Compute new values
    auto new_data_size = std::accumulate(new_offsets.begin(), new_offsets.end(),0);
    std::vector<DataType> new_data(new_data_size);
    // copy new_values
    auto global_index = 0;
    std::vector<bool> marked_items(new_offsets.size(),false);
    for (auto item : item_range) {
      marked_items[item] = true;
      auto item_index = _getItemIndexInData(item, new_offsets);
      for (auto connected_item_index = item_index; connected_item_index < item_index + new_offsets[item]; ++connected_item_index) {
        new_data[connected_item_index] = values[global_index++];
      }
    }
    // copy old values
    ItemRange old_values_range{ItemIndexes{{},0,m_offsets.size()}};
    for (auto item : old_values_range) {
      if (!marked_items[item]) {
        auto connected_items = (*this)[item];
        auto connected_item_index = _getItemIndexInData(item,new_offsets);
        for (auto connected_item : connected_items){
          new_data[connected_item_index++] = connected_item;
        }
      }
    }
    m_offsets = std::move(new_offsets);
    m_data    = std::move(new_data);
  }

  void _appendByBackInsertion(ItemRange const& item_range, std::vector<DataType> const& values, std::vector<std::size_t> const& nb_connected_item_per_item){
    if (item_range.isContiguous()) {
      std::cout << "Append in ArrayProperty by back insertion, contiguous range" << std::endl;
      std::copy(nb_connected_item_per_item.begin(),
                nb_connected_item_per_item.end(),
                std::back_inserter(m_offsets));
      std::copy(values.begin(), values.end(), std::back_inserter(m_data));
    }
    else {
      std::cout << "Append in ArrayProperty by back insertion, non contiguous range" << std::endl;
      m_offsets.resize(utils::maxItem(item_range) + 1);
      auto index = 0;
      for (auto item : item_range) m_offsets[item] = nb_connected_item_per_item[index++];
      m_data.resize(m_data.size()+values.size(),DataType());
      index = 0;
      for (auto item : item_range) {
        std::cout << "item is " << item<< std::endl;
        auto connected_items = (*this)[item];
        std::cout << " item " << item << " index in data " << _getItemIndexInData(item) << std::endl;
        for (auto& connected_item : connected_items) {
          connected_item = values[index++];
        }
      }
    }
  }

  utils::ArrayView<DataType> operator[](const utils::Int32 item) {
    return utils::ArrayView<DataType>{m_offsets[item],&m_data[_getItemIndexInData(item)]};
  }

  utils::ConstArrayView<DataType> operator[](const utils::Int32 item) const {
    return utils::ConstArrayView<DataType>{m_offsets[item],&m_data[_getItemIndexInData(item)]};
  }

  void debugPrint() const {
    std::cout << "= Print array property " << m_name << " =" << std::endl;
    for (auto &val : m_data) {
      std::cout << "\"" << val << "\" ";
    }
    std::cout << std::endl;
  }

  // todo should be computed only when m_offsets is updated, at least implement an array version
  utils::Int32 _getItemIndexInData(const utils::Int32 item) const{
    std::accumulate(m_offsets.begin(),m_offsets.begin()+item,0);
  }

  utils::Int32 _getItemIndexInData(const utils::Int32 item, const std::vector<std::size_t>& offsets) const{
    std::accumulate(offsets.begin(),offsets.begin()+item,0);
  }

  std::size_t size() const {
    return std::accumulate(m_offsets.begin(), m_offsets.end(), 0);
  }


//  private:
  std::vector<DataType> m_data;
  std::vector<std::size_t> m_offsets;

};

// special case of local ids property
class ItemLidsProperty : public PropertyBase {
public:
  explicit ItemLidsProperty(std::string const& name) : PropertyBase{name}{};

  ItemRange append(std::vector<Neo::utils::Int64> const& uids) {
    std::size_t counter = 0;
    ItemIndexes item_indexes{};
    auto& non_contiguous_lids = item_indexes.m_non_contiguous_indexes;
    non_contiguous_lids.reserve(m_empty_lids.size());
    if (uids.size() >= m_empty_lids.size()) {
      for (auto empty_lid : m_empty_lids) {
        m_uid2lid[uids[counter++]] = empty_lid;
        non_contiguous_lids.push_back(empty_lid);
      }
      item_indexes.m_first_contiguous_index = m_last_id +1;
      for (auto uid = uids.begin() + counter; uid != uids.end(); ++uid) { // todo use span
        m_uid2lid[*uid] = ++m_last_id;
      }
      item_indexes.m_nb_contiguous_indexes = m_last_id - item_indexes.m_first_contiguous_index +1 ;
      m_empty_lids.clear();
    }
    else {// empty_lids.size > uids.size
      for(auto uid : uids) {
        m_uid2lid[uid] = m_empty_lids.back();
        non_contiguous_lids.push_back(m_empty_lids.back());
        m_empty_lids.pop_back();
      }
    }
    return ItemRange{std::move(item_indexes)};
  }

  ItemRange remove(std::vector<utils::Int64> const& uids){
    ItemIndexes item_indexes{};
    item_indexes.m_non_contiguous_indexes.resize(uids.size());
    auto empty_lids_size = m_empty_lids.size();
    m_empty_lids.resize( empty_lids_size + uids.size());
    auto counter = 0;
    auto empty_lids_index = empty_lids_size;
    for (auto uid : uids) {
      // remove from map (set NULL_ITEM_LID)
      // add in range and in empty_lids
      auto& lid = m_uid2lid.at(uid); // checks bound. To see whether costly
      m_empty_lids[empty_lids_index++] = lid;
      item_indexes.m_non_contiguous_indexes[counter++] = lid;
      lid = utils::NULL_ITEM_LID;
    }
    return ItemRange{std::move(item_indexes)};
    // todo handle last_id ??
  }

  std::size_t size() const {
    return m_last_id+1-m_empty_lids.size();
  }

  ItemRange values(){
    // TODO...; + il faut mettre en cache (dans la famille ?). ? de la mise à jour (la Propriété peut dire si la range est à jour)
    // 2 stratégies : on crée l'étendue continue avant ou après les non contigus...
    // (on estime que l'on décime les id les plus élevés ou les plus faibles), avoir le choix (avec un paramètre par défaut)
    ItemIndexes item_indexes{};
    if (m_empty_lids.empty()) { // range contiguous
      item_indexes = ItemIndexes{{},0,(std::size_t)(m_last_id+1)};
    }
    else { // mix a contiguous range and non contiguous elements. Choose the larger contiguous range
      auto max_empty_lid = *std::max_element(m_empty_lids.begin(),m_empty_lids.end());
      auto min_empty_lid = *std::min_element(m_empty_lids.begin(),m_empty_lids.end());
      auto left_contiguous_range_size = min_empty_lid;
      auto right_contiguous_range_size = m_last_id-max_empty_lid;
      auto choose_left_contiguous_range = (left_contiguous_range_size > right_contiguous_range_size) ? true : false;
      if (choose_left_contiguous_range) {
        item_indexes.m_first_contiguous_index = 0;
        item_indexes.m_nb_contiguous_indexes = left_contiguous_range_size;
        // build non contiguous
        auto empty_lids_sorted = m_empty_lids;
        std::sort(empty_lids_sorted.begin(), empty_lids_sorted.end());
        auto empty_lid_index = 0;
        auto non_contiguous_lid_index = 0;
        auto& non_contiguous_lids = item_indexes.m_non_contiguous_indexes;
        non_contiguous_lids.reserve(left_contiguous_range_size+m_empty_lids.size());
        for (auto lid = item_indexes.m_nb_contiguous_indexes;
             lid < m_last_id + 1; ++lid) {
          if (lid != empty_lids_sorted[empty_lid_index]) non_contiguous_lids[non_contiguous_lid_index++] = lid;
          else ++empty_lid_index;
        }
      } else {
        item_indexes.m_first_contiguous_index = max_empty_lid +1;
        item_indexes.m_nb_contiguous_indexes = right_contiguous_range_size;
        // build non contiguous
        auto empty_lids_sorted = m_empty_lids;
        std::sort(empty_lids_sorted.begin(), empty_lids_sorted.end());
        auto empty_lid_index = 0;
        auto non_contiguous_lid_index = 0;
        auto& non_contiguous_lids = item_indexes.m_non_contiguous_indexes;
        non_contiguous_lids.reserve(right_contiguous_range_size+m_empty_lids.size());
        for (auto lid = 0;
             lid < max_empty_lid +1; ++lid) {
          if (lid != empty_lids_sorted[empty_lid_index]) non_contiguous_lids[non_contiguous_lid_index++] = lid;
          else ++empty_lid_index;
        }
      }
    }
    return ItemRange{std::move(item_indexes)};
  }

  void debugPrint() const {
    std::cout << "= Print property " << m_name << " =" << std::endl;
    for (auto uid : m_uid2lid){
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

using Property = std::variant<PropertyT<utils::Int32>, PropertyT<utils::Real3>,PropertyT<utils::Int64>,ItemLidsProperty, ArrayProperty<utils::Int32>>;

namespace tye {
template <typename... T> struct VisitorOverload : public T... {
  using T::operator()...;
};

template <typename Func, typename Variant>
void apply(Func &func, Variant &arg) {
  auto default_func = [](auto arg) {
    std::cout << "Wrong Property Type" << std::endl;
  }; // todo: prevent this behavior (statically ?)
  std::visit(VisitorOverload{default_func, func}, arg);
}

template <typename Func, typename Variant>
void apply(Func &func, Variant& arg1, Variant& arg2) {
  std::visit([&arg2, &func](auto& concrete_arg1) {
    std::visit([&concrete_arg1, &func](auto& concrete_arg2){
      auto functor = VisitorOverload{[](const auto& arg1, const auto& arg2) {std::cout << "Wrong one." << std::endl;},func}; // todo: prevent this behavior (statically ?)
      functor(concrete_arg1,concrete_arg2);// arg1 & arg2 are variants, concrete_arg* are concrete arguments
    },arg2);
  },arg1);
}

template <typename Func, typename Variant>
void apply(Func& func, Variant& arg1, Variant& arg2, Variant& arg3) {
  std::visit([&arg2, &arg3, &func](auto& concrete_arg1) {
    std::visit([&concrete_arg1, &arg3, &func](auto &concrete_arg2) {
      std::visit([&concrete_arg1, &concrete_arg2, &func](auto &concrete_arg3) {
        auto functor = VisitorOverload{[](const auto &arg1, const auto &arg2, const auto &arg3) {std::cout << "Wrong one." << std::endl;},func}; // todo: prevent this behavior (statically ?)
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

  template<typename T>
  void addProperty(std::string const& name){
    auto [iter,is_inserted] = m_properties.insert(std::make_pair(name,PropertyT<T>{name}));
    if (is_inserted) std::cout << "Add property " << name << " in Family " << m_name
                << std::endl;
  };

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
  Family& operator()(ItemKind const & ik,std::string const& name) const
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

struct InProperty{

  auto& operator() () {
    return m_family.getProperty(m_name);
  }
  Family& m_family;
  std::string m_name;

};

//  template <typename DataType, typename DataIndex=int> // sans doute inutile, on devrait se poser la question du type (et meme on n'en a pas besoin) dans lalgo. on auranautomatiquement le bon type
struct OutProperty{

  auto& operator() () {
    return m_family.getProperty(m_name);
  }
  Family& m_family;
  std::string m_name;

}; // faut-il 2 types ?

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

class ItemRangeUnlocker {
  constexpr ItemRangeUnlocker() {}
  friend class MeshBase;
};


struct ScheduledItemRange {

  ItemRange &get(ItemRangeUnlocker const &valid_mesh_state){
    return new_items;
  }
  ItemRange new_items;
};

class MeshBase {
public:
  Family& addFamily(ItemKind ik, std::string&& name) {
    std::cout << "Add Family " << name << " in mesh " << m_name << std::endl;
    return m_families.push_back(ik, name);
  }

  Family& getFamily(ItemKind ik, std::string const& name) const { return m_families.operator()(ik,name);}


  template <typename Algorithm>
  void addAlgorithm(InProperty&& in_property, OutProperty&& out_property, Algorithm algo){// problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
    //?? ajout dans le graphe. recuperer les prop...à partir nom et kind…
    // mock the graph : play the action in the given order...
    m_algos.push_back(std::make_unique<AlgoHandler<decltype(algo)>>(std::move(in_property),std::move(out_property),std::forward<Algorithm>(algo)));
  }
  template <typename Algorithm>
  void addAlgorithm(OutProperty&& out_property, Algorithm algo) { // problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
    m_algos.push_back(std::make_unique<NoDepsAlgoHandler<decltype(algo)>>(std::move(out_property),std::forward<Algorithm>(algo)));
  }

  template <typename Algorithm>
  void addAlgorithm(InProperty&& in_property1, InProperty&& in_property2, OutProperty&& out_property, Algorithm algo){// problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
    //?? ajout dans le graphe. recuperer les prop...à partir nom et kind…
    // mock the graph : play the action in the given order...
    m_algos.push_back(std::make_unique<DualInAlgoHandler<decltype(algo)>>(
        std::move(in_property1),
        std::move(in_property2),
        std::move(out_property),
        std::forward<Algorithm>(algo)));
  }

  template <typename Algorithm>
  void addAlgorithm(OutProperty&& out_property1, OutProperty&& out_property2, Algorithm algo) {// problem when putting Algorithm&& (references captured by lambda are invalidated...Todo see why)
    m_algos.push_back(std::make_unique<NoDepsDualOutAlgoHandler<decltype(algo)>>(std::move(out_property1),std::move(out_property2),std::forward<Algorithm>(algo)));
  }

  ItemRangeUnlocker applyAlgorithms() {
    std::cout << "apply added algorithms" << std::endl;
    std::for_each(m_algos.begin(),m_algos.end(),[](auto& algo){(*algo.get())();});
    m_algos.clear();
    return ItemRangeUnlocker{};
  }


  std::string m_name;
  FamilyMap m_families;
  std::list<std::unique_ptr<IAlgorithm>> m_algos;
};

} // end namespace Neo


#endif // SRC_NEO_H
