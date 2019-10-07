
// #include "neo/neo.h" // unique file for itinerant developping

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

/*-------------------------
 * sdc - (C)-2019 -
 * NEtwork Oriented kernel
 * POC version 0.0
 * true version could be derived
 * from ItemFamilyNetwork
 *--------------------------
 */
 
namespace neo{
  
  enum class ItemKind {
    IK_Node, IK_Edge, IKFace, IK_Cell, IK_Dof, IK_None
  };
  
  
  
  namespace utils {
    using Int64 = long int;
    using Int32 = int;
    struct Real3 { double x,y,z;};
    template <typename T>
    struct ArrayView {
      T& operator[](int const i) {assert(i<m_size); return *(m_ptr+i);}
      T* begin() {return m_ptr;}
      T* end()   {return m_ptr+m_size;}
      std::size_t m_size;
      T* m_ptr;
    };
static constexpr utils::Int32 NULL_ITEM_LID = -1;
  }// end namespace utils

  std::ostream& operator<<(std::ostream& oss, const neo::utils::Real3& real3){
    oss << "{" << real3.x  << ","  << real3.y << "," << real3.z << "}";
    return oss;
  }
  
  struct ItemLocalId {};
  struct ItemUniqueId {};

  // todo: check if used ??
  using DataType = std::variant<utils::Int32, utils::Int64, utils::Real3>;// ajouter des types dans la def de famille si necessaire
  using DataIndex = std::variant<int,ItemUniqueId>;

struct ItemIndexes {
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

  std::ostream &operator<<(std::ostream &os, const ItemRange &item_range){
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

  namespace utils {
    Int32 maxItem(ItemRange const &item_range) {
      if (item_range.isEmpty())
        return utils::NULL_ITEM_LID;
      return *std::max_element(item_range.begin(), item_range.end());
    }
    Int32 minItem(ItemRange const &item_range) {
      if (item_range.isEmpty())
        return utils::NULL_ITEM_LID;
      return *std::min_element(item_range.begin(), item_range.end());
    }
  }

  class PropertyBase{
    public:
    std::string m_name;
    };

  template <typename DataType, typename IndexType=int>
  class PropertyT : public PropertyBase  {
    public:

    void append(const ItemRange& item_range, const std::vector<DataType>& values) {
      assert(item_range.size() == values.size());
      auto max_item = utils::maxItem(item_range);
      if (max_item > m_data.size()) m_data.resize(max_item+1);
      std::size_t counter{0};
      for (auto item : item_range) {
        m_data[item] = values[counter++];
      }
    }

    bool isInitializableFrom(const ItemRange& item_range) {return item_range.isContiguous() && (*item_range.begin() ==0) && m_data.empty() ;}

    void init(const ItemRange& item_range, std::vector<DataType> values){
      // data must be empty
      assert(item_range.isContiguous() && (*item_range.begin() ==0) && m_data.empty()); // todo comprehensive test (message for user)
      m_data = std::move(values);
    }

    void debugPrint() const {
      std::cout << "= Print property " << m_name << " =" << std::endl;
      for (auto &val : m_data) {
        std::cout << "\"" << val << "\" ";
      }
      std::cout << std::endl;
    }
    std::vector<DataType> m_data;
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

    void debugPrint() const {
      std::cout << "= Print array property " << m_name << " =" << std::endl;
      for (auto &val : m_data) {
        std::cout << "\"" << val << "\" ";
      }
      std::cout << std::endl;
    }

    // todo should be computed only when m_offsets is updated, at least implement an array version
    utils::Int32 _getItemIndexInData(const utils::Int32 item){
      std::accumulate(m_offsets.begin(),m_offsets.begin()+item,0);
    }

    utils::Int32 _getItemIndexInData(const utils::Int32 item, const std::vector<std::size_t>& offsets){
      std::accumulate(offsets.begin(),offsets.begin()+item,0);
    }


//  private:
    std::vector<DataType> m_data;
    std::vector<std::size_t> m_offsets;

  };

  using Property = std::variant<PropertyT<utils::Int32>, PropertyT<utils::Real3>,PropertyT<utils::Int64>,PropertyT<ItemLocalId,ItemUniqueId>, ArrayProperty<utils::Int32>, PropertyT<bool>>;

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
    template<typename T, typename IndexType =int>
    void addProperty(std::string const& name){
      m_properties[name] = PropertyT<T,IndexType>{name};
      std::cout << "Add property " << name << " in Family " << m_name<< std::endl;
      };
    Property& getProperty(const std::string& name) {
      return m_properties[name];
    }

    template <typename T>
    void addArrayProperty(std::string const& name){
      m_properties[name] = ArrayProperty<T>{name};
      std::cout << "Add array property " << name << " in Family " << m_name<< std::endl;
    }

    ItemKind m_ik;
    std::string m_name;
    std::map<std::string, Property> m_properties;
  };

  class FamilyMap {
  public:
    Family& operator() (ItemKind const & ik,std::string const& name)
    {
      return m_families[std::make_pair(ik,name)];
    }
    Family& push_back(ItemKind const & ik,std::string const& name)
    {
      return m_families[std::make_pair(ik,name)] = Family{ik,name};
    }

    auto begin() noexcept {return m_families.begin();}
    auto begin() const noexcept {return m_families.begin();}
    auto end() noexcept { return m_families.end();}
    auto end() const noexcept {return m_families.end();}

  private:
    std::map<std::pair<ItemKind,std::string>, Family> m_families;

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
    
  class Mesh {
public:
    Family& addFamily(ItemKind ik, std::string&& name) {
    std::cout << "Add Family " << name << " in mesh " << m_name << std::endl;
    return m_families.push_back(ik, name);
  }

    Family& getFamily(ItemKind ik, std::string&& name){ return m_families.operator()(ik,name);}


    template <typename Algorithm>
    void addAlgorithm(InProperty&& in_property, OutProperty&& out_property, Algorithm&& algo){
      //?? ajout dans le graphe. recuperer les prop...à partir nom et kind…
      // mock the graph : play the action in the given order...
      m_algos.push_back(std::make_unique<AlgoHandler<decltype(algo)>>(std::move(in_property),std::move(out_property),std::forward<Algorithm>(algo)));
      }
    template <typename Algorithm>
    void addAlgorithm(OutProperty&& out_property, Algorithm&& algo) {
      m_algos.push_back(std::make_unique<NoDepsAlgoHandler<decltype(algo)>>(std::move(out_property),std::forward<Algorithm>(algo)));
    }

    template <typename Algorithm>
    void addAlgorithm(InProperty&& in_property1, InProperty&& in_property2, OutProperty&& out_property, Algorithm&& algo){
      //?? ajout dans le graphe. recuperer les prop...à partir nom et kind…
      // mock the graph : play the action in the given order...
      m_algos.push_back(std::make_unique<DualInAlgoHandler<decltype(algo)>>(
          std::move(in_property1),
          std::move(in_property2),
          std::move(out_property),
          std::forward<Algorithm>(algo)));
    }

    template <typename Algorithm>
    void addAlgorithm(OutProperty&& out_property1, OutProperty&& out_property2, Algorithm&& algo) {
      m_algos.push_back(std::make_unique<NoDepsDualOutAlgoHandler<decltype(algo)>>(std::move(out_property1),std::move(out_property2),std::forward<Algorithm>(algo)));
    }
    
    void beginUpdate() { std::cout << "begin mesh update" << std::endl;}
    void endUpdate() {
      std::cout << "end mesh update" << std::endl;
      std::for_each(m_algos.begin(),m_algos.end(),[](auto& algo){(*algo.get())();});
    }
      
    
    std::string m_name;
    FamilyMap m_families;
    std::list<std::unique_ptr<IAlgorithm>> m_algos;
  };

  // special case of local ids property
  template <>
  class PropertyT<ItemLocalId,ItemUniqueId> : public PropertyBase {
    public:
    explicit PropertyT(std::string const& name) : PropertyBase{name}{};

    ItemRange append(std::vector<neo::utils::Int64> const& uids) {
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
    std::vector<neo::utils::Int32> m_empty_lids;
    std::map<neo::utils::Int64, neo::utils::Int32 > m_uid2lid; // todo at least unordered_map
    int m_last_id = -1;

  };
  
  using ItemLidsProperty = PropertyT<ItemLocalId,ItemUniqueId>;
 
} // end namespace Neo

/*-------------------------
 * Neo library first test
 * sdc (C)-2019
 *
 *-------------------------
 */

void test_item_range(){
  // Test with only contiguous indexes
  std::cout << "== Testing contiguous item range from 0 with 5 items =="<< std::endl;
  auto ir = neo::ItemRange{neo::ItemIndexes{{},0,5}};
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
  }
  // Test with only non contiguous indexes
  std::cout << "== Testing non contiguous item range {3,5,7} =="<< std::endl;
  ir = neo::ItemRange{neo::ItemIndexes{{3,5,7},0,0}};
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
  }
  // Test range mixing contiguous and non contiguous indexes
  std::cout << "== Testing non contiguous item range {3,5,7} + 8 to 11 =="<< std::endl;
  ir = neo::ItemRange{neo::ItemIndexes{{3,5,7},8,4}};
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
  }
  // Internal test for out of bound
  std::cout << "Get out of bound values (index > size) " << ir.m_indexes(100) << std::endl;
  std::cout << "Get out of bound values (index < 0) " << ir.m_indexes(-100) << std::endl;


  // todo test out reverse range
}

void test_array_property()
{
  auto array_property = neo::ArrayProperty<neo::utils::Int32>{"test_array_property"};
  // add elements: 5 items with one value
  neo::ItemRange item_range{neo::ItemIndexes{{},0,5}};
  std::vector<neo::utils::Int32> values{0,1,2,3,4};
  array_property.resize({1,1,1,1,1});
  array_property.init(item_range,values);
  array_property.debugPrint();
  // Add 3 items
  std::vector<std::size_t> nb_element_per_item{0,3,1};
  item_range = {neo::ItemIndexes{{5,6,7}}};
  std::vector<neo::utils::Int32> values_added{6,6,6,7};
  array_property.append(item_range, values_added, nb_element_per_item);
  array_property.debugPrint(); // expected result: "0" "1" "2" "3" "4" "6" "6" "6" "7" (check with test framework)
  // Add three more items
  item_range = {neo::ItemIndexes{{},8,3}};
  std::for_each(values_added.begin(), values_added.end(), [](auto &elt) {return elt += 2;});
  array_property.append(item_range, values_added, nb_element_per_item);
  array_property.debugPrint(); // expected result: "0" "1" "2" "3" "4" "6" "6" "6" "7" "8" "8" "8" "9"
  // Add items and modify existing item
  item_range = {neo::ItemIndexes{{0,8,5},11,1}};
  nb_element_per_item = {3,3,2,1};
  values_added = {10,10,10,11,11,11,12,12,13};
  array_property.append(item_range, values_added, nb_element_per_item); // expected result: "10" "10" "10" "1" "2" "3" "4" "12" "12" "6" "6" "6" "7" "11" "11" "11" "8" "8" "8" "9" "13"
  array_property.debugPrint();
}

void test_property_graph()
{
  std::cout << "Test Property Graph" << std::endl;
  // to be done when the graph impl will be available
}
 
void test_lids_property()
{
  std::cout << "Test lids Property" << std::endl;
  
}

void property_test(const neo::Mesh& mesh){
  std::cout << "== Print Mesh " << mesh.m_name << " Properties =="<< std::endl;
  for (const auto& [kind_name_pair,family] : mesh.m_families) {
    std::cout << "= In family " << kind_name_pair.second << " =" << std::endl;
    for (const auto& [prop_name,property] : family.m_properties){
      std::cout << prop_name << std::endl;
    }
  }
  std::cout << "== End Print Mesh " << mesh.m_name << " Properties =="<< std::endl;
}

void prepare_mesh(neo::Mesh& mesh){

// Adding node family and properties
auto& node_family = mesh.getFamily(neo::ItemKind::IK_Node,"NodeFamily");
std::cout << "Find family " << node_family.m_name << std::endl;
node_family.addProperty<neo::utils::Real3>(std::string("node_coords"));
node_family.addProperty<neo::ItemLocalId,neo::ItemUniqueId>("node_lids");
node_family.addProperty<neo::utils::Int64>("node_uids");
node_family.addArrayProperty<neo::utils::Int32>("node2cells");
node_family.addProperty<bool>("internal_end_of_remove_tag"); // not a user-defined property

// Test adds
auto& property = node_family.getProperty("node_lids");

// Adding cell family and properties
auto& cell_family = mesh.getFamily(neo::ItemKind::IK_Cell,"CellFamily");
std::cout << "Find family " << cell_family.m_name << std::endl;
cell_family.addProperty<neo::ItemLocalId,neo::ItemUniqueId>("cell_lids");
cell_family.addProperty<neo::utils::Int64>("cell_uids");
cell_family.addArrayProperty<neo::utils::Int32>("cell2nodes");
}
 
void base_mesh_creation_test() {


// creating mesh
auto mesh = neo::Mesh{"my_neo_mesh"};
auto& node_family = mesh.addFamily(neo::ItemKind::IK_Node,"NodeFamily");
auto& cell_family = mesh.addFamily(neo::ItemKind::IK_Cell,"CellFamily");

prepare_mesh(mesh);
// return;

// given data to create mesh. After mesh creation data is no longer available
std::vector<neo::utils::Int64> node_uids{0,1,2};
std::vector<neo::utils::Real3> node_coords{{0,0,0}, {0,1,0}, {0,0,1}};
std::vector<neo::utils::Int64> cell_uids{0,2,7,9};

// add algos: 
mesh.beginUpdate();

// create nodes
auto added_nodes = neo::ItemRange{};
mesh.addAlgorithm(neo::OutProperty{node_family,"node_lids"},
  [&node_uids,&added_nodes](neo::PropertyT<neo::ItemLocalId,neo::ItemUniqueId> & node_lids_property){
  std::cout << "Algorithm: create nodes" << std::endl;
  added_nodes = node_lids_property.append(node_uids);
  node_lids_property.debugPrint();
  std::cout << "Inserted item range : " << added_nodes;
  });

// register node uids
mesh.addAlgorithm(neo::InProperty{node_family,"node_lids"},neo::OutProperty{node_family,"node_uids"},
  [&node_uids,&added_nodes](neo::PropertyT<neo::ItemLocalId,neo::ItemUniqueId> const& node_lids_property, neo::PropertyT<neo::utils::Int64>& node_uids_property){
    std::cout << "Algorithm: register node uids" << std::endl;
  if (node_uids_property.isInitializableFrom(added_nodes))  node_uids_property.init(added_nodes,std::move(node_uids)); // init can steal the input values
  else node_uids_property.append(added_nodes, node_uids);
  node_uids_property.debugPrint();
    });// need to add a property check for existing uid

// register node coords
mesh.addAlgorithm(neo::InProperty{node_family,"node_lids"},neo::OutProperty{node_family,"node_coords"},
  [&node_coords,&added_nodes](neo::PropertyT<neo::ItemLocalId,neo::ItemUniqueId> const& node_lids_property, neo::PropertyT<neo::utils::Real3> & node_coords_property){
    std::cout << "Algorithm: register node coords" << std::endl;
    if (node_coords_property.isInitializableFrom(added_nodes))  node_coords_property.init(added_nodes,std::move(node_coords)); // init can steal the input values
    else node_coords_property.append(added_nodes, node_coords);
    node_coords_property.debugPrint();
  });
//
// Add cells and connectivity

// create cells
auto added_cells = neo::ItemRange{};
mesh.addAlgorithm(neo::OutProperty{cell_family,"cell_lids"},
  [&cell_uids,&added_cells](neo::ItemLidsProperty& cell_lids_property) {
    std::cout << "Algorithm: create cells" << std::endl;
    added_cells = cell_lids_property.append(cell_uids);
    cell_lids_property.debugPrint();
    std::cout << "Inserted item range : " << added_cells;
  });

// register cell uids
mesh.addAlgorithm(neo::InProperty{cell_family,"cell_lids"},neo::OutProperty{cell_family,"cell_uids"},
    [&cell_uids,&added_cells](neo::ItemLidsProperty const& cell_lids_property, neo::PropertyT<neo::utils::Int64>& cell_uids_property){
      std::cout << "Algorithm: register cell uids" << std::endl;
      if (cell_uids_property.isInitializableFrom(added_cells))  cell_uids_property.init(added_cells,std::move(cell_uids)); // init can steal the input values
      else cell_uids_property.append(added_cells, cell_uids);
      cell_uids_property.debugPrint();
    });

// register connectivity
// node to cell
std::vector<neo::utils::Int64> connected_cell_uids{0,0,2,2,7,9};
std::vector<std::size_t> nb_cell_per_node{1,2,3};
mesh.addAlgorithm(neo::InProperty{node_family,"node_lids"},
                  neo::InProperty{cell_family,"cell_lids"},
                  neo::OutProperty{node_family,"node2cells"},
    [&connected_cell_uids, &nb_cell_per_node,& added_nodes]
    (neo::ItemLidsProperty const& node_lids_property, neo::ItemLidsProperty const& cell_lids_property, neo::ArrayProperty<neo::utils::Int32> & node2cells){
      std::cout << "Algorithm: register node-cell connectivity" << std::endl;
      auto connected_cell_lids = cell_lids_property[connected_cell_uids];
      if (node2cells.isInitializableFrom(added_nodes)) {
        node2cells.resize(std::move(nb_cell_per_node));
        node2cells.init(added_nodes,std::move(connected_cell_lids));
      }
      else {
        node2cells.append(added_nodes,connected_cell_lids, nb_cell_per_node);
      }
      node2cells.debugPrint();
});

// cell to node
std::vector<neo::utils::Int64> connected_node_uids{0,1,2,1,2,0,2,1,0};// on ne connecte volontairement pas toutes les mailles pour vérifier initialisation ok sur la famille
std::vector<std::size_t> nb_node_per_cell{3,0,3,3};
mesh.addAlgorithm(neo::InProperty{node_family,"node_lids"},
                  neo::InProperty{cell_family,"cell_lids"},
                  neo::OutProperty{cell_family,"cell2nodes"},
                  [&connected_node_uids, &nb_node_per_cell,& added_cells]
                          (neo::ItemLidsProperty const& node_lids_property, neo::ItemLidsProperty const& cell_lids_property, neo::ArrayProperty<neo::utils::Int32> & cells2nodes){
                    std::cout << "Algorithm: register cell-node connectivity" << std::endl;
                    auto connected_node_lids = node_lids_property[connected_node_uids];
                    if (cells2nodes.isInitializableFrom(added_cells)) {
                      cells2nodes.resize(std::move(nb_node_per_cell));
                      cells2nodes.init(added_cells,std::move(connected_node_lids));
                    }
                    else cells2nodes.append(added_cells,connected_node_lids, nb_node_per_cell);
                    cells2nodes.debugPrint();
                  });

// try to modify an existing property
// add new cells
std::vector<neo::utils::Int64> new_cell_uids{10,11,12}; // elles seront toutes rouges
auto new_cell_added =neo::ItemRange{};
mesh.addAlgorithm(neo::OutProperty{cell_family,"cell_lids"},
                [&new_cell_uids,&new_cell_added](neo::ItemLidsProperty& cell_lids_property) {
                  std::cout << "Algorithm: add new cells" << std::endl;
                  new_cell_added = cell_lids_property.append(new_cell_uids);
                  cell_lids_property.debugPrint();
                  std::cout << "Inserted item range : " << new_cell_added;
                });

// register new cell uids
mesh.addAlgorithm(neo::InProperty{cell_family,"cell_lids"},neo::OutProperty{cell_family,"cell_uids"},
                  [&new_cell_uids,&new_cell_added](neo::ItemLidsProperty const& cell_lids_property, neo::PropertyT<neo::utils::Int64>& cell_uids_property){
                    std::cout << "Algorithm: register new cell uids" << std::endl;
                    // must append and not initialize
                    if (cell_uids_property.isInitializableFrom(new_cell_added))  cell_uids_property.init(new_cell_added,std::move(new_cell_uids)); // init can steal the input values
                    else cell_uids_property.append(new_cell_added, new_cell_uids);
                    cell_uids_property.debugPrint();
                  });

// add connectivity to new cells
std::vector<neo::utils::Int64> new_cell_connected_node_uids{0,1,2,1,2};// on ne connecte volontairement pas toutes les mailles pour vérifier initialisation ok sur la famille
std::vector<std::size_t> nb_node_per_new_cell{0,3,2};
mesh.addAlgorithm(neo::InProperty{node_family,"node_lids"},
                  neo::InProperty{cell_family,"cell_lids"},
                  neo::OutProperty{cell_family,"cell2nodes"},
                  [&new_cell_connected_node_uids, &nb_node_per_new_cell,& new_cell_added]
                          (neo::ItemLidsProperty const& node_lids_property, neo::ItemLidsProperty const& cell_lids_property, neo::ArrayProperty<neo::utils::Int32> & cells2nodes){
                    std::cout << "Algorithm: register new cell-node connectivity" << std::endl;
                    auto connected_node_lids = node_lids_property[new_cell_connected_node_uids];
                    if (cells2nodes.isInitializableFrom(new_cell_added)) {
                      cells2nodes.resize(std::move(nb_node_per_new_cell));
                      cells2nodes.init(new_cell_added,std::move(connected_node_lids));
                    }
                    else cells2nodes.append(new_cell_added,connected_node_lids,nb_node_per_new_cell);
                    cells2nodes.debugPrint();
                  });

// remove nodes
std::vector<neo::utils::Int64> removed_node_uids{1,2};
auto removed_nodes = neo::ItemRange{};
mesh.addAlgorithm(neo::OutProperty{node_family,"node_lids"}, neo::OutProperty{node_family,"internal_end_of_remove_tag"},
                  [&removed_node_uids,&removed_nodes, &node_family](neo::ItemLidsProperty& node_lids_property, neo::PropertyT<bool> & internal_end_of_remove_tag){
                    std::cout << "Algorithm: remove nodes" << std::endl;
                    removed_nodes = node_lids_property.remove(removed_node_uids);
                    node_lids_property.debugPrint();
                    std::cout << "removed item range : " << removed_nodes;
                  });


// launch algos
mesh.endUpdate();

// test properties
property_test(mesh);
}


void partial_mesh_modif_test() {
  
// modify node coords
// input data
std::array<int,3> node_uids {0,1,3};
neo::utils::Real3 r = {0,0,0};
std::array<neo::utils::Real3,3> node_coords = {r,r,r};// don't get why I can't write {{0,0,0},{0,0,0},{0,0,0}}; ...??

// creating mesh
auto mesh = neo::Mesh{"my_neo_mesh"};
auto& node_family = mesh.addFamily(neo::ItemKind::IK_Node,"NodeFamily");
auto& cell_family = mesh.addFamily(neo::ItemKind::IK_Cell,"CellFamily");


prepare_mesh(mesh);

mesh.beginUpdate();

mesh.addAlgorithm(neo::InProperty{node_family,"node_lids"},neo::OutProperty{node_family,"node_coords"},
  [&node_coords,&node_uids](neo::PropertyT<neo::ItemLocalId,neo::ItemUniqueId> const& node_lids_property, neo::PropertyT<neo::utils::Real3> & node_coords_property){
    std::cout << "Algorithm: register node coords" << std::endl;
    //auto& lids = node_lids_property[node_uids];//todo
    //node_coords_property.appendAt(lids, node_coords);// steal node_coords memory//todo
  });

mesh.endUpdate();
  
}

// prepare a second partial scenario
 
int main() {
  
  std::cout << "*------------------------------------*"<< std::endl;
  std::cout << "* Test framework Neo thoughts " << std::endl;
  std::cout << "*------------------------------------*" << std::endl;

  test_item_range();

  test_array_property();

  partial_mesh_modif_test();

  return 0;
}