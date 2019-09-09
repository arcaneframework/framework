
// #include "neo/neo.h" // unique file for itinerant developping

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
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
  }// end namespace utils

  std::ostream& operator<<(std::ostream& oss, const neo::utils::Real3& real3){
    oss << "{" << real3.x  << ","  << real3.y << "," << real3.z << "}";
    return oss;
  }
  
  struct ItemLocalId {};
  struct ItemUniqueId {};
  
  using DataType = std::variant<utils::Int32, utils::Int64, utils::Real3>;// ajouter des types dans la def de famille si necessaire
  using DataIndex = std::variant<int,ItemUniqueId>;

struct ItemIndexes {
  std::vector<std::size_t> m_non_contiguous_indexes;
  std::size_t m_first_contiguous_index =0;
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
    bool isContiguous() const {return true;};
    ItemIterator begin() const {return ItemIterator{m_indexes,0};}
    ItemIterator end() const {return ItemIterator{m_indexes,int(m_indexes.size())};} // todo : consider reverse range : constructeur (ItemIndexes, traversal_order=forward) enum à faire
    std::size_t size() const { return m_indexes.size();}
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

  class PropertyBase{
    public:
    std::string m_name;
    };

  template <typename DataType, typename IndexType=int>
  class PropertyT : public PropertyBase  {
    public:

    void append(const ItemRange& item_range, const std::vector<DataType>& values) {
      assert(item_range.size() == values.size());
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

  using Property = std::variant<PropertyT<utils::Int32>, PropertyT<utils::Real3>,PropertyT<utils::Int64>,PropertyT<ItemLocalId,ItemUniqueId>>;

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

    ItemRange append(const std::vector<neo::utils::Int64>& uids) {
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

    void debugPrint() const {
      std::cout << "= Print property " << m_name << " =" << std::endl;
      for (auto uid : m_uid2lid){
        std::cout << " uid to lid  " << uid.first << " : " << uid.second;
      }
      std::cout << std::endl;
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
  ir = neo::ItemRange{neo::ItemIndexes{{3,5,7}}};
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
node_family.addProperty<neo::utils::Int32>("node_con");

// Test adds
auto& property = node_family.getProperty("node_lids");

// Adding cell family and properties
auto& cell_family = mesh.getFamily(neo::ItemKind::IK_Cell,"CellFamily");
std::cout << "Find family " << cell_family.m_name << std::endl;
cell_family.addProperty<neo::ItemLocalId,neo::ItemUniqueId>("cell_lids");
cell_family.addProperty<neo::utils::Int64>("cell_uids");
cell_family.addProperty<neo::utils::Int32>("cell_con");
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
std::vector<neo::utils::Int64> cell_uids{0};

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
//mesh.addAlgorithm(neo::OutProperty{neo::ItemKind::IK_Cell,"cell_lids"},
//  [&cell_uids](neo::ItemLidsProperty& lids_property) {
//    //lids_property.append(cell_uids);//todo
//  });

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

  base_mesh_creation_test();

  partial_mesh_modif_test();

  return 0;
}