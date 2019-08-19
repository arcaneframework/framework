
// #include "neo/neo.h" // unique file for itinerant developping

#include <algorithm>
#include <array>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <variant>

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
  
  struct ItemLocalId {};
  struct ItemUniqueId {};
  
  using DataType = std::variant<utils::Int32, utils::Int64, utils::Real3>;// ajouter des types dans la def de famille si necessaire
  using DataIndex = std::variant<int,ItemUniqueId>;
  
  
  
  class PropertyBase{
    public:
    std::string m_name;
    };

  template <typename DataType, typename IndexType=int>
  class PropertyT : public PropertyBase  {
    public:
    std::vector<DataType> m_data;
    };

  using Property = std::variant<PropertyT<utils::Int32>, PropertyT<utils::Real3>,PropertyT<ItemUniqueId>,PropertyT<ItemLocalId,ItemUniqueId>>;

  class Family {
  public:
    template<typename T, typename IndexType =int>
    void addProperty(std::string const& name){
      m_properties[name] = PropertyT<T,IndexType>{name};
      std::cout << "Add property " << name << " in Family " << m_name<< std::endl;
      };
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
      m_families[std::make_pair(ik,name)] = Family{ik,name};
    }
  private:
    std::map<std::pair<ItemKind,std::string>, Family> m_families;

  };
  
  struct InProperty{
    ItemKind m_ik = ItemKind::IK_None;
    std::string m_name = "empty_property";
    
    };
  
  //template <typename DataType, typename DataIndex=int> // sans doute inutile, on devrait se poser la question du type (et meme on n'en a pas besoin) dans lalgo. on auranautomatiquement le bon type
  struct OutProperty{

//    auto& operator() () {
//      return family.getProperty(m_name);
//    }
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
//      m_algo(m_in_property,m_out_property);
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
//      m_algo(m_out_property());
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
    explicit PropertyT(std::string const& name) : PropertyBase{name}{}; // with a more recent compiler, could be possible to avoid to explicitly write constructor
  };
  
  using ItemLidsProperty = PropertyT<ItemLocalId,ItemUniqueId>;
 
} // end namespace Neo



/*-------------------------
 * Neo library first test
 * sdc (C)-2019
 *
 *-------------------------
 */
 
void test_property_graph()
{
  std::cout << "Test Property Graph" << std::endl;
  // to be done when the graph impl will be available
}
 
void test_lids_property()
{
  std::cout << "Test lids Property" << std::endl;
  
}

void prepare_mesh(neo::Mesh& mesh){

// Adding node family and properties
auto& node_family = mesh.getFamily(neo::ItemKind::IK_Node,"NodeFamily");
std::cout << "Find family " << node_family.m_name << std::endl;
node_family.addProperty<neo::utils::Real3>(std::string("node_coord"));
node_family.addProperty<neo::ItemLocalId,neo::ItemUniqueId>("node_lids");
node_family.addProperty<neo::ItemUniqueId>("node_uids");
node_family.addProperty<neo::utils::Int32>("node_con");

// Adding cell family and properties
auto& cell_family = mesh.getFamily(neo::ItemKind::IK_Cell,"CellFamily");
std::cout << "Find family " << cell_family.m_name << std::endl;
cell_family.addProperty<neo::ItemLocalId,neo::ItemUniqueId>("cell_lids");
cell_family.addProperty<neo::ItemUniqueId>("cell_uids");
cell_family.addProperty<neo::utils::Int32>("cell_con");
}
 
void base_mesh_creation_test() {


// creating mesh
auto mesh = neo::Mesh{"my_neo_mesh"};
auto& node_family = mesh.addFamily(neo::ItemKind::IK_Node,"NodeFamily");
auto& cell_family = mesh.addFamily(neo::ItemKind::IK_Cell,"CellFamily");

prepare_mesh(mesh);
// return;

// given data to create mesh
std::vector<neo::utils::Int64> node_uids{0,1,2};
std::vector<neo::utils::Real3> node_coords{{0,0,0}, {0,1,0}, {0,0,1}};
std::vector<neo::utils::Int64> cell_uids{0};

// add algos: 
mesh.beginUpdate();

// create nodes 
mesh.addAlgorithm(neo::OutProperty{node_family,"node_lids"},
  [&node_uids](neo::OutProperty & node_lids_property){
  std::cout << "Algorithm: create nodes" << std::endl;
  //node_lids_property.append(node_uids);//todo
  });

// register node uids
mesh.addAlgorithm(neo::InProperty{neo::ItemKind::IK_Node,"node_lids"},neo::OutProperty{node_family,"node_uids"},
  [&node_uids](neo::InProperty const& node_lids_property, neo::OutProperty& node_uids_property){
    std::cout << "Algorithm: register node uids" << std::endl;
   //auto& added_lids = node_lids_property.lastAppended();//todo
   //node_uids_property.appendAt(added_lids,node_uids);//steal node uids memory//todo
      });// need to add a property check for existing uid

// register node coords
mesh.addAlgorithm(neo::InProperty{neo::ItemKind::IK_Node,"node_lids"},neo::OutProperty{node_family,"node_coords"},
  [&node_coords](neo::InProperty const& node_lids_property, neo::OutProperty & node_coords_property){
    std::cout << "Algorithm: register node coords" << std::endl;
    //auto& added_lids = node_lids_property.lastAppended();// todo
    // node_coords_property.appendAt(added_lids, node_coords);// steal node_coords memory//todo
  });
  
// Add cells and connectivity
//mesh.addAlgorithm(neo::OutProperty{neo::ItemKind::IK_Cell,"cell_lids"},
//  [&cell_uids](neo::ItemLidsProperty& lids_property) {
//    //lids_property.append(cell_uids);//todo
//  });

// launch algos
mesh.endUpdate();
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

mesh.addAlgorithm(neo::InProperty{neo::ItemKind::IK_Node,"node_lids"},neo::OutProperty{node_family,"node_coords"},
  [&node_coords,&node_uids](neo::InProperty const& node_lids_property, neo::OutProperty & node_coords_property){
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

base_mesh_creation_test();

//partial_mesh_modif_test();

return 0;
}