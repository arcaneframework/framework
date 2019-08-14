
// #include "neo/neo.h" // unique file for itinerant developping

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <list>
#include <memory>

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
  
  
  class Property{
    public: 
    Property (Property const&) = delete;
    Property& operator= (Property const&) = delete;
    std::string m_name;
    };
   
   
   
  template <typename DataType, typename IndexType=int>
  class PropertyT  : public Property {
    public:
    std::vector<DataType> m_data;
    };
    

  class Family {
  public:
    template<typename T, typename IndexType =int>
    void addProperty(std::string const& name){
//      m_properties[name] = PropertyT<T,IndexType>{name}; // nonsense
      std::cout << "Add property " << name << " in Family " << m_name<< std::endl;
      };
    std::string m_name;
//    std::map<std::string,Property> m_properties;
  };
  
  struct InProperty{
    ItemKind m_ik = ItemKind::IK_None;
    std::string m_name = "empty_property";
    
    };
    
  struct OutProperty{
    ItemKind m_ik = ItemKind::IK_None;
    std::string m_name;
    
    }; // faut-il 2 types ?

  struct IAlgorithm {
    virtual void operator() () =0;
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
      m_algo(m_in_property,m_out_property);
    }
  };

  template <typename Algorithm>
  struct NoDepsAlgoHandler : public IAlgorithm {
    NoDepsAlgoHandler(OutProperty&& out_prop, Algorithm&& algo) : m_out_property(std::move(out_prop)), m_algo(std::forward<Algorithm>(algo)){}
    OutProperty m_out_property;
    Algorithm m_algo;
    void operator() () override {
      m_algo(m_out_property);
    }
  };
    
  class Mesh {
public:
    Family& addFamily(ItemKind ik, std::string&& name){
      std::cout << "Add Family " << name << " in mesh " << m_name << std::endl;
      m_families.push_back(Family{std::move(name)}); //todo if name+kind exists: what we do ?
      return m_families.back();
      
}
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
    void endUpdate() { std::cout << "end mesh update" << std::endl;}
      
    
    std::string m_name;
    std::vector<Family> m_families;// todo map ?(search for families) ? quelle cle ? il faut ik + nom
    std::list<std::unique_ptr<IAlgorithm>> m_algos;
  };
  
  struct ItemLocalId {};
  struct ItemUniqueId {};
  
  // special case of local ids property
  template <>
  class PropertyT<ItemLocalId,ItemUniqueId> : public Property{
    public:
    explicit PropertyT(std::string const& name) : Property{name}{}; // with a more recent compiler, could be possible to avoid to explicitly write constructor
  };
  
  using ItemLidsProperty = PropertyT<ItemLocalId,ItemUniqueId>;
  
  namespace utils {
    using Int64 = long int;
    using Int32 = int;
    struct Real3 { double x,y,z;};
  }// end namespace utils
 
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
auto node_family = mesh.addFamily(neo::ItemKind::IK_Node,"NodeFamily");
node_family.addProperty<neo::utils::Real3>(std::string("node_coord"));
node_family.addProperty<neo::ItemLocalId,neo::ItemUniqueId>("node_lids");
node_family.addProperty<neo::ItemUniqueId>("node_uids");
node_family.addProperty<neo::utils::Int32>("node_con");

// Adding cell family and properties
auto cell_family = mesh.addFamily(neo::ItemKind::IK_Cell,"CellFamily");
cell_family.addProperty<neo::ItemLocalId>("cell_lids");
cell_family.addProperty<neo::ItemUniqueId>("cell_uids");
cell_family.addProperty<neo::utils::Int32>("node_con");
}
 
void base_mesh_creation_test() {


// creating mesh
auto mesh = neo::Mesh{"my_neo_mesh"};

prepare_mesh(mesh);
// return;

// given data to create mesh
std::vector<neo::utils::Int64> node_uids{0,1,2};
std::vector<neo::utils::Real3> node_coords{{0,0,0}, {0,1,0}, {0,0,1}};
std::vector<neo::utils::Int64> cell_uids{0};

// add algos: 
mesh.beginUpdate();

// create nodes 
mesh.addAlgorithm(neo::OutProperty{neo::ItemKind::IK_Node,"node_lids"},
  [&node_uids](neo::OutProperty& node_lids_property){
  //node_lids_property.append(node_uids);//todo
  });

// register node uids
mesh.addAlgorithm(neo::InProperty{neo::ItemKind::IK_Node,"node_lids"},neo::OutProperty{neo::ItemKind::IK_Node,"node_uids"},
  [&node_uids](neo::InProperty const& node_lids_property, neo::OutProperty& node_uids_property){
   //auto& added_lids = node_lids_property.lastAppended();//todo
   //node_uids_property.appendAt(added_lids,node_uids);//steal node uids memory//todo
      });// need to add a property check for existing uid

// register node coords
mesh.addAlgorithm(neo::InProperty{neo::ItemKind::IK_Node,"node_lids"},neo::OutProperty{neo::ItemKind::IK_Node,"node_coords"},
  [&node_coords](neo::InProperty const& node_lids_property, neo::OutProperty & node_coords_property){
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

prepare_mesh(mesh);

mesh.beginUpdate();

mesh.addAlgorithm(neo::InProperty{neo::ItemKind::IK_Node,"node_lids"},neo::OutProperty{neo::ItemKind::IK_Node,"node_coords"},
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