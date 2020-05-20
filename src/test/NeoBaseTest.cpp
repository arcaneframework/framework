
 #include "neo/Neo.h"

/*-------------------------
 * Neo library first test
 * sdc (C)-2019
 *
 *-------------------------
 */

TEST(NeoUtils,test_array_view){
  std::vector<int> vec{0,1,2};
  // build a view from a vector
  Neo::utils::ArrayView<int> view{vec.size(), vec.data()};
  Neo::utils::ConstArrayView<int> constview{vec.size(), vec.data()};
  EXPECT_TRUE(std::equal(view.begin(),view.end(),vec.begin()));
  EXPECT_TRUE(std::equal(constview.begin(),constview.end(),vec.begin()));
  // build a vector from a view
  std::vector<int> vec2 = view.copy();
  std::vector<int> vec3(view.copy());
  std::vector<int> vec4(constview.copy());
  EXPECT_TRUE(std::equal(vec2.begin(),vec2.end(),view.begin()));
  EXPECT_TRUE(std::equal(vec3.begin(),vec3.end(),view.begin()));
  EXPECT_TRUE(std::equal(vec4.begin(),vec4.end(),constview.begin()));
  std::vector<int> dim2_vec{0,1,2,3,4,5};
  // build a dim2 view from vector
  auto dim1_size = 2;
  auto dim2_size = 3;
  Neo::utils::Array2View<int> dim2_view{dim1_size,dim2_size,dim2_vec.data()};
  Neo::utils::ConstArray2View<int> dim2_const_view{dim2_size,dim1_size,dim2_vec.data()};
  for (auto i = 0; i < dim1_size; ++i) {
    for (auto j = 0; j < dim2_size; ++j) {
      EXPECT_EQ(dim2_view[i][j], dim2_vec[i*dim2_size+j]);
      EXPECT_EQ(dim2_const_view[j][i], dim2_vec[j*dim1_size+i]);
    }
  }
  // Copy all Array2View data into a 1D vector
  std::vector<int> dim2_view_vector_copy{dim2_view.copy()};
  std::vector<int> dim2_const_view_vector_copy{dim2_const_view.copy()};
  EXPECT_TRUE(std::equal(dim2_vec.begin(),dim2_vec.end(),dim2_const_view_vector_copy.begin()));
}

TEST(NeoTestItemRange,test_item_range){
  // Test with only contiguous indexes
  std::cout << "== Testing contiguous item range from 0 with 5 items =="<< std::endl;
  auto ir = Neo::ItemRange{Neo::ItemIndexes{{},0,5}};
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
  }
  // Test with only non contiguous indexes
  std::cout << "== Testing non contiguous item range {3,5,7} =="<< std::endl;
  ir = Neo::ItemRange{Neo::ItemIndexes{{3,5,7},0,0}};
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
  }
  // Test range mixing contiguous and non contiguous indexes
  std::cout << "== Testing non contiguous item range {3,5,7} + 8 to 11 =="<< std::endl;
  ir = Neo::ItemRange{Neo::ItemIndexes{{3,5,7},8,4}};
  for (auto item : ir) {
    std::cout << "item lid " << item << std::endl;
  }
  // Internal test for out of bound
  std::cout << "Get out of bound values (index > size) " << ir.m_indexes(100) << std::endl;
  std::cout << "Get out of bound values (index < 0) " << ir.m_indexes(-100) << std::endl;


  // todo test out reverse range
}

TEST(NeoTestProperty,test_property)
 {
   Neo::PropertyT<Neo::utils::Int32> property{"name"};
   std::vector<Neo::utils::Int32> values {1,2,3};
   Neo::ItemRange item_range{Neo::ItemIndexes{{},0,3}};
   EXPECT_TRUE(property.isInitializableFrom(item_range));
   property.init(item_range,values);
   EXPECT_EQ(values.size(),property.size());
   std::vector<Neo::utils::Int32> new_values {4,5,6};
   Neo::ItemRange new_item_range{Neo::ItemIndexes{{},3,3}};
   property.append(new_item_range, new_values);
   property.debugPrint();
   EXPECT_EQ(values.size()+new_values.size(),property.size());
   auto property_values = property.values();
   for (auto i =0; i < values.size(); ++i){
     EXPECT_EQ(property_values[i],values[i]);
   }
}

TEST(NeoTestArrayProperty,test_array_property)
{
  auto array_property =
      Neo::ArrayProperty<Neo::utils::Int32>{"test_array_property"};
  // add elements: 5 items with one value
  Neo::ItemRange item_range{Neo::ItemIndexes{{},0,5}};
  std::vector<Neo::utils::Int32> values{0,1,2,3,4};
  array_property.resize({1,1,1,1,1});
  array_property.init(item_range,values);
  array_property.debugPrint();
  EXPECT_EQ(values.size(),array_property.size());
  // Add 3 items
  std::vector<std::size_t> nb_element_per_item{0,3,1};
  item_range = {Neo::ItemIndexes{{5,6,7}}};
  std::vector<Neo::utils::Int32> values_added{6,6,6,7};
  array_property.append(item_range, values_added, nb_element_per_item);
  array_property.debugPrint(); // expected result: "0" "1" "2" "3" "4" "6" "6" "6" "7" (check with test framework)
  EXPECT_EQ(values.size()+values_added.size(),array_property.size());
  // Add three more items
  item_range = {Neo::ItemIndexes{{},8,3}};
  std::for_each(values_added.begin(), values_added.end(), [](auto &elt) {return elt += 2;});
  array_property.append(item_range, values_added, nb_element_per_item);
  array_property.debugPrint(); // expected result: "0" "1" "2" "3" "4" "6" "6" "6" "7" "8" "8" "8" "9"
  EXPECT_EQ(values.size()+2*values_added.size(),array_property.size());
  // Add items and modify existing item
  item_range = {Neo::ItemIndexes{{0,8,5},11,1}};
  nb_element_per_item = {3,3,2,1};
  values_added = {10,10,10,11,11,11,12,12,13};
  array_property.append(item_range, values_added, nb_element_per_item); // expected result: "10" "10" "10" "1" "2" "3" "4" "12" "12" "6" "6" "6" "7" "11" "11" "11" "8" "8" "8" "9" "13"
  array_property.debugPrint();
  EXPECT_EQ(21,array_property.size());
}

TEST(NeoTestPropertyView, test_property_view)
{
  Neo::PropertyT<Neo::utils::Int32> property{"name"};
  std::vector<Neo::utils::Int32> values {1,2,3,10,100,1000};
  Neo::ItemRange item_range{Neo::ItemIndexes{{},0,6}};
  property.init(item_range,values);
  auto prop_view = property.view();
  auto partial_prop_view = property.view({Neo::ItemIndexes{{1,3,5},0,0}});
}

TEST(NeoTestPropertyGraph,test_property_graph)
{
  std::cout << "Test Property Graph" << std::endl;
  // to be done when the graph impl will be available
}
 
TEST(NeoTestLidsProperty,test_lids_property)
{
  std::cout << "Test lids Property" << std::endl;
  auto lid_prop = Neo::ItemLidsProperty{"test_property"};
  std::vector<Neo::utils::Int64 > uids {1,2,3,4,5};
  lid_prop.append(uids);
  lid_prop.debugPrint();
  for (auto item : lid_prop.values()) {
    std::cout << "Item range, lid " << item << std::endl;
  }
  EXPECT_EQ(lid_prop.size(),5);
  uids = {1,3,5};
  lid_prop.remove(uids);
  std::cout << "new range size " << lid_prop.values().size();
  for (auto item : lid_prop.values()) {
    std::cout << "Item range, lid " << item << std::endl;
  }
  EXPECT_EQ(lid_prop.size(),2);
}

TEST(NeoTestFamily,test_family)
{
  Neo::Family family(Neo::ItemKind::IK_Dof,"MyFamily");
  EXPECT_EQ(family.lidPropName(),family._lidProp().m_name);
  family._lidProp().append({0,1,2}); // internal
  EXPECT_EQ(3,family.nbElements());
  std::string scalar_prop_name("MyScalarProperty");
  std::string array_prop_name("MyArrayProperty");
  family.addProperty<Neo::utils::Int32>(scalar_prop_name);
  family.addArrayProperty<Neo::utils::Int32>(array_prop_name);
  EXPECT_NO_THROW(family.getProperty(scalar_prop_name));
  EXPECT_NO_THROW(family.getProperty(array_prop_name));
  EXPECT_THROW(family.getProperty("UnexistingProperty"),std::invalid_argument);
  EXPECT_EQ(scalar_prop_name,family.getConcreteProperty<Neo::PropertyT<Neo::utils::Int32>>(scalar_prop_name).m_name);
  EXPECT_EQ(array_prop_name,family.getConcreteProperty<Neo::ArrayProperty<Neo::utils::Int32>>(array_prop_name).m_name);
  EXPECT_EQ(3,family.all().size());
  auto i = 0;
  for (auto item : family.all() ) {
    EXPECT_EQ(i++,item);
  }
}

void mesh_property_test(const Neo::MeshBase & mesh){
  std::cout << "== Print Mesh " << mesh.m_name << " Properties =="<< std::endl;
  for (const auto& [kind_name_pair,family] : mesh.m_families) {
    std::cout << "= In family " << kind_name_pair.second << " =" << std::endl;
    for (const auto& [prop_name,property] : family->m_properties){
      std::cout << prop_name << std::endl;
    }
  }
  std::cout << "== End Print Mesh " << mesh.m_name << " Properties =="<< std::endl;
}

void prepare_mesh(Neo::MeshBase & mesh){

// Adding node family and properties
auto& node_family = mesh.getFamily(Neo::ItemKind::IK_Node,"NodeFamily");
std::cout << "Find family " << node_family.m_name << std::endl;
node_family.addProperty<Neo::utils::Real3>(std::string("node_coords"));
node_family.addProperty<Neo::utils::Int64>("node_uids");
node_family.addArrayProperty<Neo::utils::Int32>("node2cells");
node_family.addProperty<Neo::utils::Int32>("internal_end_of_remove_tag"); // not a user-defined property // todo use byte ?

// Test adds
auto& property = node_family.getProperty("node_uids");

// Adding cell family and properties
auto& cell_family = mesh.getFamily(Neo::ItemKind::IK_Cell,"CellFamily");
std::cout << "Find family " << cell_family.m_name << std::endl;
cell_family.addProperty<Neo::utils::Int64>("cell_uids");
cell_family.addArrayProperty<Neo::utils::Int32>("cell2nodes");
}
 
TEST(NeoTestBaseMeshCreation,base_mesh_creation_test) {

  std::cout << "*------------------------------------*"<< std::endl;
  std::cout << "* Test framework Neo thoughts " << std::endl;
  std::cout << "*------------------------------------*" << std::endl;


// creating mesh
auto mesh = Neo::MeshBase{"my_neo_mesh"};
auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node,"NodeFamily");
auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,"CellFamily");

prepare_mesh(mesh);
// return;

// given data to create mesh. After mesh creation data is no longer available
std::vector<Neo::utils::Int64> node_uids{0,1,2};
std::vector<Neo::utils::Real3> node_coords{{0,0,0}, {0,1,0}, {0,0,1}};
std::vector<Neo::utils::Int64> cell_uids{0,2,7,9};

// add algos: 

// create nodes
auto added_nodes = Neo::ItemRange{};
mesh.addAlgorithm(
    Neo::OutProperty{node_family,node_family.lidPropName()},
  [&node_uids,&added_nodes](Neo::ItemLidsProperty & node_lids_property){
  std::cout << "Algorithm: create nodes" << std::endl;
  added_nodes = node_lids_property.append(node_uids);
  node_lids_property.debugPrint();
  std::cout << "Inserted item range : " << added_nodes;
  });

// register node uids
mesh.addAlgorithm(
    Neo::InProperty{node_family,node_family.lidPropName()},
    Neo::OutProperty{node_family,"node_uids"},
  [&node_uids,&added_nodes](Neo::ItemLidsProperty const& node_lids_property,
                   Neo::PropertyT<Neo::utils::Int64>& node_uids_property){
    std::cout << "Algorithm: register node uids" << std::endl;
  if (node_uids_property.isInitializableFrom(added_nodes))  node_uids_property.init(added_nodes,std::move(node_uids)); // init can steal the input values
  else node_uids_property.append(added_nodes, node_uids);
  node_uids_property.debugPrint();
    });// need to add a property check for existing uid

// register node coords
mesh.addAlgorithm(
    Neo::InProperty{node_family,node_family.lidPropName()},
    Neo::OutProperty{node_family,"node_coords"},
  [&node_coords,&added_nodes](Neo::ItemLidsProperty const& node_lids_property,
                   Neo::PropertyT<Neo::utils::Real3> & node_coords_property){
    std::cout << "Algorithm: register node coords" << std::endl;
    if (node_coords_property.isInitializableFrom(added_nodes))  node_coords_property.init(added_nodes,std::move(node_coords)); // init can steal the input values
    else node_coords_property.append(added_nodes, node_coords);
    node_coords_property.debugPrint();
  });
//
// Add cells and connectivity

// create cells
auto added_cells = Neo::ItemRange{};
mesh.addAlgorithm(
    Neo::OutProperty{cell_family,cell_family.lidPropName()},
  [&cell_uids,&added_cells](Neo::ItemLidsProperty& cell_lids_property) {
    std::cout << "Algorithm: create cells" << std::endl;
    added_cells = cell_lids_property.append(cell_uids);
    cell_lids_property.debugPrint();
    std::cout << "Inserted item range : " << added_cells;
  });

// register cell uids
mesh.addAlgorithm(
    Neo::InProperty{cell_family,cell_family.lidPropName()},
    Neo::OutProperty{cell_family,"cell_uids"},
    [&cell_uids,&added_cells](Neo::ItemLidsProperty const& cell_lids_property,
                   Neo::PropertyT<Neo::utils::Int64>& cell_uids_property){
      std::cout << "Algorithm: register cell uids" << std::endl;
      if (cell_uids_property.isInitializableFrom(added_cells))  cell_uids_property.init(added_cells,std::move(cell_uids)); // init can steal the input values
      else cell_uids_property.append(added_cells, cell_uids);
      cell_uids_property.debugPrint();
    });

// register connectivity
// node to cell
std::vector<Neo::utils::Int64> connected_cell_uids{0,0,2,2,7,9};
std::vector<std::size_t> nb_cell_per_node{1,2,3};
mesh.addAlgorithm(
    Neo::InProperty{node_family,node_family.lidPropName()},
    Neo::InProperty{cell_family,cell_family.lidPropName()},
    Neo::OutProperty{node_family,"node2cells"},
    [&connected_cell_uids, &nb_cell_per_node,& added_nodes]
    (Neo::ItemLidsProperty const& node_lids_property,
                   Neo::ItemLidsProperty const& cell_lids_property,
                   Neo::ArrayProperty<Neo::utils::Int32> & node2cells){
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
std::vector<Neo::utils::Int64> connected_node_uids{0,1,2,1,2,0,2,1,0};// on ne connecte volontairement pas toutes les mailles pour vérifier initialisation ok sur la famille
std::vector<std::size_t> nb_node_per_cell{3,0,3,3};
mesh.addAlgorithm(Neo::InProperty{node_family,node_family.lidPropName()},
                  Neo::InProperty{cell_family,cell_family.lidPropName()},
                  Neo::OutProperty{cell_family,"cell2nodes"},
                  [&connected_node_uids, &nb_node_per_cell,& added_cells]
                          (
                      Neo::ItemLidsProperty const& node_lids_property,
                      Neo::ItemLidsProperty const& cell_lids_property,
                      Neo::ArrayProperty<Neo::utils::Int32> & cells2nodes){
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
std::vector<Neo::utils::Int64> new_cell_uids{10,11,12}; // elles seront toutes rouges
auto new_cell_added = Neo::ItemRange{};
mesh.addAlgorithm(Neo::OutProperty{cell_family,cell_family.lidPropName()},
                [&new_cell_uids,&new_cell_added](Neo::ItemLidsProperty& cell_lids_property) {
                  std::cout << "Algorithm: add new cells" << std::endl;
                  new_cell_added = cell_lids_property.append(new_cell_uids);
                  cell_lids_property.debugPrint();
                  std::cout << "Inserted item range : " << new_cell_added;
                });

// register new cell uids
mesh.addAlgorithm(
    Neo::InProperty{cell_family,cell_family.lidPropName()},
    Neo::OutProperty{cell_family,"cell_uids"},
                  [&new_cell_uids,&new_cell_added](Neo::ItemLidsProperty const& cell_lids_property,
                      Neo::PropertyT<Neo::utils::Int64>& cell_uids_property){
                    std::cout << "Algorithm: register new cell uids" << std::endl;
                    // must append and not initialize
                    if (cell_uids_property.isInitializableFrom(new_cell_added))  cell_uids_property.init(new_cell_added,std::move(new_cell_uids)); // init can steal the input values
                    else cell_uids_property.append(new_cell_added, new_cell_uids);
                    cell_uids_property.debugPrint();
                  });

// add connectivity to new cells
std::vector<Neo::utils::Int64> new_cell_connected_node_uids{0,1,2,1,2};// on ne connecte volontairement pas toutes les mailles pour vérifier initialisation ok sur la famille
std::vector<std::size_t> nb_node_per_new_cell{0,3,2};
mesh.addAlgorithm(Neo::InProperty{node_family,node_family.lidPropName()},
                  Neo::InProperty{cell_family,cell_family.lidPropName()},
                  Neo::OutProperty{cell_family,"cell2nodes"},
                  [&new_cell_connected_node_uids, &nb_node_per_new_cell,& new_cell_added]
                          (
                      Neo::ItemLidsProperty const& node_lids_property,
                      Neo::ItemLidsProperty const& cell_lids_property,
                      Neo::ArrayProperty<Neo::utils::Int32> & cells2nodes){
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
std::vector<Neo::utils::Int64> removed_node_uids{1,2};
auto removed_nodes = Neo::ItemRange{};
mesh.addAlgorithm(
    Neo::OutProperty{node_family,node_family.lidPropName()},
    Neo::OutProperty{node_family,"internal_end_of_remove_tag"},
                  [&removed_node_uids,&removed_nodes, &node_family](
        Neo::ItemLidsProperty& node_lids_property,
        Neo::PropertyT<Neo::utils::Int32 > & internal_end_of_remove_tag){
                    std::cout << "Algorithm: remove nodes" << std::endl;
                    removed_nodes = node_lids_property.remove(removed_node_uids);
                    node_lids_property.debugPrint();
                    std::cout << "removed item range : " << removed_nodes;
                    // Store removed items in internal_end_of_remove_tag
                    internal_end_of_remove_tag.init(node_family.all(),0);
                    for (auto removed_item : removed_nodes) {
                      internal_end_of_remove_tag[removed_item] = 1;
                    }
                  });

// handle node removal in connectivity with node family = target family
mesh.addAlgorithm(
    Neo::InProperty{node_family,"internal_end_of_remove_tag"},
    Neo::OutProperty{cell_family,"cell2nodes"},
                  [&cell_family](
        Neo::PropertyT<Neo::utils::Int32> const& internal_end_of_remove_tag,
        Neo::ArrayProperty<Neo::utils::Int32> & cells2nodes){
//                    std::transform()
//                    Neo::ItemRange node_range {Neo::ItemIndexes{{},0,node_family.size()}};
                    for (auto cell : cell_family.all()) {
                      auto connected_nodes = cells2nodes[cell];
                      for (auto& connected_node : connected_nodes){
                        if (connected_node != Neo::utils::NULL_ITEM_LID && internal_end_of_remove_tag[connected_node] == 1) {
                          std::cout << "modify node : "<< connected_node << std::endl;
                          connected_node = Neo::utils::NULL_ITEM_LID;

                        }
                      }
                    }
                  });

// launch algos
mesh.applyAlgorithms();

// test properties
mesh_property_test(mesh);
}


TEST(NeoTestPartialMeshModification,partial_mesh_modif_test) {
  
// modify node coords
// input data
std::array<int,3> node_uids {0,1,3};
Neo::utils::Real3 r = {0,0,0};
std::array<Neo::utils::Real3,3> node_coords = {r,r,r};// don't get why I can't write {{0,0,0},{0,0,0},{0,0,0}}; ...??

// creating mesh
auto mesh = Neo::MeshBase{"my_neo_mesh"};
auto& node_family = mesh.addFamily(Neo::ItemKind::IK_Node,"NodeFamily");
auto& cell_family = mesh.addFamily(Neo::ItemKind::IK_Cell,"CellFamily");


prepare_mesh(mesh);


mesh.addAlgorithm(Neo::InProperty{node_family,node_family.lidPropName()},
                  Neo::OutProperty{node_family,"node_coords"},
  [&node_coords,&node_uids](
                      Neo::ItemLidsProperty const& node_lids_property,
                      Neo::PropertyT<Neo::utils::Real3> & node_coords_property){
    std::cout << "Algorithm: register node coords" << std::endl;
    //auto& lids = node_lids_property[node_uids];//todo
    //node_coords_property.appendAt(lids, node_coords);// steal node_coords memory//todo
  });

mesh.applyAlgorithms();
  
}