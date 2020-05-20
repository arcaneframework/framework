//
// Created by dechaiss on 5/6/20.
//

#include "neo/Mesh.h"
#include "neo/Neo.h"


Neo::Mesh::Mesh(const std::string& mesh_name)
 : m_mesh_graph(std::make_unique<Neo::MeshBase>(Neo::MeshBase{mesh_name})){
}

Neo::Mesh::~Mesh() = default;

std::string const& Neo::Mesh::name() const noexcept {
  return m_mesh_graph->m_name;
}

Neo::Family& Neo::Mesh::addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept
{
  auto& cell_family = m_mesh_graph->addFamily(item_kind, std::move(family_name));
  cell_family.addProperty<Neo::utils::Int64>(family_name + "_uids");
  return cell_family;
}

void Neo::Mesh::scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> const& uids, Neo::ScheduledItemRange & added_item_range) noexcept
{
  auto& added_items = added_item_range.new_items;
  // Add items
  m_mesh_graph->addAlgorithm(  Neo::OutProperty{family,family.lidPropName()},
                      [&family,&uids,&added_items](Neo::ItemLidsProperty & lids_property){
                        std::cout << "Algorithm: create items in family " << family.name() << std::endl;
                        added_items = lids_property.append(uids);
                        lids_property.debugPrint();
                        std::cout << "Inserted item range : " << added_items;
                      });
  // register their uids
  auto uid_property_name = family.name()+"_uids";
  m_mesh_graph->addAlgorithm(
      Neo::InProperty{family,family.lidPropName()},
      Neo::OutProperty{family, uid_property_name},
      [&family,&uids,&added_items](Neo::ItemLidsProperty const& item_lids_property,
                                   Neo::PropertyT<Neo::utils::Int64>& item_uids_property){
        std::cout << "Algorithm: register item uids for family " << family.name() << std::endl;
        item_uids_property.append(added_items, uids);
        item_uids_property.debugPrint();
      });// need to add a property check for existing uid
}

void Neo::Mesh::scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> && uids, Neo::ScheduledItemRange & added_item_range) noexcept
{
  auto& added_items = added_item_range.new_items;
  // Add items
  m_mesh_graph->addAlgorithm(  Neo::OutProperty{family,family.lidPropName()},
                               [&family,uids,&added_items](Neo::ItemLidsProperty & lids_property){
                                 std::cout << "Algorithm: create items in family " << family.name() << std::endl;
                                 added_items = lids_property.append(uids);
                                 lids_property.debugPrint();
                                 std::cout << "Inserted item range : " << added_items;
                               });
  // register their uids
  auto uid_property_name = family.name()+"_uids";
  m_mesh_graph->addAlgorithm(
      Neo::InProperty{family,family.lidPropName()},
      Neo::OutProperty{family, uid_property_name},
      [&family,uids,&added_items](Neo::ItemLidsProperty const& item_lids_property,
                                   Neo::PropertyT<Neo::utils::Int64>& item_uids_property){
        std::cout << "Algorithm: register item uids for family " << family.name() << std::endl;
        if (item_uids_property.isInitializableFrom(added_items)){
          item_uids_property.init(added_items,std::move(uids)); // init can steal the input values
        }
        else {
          item_uids_property.append(added_items, uids);
        }
        item_uids_property.debugPrint();
      });// need to add a property check for existing uid
}

Neo::ItemRangeUnlocker Neo::Mesh::applyScheduledOperations() noexcept
{
  return m_mesh_graph->applyAlgorithms();
}
