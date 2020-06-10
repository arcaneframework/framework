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

std::string Neo::Mesh::uniqueIdPropertyName(const std::string& family_name) const noexcept
{
  return family_name+"_uids";
}

Neo::Family& Neo::Mesh::addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept
{
  auto& cell_family = m_mesh_graph->addFamily(item_kind, std::move(family_name));
  cell_family.addProperty<Neo::utils::Int64>(uniqueIdPropertyName(family_name));
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
  m_mesh_graph->addAlgorithm(
      Neo::InProperty{family,family.lidPropName()},
      Neo::OutProperty{family, uniqueIdPropertyName(family.name())},
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
  m_mesh_graph->addAlgorithm(
      Neo::InProperty{family,family.lidPropName()},
      Neo::OutProperty{family, uniqueIdPropertyName(family.name())},
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

void Neo::Mesh::scheduleSetItemCoords(Neo::Family& item_family, Neo::ScheduledItemRange const& future_added_item_range,std::vector<Neo::utils::Real3> const& item_coords) noexcept
{
  auto coord_prop_name = _itemCoordPropertyName(item_family);
  item_family.addProperty<Neo::utils::Real3>(coord_prop_name);
  auto& added_items = future_added_item_range.new_items;
  m_mesh_graph->addAlgorithm(
          Neo::InProperty{item_family, item_family.lidPropName()},
          Neo::OutProperty{item_family,coord_prop_name},
          [&item_coords,&added_items](Neo::ItemLidsProperty const& item_lids_property,
                                      Neo::PropertyT<Neo::utils::Real3>& item_coords_property){
            std::cout << "Algorithm: register item coords" << std::endl;
            item_coords_property.append(added_items, item_coords);
            item_coords_property.debugPrint();
          });
}

void Neo::Mesh::scheduleSetItemCoords(Neo::Family& item_family, Neo::ScheduledItemRange const&future_added_item_range,std::vector<Neo::utils::Real3>&& item_coords) noexcept
{
  auto coord_prop_name = _itemCoordPropertyName(item_family);
  item_family.addProperty<Neo::utils::Real3>(coord_prop_name);
  auto& added_items = future_added_item_range.new_items;
  m_mesh_graph->addAlgorithm(
          Neo::InProperty{item_family, item_family.lidPropName()},
          Neo::OutProperty{item_family,coord_prop_name},
          [item_coords,&added_items](Neo::ItemLidsProperty const& item_lids_property,
                                     Neo::PropertyT<Neo::utils::Real3>& item_coords_property){
            std::cout << "Algorithm: register item coords" << std::endl;
            if (item_coords_property.isInitializableFrom(added_items)) {
              item_coords_property.init(
                  added_items,
                  std::move(item_coords)); // init can steal the input values
            }
            else {
              item_coords_property.append(added_items, item_coords);
            }
            item_coords_property.debugPrint();
          });
}

Neo::ItemRangeUnlocker Neo::Mesh::applyScheduledOperations() noexcept
{
  return m_mesh_graph->applyAlgorithms();
}

Neo::Mesh::CoordPropertyType& Neo::Mesh::getItemCoordProperty(Neo::Family & family)
{
  return family.getConcreteProperty<CoordPropertyType>(_itemCoordPropertyName(family));
}

Neo::Mesh::CoordPropertyType const& Neo::Mesh::getItemCoordProperty(Neo::Family const& family) const
{
  return family.getConcreteProperty<CoordPropertyType>(_itemCoordPropertyName(family));
}