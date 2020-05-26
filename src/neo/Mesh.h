//
// Created by dechaiss on 5/6/20.
//

#ifndef NEO_MESH_H
#define NEO_MESH_H

/*-------------------------
 * sdc - (C) 2020
 * NEtwork Oriented kernel
 * POC Mesh API
 *--------------------------
 */

#include <memory>
#include <string>
#include <vector>

#include "Neo.h"
#include "neo/Utils.h"

namespace Neo {

class MeshBase;
enum class ItemKind;
class Family;
struct ScheduledItemRange;
class ItemRangeUnlocker;

class Mesh {

public:
  using UidPropertyType = Neo::PropertyT<Neo::utils::Int64>;

public:
  Mesh(std::string const& mesh_name);
  ~Mesh();

private:
  std::unique_ptr<MeshBase> m_mesh_graph;

public:
  std::string const& name() const noexcept ;

  std::string uniqueIdPropertyName(const std::string& family_name) const noexcept;

  Neo::Family&  addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept ;

  void scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> const& uids, Neo::ScheduledItemRange & future_added_item_range) noexcept ;
  void scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> && uids, Neo::ScheduledItemRange & future_added_item_range) noexcept ;

  void scheduleSetNodeCoords(Neo::Family& node_family, Neo::ScheduledItemRange const& future_added_item_range,std::vector<Neo::utils::Real3> && node_coords){}
  void scheduleSetNodeCoords(Neo::Family& node_family, Neo::ScheduledItemRange const& future_added_item_range,std::vector<Neo::utils::Real3> const& node_coords){}

  Neo::ItemRangeUnlocker applyScheduledOperations() noexcept;
};

} // end namespace Neo

#endif // NEO_MESH_H
