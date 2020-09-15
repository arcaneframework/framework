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
struct FutureItemRange;
class EndOfMeshUpdate;

/*---------------------------------------------------------------------------*/
/*!
 * \brief Asynchronous Mesh API, schedule operations and apply them with
 * \fn applyScheduledOperations
 *
 * When an operation requires an array of external data, since the call to
 * mesh operations is asynchronous the array is copied.
 * To avoid memory copy pass as much as possible your data arrays by rvalue
 * (temporary array or std::move your array);
 *
 */

class Mesh {

public:
  using UidPropertyType   = Neo::PropertyT<Neo::utils::Int64>;
  using CoordPropertyType = Neo::PropertyT<Neo::utils::Real3>;

public:
  Mesh(std::string const& mesh_name);
  ~Mesh();

private:
  std::unique_ptr<MeshBase> m_mesh_graph;

public:
  [[nodiscard]] std::string const& name() const noexcept ;

  [[nodiscard]] std::string uniqueIdPropertyName(const std::string& family_name) const noexcept ;

  Neo::Family&  addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept ;

  void scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> uids, Neo::FutureItemRange & future_added_item_range) noexcept ;

    /*!
   * @brief Ask for a fixed-size connectivity add between \a source_family and \a target_family. Source items are scheduled but not created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Connectivity fix size value.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   *
   * Connectivity with fix size (nb of connected items per item is constant).
   * Use this method to add connectivity with source items scheduled but not yet created
   * i.e addItems and addConnectivity are applied in the same call to applyScheduledOperations.
   */
  void scheduleAddConnectivity(Neo::Family& source_family, Neo::FutureItemRange const& source_items,
                               Neo::Family& target_family, int nb_connected_item_per_item,
                               std::vector<Neo::utils::Int64> connected_item_uids,
                               std::string const& connectivity_unique_name) noexcept ;

  /*!
   * @brief Ask for a fixed-size connectivity add between \a source_family and \a target_family. Source items are already created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Given via an ItemRange.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Connectivity fix size value.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   *
   * Connectivity with fix size (nb of connected items per item is constant).
   * Use this method to add connectivity with source items already created
   * in a previous call to applyScheduledOperations.
   */
  void scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                               Neo::Family& target_family, int nb_connected_item_per_item,
                               std::vector<Neo::utils::Int64> connected_item_uids,
                               std::string const& connectivity_unique_name) noexcept ;

  /*!
   * @brief Ask for a variable size connectivity add between \a source_family and \a target_family. Source items are scheduled but not created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Number of connected item per items. Array with size equal to source items number.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   *
   * Connectivity with variable size (nb of connected items per item is variable)
   * Use this method to add connectivity with source items scheduled but not yet created
   * i.e addItems and addConnectivity are applied in the same call to applyScheduledOperations.
   */
  void scheduleAddConnectivity(Neo::Family& source_family, Neo::FutureItemRange const& source_items,
                               Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                               std::vector<Neo::utils::Int64> connected_item_uids,
                               std::string const& connectivity_unique_name) noexcept ;

  /*!
    * @brief Ask for a variable size connectivity add between \a source_family and \a target_family. Source items are already created.
    * @param source_family The family of source items.
    * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
      * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
    * @param target_family The family of target items.
    * @param nb_connected_item_per_item Number of connected item per items. Array with size equal to source items number.
    * @param connected_item_uids Unique ids of the connected items.
    * @param connectivity_unique_name Connectivity name must be unique
    *
    * Connectivity with variable size (nb of connected items per item is variable)
    * Use this method to add connectivity with source items already created
    * in a previous call to applyScheduledOperations.
    */
  void scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                               Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                               std::vector<Neo::utils::Int64> connected_item_uids,
                               std::string const& connectivity_unique_name) noexcept ;

  //! Use this method to set coordinates of new items
  void scheduleSetItemCoords(Neo::Family& item_family, Neo::FutureItemRange const& future_added_item_range,std::vector<Neo::utils::Real3> item_coords) noexcept ;

  Neo::EndOfMeshUpdate applyScheduledOperations() noexcept ;

  //! Use this method to change coordinates of existing items
  [[nodiscard]] CoordPropertyType& getItemCoordProperty(Neo::Family & family);
  [[nodiscard]] CoordPropertyType const& getItemCoordProperty(Neo::Family const& family) const;

private:
  [[nodiscard]] std::string _itemCoordPropertyName(Family const& item_family) const {return item_family.name()+"_item_coordinates";}
};

} // end namespace Neo

#endif // NEO_MESH_H
