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
  using ConnectivityPropertyType = Neo::ArrayProperty<Neo::utils::Int32>;

  struct Connectivity{
    Neo::Family const& source_family;
    Neo::Family const& target_family;
    std::string const& name;
    ConnectivityPropertyType const& connectivity_value;

    Neo::utils::ConstArrayView<Neo::utils::Int32> operator[] (Neo::utils::Int32 item_lid) const noexcept {
      return connectivity_value[item_lid];
    }
  };

  enum class ConnectivityOperation { Add, Modify};

public:
  Mesh(std::string const& mesh_name);
  ~Mesh();

private:
  std::unique_ptr<MeshBase> m_mesh_graph;
  using ConnectivityMapType = std::map<std::string,Connectivity>;
  ConnectivityMapType m_connectivities;

public:
  /*!
   * @brief name of the mesh
   * @return  the name of the mesh
   */
  [[nodiscard]] std::string const& name() const noexcept ;

  [[nodiscard]] std::string uniqueIdPropertyName(const std::string& family_name) const noexcept ;

  /*!
   * @brief Add a family of kind \p item_kind and of name \p family_name
   * @param item_kind
   * @param family_name
   * @return
   */
  Neo::Family&  addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept ;

  /*!
   * @brief Schedule items creation for family \p family with unique ids \p uids
   * @param family Family where items are added
   * @param uids Uids of the added items
   * @param future_added_item_range Future ItemRange : after the call to applyScheduledOperations
   * this future range will give access to a range containing new items local ids
   */
  void scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> uids, Neo::FutureItemRange & future_added_item_range) noexcept ;

    /*!
   * @brief Ask for a fixed-size connectivity add between \p source_family and \p target_family. Source items are scheduled but not created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Connectivity fix size value.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   * @param add_or_modify Indicates whether Connectivity is added or modified (add is default)
   *
   * Connectivity with fix size (nb of connected items per item is constant).
   * Use this method to add connectivity with source items scheduled but not yet created
   * i.e addItems and addConnectivity are applied in the same call to applyScheduledOperations.
   */
  void scheduleAddConnectivity(Neo::Family& source_family, Neo::FutureItemRange& source_items,
                               Neo::Family& target_family, int nb_connected_item_per_item,
                               std::vector<Neo::utils::Int64> connected_item_uids,
                               std::string const& connectivity_unique_name,
                               ConnectivityOperation add_or_modify = ConnectivityOperation::Add);

  /*!
   * @brief Ask for a fixed-size connectivity add between \p source_family and \p target_family. Source items are already created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Given via an ItemRange.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Connectivity fix size value.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   * @param add_or_modify Indicates whether Connectivity is added or modified (add is default)
   *
   * Connectivity with fix size (nb of connected items per item is constant).
   * Use this method to add connectivity with source items already created
   * in a previous call to applyScheduledOperations.
   */
  void scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                               Neo::Family& target_family, int nb_connected_item_per_item,
                               std::vector<Neo::utils::Int64> connected_item_uids,
                               std::string const& connectivity_unique_name,
                               ConnectivityOperation add_or_modify = ConnectivityOperation::Add);

  /*!
   * @brief Ask for a variable size connectivity add between \p source_family and \p target_family. Source items are scheduled but not created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Number of connected item per items. Array with size equal to source items number.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   * @param add_or_modify Indicates whether Connectivity is added or modified (add is default)
   *
   * Connectivity with variable size (nb of connected items per item is variable)
   * Use this method to add connectivity with source items scheduled but not yet created
   * i.e addItems and addConnectivity are applied in the same call to applyScheduledOperations.
   */
  void scheduleAddConnectivity(Neo::Family& source_family, Neo::FutureItemRange& source_items,
                               Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                               std::vector<Neo::utils::Int64> connected_item_uids,
                               std::string const& connectivity_unique_name,
                               ConnectivityOperation add_or_modify = ConnectivityOperation::Add);

  /*!
    * @brief Ask for a variable size connectivity add between \p source_family and \p target_family. Source items are already created.
    * @param source_family The family of source items.
    * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
      * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
    * @param target_family The family of target items.
    * @param nb_connected_item_per_item Number of connected item per items. Array with size equal to source items number.
    * @param connected_item_uids Unique ids of the connected items.
    * @param connectivity_unique_name Connectivity name must be unique
    * @param add_or_modify Indicates whether Connectivity is added or modified (add is default)
    *
    * Connectivity with variable size (nb of connected items per item is variable)
    * Use this method to add connectivity with source items already created
    * in a previous call to applyScheduledOperations.
    */
  void scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                               Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                               std::vector<Neo::utils::Int64> connected_item_uids,
                               std::string const& connectivity_unique_name,
                               ConnectivityOperation add_or_modify = ConnectivityOperation::Add);

  /*!
   * @brief Schedule an set item coordinates. Will be applied when applyScheduledOperations will be called
   * @param item_family Family of the items whose coords will be modified
   * @param future_added_item_range Set of the items whose coords will be modified. These items are not created yet.
   * They will be created in applyScheduledOperations before the call to this coords set.
   * @param item_coords Value of the items coordinates
   */
  void scheduleSetItemCoords(Neo::Family& item_family, Neo::FutureItemRange& future_added_item_range,std::vector<Neo::utils::Real3> item_coords) noexcept ;

  /*!
   * @brief Get item connectivity between \p source_family and \p target_family with name \p name
   * @param source_family Family of the source items
   * @param target_family Family of the target items
   * @param connectivity_name Name of the connectivity
   * @return Connectivity, a connectivity wrapper object
   * @throw a std::invalid_argument if the connectivity is not found
   */
  Connectivity const getConnectivity(Neo::Family const& source_family,Neo::Family const& target_family,std::string const& connectivity_name);

  /*!
   * @brief Apply all scheduled operations (addItems, addConnectivities, setItemCoords)
   * @return An object allowing to get the new items ItemRange from the FutureItemRange
   */
  Neo::EndOfMeshUpdate applyScheduledOperations() noexcept ;

  //! Use this method to change coordinates of existing items
  [[nodiscard]] CoordPropertyType& getItemCoordProperty(Neo::Family & family);
  [[nodiscard]] CoordPropertyType const& getItemCoordProperty(Neo::Family const& family) const;

  /*!
   * @brief Get items of kind \p item_kind connected to family \p source_family
   * @param source_family Family of source items
   * @param item_kind Kind of connected items
   * @return A vector of all Connectivities connecting \p source_family to a target family with kind \p item_kind
   */
  std::vector<Connectivity> items(Neo::Family const& source_family, Neo::ItemKind item_kind) const noexcept;

  std::vector<Connectivity> edges(Neo::Family const& source_family) const noexcept;
  std::vector<Connectivity> nodes(Neo::Family const& source_family) const noexcept;
  std::vector<Connectivity> faces(Neo::Family const& source_family) const noexcept;
  std::vector<Connectivity> cells(Neo::Family const& source_family) const noexcept;

  std::vector<Connectivity> dofs(Neo::Family const& source_family) const noexcept;
private:

  [[nodiscard]] std::string _itemCoordPropertyName(Family const& item_family) const {return item_family.name()+"_item_coordinates";}

};

} // end namespace Neo

#endif // NEO_MESH_H
