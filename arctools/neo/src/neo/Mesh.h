// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Mesh.h                                          (C) 2000-2025             */
/*                                                                           */
/* Asynchronous Mesh structure based on Neo kernel                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef NEO_MESH_H
#define NEO_MESH_H

#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include "Neo.h"
#include "Utils.h"
#include "MeshKernel.h" //todo remove when replaced by new addMeshOperation API

namespace Neo
{

namespace MeshKernel
{
  class AlgorithmPropertyGraph;
}

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

class Mesh
{

 public:
  using LocalIdPropertyType = Neo::ItemLidsProperty;
  using UniqueIdPropertyType = Neo::MeshScalarPropertyT<Neo::utils::Int64>;
  using CoordPropertyType = Neo::MeshScalarPropertyT<Neo::utils::Real3>;
  using ConnectivityPropertyType = Neo::MeshArrayPropertyT<Neo::utils::Int32>;

  template <typename... DataTypes>
  using MeshOperationT = std::variant<std::function<void(LocalIdPropertyType const&, Neo::ScalarPropertyT<DataTypes>&)>...,
                                      std::function<void(LocalIdPropertyType const&, Neo::MeshScalarPropertyT<DataTypes>&)>...,
                                      std::function<void(LocalIdPropertyType const&, Neo::MeshArrayPropertyT<DataTypes>&)>...,
                                      std::function<void(ConnectivityPropertyType const&, Neo::ScalarPropertyT<DataTypes>&)>...,
                                      std::function<void(ConnectivityPropertyType const&, Neo::MeshScalarPropertyT<DataTypes>&)>...,
                                      std::function<void(ConnectivityPropertyType const&, Neo::MeshArrayPropertyT<DataTypes>&)>...,
                                      std::function<void(CoordPropertyType const&, Neo::ScalarPropertyT<DataTypes>&)>...,
                                      std::function<void(CoordPropertyType const&, Neo::MeshScalarPropertyT<DataTypes>&)>...,
                                      std::function<void(CoordPropertyType const&, Neo::MeshArrayPropertyT<DataTypes>&)>...>;

  using MeshOperation = MeshOperationT<utils::Int32, utils::Real3, utils::Int64>;

  struct Connectivity
  {
    Neo::Family const& source_family;
    Neo::Family const& target_family;
    std::string name;
    ConnectivityPropertyType const& connectivity_value;
    ConnectivityPropertyType const& connectivity_orientation;

    Neo::utils::ConstSpan<Neo::utils::Int32> operator[](Neo::utils::Int32 item_lid) const noexcept {
      return connectivity_value[item_lid];
    }

    int maxNbConnectedItems() const {
      auto nb_connected_elements = connectivity_value.sizes();
      if (nb_connected_elements.size() == 0) {return 0;}
      return *std::max_element(nb_connected_elements.begin(), nb_connected_elements.end());
    }

    bool isEmpty() const {
      return maxNbConnectedItems() == 0;
    }

    friend bool operator==(Connectivity const& lhs, Connectivity const& rhs) {
      return (lhs.source_family == rhs.source_family) && (lhs.target_family == rhs.target_family) && lhs.name == rhs.name;
    }
  };

  enum class ConnectivityOperation
  {
    Add,
    Modify
  };

  explicit Mesh(std::string const& mesh_name);
  ~Mesh();

 private:
  std::unique_ptr<MeshKernel::AlgorithmPropertyGraph> m_mesh_graph;
  FamilyMap m_families;
  using ConnectivityMapType = std::map<std::string, Connectivity, std::less<>>;
  ConnectivityMapType m_connectivities;
  using ConnectivityPerFamilyMapType = std::map<std::pair<ItemKind, std::string>, std::vector<Connectivity>>;
  ConnectivityPerFamilyMapType m_connectivities_per_family;
  int m_dimension = 3;

  template <typename ItemRangeT>
  void _scheduleAddConnectivity(Neo::Family& source_family, Neo::ItemRangeWrapper<ItemRangeT> source_items,
                                Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                std::vector<Neo::utils::Int64> connected_item_uids,
                                std::string const& connectivity_unique_name,
                                ConnectivityOperation add_or_modify);
  template <typename ItemRangeT>
  void _scheduleAddConnectivityOrientation(Neo::Family& source_family, Neo::ItemRangeWrapper<ItemRangeT> source_items,
                                           Neo::Family& target_family, std::vector<int> nb_connected_item_per_items,
                                           std::vector<int> source_item_orientation_in_target_item, bool do_check_orientation);

  void _addConnectivityOrientationCheck(Neo::Family& source_family, const Neo::Family& target_family);
  static std::string _connectivityOrientationPropertyName(std::string const& source_family_name, std::string const& target_family);

 public:
  /*!
   * @brief name of the mesh
   * @return  the name of the mesh
   */
  [[nodiscard]] std::string const& name() const noexcept;

  /*!
   * @brief mesh dimension
   * @return the dimension of the mesh {1,2,3}
   */
  [[nodiscard]] int dimension() const noexcept {
    return m_dimension;
  }

  /*!
   * @brief mesh node number
   * @return number of nodes in the mesh
   */
  [[nodiscard]] int nbNodes() const noexcept {
    return nbItems(Neo::ItemKind::IK_Node);
  }

  /*!
   * @brief mesh edge number
   * @return number of edges in the mesh
   */
  [[nodiscard]] int nbEdges() const noexcept {
    return nbItems(Neo::ItemKind::IK_Edge);
  }

  /*!
   * @brief mesh face number
   * @return number of faces in the mesh
   */
  [[nodiscard]] int nbFaces() const noexcept {
    return nbItems(Neo::ItemKind::IK_Face);
  }

  /*!
   * @brief mesh cell number
   * @return number of cells in the mesh
   */
  [[nodiscard]] int nbCells() const noexcept {
    return nbItems(Neo::ItemKind::IK_Cell);
  }

  /*!
   * @brief mesh dof number
   * @return number of dofs in the mesh
   */
  [[nodiscard]] int nbDoFs() const noexcept {
    return nbItems(Neo::ItemKind::IK_Dof);
  }

  /*!
   * @brief mesh item with kind \p ik number
   * @param ik : kind of the counted item
   * @return number of item with item kind \p ik in the mesh (may sum over several families with the same kind)
   */
  [[nodiscard]] int nbItems(Neo::ItemKind ik) const noexcept {
    return std::accumulate(m_families.begin(), m_families.end(), 0,
                           [ik](auto const& nb_item, auto const& family_map_element) {
                             return (family_map_element.first.first == ik) ? nb_item + family_map_element.second->nbElements() : nb_item;
                           });
  }

  /*!
   * @brief unique id property name for a given family \p family_name
   * @param family_name
   * @return the name of the unique id property for a family with name \p family_name whatever its kind.
   */
  [[nodiscard]] static std::string uniqueIdPropertyName(std::string const& family_name) noexcept;

  /*!
   * @brief find an existing family given its name \p family_name and kind \p family_kind
   * If the family does not exist
   * @param family_name
   * @param family_kind
   * @return
   */
  Family& findFamily(Neo::ItemKind family_kind, std::string const& family_name) const noexcept(ndebug);

  /*!
   * @brief Add a family of kind \p item_kind and of name \p family_name
   * @param item_kind
   * @param family_name
   * @return
   */
  Neo::Family& addFamily(Neo::ItemKind item_kind, std::string family_name) noexcept;

  /*!
   * @brief Schedule items creation for family \p family with unique ids \p uids
   * @param family Family where items are added
   * @param uids Uids of the added items
   * @param future_added_item_range Future ItemRange : after the call to applyScheduledOperations
   * this future range will give access to a range containing new items local ids
   */

  void scheduleAddItems(Neo::Family& family, std::vector<Neo::utils::Int64> uids, Neo::FutureItemRange& future_added_item_range) noexcept;

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
   * @brief Ask for a fixed-size oriented connectivity add between \p source_family and \p target_family. Source items are scheduled but not created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Connectivity fix size value.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   * @param source_item_orientation_in_target_item Orientation the source items into their items (ex a face in its cells), must be -1 or 1.
   * @param add_or_modify Indicates whether Connectivity is added or modified (add is default)
   * @param do_check_orientation If true an operation is added to check if the given orientation is correct, ie the sum of the orientations for an item belongs to {-1,0,1}
   *
   * Oriented connectivity with fix size (nb of connected items per item is constant).
   * Use this method to add oriented connectivity with source items scheduled but not yet created
   * i.e addItems and addConnectivity are applied in the same call to applyScheduledOperations.
   * Adding an oriented connectivity simply register in addition to the connectivity, a property containing the orientation of the source items within the target items
   * they are connected to.
   */
  void scheduleAddOrientedConnectivity(Neo::Family& source_family, Neo::FutureItemRange& source_items,
                                       Neo::Family& target_family, int nb_connected_item_per_item,
                                       std::vector<Neo::utils::Int64> connected_item_uids,
                                       std::string const& connectivity_unique_name,
                                       std::vector<Neo::utils::Int32> source_item_orientation_in_target_item,
                                       ConnectivityOperation add_or_modify = ConnectivityOperation::Add,
                                       bool do_check_orientation = false);

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
   * @brief Ask for a fixed-size oriented connectivity add between \p source_family and \p target_family. Source items are already created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Given via an ItemRange.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Connectivity fix size value.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   * @param source_item_orientation_in_target_item Orientation the source items into their items (ex a face in its cells), must be -1 or 1.
   * @param add_or_modify Indicates whether Connectivity is added or modified (add is default)
   * @param do_check_orientation If true an operation is added to check if the given orientation is correct, ie the sum of the orientations for an item belongs to {-1,0,1}
   *
   * Oriented connectivity with fix size (nb of connected items per item is constant).
   * Use this method to add oriented connectivity with source items already created
   * in a previous call to applyScheduledOperations.
   * Adding an oriented connectivity simply register in addition to the connectivity, a property containing the orientation of the source items within the target items
   * they are connected to.
   */
  void scheduleAddOrientedConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                                       Neo::Family& target_family, int nb_connected_item_per_item,
                                       std::vector<Neo::utils::Int64> connected_item_uids,
                                       std::string const& connectivity_unique_name,
                                       std::vector<Neo::utils::Int32> source_item_orientation_in_target_item,
                                       ConnectivityOperation add_or_modify = ConnectivityOperation::Add,
                                       bool do_check_orientation = false);

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
   * @brief Ask for a variable size oriented connectivity add between \p source_family and \p target_family. Source items are scheduled but not created.
   * @param source_family The family of source items.
   * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
     * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
   * @param target_family The family of target items.
   * @param nb_connected_item_per_item Number of connected item per items. Array with size equal to source items number.
   * @param connected_item_uids Unique ids of the connected items.
   * @param connectivity_unique_name Connectivity name must be unique
   * @param source_item_orientation_in_target_item Orientation the source items into their items (ex a face in its cells), must be -1 or 1.
   * @param add_or_modify Indicates whether Connectivity is added or modified (add is default)
   * @param do_check_orientation If true an operation is added to check if the given orientation is correct, ie the sum of the orientations for an item belongs to {-1,0,1}
   *
   * Oriented connectivity with variable size (nb of connected items per item is variable)
   * Use this method to oriented add connectivity with source items scheduled but not yet created
   * i.e addItems and addConnectivity are applied in the same call to applyScheduledOperations.
   * Adding an oriented connectivity simply register in addition to the connectivity, a property containing the orientation of the source items within the target items
   * they are connected to.
   */
  void scheduleAddOrientedConnectivity(Neo::Family& source_family, Neo::FutureItemRange& source_items,
                                       Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                       std::vector<Neo::utils::Int64> connected_item_uids,
                                       std::string const& connectivity_unique_name,
                                       std::vector<Neo::utils::Int32> source_item_orientation_in_target_item,
                                       ConnectivityOperation add_or_modify = ConnectivityOperation::Add,
                                       bool do_check_orientation = false);

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
    * @brief Ask for a variable size oriented connectivity add between \p source_family and \p target_family. Source items are already created.
    * @param source_family The family of source items.
    * @param source_items Items to be connected. Use of a FutureItemRange means these items come from a the AddItems operation not yet applied.
      * (i.e addItems and addConnectivity are applied with the same call to applyScheduledOperations)
    * @param target_family The family of target items.
    * @param nb_connected_item_per_item Number of connected item per items. Array with size equal to source items number.
    * @param connected_item_uids Unique ids of the connected items.
    * @param connectivity_unique_name Connectivity name must be unique
    * @param source_item_orientation_in_target_item Orientation the source items into their items (ex a face in its cells), must be -1 or 1.
    * @param add_or_modify Indicates whether Connectivity is added or modified (add is default)
    * @param do_check_orientation If true an operation is added to check if the given orientation is correct, ie the sum of the orientations for an item belongs to {-1,0,1}
    *
    * Oriented connectivity with variable size (nb of connected items per item is variable)
    * Use this method to add oriented connectivity with source items already created
    * in a previous call to applyScheduledOperations.
    * Adding an oriented connectivity simply register in addition to the connectivity, a property containing the orientation of the source items within the target items
    * they are connected to.
    */
  void scheduleAddOrientedConnectivity(Neo::Family& source_family, Neo::ItemRange const& source_items,
                                       Neo::Family& target_family, std::vector<int> nb_connected_item_per_item,
                                       std::vector<Neo::utils::Int64> connected_item_uids,
                                       std::string const& connectivity_unique_name,
                                       std::vector<int> source_item_orientation_in_target_item,
                                       ConnectivityOperation add_or_modify = ConnectivityOperation::Add,
                                       bool do_check_orientation = false);

  /*!
   * @brief Schedule a set item coordinates. Will be applied when applyScheduledOperations will be called
   * @param item_family Family of the items whose coords will be modified
   * @param future_added_item_range Set of the items whose coords will be modified. These items are not created yet.
   * They will be created in applyScheduledOperations before the call to this coords set.
   * @param item_coords Value of the items coordinates
   *
   * To use for first definition of item coordinates. To change coordinates, use scheduleMoveItems
   */
  void scheduleSetItemCoords(Neo::Family& item_family, Neo::FutureItemRange& future_added_item_range, std::vector<Neo::utils::Real3> item_coords) noexcept;

  /*!
   * @brief Get item connectivity between \p source_family and \p target_family with name \p name
   * @param source_family Family of the source items
   * @param target_family Family of the target items
   * @param connectivity_name Name of the connectivity
   * @return Connectivity, a connectivity wrapper object
   * @throw a std::invalid_argument if the connectivity is not found
   */
  Connectivity getConnectivity(Neo::Family const& source_family, Neo::Family const& target_family, std::string const& connectivity_name) const;

  /*!
   * @brief Get item connectivities with \p source_family as source
   * @param source_family Family of the source items
   * @return a span of Connectivities (a connectivity wrapper object). The span will be empty if no connectivities are found.
   */
  Neo::utils::ConstSpan<Connectivity> getConnectivities(Neo::Family const& source_family) const;

  /*!
   * @brief Apply all scheduled operations (addItems, addConnectivities, setItemCoords)
   * @return An object allowing to get the new items ItemRange from the FutureItemRange
   */
  Neo::EndOfMeshUpdate applyScheduledOperations();

  /*!
   * Use this method to change coordinates of existing items
   * @param family ItemFamily of item to change coordinates
   * @return Coordinates property. Usage : Real3 coord = coord_prop[item_lid];
   */
  [[nodiscard]] CoordPropertyType& getItemCoordProperty(Neo::Family& family);
  [[nodiscard]] CoordPropertyType const& getItemCoordProperty(Neo::Family const& family) const;

  /*!
   * Get unique id property of a family
   * @param item_family ItemFamily of concerned items
   * @return UniqueId property. Udage Neo::utils::Int64 uid = uid_prop[item_lid]
   *
   * A more direct usage is to call directy \fn uniqueIds(item_family,item_lids)
   * to get uids of given lids
   */
  [[nodiscard]] UniqueIdPropertyType const& getItemUidsProperty(const Family& item_family) const noexcept;

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

  /*!
   * Get unique ids from \p item_lids (local ids) in \p item_family
   * @param item_family Family of given items
   * @param item_lids Given item local ids
   * @return Given item unique ids
   */
  std::vector<Neo::utils::Int64> uniqueIds(Family const& item_family, std::vector<Neo::utils::Int32> const& item_lids) const noexcept;

  std::vector<Neo::utils::Int64> uniqueIds(Family const& item_family, Neo::utils::Int32ConstSpan item_lids) const noexcept;

  /*!
   * Get local ids from \p item_uids (unique ids) in \p item_family
   * @param item_family Family of given items
   * @param item_uids Given item unique ids
   * @return Given item local ids
   */
  std::vector<Neo::utils::Int32> localIds(Family const& item_family, const std::vector<Neo::utils::Int64>& item_uids) const noexcept;

  /*!
   * @brief prepare evolutive mesh api : schedule move node operation. Will be applied when applyScheduledOperations will be called
   * @param mesh
   * @param item_family item family with moving items
   * @param moved_item_uids uids of the moving items
   * @param moved_item_new_coords new coordinates of the moving nodes given in \p node_uids
   */
  void scheduleMoveItems(Neo::Family& item_family, std::vector<Neo::utils::Int64> const& moved_item_uids, std::vector<Neo::utils::Real3> const& moved_item_new_coords);

  /*!
   * @brief prepare evolutive mesh api : schedule remove item operation. Will be applied when applyScheduledOperations will be called
   * @param item_family item family of removed items
   * @param removed_item_uids unique ids of removed items
   */
  void scheduleRemoveItems(Neo::Family& item_family, std::vector<Neo::utils::Int64> const& removed_item_uids);

  /*!
   * @brief prepare evolutive mesh api : schedule remove item operation. Will be applied when applyScheduledOperations will be called
   * @param item_family item family of removed items
   * @param removed_items item range of removed items
   */
  void scheduleRemoveItems(Neo::Family& item_family, Neo::ItemRange const& removed_items);

  /*!
   * @brief schedule a user operation on mesh.
   * @param input_property_family item family owning the input property
   * @param input_property_name
   * @param output_property_family item family owning output property
   * @param output_property_name
   * @param mesh_operation may be given by a lambda function (will be casted in std::function, see detailed description)
   *
   * This operation may depend on an input property (Item local/unique ids, item coordinates, item connectivities...),
   * and may produce an output property, that can be used as an input property on another mesh operation.
   * The mesh operation signature is operation(ConcreteInputPropertyType const& input_prop, ConcreteOutputPropertyType & output_prop.
   * A std::function is used to avoid template contamination from low level MeshKernel.
   */
  void scheduleAddMeshOperation(Family& input_property_family,
                                std::string const& input_property_name,
                                Family& output_property_family,
                                std::string const& output_property_name,
                                MeshOperation mesh_operation);
  /*!
   * Access to internal structure, for advanced use
   * @return Reference toward internal structure
   * Todo : remove !
   */
  Neo::MeshKernel::AlgorithmPropertyGraph& internalMeshGraph() noexcept {
    return *m_mesh_graph;
  }

  [[nodiscard]] std::string _itemCoordPropertyName(Family const& item_family) const {
    return item_family.name() + "_item_coordinates";
  }

 public:
  [[nodiscard]] std::string _removeItemPropertyName(Family const& item_family) const {
    return "removed_" + item_family.name() + "_items";
  }
};

} // end namespace Neo

#endif // NEO_MESH_H
