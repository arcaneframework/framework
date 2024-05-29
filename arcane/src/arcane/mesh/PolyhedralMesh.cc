// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMesh.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Polyhedral mesh impl using Neo data structure.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/PolyhedralMesh.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/ItemSharedInfo.h"
#include "arcane/core/ItemTypeInfo.h"
#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/mesh/ItemFamily.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/IMeshFactory.h"
#include "arcane/core/ItemInternal.h"
#include "arcane/core/internal/IVariableMngInternal.h"

#ifdef ARCANE_HAS_POLYHEDRAL_MESH_TOOLS

#include "arcane/IMeshMng.h"
#include "arcane/MeshHandle.h"
#include "arcane/IItemFamily.h"
#include "arcane/mesh/ItemFamily.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/List.h"

#include "neo/Mesh.h"

#ifdef ARCANE_HAS_VTKIO
#include "arcane/mesh/PolyhedralMeshTools.h" // non utilisé ??
#include "arcane/core/IMeshFactory.h"

#endif

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
_errorEmptyMesh() const
{
  ARCANE_FATAL("Cannot use PolyhedralMesh if Arcane is not linked with lib Neo");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace mesh
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class PolyhedralFamily : public ItemFamily
  {
    ItemSharedInfoWithType* m_shared_info = nullptr;
    Int32UniqueArray m_empty_connectivity{ 0 };
    Int32UniqueArray m_empty_connectivity_indexes;
    Int32UniqueArray m_empty_connectivity_nb_item;

   public:

    inline static const String m_arcane_item_lids_property_name{ "Arcane_Item_Lids" }; // inline used to initialize within the declaration

   public:

    PolyhedralFamily(IMesh* mesh, eItemKind ik, String name)
    : ItemFamily(mesh, ik, name)
    {
      ItemFamily::build();
      m_sub_domain_id = subDomain()->subDomainId();
      ItemTypeMng* itm = m_mesh->itemTypeMng();
      ItemTypeInfo* dof_type_info = itm->typeFromId(IT_NullType);
      m_shared_info = _findSharedInfo(dof_type_info);
      _updateEmptyConnectivity();
    }

   public:

    void preAllocate(Integer nb_item)
    {
      Integer nb_hash = itemsMap().buckets().size();
      Integer wanted_size = 2 * (nb_item + infos().nbItem());
      if (nb_hash < wanted_size)
        itemsMap().resize(wanted_size, true);
      m_empty_connectivity_indexes.resize(nb_item + nbItem(), 0);
      m_empty_connectivity_nb_item.resize(nb_item + nbItem(), 0);
      _updateEmptyConnectivity();
    }

    ItemInternal* _allocItem(const Int64 uid)
    {
      bool need_alloc; // given by alloc
      ItemInternal* item_internal = ItemFamily::_allocOne(uid, need_alloc);
      if (!need_alloc)
        item_internal->setUniqueId(uid);
      else {
        _allocateInfos(item_internal, uid, m_shared_info);
      }
      item_internal->setOwner(m_sub_domain_id, m_sub_domain_id);
      return item_internal;
    }

    void addItems(Int64ConstSmallSpan uids, Int32ArrayView items)
    {
      if (uids.empty())
        return;
      ARCANE_ASSERT((uids.size() == items.size()), ("one must have items.size==uids.size()"));
      m_mesh->traceMng()->info() << " PolyhedralFamily::ADDITEMS ";
      preAllocate(uids.size());
      auto index{ 0 };
      for (auto uid : uids) {
        ItemInternal* ii = _allocItem(uid);
        items[index++] = ii->localId();
      }
      m_need_prepare_dump = true;
      _updateItemInternalList();
    }

    void _updateItemInternalList()
    {
      switch (itemKind()) {
      case IK_Cell:
        m_item_internal_list->cells = _itemsInternal();
        break;
      case IK_Face:
        m_item_internal_list->faces = _itemsInternal();
        break;
      case IK_Edge:
        m_item_internal_list->edges = _itemsInternal();
        break;
      case IK_Node:
        m_item_internal_list->nodes = _itemsInternal();
        break;
      case IK_DoF:
      case IK_Particle:
      case IK_Unknown:
        break;
      }
    }

    void _updateEmptyConnectivity()
    {
      auto item_internal_connectivity_list = itemInternalConnectivityList();
      for (auto item_kind = 0; item_kind < ItemInternalConnectivityList::MAX_ITEM_KIND; ++item_kind) {
        item_internal_connectivity_list->_setConnectivityList(item_kind, m_empty_connectivity);
        item_internal_connectivity_list->_setConnectivityIndex(item_kind, m_empty_connectivity_indexes);
        item_internal_connectivity_list->_setConnectivityNbItem(item_kind, m_empty_connectivity_nb_item);
      }
    }
  };

} // namespace mesh
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_POLYHEDRAL_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace mesh
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class PolyhedralMeshImpl
  {
    ISubDomain* m_subdomain;
    Neo::Mesh m_mesh{ "Test" };

    template <eItemKind IK>
    class ItemKindTraits
    {
      static const Neo::ItemKind item_kind = Neo::ItemKind::IK_None;
    };

    static Neo::ItemKind itemKindArcaneToNeo(eItemKind ik)
    {
      switch (ik) {
      case IK_Cell:
        return Neo::ItemKind::IK_Cell;
      case IK_Face:
        return Neo::ItemKind::IK_Face;
      case IK_Edge:
        return Neo::ItemKind::IK_Edge;
      case IK_Node:
        return Neo::ItemKind::IK_Node;
      case IK_DoF:
        return Neo::ItemKind::IK_Dof;
      case IK_Unknown:
      case IK_Particle:
        return Neo::ItemKind::IK_None;
      }
      return Neo::ItemKind::IK_Node;
    }

   public:

    class ItemLocalIds
    {
      Neo::FutureItemRange m_future_items;

     public:

      friend class PolyhedralMeshImpl;
    };

   public:

    explicit PolyhedralMeshImpl(ISubDomain* subDomain)
    : m_subdomain(subDomain)
    {}

   public:

    String name() const { return m_mesh.name(); }

    Integer dimension() const { return m_mesh.dimension(); }

    Integer nbNode() const { return m_mesh.nbNodes(); }
    Integer nbEdge() const { return m_mesh.nbEdges(); }
    Integer nbFace() const { return m_mesh.nbFaces(); }
    Integer nbCell() const { return m_mesh.nbCells(); }
    Integer nbItem(eItemKind ik) const { return m_mesh.nbItems(itemKindArcaneToNeo(ik)); }

    static void _setFaceInfos(Int32 mod_flags, Face& face)
    {
      Int32 face_flags = face.itemBase().flags();
      face_flags &= ~ItemFlags::II_InterfaceFlags;
      face_flags |= mod_flags;
      face.mutableItemBase().setFlags(face_flags);
    }

    /*---------------------------------------------------------------------------*/

    void addFamily(eItemKind ik, const String& name)
    {
      m_mesh.addFamily(itemKindArcaneToNeo(ik), name.localstr());
    }

    /*---------------------------------------------------------------------------*/

    void scheduleAddItems(PolyhedralFamily* arcane_item_family,
                          Int64ConstSmallSpan uids,
                          ItemLocalIds& item_local_ids)
    {
      auto& added_items = item_local_ids.m_future_items;
      auto& item_family = m_mesh.findFamily(itemKindArcaneToNeo(arcane_item_family->itemKind()),
                                            arcane_item_family->name().localstr());
      m_mesh.scheduleAddItems(item_family, std::vector<Int64>{ uids.begin(), uids.end() }, added_items);
      // add arcane items
      auto& mesh_graph = m_mesh.internalMeshGraph();
      item_family.addMeshScalarProperty<Neo::utils::Int32>(PolyhedralFamily::m_arcane_item_lids_property_name.localstr());
      mesh_graph.addAlgorithm(Neo::MeshKernel::InProperty{ item_family, item_family.lidPropName() },
                              Neo::MeshKernel::OutProperty{ item_family, PolyhedralFamily::m_arcane_item_lids_property_name.localstr() },
                              [arcane_item_family, uids](Neo::ItemLidsProperty const& lids_property,
                                                         Neo::MeshScalarPropertyT<Neo::utils::Int32>&) {
                                Int32UniqueArray arcane_items(uids.size());
                                arcane_item_family->addItems(uids, arcane_items);
                                arcane_item_family->traceMng()->info() << arcane_items;
                                // debug check lid matching. maybe to remove if too coostly
                                auto neo_lids = lids_property.values();
                                if (!std::equal(neo_lids.begin(), neo_lids.end(), arcane_items.begin()))
                                  arcane_item_family->traceMng()->fatal() << "Inconsistent item lids generation between Arcane and Neo.";
                              });
    }

    /*---------------------------------------------------------------------------*/

    void scheduleAddConnectivity(PolyhedralFamily* arcane_source_item_family,
                                 ItemLocalIds& source_items,
                                 Integer nb_connected_items_per_item,
                                 PolyhedralFamily* arcane_target_item_family,
                                 Int64ConstArrayView target_items_uids,
                                 String const& name)
    {
      // add connectivity in Neo
      _scheduleAddConnectivity(arcane_source_item_family,
                               source_items,
                               nb_connected_items_per_item,
                               arcane_target_item_family,
                               target_items_uids,
                               name);
    }

    /*---------------------------------------------------------------------------*/

    void scheduleAddConnectivity(PolyhedralFamily* arcane_source_item_family,
                                 ItemLocalIds& source_items,
                                 Int32ConstSmallSpan nb_connected_items_per_item,
                                 PolyhedralFamily* arcane_target_item_family,
                                 Int64ConstSmallSpan target_items_uids,
                                 String const& connectivity_name)
    {
      // debug
      std::cout << "====================VALIDATE CONNECTIVITY  ==================" << std::endl;
      std::cout << "== " << connectivity_name << std::endl;
      auto item_index = 0;
      std::cout << "==== nb connected items ====" << std::endl;
      std::copy(nb_connected_items_per_item.begin(), nb_connected_items_per_item.end(), std::ostream_iterator<Int32>(std::cout, " "));
      std::cout << std::endl
                << "==== connected items ====" << std::endl;
      std::copy(target_items_uids.begin(), target_items_uids.end(), std::ostream_iterator<Int64>(std::cout, " "));
      std::cout << std::endl;
      for (auto&& nb_connected_items : nb_connected_items_per_item) {
        Int64ConstArrayView connected_items{ nb_connected_items, &target_items_uids[item_index] };
        std::copy(connected_items.begin(), connected_items.end(), std::ostream_iterator<Int64>{ std::cout, " " });
        item_index += nb_connected_items;
        std::cout << "\n";
      }
      _scheduleAddConnectivity(arcane_source_item_family,
                               source_items,
                               std::vector<Int32>{ nb_connected_items_per_item.begin(), nb_connected_items_per_item.end() },
                               arcane_target_item_family,
                               target_items_uids,
                               connectivity_name);
    }

    /*---------------------------------------------------------------------------*/

    // template to handle nb_items_per_item type (an int or an array)
    template <typename ConnectivitySizeType>
    void _scheduleAddConnectivity(PolyhedralFamily* arcane_source_item_family,
                                  ItemLocalIds& source_items,
                                  ConnectivitySizeType&& nb_connected_items_per_item,
                                  PolyhedralFamily* arcane_target_item_family,
                                  Int64ConstSmallSpan target_item_uids,
                                  String const& connectivity_name)
    {
      // add connectivity in Neo
      auto& source_family = m_mesh.findFamily(itemKindArcaneToNeo(arcane_source_item_family->itemKind()),
                                              arcane_source_item_family->name().localstr());
      auto& target_family = m_mesh.findFamily(itemKindArcaneToNeo(arcane_target_item_family->itemKind()),
                                              arcane_target_item_family->name().localstr());
      // Remove connectivities with a null item
      std::vector<Int64> target_item_uids_filtered;
      target_item_uids_filtered.reserve(target_item_uids.size());
      std::copy_if(target_item_uids.begin(),
                   target_item_uids.end(),
                   std::back_inserter(target_item_uids_filtered),
                   [](auto uid) { return uid != NULL_ITEM_UNIQUE_ID; });
      // Add connectivity in Neo (async)
      m_mesh.scheduleAddConnectivity(source_family, source_items.m_future_items, target_family,
                                     std::forward<ConnectivitySizeType>(nb_connected_items_per_item),
                                     std::move(target_item_uids_filtered),
                                     connectivity_name.localstr());
      // Register Neo connectivities in Arcane
      auto& mesh_graph = m_mesh.internalMeshGraph();
      std::string connectivity_add_output_property_name = std::string{ "EndOf" } + connectivity_name.localstr() + "Add";
      source_family.addScalarProperty<Neo::utils::Int32>(connectivity_add_output_property_name);
      mesh_graph.addAlgorithm(Neo::MeshKernel::InProperty{ source_family, connectivity_name.localstr() },
                              Neo::MeshKernel::OutProperty{ source_family, connectivity_add_output_property_name },
                              [arcane_source_item_family, arcane_target_item_family, &source_family, &target_family, this, connectivity_name](Neo::Mesh::ConnectivityPropertyType const&,
                                                                                                                              Neo::ScalarPropertyT<Neo::utils::Int32>&) {
                                this->m_subdomain->traceMng()->info() << "ADD CONNECTIVITY";
                                auto item_internal_connectivity_list = arcane_source_item_family->itemInternalConnectivityList();
                                // todo check if families are default families
                                auto connectivity = m_mesh.getConnectivity(source_family, target_family, connectivity_name.localstr());
                                // to access connectivity data (for initializing Arcane connectivities) create a proxy on Neo connectivity
                                auto& connectivity_values = source_family.getConcreteProperty<Neo::Mesh::ConnectivityPropertyType>(connectivity_name.localstr());
                                Neo::MeshArrayPropertyProxyT<Neo::Mesh::ConnectivityPropertyType::PropertyDataType> connectivity_proxy{connectivity_values};
                                auto nb_item_data = connectivity_proxy.arrayPropertyOffsets();
                                auto nb_item_size = connectivity_proxy.arrayPropertyOffsetsSize(); // todo check MeshArrayProperty::size
                                item_internal_connectivity_list->_setConnectivityNbItem(arcane_target_item_family->itemKind(),
                                                                                       Int32ArrayView{ Integer(nb_item_size), nb_item_data });
                                auto max_nb_connected_items = connectivity.maxNbConnectedItems();
                                item_internal_connectivity_list->_setMaxNbConnectedItem(arcane_target_item_family->itemKind(), max_nb_connected_items);
                                auto connectivity_values_data = connectivity_proxy.arrayPropertyData();
                                auto connectivity_values_size = connectivity_proxy.arrayPropertyDataSize();
                                item_internal_connectivity_list->_setConnectivityList(arcane_target_item_family->itemKind(),
                                                                                      Int32ArrayView{ Integer(connectivity_values_size), connectivity_values_data });
                                auto connectivity_index_data = connectivity_proxy.arrayPropertyIndex();
                                auto connectivity_index_size = connectivity_proxy.arrayPropertyIndexSize();
                                item_internal_connectivity_list->_setConnectivityIndex(arcane_target_item_family->itemKind(),
                                                                                       Int32ArrayView{ Integer(connectivity_index_size), connectivity_index_data });
                              });
      // If FaceToCellConnectivity Add face flags II_Boundary, II_SubdomainBoundary, II_HasFrontCell, II_HasBackCell
      if (arcane_source_item_family->itemKind() == IK_Face && arcane_target_item_family->itemKind() == IK_Cell) {
        std::string flag_definition_output_property_name{ "EndOfFlagDefinition" };
        source_family.addScalarProperty<Neo::utils::Int32>(flag_definition_output_property_name);
        mesh_graph.addAlgorithm(Neo::MeshKernel::InProperty{ source_family, connectivity_add_output_property_name }, Neo::MeshKernel::OutProperty{ source_family, flag_definition_output_property_name },
                                [arcane_source_item_family, this, target_item_uids, &source_items](Neo::ScalarPropertyT<Neo::utils::Int32> const&, Neo::ScalarPropertyT<Neo::utils::Int32> const&) {
                                  auto current_face_index = 0;
                                  auto arcane_faces = arcane_source_item_family->itemInfoListView();
                                  for (auto face_lid : source_items.m_future_items.new_items) {
                                    Face current_face = arcane_faces[face_lid].toFace();
                                    if (target_item_uids[2 * current_face_index + 1] == NULL_ITEM_LOCAL_ID) {
                                      //                                    if (current_face.frontCell().null()) {
                                      // Reste uniquement la back_cell ou aucune maille.
                                      Int32 mod_flags = (target_item_uids[2 * current_face_index] != NULL_ITEM_LOCAL_ID) ? (ItemFlags::II_Boundary | ItemFlags::II_HasBackCell | ItemFlags::II_BackCellIsFirst) : 0;
                                      _setFaceInfos(mod_flags, current_face);
                                    }
                                    else if (target_item_uids[2 * current_face_index] == NULL_ITEM_LOCAL_ID) {
                                      // Reste uniquement la front cell
                                      _setFaceInfos(ItemFlags::II_Boundary | ItemFlags::II_HasFrontCell | ItemFlags::II_FrontCellIsFirst, current_face);
                                    }
                                    else {
                                      // Il y a deux mailles connectées.
                                      _setFaceInfos(ItemFlags::II_HasFrontCell | ItemFlags::II_HasBackCell | ItemFlags::II_BackCellIsFirst, current_face);
                                    }
                                    ++current_face_index;
                                  }
                                });
      }
    }

    /*---------------------------------------------------------------------------*/

    void scheduleSetItemCoordinates(PolyhedralFamily* item_family, ItemLocalIds& local_ids, Real3ConstSmallSpan item_coords, VariableItemReal3& arcane_coords)
    {
      auto& _item_family = m_mesh.findFamily(itemKindArcaneToNeo(item_family->itemKind()), item_family->name().localstr());
      std::vector<Neo::utils::Real3> _node_coords(item_coords.size());
      auto node_index = 0;
      for (auto&& node_coord : item_coords) {
        _node_coords[node_index++] = Neo::utils::Real3{ node_coord.x, node_coord.y, node_coord.z };
      }
      m_mesh.scheduleSetItemCoords(_item_family, local_ids.m_future_items, _node_coords);
      // Fill Arcane Variable
      auto& mesh_graph = m_mesh.internalMeshGraph();
      _item_family.addScalarProperty<Int32>("NoOutProperty42"); // todo remove : create noOutput algo in Neo
      mesh_graph.addAlgorithm(Neo::MeshKernel::InProperty{ _item_family, m_mesh._itemCoordPropertyName(_item_family) },
                              Neo::MeshKernel::OutProperty{ _item_family, "NoOutProperty42" },
                              [this, item_family, &_item_family, &arcane_coords](Neo::Mesh::CoordPropertyType const& item_coords_property,
                                                                                 Neo::ScalarPropertyT<Neo::utils::Int32>&) {
                                this->m_subdomain->traceMng()->info() << "= Update Arcane Coordinates =";
                                // enumerate nodes : ensure again Arcane/Neo local_ids are identicals
                                auto& all_items = _item_family.all();
                                VariableNodeReal3 node_coords{ VariableBuildInfo{ item_family->mesh(), "NodeCoord" } };
                                for (auto item : all_items) {
                                  arcane_coords[ItemLocalId{ item }] = { item_coords_property[item].x,
                                                                         item_coords_property[item].y,
                                                                         item_coords_property[item].z };
                                  //                                    std::cout << "x y z : " << item_coords_property[item].x << " "
                                  //                                                            << item_coords_property[item].y << " "
                                  //                                                            << item_coords_property[item].z;
                                }
                              });
    }

    /*---------------------------------------------------------------------------*/

    void applyScheduledOperations() noexcept
    {
      m_mesh.applyScheduledOperations();
    }

   private:

    void _createSingleCellTest()
    {
      auto& cell_family = m_mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
      auto& node_family = m_mesh.addFamily(Neo::ItemKind::IK_Node, "node_family");
      auto added_cells = Neo::FutureItemRange{};
      m_mesh.scheduleAddItems(cell_family, { 0 }, added_cells);
      auto added_nodes = Neo::FutureItemRange{};
      m_mesh.scheduleAddItems(node_family, { 0, 1, 2, 3, 4, 5 }, added_nodes);
      m_mesh.applyScheduledOperations();
    }
  };

  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_Cell>
  {
    static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Cell;
  };
  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_Face>
  {
    static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Face;
  };
  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_Edge>
  {
    static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Edge;
  };
  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_Node>
  {
    static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Node;
  };
  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_DoF>
  {
    static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Dof;
  };

} // End namespace mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mesh::PolyhedralMesh::
~PolyhedralMesh()
{
  m_mesh_handle._setMesh(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* mesh::PolyhedralMesh::
traceMng()
{
  return m_subdomain->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshHandle mesh::PolyhedralMesh::
handle() const
{
  return m_mesh_handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mesh::PolyhedralMesh::
PolyhedralMesh(ISubDomain* subdomain, const MeshBuildInfo& mbi)
: EmptyMesh{ subdomain->traceMng() }
, m_name{ mbi.name() }
, m_subdomain{ subdomain }
, m_mesh_handle{ m_subdomain->defaultMeshHandle() }
, m_properties(std::make_unique<Properties>(subdomain->propertyMng(), String("ArcaneMeshProperties_") + m_name))
, m_mesh{ std::make_unique<mesh::PolyhedralMeshImpl>(m_subdomain) }
, m_parallel_mng{ mbi.parallelMngRef().get() }
, m_mesh_part_info{ makeMeshPartInfoFromParallelMng(m_parallel_mng) }
, m_item_type_mng(ItemTypeMng::_singleton())
, m_initial_allocator(*this)
, m_variable_mng{ subdomain->variableMng() }
, m_mesh_checker{ this }
{
  m_mesh_handle._setMesh(this);
  m_mesh_item_internal_list.mesh = this;
  m_default_arcane_families.fill(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
read(const String& filename)
{
  // First step: create manually a unit mesh
  //  ARCANE_UNUSED(filename); // temporary
  //  _createUnitMesh();
  // Second step read a vtk polyhedral mesh
  m_subdomain->traceMng()->info() << "--PolyhedralMesh : reading " << filename;
  // First step create a single cell
  //      _createSingleCellTest();
  // Second step : read a vtk polyhedral file
  UniqueArray<String> splitted_filename{};
  filename.split(splitted_filename, '.');
  const auto& file_extension = splitted_filename.back();
  if (file_extension != "vtk")
    m_subdomain->traceMng()->fatal() << "Only vtk file format supported for polyhedral mesh";

  createItemFamily(IK_Cell, "Cell");
  createItemFamily(IK_Node, "Node");
  createItemFamily(IK_Face, "Face");
  createItemFamily(IK_Edge, "Edge");
  [[maybe_unused]] auto cell_family = arcaneDefaultFamily(IK_Cell);
  [[maybe_unused]] auto node_family = arcaneDefaultFamily(IK_Node);
  [[maybe_unused]] auto face_family = arcaneDefaultFamily(IK_Face);
  [[maybe_unused]] auto edge_family = arcaneDefaultFamily(IK_Edge);
#ifdef ARCANE_HAS_VTKIO
  PolyhedralMeshTools::VtkReader reader{ filename };
  PolyhedralMeshImpl::ItemLocalIds cell_lids, node_lids, face_lids, edge_lids;
  // todo separate Link with Arcane and link with vtk structure
  // Add items
  m_mesh->scheduleAddItems(cell_family, reader.cellUids(), cell_lids);
  m_mesh->scheduleAddItems(node_family, reader.nodeUids(), node_lids);
  m_mesh->scheduleAddItems(face_family, reader.faceUids(), face_lids);
  m_mesh->scheduleAddItems(edge_family, reader.edgeUids(), edge_lids);
  // Add connectivities
  m_mesh->scheduleAddConnectivity(cell_family, cell_lids, reader.cellNbNodes(), node_family, reader.cellNodes(), String{ "CellToNodes" });
  m_mesh->scheduleAddConnectivity(cell_family, cell_lids, reader.cellNbFaces(), face_family, reader.cellFaces(), String{ "CellToFaces" });
  m_mesh->scheduleAddConnectivity(face_family, face_lids, reader.faceNbNodes(), node_family, reader.faceNodes(), String{ "FaceToNodes" });
  m_mesh->scheduleAddConnectivity(edge_family, edge_lids, 2, node_family, reader.edgeNodes(), String{ "EdgeToNodes" });
  m_mesh->scheduleAddConnectivity(face_family, face_lids, reader.faceNbCells(), cell_family, reader.faceCells(), String{ "FaceToCells" });
  m_mesh->scheduleAddConnectivity(edge_family, edge_lids, reader.edgeNbCells(), cell_family, reader.edgeCells(), String{ "EdgeToCells" });
  m_mesh->scheduleAddConnectivity(edge_family, edge_lids, reader.edgeNbFaces(), face_family, reader.edgeFaces(), String{ "EdgeToFaces" });
  m_mesh->scheduleAddConnectivity(face_family, face_lids, reader.faceNbEdges(), edge_family, reader.faceEdges(), String{ "FaceToEdges" });
  m_mesh->scheduleAddConnectivity(cell_family, cell_lids, reader.cellNbEdges(), edge_family, reader.cellEdges(), String{ "CellToEdges" });
  m_mesh->scheduleAddConnectivity(node_family, node_lids, reader.nodeNbCells(), cell_family, reader.nodeCells(), String{ "NodeToCells" });
  m_mesh->scheduleAddConnectivity(node_family, node_lids, reader.nodeNbFaces(), face_family, reader.nodeFaces(), String{ "NodeToFaces" });
  m_mesh->scheduleAddConnectivity(node_family, node_lids, reader.nodeNbEdges(), edge_family, reader.nodeEdges(), String{ "NodeToEdges" });
  // Add node coordinates
  m_arcane_node_coords = std::make_unique<VariableNodeReal3>(VariableBuildInfo(this, "NodeCoord"));
  m_arcane_node_coords->setUsed(true);
  m_mesh->applyScheduledOperations();
  cell_family->endUpdate();
  node_family->endUpdate();
  face_family->endUpdate();
  edge_family->endUpdate();
  endUpdate();
  m_mesh->scheduleSetItemCoordinates(node_family, node_lids, reader.nodeCoords(), *m_arcane_node_coords);
  m_mesh->applyScheduledOperations();
  m_is_allocated = true;
  // indicates mesh contains general Cells
  itemTypeMng()->setMeshWithGeneralCells(this);
#else
  ARCANE_FATAL("Need VTKIO to read polyhedral mesh");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
allocateItems(const Arcane::ItemAllocationInfo& item_allocation_info)
{
  // Second step read a vtk polyhedral mesh
  m_subdomain->traceMng()->info() << "--PolyhedralMesh : allocate items --";
  UniqueArray<PolyhedralMeshImpl::ItemLocalIds> item_local_ids(item_allocation_info.family_infos.size());
  auto family_index = 0;
  // Prepare item creation
  for (auto& family_info : item_allocation_info.family_infos) {
    auto item_family = _createItemFamily(family_info.item_kind, family_info.name);
    m_trace_mng->info() << " Create items " << family_info.name;
    m_mesh->scheduleAddItems(item_family, family_info.item_uids, item_local_ids[family_index++]);
  }
  // Prepare connectivity creation
  family_index = 0;
  for (auto& family_info : item_allocation_info.family_infos) {
    auto item_family = _findItemFamily(family_info.item_kind, family_info.name);
    m_trace_mng->info() << "Current family " << family_info.name;
    for (auto& current_connected_family_info : family_info.connected_family_info) {
      auto connected_family = _findItemFamily(current_connected_family_info.item_kind, current_connected_family_info.name);
      m_trace_mng->info() << " Create connectivity " << current_connected_family_info.connectivity_name;
      // check if connected family exists
      if (!connected_family) {
        ARCANE_WARNING((String::format("Cannot find family {0} with kind {1} "
                                       "The connectivity between {1} and this family is skipped",
                                       current_connected_family_info.name,
                                       current_connected_family_info.item_kind,
                                       item_family->name())
                        .localstr()));
        continue;
      }
      m_mesh->scheduleAddConnectivity(item_family,
                                      item_local_ids[family_index],
                                      current_connected_family_info.nb_connected_items_per_item,
                                      connected_family,
                                      current_connected_family_info.connected_items_uids,
                                      current_connected_family_info.connectivity_name);
    }
    ++family_index;
  }
  // Create items and connectivities
  m_mesh->applyScheduledOperations();
  // Create variable for coordinates. This has to be done before call to family::endUpdate. Todo add to the graph
  for (auto& family_info : item_allocation_info.family_infos) {
    if (family_info.item_coordinates.empty()) {
      ++family_index;
      continue;
    }
    auto item_family = _findItemFamily(family_info.item_kind, family_info.name);
    if (item_family == itemFamily(IK_Node)) { // mesh node coords
      m_arcane_node_coords = std::make_unique<VariableNodeReal3>(VariableBuildInfo(this, family_info.item_coordinates_variable_name));
      m_arcane_node_coords->setUsed(true);
    }
    else {
      auto arcane_item_coords_var_ptr = std::make_unique<VariableItemReal3>(VariableBuildInfo(this, family_info.item_coordinates_variable_name),
                                                                            item_family->itemKind());
      arcane_item_coords_var_ptr->setUsed(true);
      m_arcane_item_coords.push_back(std::move(arcane_item_coords_var_ptr));
    }
  }
  // Call Arcane ItemFamily endUpdate
  for (auto& family : m_arcane_families) {
    family->endUpdate();
  }
  endUpdate();
  // Add coordinates when needed (nodes, or dof, or particles...)
  family_index = 0;
  auto index = 0;
  for (auto& family_info : item_allocation_info.family_infos) {
    if (family_info.item_coordinates.empty()) {
      ++family_index;
      continue;
    }
    auto item_family = _findItemFamily(family_info.item_kind, family_info.name);
    if (item_family == itemFamily(IK_Node)) { // mesh node coords
      m_mesh->scheduleSetItemCoordinates(item_family, item_local_ids[family_index], family_info.item_coordinates, *m_arcane_node_coords);
    }
    else
      m_mesh->scheduleSetItemCoordinates(item_family, item_local_ids[family_index], family_info.item_coordinates, *m_arcane_item_coords[index++].get());
  }
  m_mesh->applyScheduledOperations();
  m_is_allocated = true;
  // indicates mesh contains general Cells
  itemTypeMng()->setMeshWithGeneralCells(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Arcane::mesh::PolyhedralMesh::
name() const
{
  return m_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
dimension()
{
  return m_mesh->dimension();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbNode()
{
  return m_mesh->nbNode();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbEdge()
{
  return m_mesh->nbEdge();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbFace()
{
  return m_mesh->nbFace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbCell()
{
  return m_mesh->nbCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbItem(eItemKind ik)
{
  return m_mesh->nbItem(ik);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup mesh::PolyhedralMesh::
allNodes()
{
  if (m_default_arcane_families[IK_Node])
    return m_default_arcane_families[IK_Node]->allItems();
  else
    return NodeGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EdgeGroup mesh::PolyhedralMesh::
allEdges()
{
  if (m_default_arcane_families[IK_Edge])
    return m_default_arcane_families[IK_Edge]->allItems();
  else
    return EdgeGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup mesh::PolyhedralMesh::
allFaces()
{
  if (m_default_arcane_families[IK_Face])
    return m_default_arcane_families[IK_Face]->allItems();
  else
    return FaceGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup mesh::PolyhedralMesh::
allCells()
{
  if (m_default_arcane_families[IK_Cell])
    return m_default_arcane_families[IK_Cell]->allItems();
  else
    return CellGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup mesh::PolyhedralMesh::
ownNodes()
{
  if (m_default_arcane_families[IK_Node])
    return m_default_arcane_families[IK_Node]->allItems().own();
  else
    return NodeGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EdgeGroup mesh::PolyhedralMesh::
ownEdges()
{
  if (m_default_arcane_families[IK_Edge])
    return m_default_arcane_families[IK_Edge]->allItems().own();
  else
    return EdgeGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup mesh::PolyhedralMesh::
ownFaces()
{
  if (m_default_arcane_families[IK_Face])
    return m_default_arcane_families[IK_Face]->allItems().own();
  else
    return FaceGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup mesh::PolyhedralMesh::
ownCells()
{
  if (m_default_arcane_families[IK_Cell])
    return m_default_arcane_families[IK_Cell]->allItems().own();
  else
    return CellGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup mesh::PolyhedralMesh::
outerFaces()
{
  if (m_default_arcane_families[IK_Cell])
    return m_default_arcane_families[IK_Cell]->allItems().outerFaceGroup();
  else
    return FaceGroup{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mesh::PolyhedralFamily* mesh::PolyhedralMesh::
_createItemFamily(eItemKind ik, const String& name)
{
  m_mesh->addFamily(ik, name);
  m_arcane_families.push_back(std::make_unique<PolyhedralFamily>(this, ik, name));
  auto current_family = m_arcane_families.back().get();
  if (m_default_arcane_families[ik] == nullptr) {
    m_default_arcane_families[ik] = current_family;
    _updateMeshInternalList(ik);
  }
  m_item_family_collection.add(current_family);
  return current_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
createItemFamily(eItemKind ik, const String& name)
{
  return _createItemFamily(ik, name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mesh::PolyhedralMesh::
_createUnitMesh()
{
  createItemFamily(IK_Cell, "CellFamily");
  createItemFamily(IK_Node, "NodeFamily");
  auto cell_family = m_default_arcane_families[IK_Cell];
  auto node_family = m_default_arcane_families[IK_Node];
  Int64UniqueArray cell_uids{ 0 }, node_uids{ 0, 1, 2, 3, 4, 5 };
  // todo add a cell_lids struct (containing future)
  PolyhedralMeshImpl::ItemLocalIds cell_lids, node_lids;
  m_mesh->scheduleAddItems(cell_family, cell_uids.constView(), cell_lids);
  m_mesh->scheduleAddItems(node_family, node_uids.constView(), node_lids);
  int nb_node = 6;
  m_mesh->scheduleAddConnectivity(cell_family, cell_lids, nb_node, node_family, node_uids, String{ "CellToNodes" });
  m_mesh->scheduleAddConnectivity(node_family, node_lids, 1, cell_family,
                                  Int64UniqueArray{ 0, 0, 0, 0, 0, 0 }, String{ "NodeToCells" });
  m_mesh->applyScheduledOperations();
  cell_family->endUpdate();
  node_family->endUpdate();
  endUpdate();
  // Mimic what IMeshModifier::endUpdate would do => default families are completed.
  // Families created after a first endUpdate call are not default families
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mesh::PolyhedralMesh::
endUpdate()
{
  // create empty default families not already created
  for (auto ik = 0; ik < NB_ITEM_KIND; ++ik) {
    if (m_default_arcane_families[ik] == nullptr && ik != eItemKind::IK_DoF) {
      String name = String::concat(itemKindName((eItemKind)ik), "EmptyFamily");
      m_empty_arcane_families[ik] = std::make_unique<mesh::PolyhedralFamily>(this, (eItemKind)ik, name);
      m_default_arcane_families[ik] = m_empty_arcane_families[ik].get();
      m_subdomain->traceMng()->info() << " Create empty family " << itemKindName((eItemKind)ik);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
nodeFamily()
{
  return m_default_arcane_families[IK_Node];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
edgeFamily()
{
  return m_default_arcane_families[IK_Edge];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
faceFamily()
{
  return m_default_arcane_families[IK_Face];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
cellFamily()
{
  return m_default_arcane_families[IK_Cell];
}

void mesh::PolyhedralMesh::
_updateMeshInternalList(eItemKind kind)
{
  switch (kind) {
  case IK_Cell:
    m_mesh_item_internal_list.cells = m_default_arcane_families[kind]->itemsInternal();
    m_mesh_item_internal_list._internalSetCellSharedInfo(m_default_arcane_families[kind]->commonItemSharedInfo());
    break;
  case IK_Face:
    m_mesh_item_internal_list.faces = m_default_arcane_families[kind]->itemsInternal();
    m_mesh_item_internal_list._internalSetFaceSharedInfo(m_default_arcane_families[kind]->commonItemSharedInfo());
    break;
  case IK_Edge:
    m_mesh_item_internal_list.edges = m_default_arcane_families[kind]->itemsInternal();
    m_mesh_item_internal_list._internalSetEdgeSharedInfo(m_default_arcane_families[kind]->commonItemSharedInfo());
    break;
  case IK_Node:
    m_mesh_item_internal_list.nodes = m_default_arcane_families[kind]->itemsInternal();
    m_mesh_item_internal_list._internalSetNodeSharedInfo(m_default_arcane_families[kind]->commonItemSharedInfo());
    break;
  case IK_DoF:
  case IK_Particle:
  case IK_Unknown:
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mesh::PolyhedralFamily* mesh::PolyhedralMesh::
_itemFamily(eItemKind ik)
{
  return m_default_arcane_families[ik];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
itemFamily(eItemKind ik)
{
  return _itemFamily(ik);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemTypeMng* mesh::PolyhedralMesh::
itemTypeMng() const
{
  return m_item_type_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mesh::PolyhedralFamily* mesh::PolyhedralMesh::
_findItemFamily(eItemKind ik, const String& name, bool create_if_needed)
{
  // Check if is a default family
  auto found_family = _itemFamily(ik);
  if (found_family) {
    if (found_family->name() == name)
      return found_family;
  }
  for (auto& family : m_arcane_families) {
    if (family->itemKind() == ik && family->name() == name)
      return family.get();
  }
  if (!create_if_needed)
    return nullptr;
  return _createItemFamily(ik, name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
findItemFamily(eItemKind ik, const String& name, bool create_if_needed, bool register_modifier_if_created)
{
  ARCANE_UNUSED(register_modifier_if_created); // IItemFamilyModifier not yet used in polyhedral mesh
  return _findItemFamily(ik, name, create_if_needed);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mesh::PolyhedralFamily* mesh::PolyhedralMesh::
arcaneDefaultFamily(eItemKind ik)
{
  return m_default_arcane_families[ik];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableNodeReal3& mesh::PolyhedralMesh::
nodesCoordinates()
{
  ARCANE_ASSERT(m_arcane_node_coords, ("Node coordinates not yet loaded."));
  return *m_arcane_node_coords;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup mesh::PolyhedralMesh::
findGroup(const String& name)
{
  ItemGroup group;
  for (auto& family : m_arcane_families) {
    group = family->findGroup(name);
    if (!group.null())
      return group;
  }
  return group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupCollection mesh::PolyhedralMesh::
groups()
{
  m_all_groups.clear();
  for (auto& family : m_arcane_families) {
    for (ItemGroupCollection::Enumerator i_group(family->groups()); ++i_group;)
      m_all_groups.add(*i_group);
  }
  return m_all_groups;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mesh::PolyhedralMesh::
destroyGroups()
{
  for (auto& family : m_arcane_families) {
    family->destroyGroups();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilyCollection mesh::PolyhedralMesh::
itemFamilies()
{
  return m_item_family_collection;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#else // ARCANE_HAS_POLYHEDRAL_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class PolyhedralMeshImpl
{};
} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::mesh::PolyhedralMesh::
~PolyhedralMesh() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::mesh::PolyhedralMesh::
PolyhedralMesh(ISubDomain* subdomain, const MeshBuildInfo& mbi)
: EmptyMesh{ subdomain->traceMng() }
, m_subdomain{ subdomain }
, m_mesh{ nullptr }
, m_mesh_kind(mbi.meshKind())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
read([[maybe_unused]] const String& filename)
{
  _errorEmptyMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
allocateItems(const Arcane::ItemAllocationInfo& item_allocation_info)
{
  ARCANE_UNUSED(item_allocation_info);
  _errorEmptyMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_HAS_POLYHEDRAL_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ARCANE_MESH_EXPORT PolyhedralMeshFactory
: public AbstractService
, public IMeshFactory
{
 public:

  explicit PolyhedralMeshFactory(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}

 public:

  void build() override {}
  IPrimaryMesh* createMesh(IMeshMng* mm, const MeshBuildInfo& build_info) override
  {
    ISubDomain* sd = mm->variableMng()->_internalApi()->internalSubDomain();
    return new mesh::PolyhedralMesh(sd, build_info);
  }

  static String name() { return "ArcanePolyhedralMeshFactory"; }
};

ARCANE_REGISTER_SERVICE(PolyhedralMeshFactory,
                        ServiceProperty(PolyhedralMeshFactory::name().localstr(), ST_Application),
                        ARCANE_SERVICE_INTERFACE(IMeshFactory));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if ARCANE_HAS_POLYHEDRAL_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String mesh::PolyhedralMesh::
factoryName() const
{
  return PolyhedralMeshFactory::name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_HAS_POLYHEDRAL_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
