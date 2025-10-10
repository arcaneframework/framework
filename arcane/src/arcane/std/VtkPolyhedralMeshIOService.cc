// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkPolyhedralMeshIOService                      (C) 2000-2025             */
/*                                                                           */
/* Read/write fools for polyhedral mesh with vtk file format                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include <iostream>
#include <numeric>
#include <functional>
#include <memory>
#include <iterator>

#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>
#include <vtkNew.h>
#include <vtkCellIterator.h>
#include <vtkIdTypeArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkDataSetAttributes.h>
#include <vtkArrayDispatch.h>
#include <vtkDataArrayAccessor.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkVersion.h>
#if VTK_VERSION_NUMBER >= 900000000
using vtkIdType_generic = vtkIdType const;
#else
using vtkIdType_generic = vtkIdType;
#endif

#include <arccore/base/Ref.h>
#include <arccore/base/String.h>
#include <arccore/base/FatalErrorException.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLPolyDataReader.h>

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IMeshInitialAllocator.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/Real3.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/core/MeshVariableScalarRef.h"
#include "arcane/core/MeshVariableArrayRef.h"

#include "arcane/core/ItemAllocationInfo.h"
#include "arcane/core/VariableBuildInfo.h"

#include "arcane/utils/OStringStream.h"

#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/internal/IVariableMngInternal.h"
#include "arcane/core/datatype/DataTypeTraits.h"

#include "arcane/std/VtkPolyhedralMeshIO_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace VtkPolyhedralTools
{
  struct ReadStatus
  {
    bool failure = false;
    String failure_message;
    String info_message;
  };

  struct PrintInfoLevel
  {
    bool print_mesh_info = false;
    bool print_debug_info = false;
  };
} // namespace VtkPolyhedralTools

class VtkPolyhedralMeshIOService
: public TraceAccessor
{
 public:

  explicit VtkPolyhedralMeshIOService(ITraceMng* trace_mng, VtkPolyhedralTools::PrintInfoLevel print_info_level)
  : TraceAccessor(trace_mng)
  , m_print_info_level(print_info_level)
  {}

  class VtkReader
  {

   public:

    explicit VtkReader(const String& filename, VtkPolyhedralTools::PrintInfoLevel print_info_level = VtkPolyhedralTools::PrintInfoLevel{});

    VtkReader() = default;

    static String supportedVtkExtensions() noexcept {return "vtk,vtu";};

    Int64ConstArrayView cellUids();
    Int64ConstArrayView nodeUids();
    Int64ConstArrayView faceUids();
    Int64ConstArrayView edgeUids();

    Integer nbNodes();

    Int64ConstArrayView cellNodes();
    Int32ConstArrayView cellNbNodes();

    Int64ConstArrayView faceNodes();
    Int32ConstArrayView faceNbNodes();

    Int64ConstArrayView faceNodesInFaceMesh();
    Int32ConstArrayView faceNbNodesInFaceMesh();

    Int64ConstArrayView edgeNodes();
    Int32ConstArrayView edgeNbNodes();

    Int64ConstArrayView faceCells();
    Int32ConstArrayView faceNbCells();

    Int32ConstArrayView edgeNbCells();
    Int64ConstArrayView edgeCells();

    Int32ConstArrayView cellNbFaces();
    Int64ConstArrayView cellFaces();

    Int32ConstArrayView edgeNbFaces();
    Int64ConstArrayView edgeFaces();

    Int32ConstArrayView cellNbEdges();
    Int64ConstArrayView cellEdges();

    Int32ConstArrayView faceNbEdges();
    Int64ConstArrayView faceEdges();

    Int32ConstArrayView nodeNbCells();
    Int64ConstArrayView nodeCells();

    Int32ConstArrayView nodeNbFaces();
    Int64ConstArrayView nodeFaces();

    Int32ConstArrayView nodeNbEdges();
    Int64ConstArrayView nodeEdges();

    Real3ArrayView nodeCoords();

    bool readHasFailed() const noexcept { return m_read_status.failure; }
    VtkPolyhedralTools::ReadStatus readStatus() const noexcept { return m_read_status; }

    vtkCellData* cellData();
    vtkPointData* pointData();
    vtkCellData* faceData();

    bool isEmpty() const noexcept { return m_is_empty; }
    bool doRead() const noexcept { return m_do_read; }

   private:

    bool m_is_empty = true;
    bool m_do_read = false;
    String m_filename;
    VtkPolyhedralTools::PrintInfoLevel m_print_info_level;
    VtkPolyhedralTools::ReadStatus m_read_status;
    vtkNew<vtkUnstructuredGridReader> m_vtk_grid_reader;
    vtkNew<vtkXMLUnstructuredGridReader> m_vtk_xml_grid_reader;
    vtkNew<vtkPolyDataReader> m_vtk_face_grid_reader;
    vtkNew<vtkXMLPolyDataReader> m_vtk_xml_face_grid_reader;
    vtkUnstructuredGrid* m_vtk_grid = nullptr;
    vtkPolyData* m_vtk_face_grid = nullptr;
    Int64UniqueArray m_cell_uids, m_node_uids, m_face_uids, m_edge_uids;
    Int64UniqueArray m_face_node_uids, m_edge_node_uids, m_cell_node_uids;
    Int64UniqueArray m_face_cell_uids, m_edge_cell_uids, m_edge_face_uids;
    Int64UniqueArray m_cell_face_uids, m_cell_edge_uids, m_face_edge_uids;
    Int64UniqueArray m_node_cell_uids, m_node_face_uids, m_node_edge_uids;
    Int32UniqueArray m_face_nb_nodes, m_cell_nb_nodes, m_face_nb_cells;
    Int32UniqueArray m_edge_nb_cells, m_edge_nb_faces, m_cell_nb_faces;
    Int32UniqueArray m_node_nb_cells, m_node_nb_faces, m_node_nb_edges;
    Int32UniqueArray m_cell_nb_edges, m_face_nb_edges, m_face_uid_indexes;
    Int32UniqueArray m_cell_face_indexes, m_edge_nb_nodes;
    Int64UniqueArray m_face_node_uids_in_face_mesh;
    Int32UniqueArray m_face_nb_nodes_in_face_mesh;
    using NodeUidToIndexMap = Int32UniqueArray; // use a map when no longer possible to index with node uids
    using FaceUidToIndexMap = Int32UniqueArray; // use a map when no longer possible to index with face uids
    using EdgeUidToIndexMap = Int32UniqueArray; // use a map when no longer possible to index with edge uids
    NodeUidToIndexMap m_node_uid_to_index;
    Real3UniqueArray m_node_coordinates;
    vtkCellData* m_cell_data = nullptr;
    vtkPointData* m_point_data = nullptr;
    vtkCellData* m_face_data = nullptr;
    vtkCellArray* m_poly_data = nullptr; // to store faces from face mesh file

    void _printMeshInfos() const;

    std::pair<bool, Int32> _findFace(const Int64UniqueArray& sorted_face_nodes, const UniqueArray<Int64UniqueArray>& node_face_uids, const NodeUidToIndexMap& node_uid_to_index, const Int32UniqueArray& face_nb_nodes, const FaceUidToIndexMap& face_uid_to_index, const UniqueArray<Int32>& face_offsets, const Int64UniqueArray& face_node_uids);
    template <typename Connectivity2DArray>
    static void _flattenConnectivity(Connectivity2DArray connected_item_2darray, Int32Span nb_connected_item_per_source_item, Int64UniqueArray& connected_item_array);
    void _readPlainTextVtkGrid(const String& filename);
    void _readXlmVtkGrid(const String& filename);
    void _checkVtkGrid() const;
    void _readPlainTextVtkFaceGrid(const String& faces_filename);
    void _readXmlVtkFaceGrid(const String& faces_filename);
    void _readfaceNodesInFaceMesh();
  };

 public:

  VtkPolyhedralTools::ReadStatus read(IPrimaryMesh* mesh, const String& filename, bool is_parallel_read)
  {
    ARCANE_CHECK_POINTER(mesh);
    // warning meaning of parallel_read not obvious :
    // if is parallel_read => master reads + broadcast, if not all ranks read a pre-distributed mesh
    bool do_read = is_parallel_read ? mesh->parallelMng()->isMasterIO() : true;
    using VtkReaderPtr = std::unique_ptr<VtkReader>;
    VtkReaderPtr reader = std::make_unique<VtkReader>();
    if (do_read) reader = std::make_unique<VtkReader>( filename, m_print_info_level );
    if (reader->readHasFailed())
      return reader->readStatus();
    ItemAllocationInfo item_allocation_info;
    _fillItemAllocationInfo(item_allocation_info, *reader);
    auto polyhedral_mesh_allocator = mesh->initialAllocator()->polyhedralMeshAllocator();
    polyhedral_mesh_allocator->allocateItems(item_allocation_info);
    _readVariablesAndGroups(mesh, *reader);
    return reader->readStatus();
  }

 private:

  VtkPolyhedralTools::PrintInfoLevel m_print_info_level;

  struct VariableInfo
  {
    eDataType m_type = DT_Unknown;
    bool is_array = false;
  };

  void _readVariablesAndGroups(IPrimaryMesh* mesh, VtkReader& reader);
  void _createGroup(vtkDataArray* group_items, const String& group_name, IPrimaryMesh* mesh, IItemFamily* item_family, Int32ConstSpan vtkToArcaneLid) const;
  VariableInfo _createVariable(vtkDataArray* item_values, const String& variable_name, IMesh* mesh, IItemFamily* item_family, Int32ConstSpan arcane_to_vtk_lids) const;

  static void _fillItemAllocationInfo(ItemAllocationInfo& item_allocation_info, VtkReader& vtk_reader);
  void _computeFaceVtkArcaneLidConversion(Int32Span face_vtk_to_arcane_lids, Int32Span arcane_to_vtk_lids, VtkPolyhedralMeshIOService::VtkReader& reader, IPrimaryMesh* mesh) const;
  void _createEmptyVariablesAndGroups(IMesh* mesh, XmlNode::const_reference xml_node) const;
  template <template <class> class  VariableRootType , template <class> class  ArrayVariableRootType>
  void _createEmptyVariables(IMesh* mesh, const XmlNodeList& cell_variables_node, eItemKind item_kind) const;
  void _createEmptyGroups(IMesh* mesh, const XmlNodeList& children, IItemFamily* item_family) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkPolyhedralCaseMeshReader
: public ArcaneVtkPolyhedralMeshIOObject
{
 public:

  class Builder : public IMeshBuilder
  {
   public:

    explicit Builder(ITraceMng* tm, const CaseMeshReaderReadInfo& read_info, VtkPolyhedralTools::PrintInfoLevel print_info_level)
    : m_trace_mng(tm)
    , m_read_info(read_info)
    , m_print_info_level(print_info_level)
    {}

    void fillMeshBuildInfo(MeshBuildInfo& build_info) override
    {
      build_info.addFactoryName("ArcanePolyhedralMeshFactory");
      build_info.addNeedPartitioning(false);
      MeshKind mk;
      mk.setMeshStructure(eMeshStructure::Polyhedral);
      build_info.addMeshKind(mk);
    }

    void allocateMeshItems(IPrimaryMesh* pm) override
    {
      ARCANE_CHECK_POINTER(pm);
      m_trace_mng->info() << "---Create Polyhedral mesh: " << pm->name() << "---";
      m_trace_mng->info() << "--Read mesh file " << m_read_info.fileName();
      VtkPolyhedralMeshIOService polyhedral_vtk_service{ m_trace_mng, m_print_info_level };
      auto read_status = polyhedral_vtk_service.read(pm, m_read_info.fileName(),m_read_info.isParallelRead());
      if (read_status.failure)
        ARCANE_FATAL(read_status.failure_message);
      m_trace_mng->info() << read_status.info_message;
    }

   private:

    ITraceMng* m_trace_mng;
    CaseMeshReaderReadInfo m_read_info;
    VtkPolyhedralTools::PrintInfoLevel m_print_info_level;
  };

  explicit VtkPolyhedralCaseMeshReader(const ServiceBuildInfo& sbi)
  : ArcaneVtkPolyhedralMeshIOObject(sbi)
  {}

 public:

  Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const override
  {
    IMeshBuilder* builder = nullptr;
    if (VtkPolyhedralMeshIOService::VtkReader::supportedVtkExtensions().contains(read_info.format()))
      builder = new Builder(traceMng(), read_info, VtkPolyhedralTools::PrintInfoLevel{ options()->getPrintMeshInfos(), options()->getPrintDebugInfos() });
    return makeRef(builder);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(VtkPolyhedralCaseMeshReader,
                        ServiceProperty("VtkPolyhedralCaseMeshReader", ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(ICaseMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::
_fillItemAllocationInfo(ItemAllocationInfo& item_allocation_info, VtkReader& vtk_reader)
{
  auto nb_item_family = 4;
  auto nb_connected_family = 3;
  item_allocation_info.family_infos.resize(nb_item_family);
  for (auto& family_info : item_allocation_info.family_infos) {
    family_info.connected_family_infos.resize(nb_connected_family);
  }
  // Create regular item families and connectivities
  auto& cell_family_info = item_allocation_info.family_infos[0];
  cell_family_info.name = "Cell";
  cell_family_info.item_kind = IK_Cell;
  cell_family_info.item_uids = vtk_reader.cellUids();
  auto& node_family_info = item_allocation_info.family_infos[1];
  node_family_info.name = "Node";
  node_family_info.item_kind = IK_Node;
  node_family_info.item_uids = vtk_reader.nodeUids();
  auto& face_family_info = item_allocation_info.family_infos[2];
  face_family_info.name = "Face";
  face_family_info.item_kind = IK_Face;
  face_family_info.item_uids = vtk_reader.faceUids();
  auto& edge_family_info = item_allocation_info.family_infos[3];
  edge_family_info.name = "Edge";
  edge_family_info.item_kind = IK_Edge;
  edge_family_info.item_uids = vtk_reader.edgeUids();
  // Cell to nodes connectivity
  auto cell_connected_family_index = 0;
  auto& cell_connected_node_family_info = cell_family_info.connected_family_infos[cell_connected_family_index++];
  cell_connected_node_family_info.name = node_family_info.name;
  cell_connected_node_family_info.item_kind = node_family_info.item_kind;
  cell_connected_node_family_info.connectivity_name = "CellToNodes";
  cell_connected_node_family_info.nb_connected_items_per_item = vtk_reader.cellNbNodes();
  cell_connected_node_family_info.connected_items_uids = vtk_reader.cellNodes();
  // Cell to faces connectivity
  auto& cell_connected_face_family_info = cell_family_info.connected_family_infos[cell_connected_family_index++];
  cell_connected_face_family_info.name = face_family_info.name;
  cell_connected_face_family_info.item_kind = face_family_info.item_kind;
  cell_connected_face_family_info.connectivity_name = "CellToFaces";
  cell_connected_face_family_info.nb_connected_items_per_item = vtk_reader.cellNbFaces();
  cell_connected_face_family_info.connected_items_uids = vtk_reader.cellFaces();
  // Cell to edges connectivity
  auto& cell_connected_edge_family_info = cell_family_info.connected_family_infos[cell_connected_family_index++];
  cell_connected_edge_family_info.name = edge_family_info.name;
  cell_connected_edge_family_info.item_kind = edge_family_info.item_kind;
  cell_connected_edge_family_info.connectivity_name = "CellToEdges";
  cell_connected_edge_family_info.nb_connected_items_per_item = vtk_reader.cellNbEdges();
  cell_connected_edge_family_info.connected_items_uids = vtk_reader.cellEdges();
  // Face to cells connectivity
  auto face_connected_family_index = 0;
  auto& face_connected_cell_family_info = face_family_info.connected_family_infos[face_connected_family_index++];
  face_connected_cell_family_info.name = cell_family_info.name;
  face_connected_cell_family_info.item_kind = cell_family_info.item_kind;
  face_connected_cell_family_info.connectivity_name = "FaceToCells";
  face_connected_cell_family_info.nb_connected_items_per_item = vtk_reader.faceNbCells();
  face_connected_cell_family_info.connected_items_uids = vtk_reader.faceCells();
  // Face to nodes connectivity
  auto& face_connected_node_family_info = face_family_info.connected_family_infos[face_connected_family_index++];
  face_connected_node_family_info.name = node_family_info.name;
  face_connected_node_family_info.item_kind = node_family_info.item_kind;
  face_connected_node_family_info.connectivity_name = "FaceToNodes";
  face_connected_node_family_info.nb_connected_items_per_item = vtk_reader.faceNbNodes();
  face_connected_node_family_info.connected_items_uids = vtk_reader.faceNodes();
  // Face to edges connectivity
  auto& face_connected_edge_family_info = face_family_info.connected_family_infos[face_connected_family_index];
  face_connected_edge_family_info.name = edge_family_info.name;
  face_connected_edge_family_info.item_kind = edge_family_info.item_kind;
  face_connected_edge_family_info.connectivity_name = "FaceToEdges";
  face_connected_edge_family_info.nb_connected_items_per_item = vtk_reader.faceNbEdges();
  face_connected_edge_family_info.connected_items_uids = vtk_reader.faceEdges();
  // Edge to cells connectivity
  auto edge_connected_family_index = 0;
  auto& edge_connected_cell_family_info = edge_family_info.connected_family_infos[edge_connected_family_index++];
  edge_connected_cell_family_info.name = cell_family_info.name;
  edge_connected_cell_family_info.item_kind = cell_family_info.item_kind;
  edge_connected_cell_family_info.connectivity_name = "EdgeToCells";
  edge_connected_cell_family_info.nb_connected_items_per_item = vtk_reader.edgeNbCells();
  edge_connected_cell_family_info.connected_items_uids = vtk_reader.edgeCells();
  // Edge to faces connectivity
  auto& edge_connected_face_family_info = edge_family_info.connected_family_infos[edge_connected_family_index++];
  edge_connected_face_family_info.name = face_family_info.name;
  edge_connected_face_family_info.item_kind = face_family_info.item_kind;
  edge_connected_face_family_info.connectivity_name = "EdgeToFaces";
  edge_connected_face_family_info.nb_connected_items_per_item = vtk_reader.edgeNbFaces();
  edge_connected_face_family_info.connected_items_uids = vtk_reader.edgeFaces();
  // Edge to nodes connectivity
  auto& edge_connected_node_family_info = edge_family_info.connected_family_infos[edge_connected_family_index++];
  edge_connected_node_family_info.name = node_family_info.name;
  edge_connected_node_family_info.item_kind = node_family_info.item_kind;
  edge_connected_node_family_info.connectivity_name = "EdgeToNodes";
  edge_connected_node_family_info.nb_connected_items_per_item = vtk_reader.edgeNbNodes();
  edge_connected_node_family_info.connected_items_uids = vtk_reader.edgeNodes();
  // Node to cells connectivity
  auto node_connected_family_index = 0;
  auto& node_connected_cell_family_info = node_family_info.connected_family_infos[node_connected_family_index++];
  node_connected_cell_family_info.name = cell_family_info.name;
  node_connected_cell_family_info.item_kind = cell_family_info.item_kind;
  node_connected_cell_family_info.connectivity_name = "NodeToCells";
  node_connected_cell_family_info.nb_connected_items_per_item = vtk_reader.nodeNbCells();
  node_connected_cell_family_info.connected_items_uids = vtk_reader.nodeCells();
  // Node to faces connectivity
  auto& node_connected_face_family_info = node_family_info.connected_family_infos[node_connected_family_index++];
  node_connected_face_family_info.name = face_family_info.name;
  node_connected_face_family_info.item_kind = face_family_info.item_kind;
  node_connected_face_family_info.connectivity_name = "NodeToFaces";
  node_connected_face_family_info.nb_connected_items_per_item = vtk_reader.nodeNbFaces();
  node_connected_face_family_info.connected_items_uids = vtk_reader.nodeFaces();
  // Node to edges connectivity
  auto& node_connected_edge_family_info = node_family_info.connected_family_infos[node_connected_family_index++];
  node_connected_edge_family_info.name = edge_family_info.name;
  node_connected_edge_family_info.item_kind = edge_family_info.item_kind;
  node_connected_edge_family_info.connectivity_name = "NodeToEdges";
  node_connected_edge_family_info.nb_connected_items_per_item = vtk_reader.nodeNbEdges();
  node_connected_edge_family_info.connected_items_uids = vtk_reader.nodeEdges();
  // Node coordinates
  node_family_info.item_coordinates_variable_name = "NodeCoord";
  node_family_info.item_coordinates = vtk_reader.nodeCoords();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::
_readVariablesAndGroups(IPrimaryMesh* mesh, VtkReader& reader)
{
  // Register variable and group info in an xml for non reader ranks
  OStringStream created_infos_str;
  created_infos_str() << "<?xml version='1.0' ?>\n";
  created_infos_str() << "<infos>";
  // Cell data
  if (auto* cell_data = reader.cellData(); cell_data) {
    // Read cell groups and variables
    Int32SharedArray vtk_to_arcane_lids(mesh->cellFamily()->nbItem());
    std::iota(vtk_to_arcane_lids.begin(), vtk_to_arcane_lids.end(), 0);
    Int32SharedArray arcane_to_vtk_lids{ vtk_to_arcane_lids };
    for (auto array_index = 0; array_index < cell_data->GetNumberOfArrays(); ++array_index) {
      auto* cell_array = cell_data->GetArray(array_index);
      if (!cell_array)
        continue;
      if (String name = cell_array->GetName(); name.substring(0, 6) == "GROUP_") {
        _createGroup(cell_array, name.substring(6), mesh, mesh->cellFamily(), vtk_to_arcane_lids.constSpan());
        created_infos_str() << "<cell-group name='" << name.substring(6) << "'/>";
      }
      else {
        auto var_info = _createVariable(cell_array, name, mesh, mesh->cellFamily(), arcane_to_vtk_lids);
        created_infos_str() << "<cell-variable name='" << name << "' "
                            << " data-type='" << dataTypeName(var_info.m_type) << "' "
                            << " is_array='" << std::boolalpha << var_info.is_array << "'/>";
      }
      if (m_print_info_level.print_debug_info) {
        debug(Trace::High) << "Reading property " << cell_array->GetName();
        for (auto tuple_index = 0; tuple_index < cell_array->GetNumberOfTuples(); ++tuple_index) {
          for (auto component_index = 0; component_index < cell_array->GetNumberOfComponents(); ++component_index) {
            debug(Trace::High) << cell_array->GetName() << "[" << tuple_index << "][" << component_index << "] = " << cell_array->GetComponent(tuple_index, component_index);
          }
        }
      }
    }
  }
  // Node data
  if (auto* point_data = reader.pointData(); point_data) {
    // Read node groups and variables
    Int32SharedArray vtk_to_arcane_lids(mesh->nodeFamily()->nbItem());
    std::iota(vtk_to_arcane_lids.begin(), vtk_to_arcane_lids.end(), 0);
    Int32SharedArray arcane_to_vtk_lids{ vtk_to_arcane_lids };
    for (auto array_index = 0; array_index < point_data->GetNumberOfArrays(); ++array_index) {
      auto* point_array = point_data->GetArray(array_index);
      if (String name = point_array->GetName(); name.substring(0, 6) == "GROUP_") {
        _createGroup(point_array, name.substring(6), mesh, mesh->nodeFamily(), vtk_to_arcane_lids.constSpan());
        created_infos_str() << "<node-group name='" << name.substring(6) << "'/>";
      }
      else {
        auto var_info = _createVariable(point_array, name, mesh, mesh->nodeFamily(), arcane_to_vtk_lids);
        created_infos_str() << "<node-variable name='" << name << "' "
                            << " data-type='" << dataTypeName(var_info.m_type) << "' "
                            << " is_array='" << std::boolalpha << var_info.is_array << "'/>";
      }
      if (m_print_info_level.print_debug_info) {
        debug(Trace::High) << "Reading property " << point_array->GetName();
        for (auto tuple_index = 0; tuple_index < point_array->GetNumberOfTuples(); ++tuple_index) {
          for (auto component_index = 0; component_index < point_array->GetNumberOfComponents(); ++component_index) {
            debug(Trace::High) << point_array->GetName() << "[" << tuple_index << "][" << component_index << "] = " << point_array->GetComponent(tuple_index, component_index);
          }
        }
      }
    }
  }
  // Face data
  if (auto* face_data = reader.faceData(); face_data) {
    // Read face groups and variables
    Int32UniqueArray vtk_to_Arcane_lids(mesh->faceFamily()->nbItem());
    Int32UniqueArray arcane_to_vtk_lids(mesh->faceFamily()->nbItem());
    _computeFaceVtkArcaneLidConversion(vtk_to_Arcane_lids, arcane_to_vtk_lids, reader, mesh);
    for (auto array_index = 0; array_index < face_data->GetNumberOfArrays(); ++array_index) {
      auto* face_array = face_data->GetArray(array_index);
      if (String name = face_array->GetName(); name.substring(0, 6) == "GROUP_") {
        _createGroup(face_array, name.substring(6), mesh, mesh->faceFamily(), vtk_to_Arcane_lids);
        created_infos_str() << "<face-group name='" << name.substring(6) << "'/>";
      }
      else {
        auto var_info = _createVariable(face_array, name, mesh, mesh->faceFamily(), arcane_to_vtk_lids);
        created_infos_str() << "<face-variable name='" << name << "' "
                            << " data-type='" << dataTypeName(var_info.m_type) << "' "
                            << " is_array='" << std::boolalpha << var_info.is_array << "'/>";
      }
      if (m_print_info_level.print_debug_info) {
        debug(Trace::High) << "Reading property " << face_array->GetName();
        for (auto tuple_index = 0; tuple_index < face_array->GetNumberOfTuples(); ++tuple_index) {
          for (auto component_index = 0; component_index < face_array->GetNumberOfComponents(); ++component_index) {
            debug(Trace::High) << face_array->GetName() << "[" << tuple_index << "][" << component_index << "] = " << face_array->GetComponent(tuple_index, component_index);
          }
        }
      }
    }
  }
  created_infos_str() << "</infos>";
  // Create empty group and variables in non reader subdomain
  ByteUniqueArray bytes;
  auto* pm = mesh->parallelMng();
  if (!reader.isEmpty() && pm->isMasterIO()) {
    String str = created_infos_str.str();
    ByteConstArrayView bv = str.utf8();
    Integer len = bv.size();
    bytes.resize(len + 1);
    bytes.copy(bv);
  }
  pm->broadcastMemoryBuffer(bytes, pm->masterIORank());
  if (reader.isEmpty()) {
    std::unique_ptr<IXmlDocumentHolder> doc(IXmlDocumentHolder::loadFromBuffer(bytes, "InternalBuffer", traceMng()));
    XmlNode doc_node = doc->documentNode();
    _createEmptyVariablesAndGroups(mesh,doc_node);

  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::
_createGroup(vtkDataArray* group_items, const String& group_name, IPrimaryMesh* mesh, IItemFamily* item_family, Int32ConstSpan vtkToArcaneLid) const
{
  ARCANE_CHECK_POINTER(group_items);
  ARCANE_CHECK_POINTER(mesh);
  ARCANE_CHECK_POINTER(item_family);
  if (group_items->GetNumberOfComponents() != 1)
    fatal() << String::format("Cannot create item group {0}. Group information in data property must be a scalar", group_name);
  debug() << "Create group " << group_name;
  Int32UniqueArray arcane_lids;
  arcane_lids.reserve((int)group_items->GetNumberOfValues());
  using GroupDispatcher = vtkArrayDispatch::DispatchByValueType<vtkArrayDispatch::Integrals>;
  auto group_creator = [&arcane_lids, &vtkToArcaneLid](auto* array) {
    vtkIdType numTuples = array->GetNumberOfTuples();
    vtkDataArrayAccessor<std::remove_pointer_t<decltype(array)>> array_accessor{ array };
    auto local_id = 0;
    for (vtkIdType tupleIdx = 0; tupleIdx < numTuples; ++tupleIdx) {
      auto value = array_accessor.Get(tupleIdx, 0);
      if (value)
        arcane_lids.push_back(vtkToArcaneLid[local_id]);
      ++local_id;
    }
  };
  if (!GroupDispatcher::Execute(group_items, group_creator))
    ARCANE_FATAL("Cannot create item group {0}. Group information in data property must be an integral type", group_name);
  debug() << " local ids for item group " << group_name << "  " << arcane_lids; // to remove
  item_family->createGroup(group_name, arcane_lids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename vtkType>
struct ToArcaneType
{
  using type = vtkType;
};

template <>
struct ToArcaneType<float>
{
  using type = Real;
};

template <>
struct ToArcaneType<long long>
{
  using type = Int64;
};

template <typename T> using to_arcane_type_t = typename ToArcaneType<T>::type;
/*---------------------------------------------------------------------------*/

VtkPolyhedralMeshIOService::VariableInfo VtkPolyhedralMeshIOService::
_createVariable(vtkDataArray* item_values, const String& variable_name, IMesh* mesh, IItemFamily* item_family, Int32ConstSpan arcane_to_vtk_lids) const
{
  ARCANE_CHECK_POINTER(item_values);
  ARCANE_CHECK_POINTER(mesh);
  ARCANE_CHECK_POINTER(item_family);
  if (item_values->GetNumberOfTuples() != item_family->nbItem())
    ARCANE_FATAL("Cannot create variable {0}, {1} values are given for {2} items in {3} family",
                 variable_name, item_values->GetNumberOfTuples(), item_family->nbItem(), item_family->name());
  debug() << "Create mesh variable " << variable_name;
  VariableInfo var_info;
  auto variable_creator = [mesh, variable_name, item_family, arcane_to_vtk_lids, this, &var_info](auto* values) {
    VariableBuildInfo vbi{ mesh, variable_name };
    using ValueType = typename std::remove_pointer_t<decltype(values)>::ValueType;
    auto* var = new ItemVariableScalarRefT<to_arcane_type_t<ValueType>>{ vbi, item_family->itemKind() };
    mesh->variableMng()->_internalApi()->addAutoDestroyVariable(var);
    vtkDataArrayAccessor<std::remove_pointer_t<decltype(values)>> values_accessor{ values };
    ENUMERATE_ITEM (item, item_family->allItems()) {
      (*var)[item] = (to_arcane_type_t<ValueType>)values_accessor.Get(arcane_to_vtk_lids[item.localId()], 0);
    }
    var_info.m_type = DataTypeTraitsT<to_arcane_type_t<ValueType>>::type();
    var_info.is_array = false;
  };
  auto array_variable_creator = [mesh, variable_name, item_family, arcane_to_vtk_lids, this, &var_info](auto* values) {
    VariableBuildInfo vbi{ mesh, variable_name };
    using ValueType = typename std::remove_pointer_t<decltype(values)>::ValueType;
    auto* var = new ItemVariableArrayRefT<to_arcane_type_t<ValueType>>{ vbi, item_family->itemKind() };
    mesh->variableMng()->_internalApi()->addAutoDestroyVariable(var);
    vtkDataArrayAccessor<std::remove_pointer_t<decltype(values)>> values_accessor{ values };
    var->resize(values->GetNumberOfComponents());
    ENUMERATE_ITEM (item, item_family->allItems()) {
      auto index = 0;
      for (auto& var_value : (*var)[item]) {
        var_value = (to_arcane_type_t<ValueType>)values_accessor.Get(arcane_to_vtk_lids[item.localId()], index++);
      }
    }
    var_info.m_type = DataTypeTraitsT<to_arcane_type_t<ValueType>>::type();
    var_info.is_array = true;
  };
  // Restrict to int and real values
  using ValueTypes = vtkTypeList_Create_6(double, float, int, long, long long, short);
  using ArrayDispatcher = vtkArrayDispatch::DispatchByValueType<ValueTypes>;
  // Create ScalarVariable
  bool is_variable_created = false;
  if (item_values->GetNumberOfComponents() == 1) {
    is_variable_created = ArrayDispatcher::Execute(item_values, variable_creator);
  }
  // Create ArrayVariable
  else { // ArrayVariable
    is_variable_created = ArrayDispatcher::Execute(item_values, array_variable_creator);
  }
  if (!is_variable_created)
    ARCANE_FATAL("Cannot create variable {0}, it's data type is not supported. Only real and integral types are supported", variable_name);
  return var_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::
_computeFaceVtkArcaneLidConversion(Int32Span face_vtk_to_arcane_lids, Int32Span arcane_to_vtk_lids, VtkPolyhedralMeshIOService::VtkReader& reader, IPrimaryMesh* mesh) const
{
  auto face_nodes_unique_ids = reader.faceNodesInFaceMesh();
  auto face_nb_nodes = reader.faceNbNodesInFaceMesh();
  auto current_face_index = 0;
  auto current_face_index_in_face_nodes = 0;
  Int32UniqueArray face_nodes_local_ids(face_nodes_unique_ids.size());
  mesh->nodeFamily()->itemsUniqueIdToLocalId(face_nodes_local_ids, face_nodes_unique_ids);
  for (auto current_face_nb_node : face_nb_nodes) {
    auto current_face_nodes = face_nodes_local_ids.subConstView(current_face_index_in_face_nodes, current_face_nb_node);
    current_face_index_in_face_nodes += current_face_nb_node;
    Node face_first_node{ mesh->nodeFamily()->view()[current_face_nodes[0]] };
    face_vtk_to_arcane_lids[current_face_index] = MeshUtils::getFaceFromNodesLocalId(face_first_node, current_face_nodes).localId();
    ++current_face_index;
  }
  auto vtk_lid = 0;
  for (auto arcane_lid : face_vtk_to_arcane_lids) {
    arcane_to_vtk_lids[arcane_lid] = vtk_lid;
    ++vtk_lid;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <template <class> class  VariableRootType>
VariableRef* _createVar(IMesh* mesh, const String& var_name, const String& var_data_type_name, eItemKind var_kind)
{
  bool has_error = false;
  eDataType var_data_type = dataTypeFromName(var_data_type_name.localstr(),has_error);
  if (has_error) {
    ARCANE_FATAL("Invalid data type name {0} for Variable creation in VtkPolyhedralMeshIOService");
  }
  switch (var_data_type){
  case DT_Int32:
    return new VariableRootType<Int32>{VariableBuildInfo{mesh,var_name},var_kind};
    break;
  case DT_Int64:
    return new VariableRootType<Int64>{VariableBuildInfo{mesh,var_name},var_kind};
    break;
  case DT_Real:
    return new VariableRootType<Real>{VariableBuildInfo{mesh,var_name},var_kind};
    break;
  default :
    ARCANE_FATAL("Handle only DT_Int32, DT_Int64, DT_Real in VtkPolyhedralMeshIOService");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <template <class> class  VariableRootType , template <class> class  ArrayVariableRootType>
void VtkPolyhedralMeshIOService::
_createEmptyVariables(IMesh* mesh, const XmlNodeList& item_variables_node, eItemKind item_kind) const
{
  ARCANE_CHECK_PTR(mesh);
  {
    for (XmlNode xnode : item_variables_node) {
      String name = xnode.attrValue("name");
      debug() << "Create mesh variable: " << name;
      String data_type_name = xnode.attrValue("data-type");
      bool is_array = xnode.attrValue("is_array") == "true";
      VariableRef* var = nullptr;
      if (is_array) {
        var = _createVar<ArrayVariableRootType>(mesh,name,data_type_name,item_kind);
      }
      else {
        var = _createVar<VariableRootType>(mesh,name, data_type_name,item_kind);
      }
      mesh->variableMng()->_internalApi()->addAutoDestroyVariable(var);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::
_createEmptyGroups(IMesh* mesh, const XmlNodeList& groups_node, IItemFamily* item_family) const
{
  for (XmlNode xnode : groups_node) {
    String name = xnode.attrValue("name");
    info() << "Building group: " << name;
    item_family->createGroup(name);
  }
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::
_createEmptyVariablesAndGroups(IMesh* mesh, XmlNode::const_reference variable_and_group_info) const
{
  ARCANE_CHECK_PTR(mesh);
  auto document_node = variable_and_group_info.documentElement();
  _createEmptyVariables<ItemVariableScalarRefT,ItemVariableArrayRefT>(mesh, document_node.children("cell-variable"), IK_Cell);
  _createEmptyVariables<ItemVariableScalarRefT,ItemVariableArrayRefT>(mesh, document_node.children("node-variable"), IK_Node);
  _createEmptyVariables<ItemVariableScalarRefT,ItemVariableArrayRefT>(mesh, document_node.children("face-variable"), IK_Face);

  _createEmptyGroups(mesh, document_node.children("cell-group"), mesh->itemFamily(IK_Cell));
  _createEmptyGroups(mesh, document_node.children("node-group"), mesh->itemFamily(IK_Node));
  _createEmptyGroups(mesh, document_node.children("face-group"), mesh->itemFamily(IK_Face));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VtkPolyhedralMeshIOService::VtkReader::
VtkReader(const String& filename, VtkPolyhedralTools::PrintInfoLevel print_info_level)
: m_filename{ filename }
, m_print_info_level{ print_info_level }
{
  m_is_empty = false;
  m_do_read = true;
  if (filename.empty()) {
    m_read_status.failure = true;
    m_read_status.failure_message = "filename for polyhedral vtk mesh is empty.";
    return;
  }
  if (filename.endsWith("vtk"))
    _readPlainTextVtkGrid(filename);
  else if (filename.endsWith("vtu"))
    _readXlmVtkGrid(filename);
  else {
    m_read_status.failure = true;
    m_read_status.failure_message = String::format("Unsupported vtk extension for file {0}. Supported vtk extension for Polyhedral meshes are {1}",
                                                   filename,VtkReader::supportedVtkExtensions());
  }
  // Check vtk grid exists and not empty
  if (!m_vtk_grid) {
    m_read_status.failure = true;
    m_read_status.failure_message = String::format("Cannot read vtk polyhedral file {0}. Vtk grid was not created.", filename);
    return;
  }
  if (m_vtk_grid->GetNumberOfCells() == 0) {
    m_read_status.failure = true;
    m_read_status.failure_message = String::format("Cannot read vtk polyhedral file {0}. No cells were found.", filename);
    return;
  }
  if (!m_vtk_grid->GetFaces()) {
    m_read_status.failure = true;
    m_read_status.failure_message = String::format("The given mesh vtk file {0} is not a polyhedral mesh, cannot read it", filename);
    return;
  }

  m_cell_data = m_vtk_grid->GetCellData();
  m_point_data = m_vtk_grid->GetPointData();

  // Read face info (for variables and groups) if present
  String faces_filename = m_filename + "faces.vtk";
  std::ifstream ifile(faces_filename.localstr());
  if (ifile) {
    _readPlainTextVtkFaceGrid(faces_filename);
  }
  else{
    faces_filename = m_filename + "faces.vtp";
    ifile = std::ifstream{ faces_filename.localstr() };
    if (ifile) {
      _readXmlVtkFaceGrid(faces_filename);
    }
  }

  StringUniqueArray faces_filename_and_extension;
  faces_filename.split(faces_filename_and_extension,'.');

  if (!ifile){
    m_read_status.info_message = String::format("Information no face mesh given {0}{1} (.vtk or .vtp) to define face variables or groups on faces.",
                                                faces_filename_and_extension[0],
                                                faces_filename_and_extension[1]);
  }
  else {
    if (m_vtk_face_grid) { // Check face vtk grid exists and not empty
      if (m_vtk_face_grid->GetNumberOfCells() == 0) {
        m_read_status.failure = true;
        m_read_status.failure_message = m_read_status.failure_message + String::format(" Error in reading face information for groups in mesh file {0} ", faces_filename);
      }
      else {
        m_face_data = m_vtk_face_grid->GetCellData();
        m_poly_data = m_vtk_face_grid->GetPolys();
      }
    }
    else{
      m_read_status.failure = true;
      m_read_status.failure_message = m_read_status.failure_message + String::format("Face data could not be built from file {0}{1} (.vtk or .vtu).",
                                                                                     faces_filename_and_extension[0],
                                                                                     faces_filename_and_extension[1]);
    }
  }

  if (m_print_info_level.print_mesh_info)
    _printMeshInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
cellUids()
{
  if (!doRead()) return m_cell_uids;
  if (m_cell_uids.empty()) {
    _checkVtkGrid();
    m_cell_uids.reserve(m_vtk_grid->GetNumberOfCells());
    m_cell_nb_nodes.reserve(m_vtk_grid->GetNumberOfCells());
    m_cell_node_uids.reserve(10 * m_vtk_grid->GetNumberOfCells()); // take a mean of 10 nodes per cell
    auto* cell_iter = m_vtk_grid->NewCellIterator();
    cell_iter->InitTraversal();
    while (!cell_iter->IsDoneWithTraversal()) {
      m_cell_uids.push_back(cell_iter->GetCellId());
      m_cell_nb_nodes.push_back(Integer(cell_iter->GetNumberOfPoints()));
      ArrayView<vtkIdType> cell_nodes{ Integer(cell_iter->GetNumberOfPoints()), cell_iter->GetPointIds()->GetPointer(0) };
      std::for_each(cell_nodes.begin(), cell_nodes.end(), [this](auto uid) { this->m_cell_node_uids.push_back(uid); });
      cell_iter->GoToNextCell();
    }
  }
  return m_cell_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
nodeUids()
{
  if (!doRead()) return m_node_uids;
  if (m_node_uids.empty()) {
    _checkVtkGrid();
    auto nb_nodes = m_vtk_grid->GetNumberOfPoints();
    m_node_uids.resize(nb_nodes);
    m_node_nb_cells.resize(nb_nodes);
    m_node_cell_uids.reserve(8 * nb_nodes);
    m_node_uid_to_index.resize(nb_nodes);
    for (int node_index = 0; node_index < nb_nodes; ++node_index) {
      Int64 node_uid = node_index;
      m_node_uids[node_index] = node_uid;
      auto cell_nodes = vtkIdList::New();
      m_vtk_grid->GetPointCells(node_index, cell_nodes);
      Int64Span cell_nodes_view((Int64*)cell_nodes->GetPointer(0), cell_nodes->GetNumberOfIds());
      m_node_cell_uids.addRange(cell_nodes_view);
      m_node_nb_cells[node_index] = (Int32)cell_nodes->GetNumberOfIds();
      // uid and index might differ in the future (ex parallel read).
      // This structure is created to be used in faceUids. Should be a map if node_uid is no longer the node index
      m_node_uid_to_index[node_uid] = node_index;
    }
  }
  return m_node_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceUids()
{
  // Needs nodeUids to be called first (no work done if already called)
  nodeUids();

  if (!doRead()) return m_face_uids;
  if (!m_face_uids.empty())
    return m_face_uids;
  _checkVtkGrid();
  auto* cell_iter = m_vtk_grid->NewCellIterator();
  cell_iter->InitTraversal();
  vtkIdType nb_face_estimation = 0;
  while (!cell_iter->IsDoneWithTraversal()) {
    vtkIdType cell_nb_faces = 0;
    vtkIdType_generic * points{ nullptr };
    m_vtk_grid->GetFaceStream(cell_iter->GetCellId(), cell_nb_faces, points);
    nb_face_estimation += cell_nb_faces;
    cell_iter->GoToNextCell();
  }
  m_face_uids.reserve(nb_face_estimation);
  auto const* faces = m_vtk_grid->GetFaces();
  // This array contains the face info per cells (cf. vtk file)
  // first_cell_nb_faces first_cell_first_face_nb_nodes first_cell_first_face_node_1 ... first_cell_first_face_node_n first_cell_second_face_nb_nodes etc

  if (!faces) {
    ARCANE_FATAL("Mesh {0} is not polyhedral: faces are not defined", m_filename);
  }
  Int64 face_uid = 0;
  auto face_info_size = faces->GetNumberOfValues();
  m_face_node_uids.reserve(face_info_size);
  m_face_nb_nodes.reserve(nb_face_estimation);
  m_face_cell_uids.reserve(2 * nb_face_estimation);
  m_face_nb_cells.reserve(nb_face_estimation);
  m_cell_face_uids.reserve(8 * m_cell_uids.size()); // take a mean of 8 faces per cell
  m_cell_nb_faces.resize(m_cell_uids.size(), 0);
  m_cell_face_indexes.resize(m_cell_uids.size(), -1);
  m_face_uid_indexes.resize(2 * nb_face_estimation, -1);
  Int64UniqueArray current_face_nodes, sorted_current_face_nodes;
  current_face_nodes.reserve(10);
  sorted_current_face_nodes.reserve(10);
  UniqueArray<UniqueArray<Int64>> node_faces(m_node_uids.size());
  UniqueArray<Int32> face_offsets;
  face_offsets.reserve(nb_face_estimation);
  face_offsets.push_back(0);
  FaceUidToIndexMap face_uid_to_index;
  face_uid_to_index.reserve(nb_face_estimation);
  auto cell_index = 0;
  auto cell_face_index = 0;
  auto global_face_index = 0;
  auto face_uid_index = 0;
  for (int face_info_index = 0; face_info_index < face_info_size; cell_index++) { // face data are given by cell
    auto current_cell_nb_faces = Int32(faces->GetValue(face_info_index++));
    m_cell_face_indexes[m_cell_uids[cell_index]] = cell_face_index;
    for (auto face_index = 0; face_index < current_cell_nb_faces; ++face_index, ++global_face_index) {
      auto current_face_nb_nodes = Int32(faces->GetValue(face_info_index++));
      m_cell_nb_faces[m_cell_uids[cell_index]] += 1;
      for (int node_index = 0; node_index < current_face_nb_nodes; ++node_index) {
        current_face_nodes.push_back(faces->GetValue(face_info_index++));
      }
      sorted_current_face_nodes.resize(current_face_nodes.size());
      auto is_front_cell = mesh_utils::reorderNodesOfFace(current_face_nodes, sorted_current_face_nodes);
      auto [is_face_found, existing_face_index] = _findFace(sorted_current_face_nodes, node_faces,
                                                             m_node_uid_to_index, m_face_nb_nodes,
                                                             face_uid_to_index, face_offsets, m_face_node_uids);
      if (!is_face_found) {
        for (auto node_uid : current_face_nodes) {
          node_faces[m_node_uid_to_index[node_uid]].push_back(face_uid);
        }
        m_cell_face_uids.push_back(face_uid);
        m_face_uids.push_back(face_uid); // todo parallel
        m_face_nb_nodes.push_back(current_face_nb_nodes);
        m_face_node_uids.addRange(sorted_current_face_nodes);
        m_face_nb_cells.push_back(1);
        m_face_uid_indexes[global_face_index] = face_uid_index;
        face_uid_to_index.push_back(face_uid_index);
        auto previous_offset = face_offsets.back();
        face_offsets.push_back(previous_offset + sorted_current_face_nodes.size());
        ++face_uid;
        ++face_uid_index;
        if (is_front_cell) {
          m_face_cell_uids.push_back(NULL_ITEM_UNIQUE_ID);
          m_face_cell_uids.push_back(m_cell_uids[cell_index]);
        }
        else {
          m_face_cell_uids.push_back(m_cell_uids[cell_index]);
          m_face_cell_uids.push_back(NULL_ITEM_UNIQUE_ID);
          }
        }
        else {
          m_cell_face_uids.push_back(m_face_uids[existing_face_index]);
          m_face_nb_cells[existing_face_index] += 1;
          m_face_uid_indexes[global_face_index] = existing_face_index;
          // add cell to face cell connectivity
          if (is_front_cell) {
            if (m_face_cell_uids[2 * existing_face_index + 1] != NULL_ITEM_UNIQUE_ID) {
              ARCANE_FATAL("Problem in face orientation, face uid {0}, nodes {1}, same orientation in cell {2} and {3}. Change mesh file.",
                           m_face_uids[existing_face_index],
                           current_face_nodes,
                           m_face_cell_uids[2 * existing_face_index + 1],
                           m_cell_uids[cell_index]);
            }
            m_face_cell_uids[2 * existing_face_index + 1] = m_cell_uids[cell_index];
          }
          else {
            if (m_face_cell_uids[2 * existing_face_index] != NULL_ITEM_UNIQUE_ID) {
              ARCANE_FATAL("Problem in face orientation, face uid {0}, nodes {1}, same orientation in cell {2} and {3}. Change mesh file.",
                           m_face_uids[existing_face_index],
                           current_face_nodes,
                           m_face_cell_uids[2 * existing_face_index],
                           m_cell_uids[cell_index]);
            }
            m_face_cell_uids[2 * existing_face_index] = m_cell_uids[cell_index];
          }
        }
        current_face_nodes.clear();
        sorted_current_face_nodes.clear();
    }
    cell_face_index += m_cell_nb_faces[m_cell_uids[cell_index]];
  }
  // fill node_face_uids and node_nb_faces from node_faces (array form [nb_nodes][nb_connected_faces])
  m_node_nb_faces.resize(m_node_uids.size(), 0);
  _flattenConnectivity(node_faces.constSpan(), m_node_nb_faces, m_node_face_uids);

  if (m_print_info_level.print_debug_info) {
    std::cout << "================FACE NODES ==============" << std::endl;
    std::copy(m_face_node_uids.begin(), m_face_node_uids.end(), std::ostream_iterator<Int64>(std::cout, " "));
    std::cout << std::endl;
    std::copy(m_face_nb_nodes.begin(), m_face_nb_nodes.end(), std::ostream_iterator<Int64>(std::cout, " "));
    std::cout << std::endl;
    std::copy(m_cell_face_indexes.begin(), m_cell_face_indexes.end(), std::ostream_iterator<Int64>(std::cout, " "));
    std::cout << std::endl;
  }

  return m_face_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
edgeUids()
{
  if (!doRead()) return m_edge_uids;
  if (!m_edge_uids.empty())
    return m_edge_uids;

  nodeUids(); // Needed to be called first. No work done if already called

  _checkVtkGrid();
  m_edge_uids.reserve(2 * m_vtk_grid->GetNumberOfPoints());
  auto const* faces = m_vtk_grid->GetFaces();
  // This array contains the face info per cells (cf. vtk file)
  // first_cell_nb_faces first_cell_first_face_nb_nodes first_cell_first_face_node_1 ... first_cell_first_face_node_n first_cell_second_face_nb_nodes etc

  if (!faces) {
    ARCANE_FATAL("Mesh {0} is not polyhedral: faces are not defined", m_filename);
  }
  Int64 edge_uid = 0;
  auto nb_edge_estimation = 2 * m_edge_uids.capacity();
  m_edge_node_uids.reserve(nb_edge_estimation);
  auto face_info_size = faces->GetNumberOfValues();
  auto cell_index = 0;
  auto global_face_index = 0;
  auto new_edge_index = 0;
  UniqueArray<std::set<Int64>> edge_cells;
  UniqueArray<Int64UniqueArray> edge_faces;
  edge_cells.reserve(m_edge_uids.capacity());
  edge_faces.reserve(m_edge_uids.capacity());
  m_cell_nb_edges.resize(m_cell_uids.size(), 0);
  m_cell_edge_uids.reserve(20 * m_cell_uids.size()); // choose a value of 20 edge per cell
  UniqueArray<std::set<Int64>> face_edges;
  face_edges.resize(m_face_uids.size());
  UniqueArray<std::set<Int64>> cell_edges;
  cell_edges.resize(m_cell_uids.size());
  UniqueArray<Int64UniqueArray> node_edges;
  node_edges.resize(m_node_uids.size());
  EdgeUidToIndexMap edge_uid_to_index;
  edge_uid_to_index.reserve(nb_edge_estimation);
  Int32UniqueArray edge_offsets;
  edge_offsets.reserve(nb_edge_estimation);
  edge_offsets.push_back(0);
  m_edge_nb_nodes.reserve(nb_edge_estimation);
  for (int face_info_index = 0; face_info_index < face_info_size; ++cell_index) {
    auto current_cell_nb_faces = Int32(faces->GetValue(face_info_index++));
    for (auto face_index = 0; face_index < current_cell_nb_faces; ++face_index, ++global_face_index) {
        auto current_face_nb_nodes = Int32(faces->GetValue(face_info_index++));
        auto first_face_node_uid = Int32(faces->GetValue(face_info_index));
        UniqueArray<Int64> current_edge(2), sorted_edge(2);
        for (int node_index = 0; node_index < current_face_nb_nodes - 1; ++node_index) {
          current_edge = UniqueArray<Int64>{ faces->GetValue(face_info_index++), faces->GetValue(face_info_index) };
          mesh_utils::reorderNodesOfFace(current_edge, sorted_edge); // works for edges
          auto [is_edge_found, existing_edge_index] = _findFace(sorted_edge, node_edges,
                                                                 m_node_uid_to_index,
                                                                 m_edge_nb_nodes,
                                                                 edge_uid_to_index, edge_offsets, m_edge_node_uids); // works for edges
          if (!is_edge_found) {
            m_cell_nb_edges[cell_index] += 1;
            face_edges[m_face_uid_indexes[global_face_index]].insert(edge_uid);
            cell_edges[cell_index].insert(edge_uid);
            for (auto node : current_edge) {
              node_edges[node].push_back(edge_uid);
            }
            edge_cells.push_back(std::set{ m_cell_uids[cell_index] });
            edge_faces.push_back(Int64UniqueArray{ m_cell_face_uids[m_cell_face_indexes[cell_index] + face_index] });
            m_edge_uids.push_back(edge_uid++); // todo parallel
            m_edge_node_uids.addRange(sorted_edge);
            edge_uid_to_index.push_back(new_edge_index);
            auto current_offset = edge_offsets.back();
            edge_offsets.push_back(current_offset + 2);
            m_edge_nb_nodes.push_back(2);
            ++new_edge_index;
          }
          else {
            edge_cells[existing_edge_index].insert(m_cell_uids[cell_index]);
            edge_faces[existing_edge_index].push_back(m_cell_face_uids[m_cell_face_indexes[cell_index] + face_index]);
            face_edges[m_face_uid_indexes[global_face_index]].insert(m_edge_uids[existing_edge_index]);
            cell_edges[cell_index].insert(m_edge_uids[existing_edge_index]);
          }
        }
        current_edge = UniqueArray<Int64>{ faces->GetValue(face_info_index++), first_face_node_uid };
        mesh_utils::reorderNodesOfFace(current_edge, sorted_edge); // works for edges
        auto [is_edge_found, existing_edge_index] = _findFace(sorted_edge, node_edges,
                                                               m_node_uid_to_index,
                                                               m_edge_nb_nodes,
                                                               edge_uid_to_index, edge_offsets, m_edge_node_uids); // works for edges
        if (!is_edge_found) {
          m_cell_nb_edges[cell_index] += 1;
          edge_cells.push_back(std::set{ m_cell_uids[cell_index] });
          edge_faces.push_back(Int64UniqueArray{ m_cell_face_uids[m_cell_face_indexes[cell_index] + face_index] });
          face_edges[m_face_uid_indexes[global_face_index]].insert(edge_uid);
          cell_edges[cell_index].insert(edge_uid);
          for (auto node : current_edge) {
            node_edges[node].push_back(edge_uid);
          }
          m_edge_uids.push_back(edge_uid++); // todo parallel
          m_edge_node_uids.addRange(sorted_edge);
          edge_uid_to_index.push_back(new_edge_index);
          auto current_offset = edge_offsets.back();
          edge_offsets.push_back(current_offset + 2);
          m_edge_nb_nodes.push_back(2);
          ++new_edge_index;
        }
        else {
          edge_cells[existing_edge_index].insert(m_cell_uids[cell_index]);
          edge_faces[existing_edge_index].push_back(m_cell_face_uids[m_cell_face_indexes[cell_index] + face_index]);
          face_edges[m_face_uid_indexes[global_face_index]].insert(m_edge_uids[existing_edge_index]);
          cell_edges[cell_index].insert(m_edge_uids[existing_edge_index]);
        }
      }
    }
    // fill edge_cell_uids and edge_nb_cells from edge_cells (array form [nb_edges][nb_connected_cells])
    m_edge_nb_cells.resize(m_edge_uids.size(), 0);
    _flattenConnectivity(edge_cells.constSpan(), m_edge_nb_cells, m_edge_cell_uids);

    // fill edge faces uids
    m_edge_nb_faces.resize(m_edge_uids.size(), 0);
    _flattenConnectivity(edge_faces.constSpan(), m_edge_nb_faces, m_edge_face_uids);

    // fill face edge uids
    m_face_nb_edges.resize(m_face_uids.size(), 0);
    _flattenConnectivity(face_edges.constSpan(), m_face_nb_edges, m_face_edge_uids);

    // fill cell edge uids
    m_cell_nb_edges.resize(m_cell_uids.size(), 0);
    _flattenConnectivity(cell_edges, m_cell_nb_edges, m_cell_edge_uids);

    // fill node edge uids
    m_node_nb_edges.resize(m_node_uids.size(), 0);
    _flattenConnectivity(node_edges, m_node_nb_edges, m_node_edge_uids);

    if (m_print_info_level.print_debug_info) {
      std::cout << "================EDGE NODES ==============" << std::endl;
      std::copy(m_edge_node_uids.begin(), m_edge_node_uids.end(), std::ostream_iterator<Int64>(std::cout, " "));
      std::cout << std::endl;
      std::cout << "================FACE EDGES ==============" << std::endl;
      std::copy(m_face_nb_edges.begin(), m_face_nb_edges.end(), std::ostream_iterator<Int32>(std::cout, " "));
      std::cout << std::endl;
      std::copy(m_face_edge_uids.begin(), m_face_edge_uids.end(), std::ostream_iterator<Int64>(std::cout, " "));
      std::cout << std::endl;
      std::cout << "================CELL EDGES ==============" << std::endl;
      std::copy(m_cell_nb_edges.begin(), m_cell_nb_edges.end(), std::ostream_iterator<Int32>(std::cout, " "));
      std::cout << std::endl;
      std::copy(m_cell_edge_uids.begin(), m_cell_edge_uids.end(), std::ostream_iterator<Int64>(std::cout, " "));
      std::cout << std::endl;
    }
    return m_edge_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<bool, Int32> VtkPolyhedralMeshIOService::VtkReader::
_findFace(const Int64UniqueArray& sorted_face_nodes,
          const UniqueArray<Int64UniqueArray>& node_face_uids,
          const NodeUidToIndexMap& node_uid_to_index,
          const Int32UniqueArray& face_nb_nodes,
          const FaceUidToIndexMap& face_uid_to_index,
          const UniqueArray<Int32>& face_offsets,
          const Int64UniqueArray& face_node_uids)
{
  auto first_node_uid = sorted_face_nodes[0];
  auto first_node_index = node_uid_to_index[first_node_uid];
  // If the face already exists it has already been registered in node_face connectivity
  for (auto face_uid : node_face_uids[first_node_index]) {
    auto face_index = face_uid_to_index[face_uid];
    auto face_offset = face_offsets[face_index];
    auto face_nb_node = face_nb_nodes[face_index];
    if (face_nb_node == sorted_face_nodes.size()) {
      bool is_same_face = true;
      for (auto index = 0; index < face_nb_node; ++index) {
        if (sorted_face_nodes[index] != face_node_uids[face_offset + index]) {
          is_same_face = false;
        }
      }
      if (is_same_face)
        return { true, face_index };
    }
  }
  return { false, -1 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer VtkPolyhedralMeshIOService::VtkReader::
nbNodes()
{
  if (m_node_uids.empty())
    nodeUids();
  return m_node_uids.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
cellNodes()
{
  if (m_cell_node_uids.empty())
    cellUids();
  return m_cell_node_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
cellNbNodes()
{
  if (m_cell_nb_nodes.empty())
    cellUids();
  return m_cell_nb_nodes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceNodes()
{
  if (m_face_node_uids.empty())
    faceUids();
  return m_face_node_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceNbNodes()
{
  if (m_face_nb_nodes.empty())
    faceUids();
  return m_face_nb_nodes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceNodesInFaceMesh()
{
  if (m_face_node_uids_in_face_mesh.empty())
    _readfaceNodesInFaceMesh();
  return m_face_node_uids_in_face_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceNbNodesInFaceMesh()
{
  if (m_face_nb_nodes_in_face_mesh.empty())
    _readfaceNodesInFaceMesh();
  return m_face_nb_nodes_in_face_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
edgeNbNodes()
{
  if (m_edge_node_uids.empty())
    edgeUids();
  return m_edge_nb_nodes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
edgeNodes()
{
  if (m_edge_node_uids.empty())
    edgeUids();
  return m_edge_node_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceCells()
{
  if (m_face_cell_uids.empty())
    faceUids();
  // debug
  if (m_print_info_level.print_debug_info) {
    std::cout << "=================FACE CELLS================="
              << "\n";
    std::copy(m_face_cell_uids.begin(), m_face_cell_uids.end(), std::ostream_iterator<Int64>(std::cout, " "));
    std::cout << "\n";
    std::cout << "=================END FACE CELLS================="
              << "\n";
  }
  return m_face_cell_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceNbCells()
{
  if (m_face_nb_cells.empty())
    faceUids();
  return m_face_nb_cells;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
edgeNbCells()
{
  if (m_edge_nb_cells.empty())
    edgeUids();
  return m_edge_nb_cells;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
edgeCells()
{
  if (m_edge_cell_uids.empty())
    edgeUids();
  return m_edge_cell_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
cellNbFaces()
{
  if (m_cell_nb_faces.empty())
    faceUids();
  return m_cell_nb_faces;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
cellFaces()
{
  if (m_cell_face_uids.empty())
    faceUids();
  return m_cell_face_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
edgeNbFaces()
{
  if (m_edge_nb_faces.empty())
    edgeUids();
  return m_edge_nb_faces;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
edgeFaces()
{
  if (m_edge_face_uids.empty())
    edgeUids();
  return m_edge_face_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
cellNbEdges()
{
  if (m_cell_nb_edges.empty())
    edgeUids();
  return m_cell_nb_edges;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
cellEdges()
{
  if (m_cell_edge_uids.empty())
    edgeUids();
  return m_cell_edge_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceNbEdges()
{
  if (m_face_nb_edges.empty())
    edgeUids();
  return m_face_nb_edges;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
faceEdges()
{
  if (m_face_edge_uids.empty())
    edgeUids();
  return m_face_edge_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Connectivity2DArray>
void VtkPolyhedralMeshIOService::VtkReader::
_flattenConnectivity(Connectivity2DArray connected_item_2darray,
                     Int32Span nb_connected_item_per_source_item,
                     Int64UniqueArray& connected_item_array)
{
  // fill nb_connected_item_per_source_items
  std::transform(connected_item_2darray.begin(), connected_item_2darray.end(), nb_connected_item_per_source_item.begin(), [](auto const& connected_items) {
    return connected_items.size();
  });
  // fill connected_item_array
  connected_item_array.reserve(std::accumulate(nb_connected_item_per_source_item.begin(), nb_connected_item_per_source_item.end(), 0));
  std::for_each(connected_item_2darray.begin(), connected_item_2darray.end(), [&connected_item_array](auto const& connected_items) {
    for (auto const& connected_item : connected_items) {
      connected_item_array.push_back(connected_item);
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
nodeNbCells()
{
  if (m_node_nb_cells.empty())
    nodeUids();
  return m_node_nb_cells;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
nodeCells()
{
  if (m_node_cell_uids.empty())
    nodeUids();
  return m_node_cell_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
nodeNbFaces()
{
  if (m_node_nb_faces.empty())
    faceUids();
  return m_node_nb_faces;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
nodeFaces()
{
  if (m_node_face_uids.empty())
    faceUids();
  return m_node_face_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
nodeNbEdges()
{
  if (m_node_nb_edges.empty())
    edgeUids();
  return m_node_nb_edges;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64ConstArrayView VtkPolyhedralMeshIOService::VtkReader::
nodeEdges()
{
  if (m_node_edge_uids.empty())
    edgeUids();
  return m_node_edge_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3ArrayView VtkPolyhedralMeshIOService::VtkReader::
nodeCoords()
{
  if (!doRead()) return m_node_coordinates;
  if (m_node_coordinates.empty()) {
    _checkVtkGrid();
    auto point_coords = m_vtk_grid->GetPoints()->GetData();
    if (m_print_info_level.print_debug_info) {
      std::cout << "======= Point COORDS ====" << std::endl;
      std::ostringstream oss;
      point_coords->PrintSelf(oss, vtkIndent{ 2 });
      std::cout << oss.str() << std::endl;
    }
    auto nb_nodes = m_vtk_grid->GetNumberOfPoints();
    for (int i = 0; i < nb_nodes; ++i) {
      if (m_print_info_level.print_debug_info) {
          std::cout << "==========current point coordinates : ( ";
          std::cout << *(point_coords->GetTuple(i)) << " , ";
          std::cout << *(point_coords->GetTuple(i) + 1) << " , ";
          std::cout << *(point_coords->GetTuple(i) + 2) << " ) ===" << std::endl;
      }
      m_node_coordinates.add({ *(point_coords->GetTuple(i)),
                               *(point_coords->GetTuple(i) + 1),
                               *(point_coords->GetTuple(i) + 2) });
    }
  }
  return m_node_coordinates;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

vtkCellData* VtkPolyhedralMeshIOService::VtkReader::
cellData()
{
  return m_cell_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

vtkPointData* VtkPolyhedralMeshIOService::VtkReader::
pointData()
{
  return m_point_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::VtkReader::
_printMeshInfos() const
{
  _checkVtkGrid();
  std::cout << "-- VTK GRID READ "
            << " NB CELLS  " << m_vtk_grid->GetNumberOfCells() << std::endl;
  // Parse cells
  auto* cell_iter = m_vtk_grid->vtkDataSet::NewCellIterator();
  cell_iter->InitTraversal();
  vtkIdType_generic* cell_faces{ nullptr };
  vtkIdType nb_faces = 0;
  while (!cell_iter->IsDoneWithTraversal()) {
    std::cout << "---- visiting cell id " << cell_iter->GetCellId() << std::endl;
    std::cout << "----   cell number of faces " << cell_iter->GetNumberOfFaces() << std::endl;
    std::cout << "----   cell number of points " << cell_iter->GetNumberOfPoints() << std::endl;
    m_vtk_grid->GetFaceStream(cell_iter->GetCellId(), nb_faces, cell_faces);
    for (auto iface = 0; iface < nb_faces; ++iface) {
      auto face_nb_nodes = *cell_faces++;
      std::cout << "----      has face with " << face_nb_nodes << " nodes. Node ids : ";
      for (int inode = 0; inode < face_nb_nodes; ++inode) {
          std::cout << *cell_faces++ << " ";
      }
      std::cout << std::endl;
    }
    cell_iter->GoToNextCell();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

vtkCellData* VtkPolyhedralMeshIOService::VtkReader::faceData()
{
  return m_face_data;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::VtkReader::
_readPlainTextVtkGrid(const String& filename)
{
  m_vtk_grid_reader->SetFileName(filename.localstr());
  m_vtk_grid_reader->ReadAllScalarsOn();
  m_vtk_grid_reader->Update();
  m_vtk_grid = m_vtk_grid_reader->GetOutput();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::VtkReader::
_readXlmVtkGrid(const String& filename)
{
  m_vtk_xml_grid_reader->SetFileName(filename.localstr());
  m_vtk_xml_grid_reader->Update();
  m_vtk_grid = m_vtk_xml_grid_reader->GetOutput();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::VtkReader::
_readPlainTextVtkFaceGrid(const String& faces_filename)
{
  m_vtk_face_grid_reader->SetFileName(faces_filename.localstr());
  m_vtk_face_grid_reader->ReadAllScalarsOn();
  m_vtk_face_grid_reader->Update();
  m_vtk_face_grid = m_vtk_face_grid_reader->GetOutput();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::VtkReader::
_readXmlVtkFaceGrid(const String& faces_filename)
{
  m_vtk_xml_face_grid_reader->SetFileName(faces_filename.localstr());
  m_vtk_xml_face_grid_reader->Update();
  m_vtk_face_grid = m_vtk_xml_face_grid_reader->GetOutput();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::VtkReader::
_readfaceNodesInFaceMesh()
{
  m_face_nb_nodes_in_face_mesh.resize(m_poly_data->GetNumberOfCells());
  m_face_node_uids_in_face_mesh.reserve(m_poly_data->GetNumberOfCells() * m_poly_data->GetMaxCellSize());
  m_poly_data->InitTraversal();
  vtkIdType face_nb_nodes;
  vtkIdType_generic* face_nodes;

  auto face_nb_node_index = 0;
  Int64UniqueArray current_face_node_uids;
  Int64UniqueArray reordered_current_face_node_uids;
  current_face_node_uids.reserve(m_poly_data->GetMaxCellSize());
  reordered_current_face_node_uids.reserve(m_poly_data->GetMaxCellSize());
  Int64UniqueArray reordered_face_node_uids(m_poly_data->GetMaxCellSize());
  while (m_poly_data->GetNextCell(face_nb_nodes, face_nodes)) {
    m_face_nb_nodes_in_face_mesh[face_nb_node_index] = face_nb_nodes;
    ConstArrayView<vtkIdType> face_nodes_view(face_nb_nodes, face_nodes);
    current_face_node_uids.resize(face_nb_nodes);
    reordered_current_face_node_uids.resize(face_nb_nodes);
    std::copy(face_nodes_view.begin(), face_nodes_view.end(), current_face_node_uids.begin());
    MeshUtils::reorderNodesOfFace(current_face_node_uids, reordered_current_face_node_uids);
    std::copy(reordered_current_face_node_uids.begin(), reordered_current_face_node_uids.end(), std::back_inserter(m_face_node_uids_in_face_mesh));
    ++face_nb_node_index;
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkPolyhedralMeshIOService::VtkReader::
_checkVtkGrid() const
{
  if (!m_vtk_grid)
    ARCANE_FATAL("Polyhedral vtk grid not loaded. Cannot continue.");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane
