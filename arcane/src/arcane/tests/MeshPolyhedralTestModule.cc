// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPolyhedralTestModule.cc                                  C) 2000-2025 */
/*                                                                           */
/* Test Module for custom mesh                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <numeric>

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/internal/IMeshInternal.h"
#include "arcane/mesh/PolyhedralMesh.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/ItemGroup.h"
#include "MeshPolyhedralTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest::MeshPolyhedral
{
using namespace Arcane;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshPolyhedralTestModule : public ArcaneMeshPolyhedralTestObject
{
 public:

  explicit MeshPolyhedralTestModule(const ModuleBuildInfo& sbi)
  : ArcaneMeshPolyhedralTestObject(sbi)
  {}

 public:

  void init()
  {
    auto mesh_handle = subDomain()->defaultMeshHandle();
    if (mesh_handle.hasMesh()) {
      info() << "-- Mesh name: " << mesh()->name();
      _testKind(mesh());
      _testDimensions(mesh());
      _testCoordinates(mesh());
      _testEnumerationAndConnectivities(mesh());
      _testVariables(mesh());
      _testGroups(mesh());
      _testMeshUtilities(mesh());
      _testMeshModifier(mesh());
    }
    else
      info() << "No Mesh";

    subDomain()->timeLoopMng()->stopComputeLoop(true);
  }

 private:

  void _testKind(IMesh* mesh);
  void _testEnumerationAndConnectivities(IMesh* mesh);
  void _testVariables(IMesh* mesh);
  void _testGroups(IMesh* mesh);
  void _testDimensions(IMesh* mesh);
  void _testCoordinates(IMesh* mesh);
  void _testMeshUtilities(IMesh* mesh);
  void _testMeshModifier(IMesh* mesh);
  void _buildGroup(IItemFamily* family, String const& group_name);
  void _checkBoundaryFaceGroup(IMesh* mesh, String const& boundary_face_group_name) const;
  void _checkInternalFaceGroup(IMesh* mesh, String const& internal_face_group_name) const;
  void _checkFlags(IMesh* mesh) const;
  template <typename VariableRefType>
  void _checkVariable(VariableRefType variable, ItemGroup item_group);
  template <typename VariableRefType>
  void _checkVariableWithRefValue(VariableRefType variable, ItemGroup item_group, const typename VariableRefType::DataType& ref_sum);
  template <typename VariableArrayRefType>
  void _checkArrayVariable(VariableArrayRefType variable, ItemGroup item_group);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_testKind(IMesh* mesh)
{
  // Check mesh kind
  if (mesh->meshKind().meshStructure() != eMeshStructure::Polyhedral) {
    ARCANE_FATAL("Mesh kind for mesh {0} is not eMeshStructure::Polyhedral",mesh->name());
  }
  // Set mesh kind (for test, done by mesh reader)
  // So far for PolyhedralMesh must be : eMeshStructure::Polyhedral, eMeshAMRKind::Node
  MeshKind kind;
  kind.setMeshStructure(eMeshStructure::Polyhedral);
  kind.setMeshAMRKind(eMeshAMRKind::None);
  mesh->_internalApi()->setMeshKind(kind);
  // Finalize internalApi check. Dof tests done in DoFTester
  auto dof_mng = mesh->_internalApi()->dofConnectivityMng();
  if (!dof_mng)
    ARCANE_FATAL("Cannot get DoFConnectivityMng from PolyhedralMesh");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_testEnumerationAndConnectivities(IMesh* mesh)
{
  info() << "- Polyhedral mesh test -";
  info() << "- Mesh dimension " << mesh->dimension();
  info() << "- Mesh nb cells  " << mesh->nbItem(IK_Cell) << " or " << mesh->nbCell();
  info() << "- Mesh nb faces  " << mesh->nbItem(IK_Face) << " or " << mesh->nbFace();
  info() << "- Mesh nb edges  " << mesh->nbItem(IK_Edge) << " or " << mesh->nbEdge();
  info() << "- Mesh nb nodes  " << mesh->nbItem(IK_Node) << " or " << mesh->nbNode();
  info() << "Cell family " << mesh->cellFamily()->name();
  info() << "Node family " << mesh->nodeFamily()->name();
  auto all_cells = mesh->allCells();
  // ALL CELLS
  ENUMERATE_CELL (icell, all_cells) {
    debug(Trace::High) << "cell with index " << icell.index();
    debug(Trace::High) << "cell with lid " << icell.localId();
    debug(Trace::High) << "cell with uid " << icell->uniqueId().asInt64();
    debug(Trace::High) << "cell number of nodes " << icell->nodes().size();
    debug(Trace::High) << "cell number of faces " << icell->faces().size();
    debug(Trace::High) << "cell number of edges " << icell->edges().size();
    for (Node node : icell->nodes()) {
      debug(Trace::High) << "cell node lid " << node.localId() << " uid " << node.uniqueId().asInt64();
    }
    for (Face face : icell->faces()) {
      debug(Trace::High) << "cell face lid " << face.localId() << " uid " << face.uniqueId().asInt64();
    }
    for (Edge edge : icell->edges()) {
      debug(Trace::High) << "cell edge lid " << edge.localId() << " uid " << edge.uniqueId().asInt64();
    }
  }
  // ALL FACES
  ENUMERATE_FACE (iface, mesh->allFaces()) {
    debug(Trace::High) << "face with index " << iface.index();
    debug(Trace::High) << "face with lid " << iface.localId();
    debug(Trace::High) << "face with uid " << iface->uniqueId().asInt64();
    debug(Trace::High) << "face number of nodes " << iface->nodes().size();
    debug(Trace::High) << "face number of cells " << iface->cells().size();
    debug(Trace::High) << "face number of edges " << iface->edges().size();
    debug(Trace::High) << "face back cell " << iface->backCell().localId();
    debug(Trace::High) << "face front cell " << iface->frontCell().localId();
    for (Node node : iface->nodes()) {
      debug(Trace::High) << "face node lid " << node.localId() << " uid " << node.uniqueId().asInt64();
    }
    auto cell_index = 0;
    bool are_face_cells_ok = true;
    for (Cell cell : iface->cells()) {
      debug(Trace::High) << "face cell lid " << cell.localId() << " uid " << cell.uniqueId().asInt64();
      if (cell_index == 0) {
        if (iface->itemBase().flags() & ItemFlags::II_FrontCellIsFirst)
          are_face_cells_ok = are_face_cells_ok && cell.uniqueId() == iface->frontCell().uniqueId();
        else
          are_face_cells_ok = are_face_cells_ok && cell.uniqueId() == iface->backCell().uniqueId();
      }
      else
        are_face_cells_ok = are_face_cells_ok && cell.uniqueId() == iface->frontCell().uniqueId();
      ++cell_index;
    }
    if (!are_face_cells_ok) {
      ARCANE_FATAL("Problem with face cells.");
    }
    for (Edge edge : iface->edges()) {
      debug(Trace::High) << "face edge lid " << edge.localId() << " uid " << edge.uniqueId().asInt64();
    }
  }
  // Check face flags
  _checkFlags(mesh);
  // ALL NODES
  ENUMERATE_NODE (inode, mesh->allNodes()) {
    debug(Trace::High) << "node with index " << inode.index();
    debug(Trace::High) << "node with lid " << inode.localId();
    debug(Trace::High) << "node with uid " << inode->uniqueId().asInt64();
    debug(Trace::High) << "node number of faces " << inode->faces().size();
    debug(Trace::High) << "node number of cells " << inode->cells().size();
    debug(Trace::High) << "node number of edges " << inode->edges().size();
    for (Face face : inode->faces()) {
      debug(Trace::High) << "node face lid " << face.localId() << " uid " << face.uniqueId().asInt64();
    }
    for (Cell cell : inode->cells()) {
      debug(Trace::High) << "node cell lid " << cell.localId() << " uid " << cell.uniqueId().asInt64();
    }
    for (Edge edge : inode->edges()) {
      debug(Trace::High) << "node edge lid " << edge.localId() << " uid " << edge.uniqueId().asInt64();
    }
  }
  // ALL EDGES
  ENUMERATE_EDGE (iedge, mesh->allEdges()) {
    debug(Trace::High) << "edge with index " << iedge.index();
    debug(Trace::High) << "edge with lid " << iedge.localId();
    debug(Trace::High) << "edge with uid " << iedge->uniqueId().asInt64();
    debug(Trace::High) << "edge number of faces " << iedge->faces().size();
    debug(Trace::High) << "edge number of cells " << iedge->cells().size();
    debug(Trace::High) << "edge number of nodes " << iedge->nodes().size();
    for (Face face : iedge->faces()) {
      debug(Trace::High) << "edge face lid " << face.localId() << " uid " << face.uniqueId();
    }
    for (Cell cell : iedge->cells()) {
      debug(Trace::High) << "edge cell lid " << cell.localId() << " uid " << cell.uniqueId();
    }
    for (Node node : iedge->nodes()) {
      debug(Trace::High) << "edge node lid " << node.localId() << " uid " << node.uniqueId();
    }
  }
  // Active items : no AMR available with polyhedral mesh but must return all items
  bool is_active_ok = (mesh->allActiveCells().size() == mesh->allCells().size());
  is_active_ok &= (mesh->ownActiveCells().size() == mesh->ownCells().size());
  is_active_ok &= (mesh->allActiveFaces().size() == mesh->allFaces().size());
  is_active_ok &= (mesh->ownFaces().size() == mesh->ownFaces().size());
  is_active_ok &= (mesh->innerActiveFaces().size() == mesh->allCells().innerFaceGroup().size());
  is_active_ok &= (mesh->outerActiveFaces().size() == mesh->outerFaces().size());
  is_active_ok &= (mesh->allLevelCells(0).size() == mesh->allCells().size());
  is_active_ok &= (mesh->ownLevelCells(0).size() == mesh->ownCells().size());
  is_active_ok &= (mesh->allLevelCells(1).empty());
  is_active_ok &= (mesh->ownLevelCells(1).empty());
  if (!is_active_ok)
    ARCANE_FATAL("Polyhedral mesh implem does not handle correctly activeCells methods. Should return all items.");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::_testVariables(IMesh* mesh)
{
  // test variables
  info() << " -- test variables -- ";
  // cell variable
  m_cell_variable.fill(1);
  _checkVariable(m_cell_variable, mesh->allCells());
  // node variable
  m_node_variable.fill(1);
  _checkVariable(m_node_variable, mesh->allNodes());
  // face variable
  m_face_variable.fill(1);
  _checkVariable(m_face_variable, mesh->allFaces());
  // edge variable
  m_edge_variable.fill(1);
  _checkVariable(m_edge_variable, mesh->allEdges());
  // Check variables defined in mesh file
  // Cell variables
  for (const auto& variable_name : options()->getCheckCellVariableReal()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableCellReal var{ VariableBuildInfo(mesh, variable_name) };
    _checkVariable(var, mesh->allCells());
  }
  for (const auto& variable_name : options()->getCheckCellVariableInteger()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableCellInteger var{ VariableBuildInfo(mesh, variable_name) };
    _checkVariable(var, mesh->allCells());
  }
  for (const auto& variable_name : options()->getCheckCellVariableArrayInteger()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableCellArrayInteger var{ VariableBuildInfo(mesh, variable_name) };
    _checkArrayVariable(var, mesh->allCells());
  }
  for (const auto& variable_name : options()->getCheckCellVariableArrayReal()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableCellArrayReal var{ VariableBuildInfo(mesh, variable_name) };
    _checkArrayVariable(var, mesh->allCells());
  }
  // Node variables
  for (const auto& variable_name : options()->getCheckNodeVariableReal()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableNodeReal var{ VariableBuildInfo(mesh, variable_name) };
    _checkVariable(var, mesh->allNodes());
  }
  for (const auto& variable_name : options()->getCheckNodeVariableInteger()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableNodeInteger var{ VariableBuildInfo(mesh, variable_name) };
    _checkVariable(var, mesh->allNodes());
  }
  for (const auto& variable_name : options()->getCheckNodeVariableArrayInteger()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableNodeArrayInteger var{ VariableBuildInfo(mesh, variable_name) };
    _checkArrayVariable(var, mesh->allNodes());
  }
  for (const auto& variable_name : options()->getCheckNodeVariableArrayReal()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableNodeArrayReal var{ VariableBuildInfo(mesh, variable_name) };
    _checkArrayVariable(var, mesh->allNodes());
  }
  // Face variables
  for (const auto& variable_name : options()->getCheckFaceVariableReal()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableFaceReal var{ VariableBuildInfo(mesh, variable_name) };
    _checkVariable(var, mesh->allFaces());
  }
  for (const auto& variable_name : options()->getCheckFaceVariableInteger()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableFaceInteger var{ VariableBuildInfo(mesh, variable_name) };
    _checkVariable(var, mesh->allFaces());
  }
  for (const auto& variable_name : options()->getCheckFaceVariableArrayInteger()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableFaceArrayInteger var{ VariableBuildInfo(mesh, variable_name) };
    _checkArrayVariable(var, mesh->allFaces());
  }
  for (const auto& variable_name : options()->getCheckFaceVariableArrayReal()) {
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh, variable_name))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableFaceArrayReal var{ VariableBuildInfo(mesh, variable_name) };
    _checkArrayVariable(var, mesh->allFaces());
  }
  for (const auto& variable_with_ref_option : options()->checkCellVariableIntegerWithRefValue()) {
    String variable_name = variable_with_ref_option->getVarName();
    if (!Arcane::AbstractModule::subDomain()->variableMng()->findMeshVariable(mesh,variable_name ))
      ARCANE_FATAL("Cannot find mesh array variable {0}", variable_name);
    VariableCellInteger var{ VariableBuildInfo(mesh, variable_name) };
    _checkVariableWithRefValue(var, mesh->allCells(),variable_with_ref_option->getVarRefSum());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_testGroups(IMesh* mesh)
{
  // AllItems groups
  ARCANE_ASSERT((!mesh->findGroup("AllCells").null()), ("Group AllCells has not been created"));
  ARCANE_ASSERT((!mesh->findGroup("AllNodes").null()), ("Group AllNodes has not been created"));
  ARCANE_ASSERT((!mesh->findGroup("AllFaces").null()), ("Group AllFaces has not been created"));
  ARCANE_ASSERT((!mesh->findGroup("AllEdges").null()), ("Group AllEdges has not been created"));
  // OwnItems groups
  if (!Arcane::AbstractModule::subDomain()->parallelMng()->isParallel()) {
    ValueChecker vc{ A_FUNCINFO };
    vc.areEqual(mesh->allCells().size(), mesh->ownCells().size(), "All and own cell group size differ in sequential.");
    vc.areEqual(mesh->allFaces().size(), mesh->ownFaces().size(), "All and own face group size differ in sequential.");
    vc.areEqual(mesh->allEdges().size(), mesh->ownEdges().size(), "All and own edge group size differ in sequential.");
    vc.areEqual(mesh->allNodes().size(), mesh->ownNodes().size(), "All and own node group size differ in sequential.");
  }
  // Cell group
  String group_name = "my_cell_group";
  _buildGroup(mesh->cellFamily(), group_name);
  ARCANE_ASSERT((!mesh->findGroup(group_name).null()), ("Group my_cell_group has not been created"));
  PartialVariableCellInt32 partial_cell_var({ mesh, "partial_cell_variable", mesh->cellFamily()->name(), group_name });
  partial_cell_var.fill(1);
  _checkVariable(partial_cell_var, partial_cell_var.itemGroup());
  // Node group
  group_name = "my_node_group";
  _buildGroup(mesh->nodeFamily(), group_name);
  ARCANE_ASSERT((!mesh->findGroup(group_name).null()), ("Group my_node_group has not been created"));
  PartialVariableNodeInt32 partial_node_var({ mesh, "partial_node_variable", mesh->nodeFamily()->name(), group_name });
  partial_node_var.fill(1);
  _checkVariable(partial_node_var, partial_node_var.itemGroup());
  // Face group
  group_name = "my_face_group";
  _buildGroup(mesh->faceFamily(), group_name);
  ARCANE_ASSERT((!mesh->findGroup(group_name).null()), ("Group my_face_group has not been created"));
  PartialVariableFaceInt32 partial_face_var({ mesh, "partial_face_variable", mesh->faceFamily()->name(), group_name });
  partial_face_var.fill(1);
  _checkVariable(partial_face_var, partial_face_var.itemGroup());
  // Edge group
  group_name = "my_edge_group";
  _buildGroup(mesh->edgeFamily(), group_name);
  ARCANE_ASSERT((!mesh->findGroup(group_name).null()), ("Group my_edge_group has not been created"));
  PartialVariableEdgeInt32 partial_edge_var({ mesh, "partial_edge_variable", mesh->edgeFamily()->name(), group_name });
  partial_edge_var.fill(1);
  _checkVariable(partial_edge_var, partial_edge_var.itemGroup());

  for (const auto& group_infos : options()->checkGroup()) {
    auto group = mesh->findGroup(group_infos->getName());
    if (group.null())
      ARCANE_FATAL("Could not find group {0}", group_infos->getName());
    ValueChecker vc{ A_FUNCINFO };
    vc.areEqual(group.size(), group_infos->getSize(), "check group size");
  }
  ValueChecker vc{ A_FUNCINFO };
  auto nb_internal_group = 19;
  if (subDomain()->parallelMng()->isParallel()) {
    nb_internal_group = 23;
  }
  auto nb_group = nb_internal_group + options()->nbMeshGroup;
  vc.areEqual(nb_group, mesh->groups().count(), "check number of groups in the mesh");

  for (const auto& boundary_face_group_name : options()->getCheckBoundaryFaceGroup()) {
    _checkBoundaryFaceGroup(mesh, boundary_face_group_name);
  }
  for (const auto& boundary_face_group_name : options()->getCheckInternalFaceGroup()) {
    _checkInternalFaceGroup(mesh, boundary_face_group_name);
  }

  _checkBoundaryFaceGroup(mesh, mesh->outerFaces().name());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_testDimensions(IMesh* mesh)
{
  auto mesh_size = options()->meshSize();
  if (mesh_size.empty())
    return;
  ValueChecker vc(A_FUNCINFO);
  vc.areEqual(mesh->nbCell(), mesh_size[0]->getNbCells(), "check number of cells");
  vc.areEqual(mesh->nbFace(), mesh_size[0]->getNbFaces(), "check number of faces");
  vc.areEqual(mesh->nbEdge(), mesh_size[0]->getNbEdges(), "check number of edges");
  vc.areEqual(mesh->nbNode(), mesh_size[0]->getNbNodes(), "check number of nodes");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_testCoordinates(Arcane::IMesh* mesh)
{
  if (options()->meshCoordinates.size() == 1) {
    if (options()->meshCoordinates[0].doCheck()) {
      auto node_coords = mesh->toPrimaryMesh()->nodesCoordinates();
      auto node_coords_ref = options()->meshCoordinates[0].coords();
      ValueChecker vc{ A_FUNCINFO };
      ENUMERATE_NODE (inode, allNodes()) {
        vc.areEqual(node_coords[inode], node_coords_ref[0]->value[inode.index()], "check coords values");
        debug(Trace::High) << " node coords  " << node_coords[inode];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_testMeshUtilities(Arcane::IMesh* mesh)
{
  auto* mesh_utilities = mesh->utilities();
  ARCANE_CHECK_POINTER(mesh_utilities);
  // Test change owner from cells
  // wip test in sequential: virtually change all cell owners to 1 and check all items have 1 as owner
  auto& cell_owners = mesh->cellFamily()->itemsNewOwner();
  auto& face_owners = mesh->faceFamily()->itemsNewOwner();
  auto& edge_owners = mesh->edgeFamily()->itemsNewOwner();
  auto& node_owners = mesh->nodeFamily()->itemsNewOwner();
  ENUMERATE_(Cell,icell,allCells()) {
    cell_owners[icell] = 1;
    info() << "Cell uid " << icell->uniqueId() << " cell_owner[icell] "  << cell_owners[icell];
    info() << "Cell uid " << icell->uniqueId() << " has owner "  << icell->owner();
  }
  mesh_utilities->changeOwnersFromCells();
  bool has_error = false;
  ENUMERATE_ (Face,iface,allFaces()) {
    has_error |= face_owners[iface] != 1;
  }
  ENUMERATE_ (Node,inode,allNodes()) {
    has_error |= node_owners[inode] != 1;
  }
  ENUMERATE_ (Edge,iedge,allEdges()) {
    has_error |= edge_owners[iedge] != 1;
  }
  if (has_error) {
    ARCANE_FATAL("changeOwnerFromCells does not work with PolyhedralMesh");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_testMeshModifier(Arcane::IMesh* mesh)
{
  auto* mesh_modifier = mesh->modifier();
  ARCANE_CHECK_POINTER(mesh_modifier);
  // Following methods not yet implemented. Do nothing in sequential and crash in parallel
  mesh_modifier->addExtraGhostCellsBuilder(nullptr);
  mesh_modifier->removeExtraGhostCellsBuilder(nullptr);
  mesh_modifier->endUpdate(false, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_buildGroup(IItemFamily* family, String const& group_name)
{
  auto group = family->findGroup(group_name, true);
  Int32UniqueArray item_lids;
  item_lids.reserve(family->nbItem());
  ENUMERATE_ITEM (iitem, family->allItems()) {
    if (iitem.localId() % 2 == 0)
      item_lids.add(iitem.localId());
  }
  group.addItems(item_lids);
  info() << itemKindName(family->itemKind()) << " group size " << group.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename VariableRefType>
void MeshPolyhedralTestModule::
_checkVariable(VariableRefType variable, ItemGroup item_group)
{
  _checkVariableWithRefValue(variable, item_group, item_group.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename VariableRefType>
void MeshPolyhedralTestModule::
_checkVariableWithRefValue(VariableRefType variable, Arcane::ItemGroup item_group, const typename VariableRefType::DataType& ref_sum)
{
  typename VariableRefType::DataType variable_sum = 0.;
  using ItemType = typename VariableRefType::ItemType;
  ENUMERATE_ (ItemType, iitem, item_group) {
    debug(Trace::High) << variable.name() << " at item " << iitem.localId() << " " << variable[iitem];
    variable_sum += variable[iitem];
  }
  if (variable_sum != ref_sum) {
    fatal() << "Error on variable " << variable.name();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename VariableArrayRefType>
void MeshPolyhedralTestModule::
_checkArrayVariable(VariableArrayRefType variable_ref, ItemGroup item_group)
{
  auto variable_sum = 0.;
  using ItemType = typename VariableArrayRefType::ItemType;
  auto array_size = variable_ref.arraySize();
  ENUMERATE_ (ItemType, iitem, item_group) {
    for (auto value : variable_ref[iitem]) {
      variable_sum += value;
    }
    debug(Trace::High) << variable_ref.name() << " at item " << iitem.localId() << variable_ref[iitem];
  }
  ValueChecker vc{ A_FUNCINFO };
  if (array_size == 0)
    ARCANE_FATAL("Array variable {0} array size is zero");
  std::vector<int> ref_sum(array_size);
  std::iota(ref_sum.begin(), ref_sum.end(), 1.);
  vc.areEqual(variable_sum, item_group.size() * std::accumulate(ref_sum.begin(), ref_sum.end(), 0.), "check array variable values");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_checkBoundaryFaceGroup(IMesh* mesh, const String& boundary_face_group_name) const
{
  auto boundary_face_group = mesh->findGroup(boundary_face_group_name);
  if (boundary_face_group.null())
    ARCANE_FATAL("Cannot find boundary face group {0}", boundary_face_group_name);
  bool are_face_boundaries = true;
  ENUMERATE_FACE (iface, boundary_face_group) {
    are_face_boundaries = are_face_boundaries && iface->isSubDomainBoundary();
    if (!iface->isSubDomainBoundary()) {
      debug(Trace::High) << String::format("Face {0} with nodes {1} is not boundary", iface->uniqueId(), iface->nodes());
    }
  }
  if (!are_face_boundaries)
    ARCANE_FATAL("Boundary face group {0} contains face(s) that are not on the subdomain boundary", boundary_face_group_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_checkInternalFaceGroup(IMesh* mesh, const String& internal_face_group_name) const
{
  auto internal_face_group = mesh->findGroup(internal_face_group_name);
  if (internal_face_group.null())
    ARCANE_FATAL("Cannot find internal face group {0}", internal_face_group_name);
  bool are_face_internals = true;
  ENUMERATE_FACE (iface, internal_face_group) {
    are_face_internals = are_face_internals && !iface->isSubDomainBoundary();
    if (iface->isSubDomainBoundary()) {
      debug(Trace::High) << String::format("Face {0} with nodes {1} is not an internal face", iface->uniqueId(), iface->nodes());
    }
  }
  if (!are_face_internals)
    ARCANE_FATAL("Internal face group {0} contains face(s) that are on the subdomain boundary", internal_face_group_name);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPolyhedralTestModule::
_checkFlags(IMesh* mesh) const
{
  bool are_flags_ok = true;
  bool has_internal_faces = false;
  ENUMERATE_FACE (iface, mesh->allFaces()) {
    Face face{ *iface };
    if (face.backCell().null()) {
      are_flags_ok = are_flags_ok && face.isSubDomainBoundary();
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_Boundary);
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_FrontCellIsFirst);
      are_flags_ok = are_flags_ok && !(face.itemBase().flags() & ItemFlags::II_BackCellIsFirst);
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_HasFrontCell);
      are_flags_ok = are_flags_ok && !(face.itemBase().flags() & ItemFlags::II_HasBackCell);
    }
    else if (face.frontCell().null()) {
      are_flags_ok = are_flags_ok && face.isSubDomainBoundary();
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_Boundary);
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_BackCellIsFirst);
      are_flags_ok = are_flags_ok && !(face.itemBase().flags() & ItemFlags::II_FrontCellIsFirst);
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_HasBackCell);
      are_flags_ok = are_flags_ok && !(face.itemBase().flags() & ItemFlags::II_HasFrontCell);
    }
    else {
      are_flags_ok = are_flags_ok && !face.isSubDomainBoundary();
      are_flags_ok = are_flags_ok && !(face.itemBase().flags() & ItemFlags::II_Boundary);
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_BackCellIsFirst);
      are_flags_ok = are_flags_ok && !(face.itemBase().flags() & ItemFlags::II_FrontCellIsFirst);
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_HasBackCell);
      are_flags_ok = are_flags_ok && (face.itemBase().flags() & ItemFlags::II_HasFrontCell);
      has_internal_faces = true;
    }
  }
  if (!are_flags_ok)
    ARCANE_FATAL("Face flags are incorrect");
  if (has_internal_faces)
    info() << "Mesh has internal faces";
  else
    info() << "Mesh has no internal face";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MESHPOLYHEDRALTEST(MeshPolyhedralTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest::MeshPolyhedral

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
