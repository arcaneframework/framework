// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMesh.cc                                              (C) 2000-2026 */
/*                                                                           */
/* Class for managing an evolving unstructured mesh.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/DynamicMesh.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/ITimeStats.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/Properties.h"
#include "arcane/core/SharedVariable.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IPropertyMng.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/MeshStats.h"
#include "arcane/core/IMeshFactory.h"
#include "arcane/core/IMeshPartitionConstraintMng.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/Connectivity.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/MeshToMeshTransposer.h"
#include "arcane/core/IItemFamilyCompactPolicy.h"
#include "arcane/core/IItemFamilyExchanger.h"
#include "arcane/core/IItemFamilySerializer.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/IMeshExchanger.h"
#include "arcane/core/IMeshCompacter.h"
#include "arcane/core/MeshVisitor.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/ICaseMng.h"

#include "arcane/core/internal/UnstructuredMeshAllocateBuildInfoInternal.h"
#include "arcane/core/internal/IItemFamilyInternal.h"
#include "arcane/core/internal/IVariableMngInternal.h"
#include "arcane/core/internal/IMeshInternal.h"
#include "arcane/core/internal/IMeshModifierInternal.h"

#include "arcane/mesh/ExtraGhostCellsBuilder.h"
#include "arcane/mesh/ExtraGhostParticlesBuilder.h"

#include "arcane/mesh/MeshPartitionConstraintMng.h"
#include "arcane/mesh/ItemGroupsSynchronize.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/OneMeshItemAdder.h"
#include "arcane/mesh/DynamicMeshChecker.h"
#include "arcane/mesh/GhostLayerMng.h"
#include "arcane/mesh/MeshUniqueIdMng.h"
#include "arcane/mesh/ItemGroupDynamicMeshObserver.h"
#include "arcane/mesh/ParticleFamily.h"
#include "arcane/mesh/MeshExchange.h"
#include "arcane/mesh/UnstructuredMeshUtilities.h"
#include "arcane/mesh/TiedInterfaceMng.h"
#include "arcane/mesh/MeshCompactMng.h"
#include "arcane/mesh/MeshExchangeMng.h"
#include "arcane/mesh/DynamicMeshMerger.h"
#include "arcane/mesh/ItemFamilyNetwork.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/MeshExchanger.h"
#include "arcane/mesh/IndexedIncrementalItemConnectivityMng.h"
#include "arcane/mesh/NodeFamily.h"
#include "arcane/mesh/EdgeFamily.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/mesh/DoFFamily.h"

//! AMR
#include "arcane/mesh/MeshRefinement.h"
#include "arcane/mesh/FaceReorienter.h"
#include "arcane/mesh/NewItemOwnerBuilder.h"

#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/ItemConnectivityMng.h"

#include "arcane/mesh/internal/DynamicMeshInternal.h"

#include <functional>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
// Public properties that the user can set
const char* PROPERTY_SORT = "sort";
const char* PROPERTY_COMPACT = "compact";
const char* PROPERTY_COMPACT_AFTER_ALLOCATE = "compact-after-allocate";
const char* PROPERTY_DUMP = "dump";
const char* PROPERTY_DISPLAY_STATS = "display-stats";

// Internal Arcane properties
const char* PROPERTY_MESH_VERSION = "mesh-version";
} // namespace

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createNodeFamilyPolicyMng(ItemFamily* family);

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createEdgeFamilyPolicyMng(ItemFamily* family);

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createFaceFamilyPolicyMng(ItemFamily* family);

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createCellFamilyPolicyMng(ItemFamily* family);

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createParticleFamilyPolicyMng(ItemFamily* family);

extern "C++" ARCANE_MESH_EXPORT IItemFamilyPolicyMng*
createDoFFamilyPolicyMng(ItemFamily* family);

extern "C++" ARCANE_MESH_EXPORT void
allocateCartesianMesh(DynamicMesh* mesh, CartesianMeshAllocateBuildInfo& build_info);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// #define ARCANE_DEBUG_DYNAMIC_MESH
// #define ARCANE_DEBUG_LOAD_BALANCING

#ifdef ARCANE_DEBUG_LOAD_BALANCING
static bool arcane_debug_load_balancing = true;
#else
static bool arcane_debug_load_balancing = false;
#endif

#ifdef ACTIVATE_PERF_COUNTER
const std::string DynamicMesh::PerfCounter::m_names[] = {
  "UPGHOSTLAYER1",
  "UPGHOSTLAYER2",
  "UPGHOSTLAYER3",
  "UPGHOSTLAYER4",
  "UPGHOSTLAYER5",
  "UPGHOSTLAYER6",
  "UPGHOSTLAYER7"
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMesh::
DynamicMesh(ISubDomain* sub_domain, const MeshBuildInfo& mbi, bool is_submesh)
: MeshVariables(sub_domain, mbi.name())
, TraceAccessor(mbi.parallelMngRef()->traceMng())
, m_sub_domain(sub_domain)
, m_mesh_mng(sub_domain->meshMng())
, m_mesh_handle(m_mesh_mng->findMeshHandle(mbi.name()))
, m_parallel_mng(mbi.parallelMngRef().get())
, m_variable_mng(sub_domain->variableMng())
, m_properties(new Properties(sub_domain->propertyMng(), String("ArcaneMeshProperties_") + mbi.name()))
, m_timestamp(0)
, m_is_allocated(false)
, m_dimension(-1)
, m_name(mbi.name())
, m_factory_name(mbi.factoryName())
, m_need_compact(true)
, m_node_family(nullptr)
, m_edge_family(nullptr)
, m_face_family(nullptr)
, m_cell_family(nullptr)
, m_parent_mesh(nullptr)
, m_parent_group(nullptr)
, m_mesh_utilities(nullptr)
, m_mesh_builder(nullptr)
, m_mesh_checker(nullptr)
, m_submesh_tools(nullptr)
//! AMR
, m_mesh_refinement(nullptr)
, m_new_item_owner_builder(nullptr)
, m_extra_ghost_cells_builder(nullptr)
, m_extra_ghost_particles_builder(nullptr)
, m_initial_allocator(this)
, m_internal_api(std::make_unique<DynamicMeshInternal>(this))
, m_is_amr_activated(mbi.meshKind().meshAMRKind() != eMeshAMRKind::None)
, m_amr_type(mbi.meshKind().meshAMRKind())
, m_is_dynamic(false)
, m_tied_interface_mng(nullptr)
, m_is_sub_connectivity_set(false)
, m_tied_interface_need_prepare_dump(true)
, m_partition_constraint_mng(nullptr)
, m_ghost_layer_mng(new GhostLayerMng(m_parallel_mng->traceMng()))
, m_mesh_unique_id_mng(new MeshUniqueIdMng(m_parallel_mng->traceMng()))
, m_mesh_exchange_mng(new MeshExchangeMng(this))
, m_mesh_compact_mng(new MeshCompactMng(this))
, m_connectivity_policy(InternalConnectivityPolicy::NewOnly)
, m_mesh_part_info(makeMeshPartInfoFromParallelMng(m_parallel_mng))
, m_item_type_mng(new ItemTypeMng())
, m_indexed_connectivity_mng(new IndexedIncrementalItemConnectivityMng(m_parallel_mng->traceMng()))
, m_mesh_kind(mbi.meshKind())
{
  m_node_family = new NodeFamily(this, "Node");
  m_edge_family = new EdgeFamily(this, "Edge");
  m_face_family = new FaceFamily(this, "Face");
  m_cell_family = new CellFamily(this, "Cell");

  _addFamily(m_node_family);
  _addFamily(m_edge_family);
  _addFamily(m_face_family);
  _addFamily(m_cell_family);

  m_properties->setBool(PROPERTY_SORT, true);
  m_properties->setBool(PROPERTY_COMPACT, true);
  m_properties->setBool(PROPERTY_COMPACT_AFTER_ALLOCATE, true);
  m_properties->setBool(PROPERTY_DUMP, true);
  m_properties->setBool(PROPERTY_DISPLAY_STATS, true);
  m_properties->setInt32(PROPERTY_MESH_VERSION, 1);

  m_item_internal_list.mesh = this;
  m_item_internal_list._internalSetNodeSharedInfo(m_node_family->commonItemSharedInfo());
  m_item_internal_list._internalSetEdgeSharedInfo(m_edge_family->commonItemSharedInfo());
  m_item_internal_list._internalSetFaceSharedInfo(m_face_family->commonItemSharedInfo());
  m_item_internal_list._internalSetCellSharedInfo(m_cell_family->commonItemSharedInfo());

  info() << "Is AMR Activated? = " << m_is_amr_activated
         << " AMR type = " << m_amr_type
         << " allow_loose_items=" << m_mesh_kind.isNonManifold();

  _printConnectivityPolicy();

  // Adding the family dependencies if asked
  if (_connectivityPolicy() == InternalConnectivityPolicy::NewWithDependenciesAndLegacy && !is_submesh && !m_is_amr_activated) {
    m_use_mesh_item_family_dependencies = true;
    m_item_family_network = new ItemFamilyNetwork(traceMng());
    _addDependency(m_cell_family, m_node_family);
    _addDependency(m_cell_family, m_face_family);
    _addDependency(m_cell_family, m_edge_family);
    _addDependency(m_face_family, m_node_family);
    _addDependency(m_edge_family, m_node_family);
    _addRelation(m_face_family, m_edge_family); // Not seen as a dependency in DynamicMesh : for example not possible to use replaceConnectedItem for Face to Edge...
    _addRelation(m_face_family, m_face_family);
    _addRelation(m_face_family, m_cell_family);
    _addRelation(m_edge_family, m_cell_family);
    _addRelation(m_edge_family, m_face_family);
    _addRelation(m_node_family, m_cell_family);
    _addRelation(m_node_family, m_face_family);
    _addRelation(m_node_family, m_edge_family);
    // The relation concerning edge family are only added when the dimension is known since they change with dimension
    // cf. 3D Cell <-> Faces <-> Edges <-> Nodes
    //     2D Cell <-> Faces <-> Nodes
    //             <-> Edges <-> Nodes
    //     1D No edge...
    m_family_modifiers.add(m_cell_family);
    m_family_modifiers.add(m_face_family);
    m_family_modifiers.add(m_node_family);
    m_family_modifiers.add(m_edge_family);
  }

  {
    String s = platform::getEnvironmentVariable("ARCANE_GRAPH_CONNECTIVITY_POLICY");
#ifdef USE_GRAPH_CONNECTIVITY_POLICY
    s = "1";
#endif
    if (s == "1") {
      m_item_family_network = new ItemFamilyNetwork(traceMng());
      info() << "Graph connectivity is activated";
      m_family_modifiers.add(m_cell_family);
      m_family_modifiers.add(m_face_family);
      m_family_modifiers.add(m_node_family);
      m_family_modifiers.add(m_edge_family);
    }
  }

  m_extra_ghost_cells_builder = new ExtraGhostCellsBuilder(this);
  m_extra_ghost_particles_builder = new ExtraGhostParticlesBuilder(this);

  // To keep compatibility with the existing code, allows not to
  // save the 'need-compact' attribute. This was added for version 3.10
  // of Arcane (June 2023). To be removed before end of 2023.
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_NO_SAVE_NEED_COMPACT", true))
    m_do_not_save_need_compact = v.value();

  // Overrides the default value for the numbering mechanism
  // of uniqueId() for edges and faces.
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_GENERATE_UNIQUE_ID_FROM_NODES", true)) {
    bool is_generate = (v.value() != 0);
    // Using free entities implies using uniqueId() generation
    // from nodes.
    if (!is_generate && meshKind().isNonManifold())
      is_generate = true;
    m_mesh_unique_id_mng->setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(is_generate);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMesh::
~DynamicMesh()
{
  // It might be better to throw an exception but in a
  // destructor this is not necessarily advisable
  if (m_extra_ghost_cells_builder->hasBuilder())
    info() << "WARNING: pending ExtraGhostCellsBuilder reference";
  if (m_extra_ghost_particles_builder->hasBuilder())
    info() << "WARNING: pending ExtraGhostParticlesBuilder reference";

  m_indexed_connectivity_mng = nullptr;
  delete m_mesh_compact_mng;
  delete m_mesh_exchange_mng;
  delete m_extra_ghost_cells_builder;
  delete m_extra_ghost_particles_builder;
  delete m_mesh_unique_id_mng;
  delete m_ghost_layer_mng;
  delete m_tied_interface_mng;
  delete m_partition_constraint_mng;
  delete m_mesh_utilities;
  delete m_mesh_builder;
  delete m_mesh_checker;
  //! AMR
  delete m_mesh_refinement;
  delete m_submesh_tools;
  delete m_new_item_owner_builder;

  // Destroys dynamically allocated families.
  for (IItemFamily* family : m_item_families) {
    eItemKind kind = family->itemKind();
    // Only particle or DoF families are dynamically allocated.
    // TODO: now they are all so they should all be
    // deallocated by this loop
    if (kind == IK_Particle || kind == IK_DoF)
      delete family;
  }
  m_properties->destroy();
  delete m_properties;

  delete m_cell_family;
  delete m_face_family;
  delete m_edge_family;
  delete m_node_family;

  delete m_item_type_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
build()
{
  Trace::Setter mci(traceMng(), _className());

  info() << "Building DynamicMesh name=" << name()
         << " ItemInternalMapImpl=" << ItemInternalMap::UseNewImpl;

  m_item_type_mng->build(this);

  m_tied_interface_mng = new TiedInterfaceMng(this);

  // IMPORTANT: the first elements of m_item_families must
  // correspond to the families associated with eItemKind values
  // (IK_Node, IK_Edge, IK_Face, IK_Cell)
  for (IItemFamily* family : m_item_families) {
    _buildAndInitFamily(family);
  }

  // Allows calling the '_prepareDump' function when a
  // dump is performed
  IVariableMng* vm = subDomain()->variableMng();
  m_observer_pool.addObserver(this,
                              &DynamicMesh::_prepareForDump,
                              vm->writeObservable());
  // Allows calling the '_readFromDump' function when a
  // dump is read
  m_observer_pool.addObserver(this,
                              &DynamicMesh::_readFromDump,
                              vm->readObservable());

  m_mesh_builder = new DynamicMeshIncrementalBuilder(this);
  m_mesh_checker = new DynamicMeshChecker(this);
  m_partition_constraint_mng = new MeshPartitionConstraintMng(this);

  if (parentMesh()) {
    // This can induce segfaults if the DynamicMesh is not used correctly
    m_submesh_tools = new SubMeshTools(this, m_mesh_builder);
    //! AMR
    m_mesh_refinement = 0;

    this->properties()->setBool(PROPERTY_COMPACT, true);
    this->properties()->setBool(PROPERTY_SORT, true);

    ItemGroupDynamicMeshObserver* obs = NULL;
    m_parent_group->attachObserver(this, (obs = new ItemGroupDynamicMeshObserver(this)));

    this->endAllocate();
    Int32ConstArrayView localIds = m_parent_group->itemsLocalId();
    obs->executeExtend(&localIds);
    this->endUpdate();
  }
  else {
    m_submesh_tools = 0;
    //! AMR

    // GG: does not build m_mesh_refinement if AMR is active
    // This avoids creating unnecessary variables when AMR is not
    // requested. The creation of m_mesh_refinement is now
    // done in readAmrActivator(). This might not be the right place
    // to do it and it can potentially be moved. It just needs
    // m_mesh_refinement to be created only if AMR is active.
    // SDC: OK. Restored because now the AMR (in)active info is known at
    // construction. Removal of readAmrActivator.

    if (m_is_amr_activated) {
      if (m_amr_type == eMeshAMRKind::None || m_amr_type == eMeshAMRKind::Cell) {
        m_mesh_refinement = new MeshRefinement(this);
      }
      else if (m_amr_type == eMeshAMRKind::Patch) {
        ARCANE_FATAL("Patch AMR type is not implemented.");
      }
      else if (m_amr_type == eMeshAMRKind::PatchCartesianMeshOnly) {
        // The AMR PatchCartesianMeshOnly is not handled by MeshRefinement().
        // See in CartesianMesh.cc.
        // TODO: CartesianMeshAMRPatchMng needs it for ghost cells.
        //        See to remove or replace the call to the method
        //        updateGhostLayerFromParent().
        m_mesh_refinement = new MeshRefinement(this);
      }
    }
  }

  m_internal_api->build();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* DynamicMesh::
parallelMng()
{
  return m_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_checkValidItem(ItemInternal* item)
{
  if (!item)
    ARCANE_FATAL("INTERNAL: DynamicMesh: invalid use of a null entity");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
prepareForDump()
{
  _prepareForDump();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
reloadMesh()
{
  Trace::Setter mci(traceMng(), _className());
  info() << "Reloading the mesh " << name();
  m_is_allocated = true;
  Timer timer(subDomain(), "DynamicMesh::reloadMesh", Timer::TimerReal);
  {
    Timer::Sentry sentry(&timer);
    computeSynchronizeInfos();
    m_mesh_checker->checkMeshFromReferenceFile();
  }
  info() << "Time to reallocate the mesh structures (direct method) (unit: second): "
         << timer.lastActivationTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
setCheckLevel(Integer level)
{
  m_mesh_checker->setCheckLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer DynamicMesh::
checkLevel() const
{
  return m_mesh_checker->checkLevel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
checkValidMesh()
{
  if (!m_is_allocated)
    return;
  m_mesh_checker->checkValidMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
checkValidMeshFull()
{
  if (!m_is_allocated)
    return;
  m_mesh_checker->checkValidMeshFull();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
checkValidConnectivity()
{
  m_mesh_checker->checkValidConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
destroyGroups()
{
  for (IItemFamily* family : m_item_families) {
    family->destroyGroups();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup DynamicMesh::
findGroup(const String& name)
{
  ItemGroup group;
  for (IItemFamily* family : m_item_families) {
    group = family->findGroup(name);
    if (!group.null())
      return group;
  }
  return group;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup DynamicMesh::
findGroup(const String& name, eItemKind ik, bool create_if_needed)
{
  _checkKindRange(ik);
  IItemFamily* family = m_item_families[ik];
  return family->findGroup(name, create_if_needed);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup DynamicMesh::
createGroup(const String& name, eItemKind ik)
{
  _checkKindRange(ik);
  IItemFamily* family = m_item_families[ik];
  ARCANE_CHECK_PTR(family);
  return family->findGroup(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup DynamicMesh::
createGroup(const String& name, const ItemGroup& parent)
{
  IItemFamily* family = parent.itemFamily();
  ARCANE_CHECK_PTR(family);
  return family->createGroup(name, parent);
}

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

void DynamicMesh::
_computeSynchronizeInfos()
{
  _computeFamilySynchronizeInfos();
  _computeGroupSynchronizeInfos();
}

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

void DynamicMesh::
_computeFamilySynchronizeInfos()
{
  info() << "Computing family synchronization information for " << name();
  for (IItemFamily* family : m_item_families) {
    family->computeSynchronizeInfos();
  }

  // Writes the topology for cell synchronization
  if (!platform::getEnvironmentVariable("ARCANE_DUMP_VARIABLE_SYNCHRONIZER_TOPOLOGY").null()) {
    auto* var_syncer = cellFamily()->allItemsSynchronizer();
    Int32 iteration = subDomain()->commonVariables().globalIteration();
    String file_name = String::format("{0}_sync_topology_iter{1}.json", name(), iteration);
    mesh_utils::dumpSynchronizerTopologyJSON(var_syncer, file_name);
  }
}

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

void DynamicMesh::
_computeGroupSynchronizeInfos()
{
  auto action = [](ItemGroup& group) {
    if (group.hasSynchronizer())
      group.synchronizer()->compute();
  };

  info() << "Computing group synchronization information for " << name();
  meshvisitor::visitGroups(this, action);
}

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

ItemGroupCollection DynamicMesh::
groups()
{
  m_all_groups.clear();
  for (IItemFamily* family : m_item_families) {
    for (ItemGroupCollection::Enumerator i_group(family->groups()); ++i_group;)
      m_all_groups.add(*i_group);
  }
  return m_all_groups;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo: Do not do this here.
 */
void DynamicMesh::
initializeVariables(const XmlNode& init_node)
{
  // TEMPORARY: copies the owner() field for each family into
  // the corresponding variable.
  for (IItemFamily* family : m_item_families) {
    VariableItemInt32& items_owner(family->itemsNewOwner());
    ENUMERATE_ITEM (iitem, family->allItems()) {
      Item item = *iitem;
      items_owner[iitem] = item.owner();
    }
  }

  ICaseDocumentFragment* doc = subDomain()->caseMng()->caseDocumentFragment();
  XmlNode root = doc->rootElement();
  if (root.null())
    return;

  bool has_error = false;
  IVariableMng* vm = m_sub_domain->variableMng();
  XmlNodeList child_list = init_node.children(String("variable"));
  for (const auto& i : child_list) {
    String var_name = i.attrValue("nom");
    IVariable* var = vm->findMeshVariable(this, var_name);
    if (!var) {
      error() << "Failed to initialize the variable '" << var_name
              << "' : no variable with that name exists";
      has_error = true;
      continue;
    }
    // Does not initialize unused variables.
    if (!var->isUsed())
      continue;

    // Tests if the variable has a family. Normally this is always
    // the case because we used findMeshVariable() to find the variable.
    IItemFamily* family = var->itemFamily();
    if (!family) {
      error() << "Variable '" << var->name() << "' has no family";
      continue;
    }
    String grp_name = i.attrValue("groupe");
    ItemGroup grp = family->findGroup(grp_name);

    if (grp.null()) {
      error() << "Failed to initialize the variable '" << var_name
              << "' on the group `" << grp_name << "' : "
              << "No group with that name exists for family '" << family->name() << "'\n";
      has_error = true;
      continue;
    }
    debug() << "Read value variable `" << grp_name
            << "' `" << var_name << "' " << var;
    String val_str = i.attrValue("valeur");
    bool ret = var->initialize(grp, val_str);
    if (ret) {
      error() << "Failed to initialized the variable '" << var_name
              << "' on the group `" << grp_name << "'";
      has_error = true;
      continue;
    }
  }
  if (has_error)
    ARCANE_FATAL("Variables initialization failed");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// NOTE: This method probably does not work if there are variables
// partial because the groups they rely on will be destroyed
// (To be checked)

void DynamicMesh::
deallocate()
{
  if (!m_is_allocated)
    ARCANE_FATAL("mesh is not allocated");

  clearItems();
  destroyGroups();
  m_mesh_builder->resetAfterDeallocate();

  m_is_allocated = false;
  m_mesh_dimension = (-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
allocateCells(Integer mesh_nb_cell, Int64ConstArrayView cells_infos, bool one_alloc)
{
  ARCANE_FATAL_IF(m_is_allocated, "mesh has already been allocated (via endAllocate() or allocateCells())");
  if (mesh_nb_cell == 0 && !cells_infos.empty())
    ARCANE_FATAL("Can not dynamically compute the number of cells");

  Trace::Setter mci(traceMng(), _className());

  setEstimatedCells(mesh_nb_cell);

  Timer timer(subDomain(), "AllocateCells", Timer::TimerReal);
  {
    Timer::Sentry sentry(&timer);
    _allocateCells(mesh_nb_cell, cells_infos);
    if (one_alloc)
      endAllocate();
  }
  info() << "Time to build the mesh structures (indirect method) (unit: second): "
         << timer.lastActivationTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
endAllocate()
{
  ARCANE_FATAL_IF(m_is_allocated, "mesh has already been allocated (via endAllocate() or allocateCells())");

  Trace::Setter mci(traceMng(), _className());

  _checkDimension(); // HP: add control if endAllocate is called
  _checkConnectivity(); // without any cell allocation

  bool print_stats = true;
  ITimeStats* ts = m_sub_domain->timeStats();
  IParallelMng* pm = parallelMng();
  if (print_stats) {
    info() << "Begin compute face unique ids";
    ts->dumpTimeAndMemoryUsage(pm);
  }
  m_mesh_builder->computeFacesUniqueIds();
  if (print_stats) {
    info() << "Begin compute ghost layer";
    ts->dumpTimeAndMemoryUsage(pm);
  }
  m_mesh_builder->addGhostLayers(true);
  if (print_stats) {
    info() << "Begin compact items";
    ts->dumpTimeAndMemoryUsage(pm);
  }
  _allocateCells2(m_mesh_builder);

  if (m_properties->getBool(PROPERTY_COMPACT_AFTER_ALLOCATE))
    _compactItems(true, false);

  _compactItemInternalReferences();
  for (IItemFamily* family : m_item_families)
    family->_internalApi()->endAllocate();

  if (print_stats)
    ts->dumpTimeAndMemoryUsage(pm);

#ifdef ARCANE_DEBUG_DYNAMIC_MESH
  {
    String file_name("mesh-end-allocate");
    if (parallelMng()->isParallel()) {
      file_name += "-";
      file_name += m_mesh_rank;
    }
    mesh_utils::writeMeshConnectivity(this, file_name);
  }
#endif

  computeSynchronizeInfos();
  m_mesh_checker->checkMeshFromReferenceFile();

  // Positions the owners so that it is like after
  // an exchange. This is not strictly necessary
  // but it ensures consistency with
  // other calls to setOwnersFromCells()
  if (parallelMng()->isParallel()) {
    String s = platform::getEnvironmentVariable("ARCANE_CHANGE_OWNER_ON_INIT");
    if (!s.null()) {
      info() << "** Set owners from cells";
      _setOwnersFromCells();
    }
  }

  // Notifies the families that the mesh has just been updated
  // So that they can recalculate the information they want.
  _notifyEndUpdateForFamilies();

  _prepareForDump();

  m_is_allocated = true;
  if (arcaneIsCheck())
    checkValidMesh();

  // Displays the statistics of the new mesh
  {
    MeshStats ms(traceMng(), this, m_parallel_mng);
    ms.dumpStats();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_allocateCells(Integer mesh_nb_cell,
               Int64ConstArrayView cells_infos,
               Int32ArrayView cells,
               bool allow_build_face)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  Int32 rank = meshRank();
  if (m_use_mesh_item_family_dependencies)
    m_mesh_builder->addCells3(mesh_nb_cell, cells_infos, rank, cells, allow_build_face);
  else
    m_mesh_builder->addCells(mesh_nb_cell, cells_infos, rank, cells, allow_build_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addCells(Integer nb_cell,
         Int64ConstArrayView cells_infos,
         Int32ArrayView cells)
{
  const bool allow_build_face = (m_parallel_mng->commSize() == 1);
  _allocateCells(nb_cell, cells_infos, cells, allow_build_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addCells(const MeshModifierAddCellsArgs& args)
{
  bool allow_build_face = args.isAllowBuildFaces();
  // In parallel, we cannot build faces on the fly
  // (because it is necessary to generate a uniqueId(), and this ID would not be consistent
  // between the sub-domains)
  if (m_parallel_mng->commSize() > 1)
    allow_build_face = false;
  _allocateCells(args.nbCell(), args.cellInfos(), args.cellLocalIds(), allow_build_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addCells(ISerializer* buffer)
{
  _addCells(buffer, 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addCells(ISerializer* buffer, Int32Array& cells_local_id)
{
  _addCells(buffer, &cells_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_addCells(ISerializer* buffer, Int32Array* cells_local_id)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  if (!itemFamilyNetwork() ||
      !(itemFamilyNetwork() && itemFamilyNetwork()->isActivated()) ||
      !IItemFamilyNetwork::plug_serializer) {
    buffer->setMode(ISerializer::ModeGet);
    ScopedPtrT<IItemFamilySerializer> cell_serializer(m_cell_family->policyMng()->createSerializer());
    cell_serializer->deserializeItems(buffer, cells_local_id);
  }
  else {
    _deserializeItems(buffer, cells_local_id, m_cell_family);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
allocate(UnstructuredMeshAllocateBuildInfo& build_info)
{
  auto* x = build_info._internal();
  setDimension(x->meshDimension());
  allocateCells(x->nbCell(), x->cellsInfos(), true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
allocate(CartesianMeshAllocateBuildInfo& build_info)
{
  // The allocation of mesh entities is performed by the
  // DynamicMeshCartesianBuilder class.
  allocateCartesianMesh(this, build_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
serializeCells(ISerializer* buffer, Int32ConstArrayView cells_local_id)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  if (!itemFamilyNetwork() ||
      !(itemFamilyNetwork() && itemFamilyNetwork()->isActivated()) ||
      !IItemFamilyNetwork::plug_serializer) {
    ScopedPtrT<IItemFamilySerializer> cell_serializer(m_cell_family->policyMng()->createSerializer());
    buffer->setMode(ISerializer::ModeReserve);
    cell_serializer->serializeItems(buffer, cells_local_id);
    buffer->allocateBuffer();
    buffer->setMode(ISerializer::ModePut);
    cell_serializer->serializeItems(buffer, cells_local_id);
  }
  else {
    _serializeItems(buffer, cells_local_id, m_cell_family);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_serializeItems(ISerializer* buffer, Int32ConstArrayView item_local_ids, IItemFamily* item_family)
{
  // 1-Get item lids for family and downard dependencies.
  // Rk downard relations are not taken here => automatically added in addItems(ItemData) :
  // this will change. Todo add relations when addItems has been updated
  using FamilyLidMap = std::map<String, Int32UniqueArray>;
  FamilyLidMap serialized_items;
  serialized_items[item_family->name()] = item_local_ids;
  for (const auto& connectivity : mesh()->itemFamilyNetwork()->getChildDependencies(item_family)) {
    ENUMERATE_ITEM (item, item_family->view(item_local_ids)) {
      auto connectivity_accessor = ConnectivityItemVector{ connectivity };
      auto& connected_family_serialized_items = serialized_items[connectivity->targetFamily()->name()];
      connected_family_serialized_items.addRange(connectivity_accessor.connectedItems(ItemLocalId(item)).localIds());
    }
  }
  // 2-Serialize each family in the buffer. The order is important: use the family graph from leaves to root
  buffer->setMode(ISerializer::ModeReserve);
  _fillSerializer(buffer, serialized_items);
  buffer->allocateBuffer();
  buffer->setMode(ISerializer::ModePut);
  _fillSerializer(buffer, serialized_items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_deserializeItems(ISerializer* buffer, Int32Array* item_local_ids, IItemFamily* item_family)
{
  ARCANE_UNUSED(item_family);
  buffer->setMode(ISerializer::ModeGet);
  mesh()->itemFamilyNetwork()->schedule([item_local_ids, buffer](IItemFamily* family) {
    auto family_serializer = std::unique_ptr<IItemFamilySerializer>{ family->policyMng()->createSerializer() }; // todo make_unique (not in 4.7.2...)
    family_serializer->deserializeItems(buffer, item_local_ids);
  },
                                        IItemFamilyNetwork::InverseTopologicalOrder);
  mesh()->itemFamilyNetwork()->schedule([item_local_ids, buffer](IItemFamily* family) {
    auto family_serializer = std::unique_ptr<IItemFamilySerializer>{ family->policyMng()->createSerializer() }; // todo make_unique (not in 4.7.2...)
    family_serializer->deserializeItemRelations(buffer, item_local_ids);
  },
                                        IItemFamilyNetwork::InverseTopologicalOrder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_fillSerializer(ISerializer* buffer, std::map<String, Int32UniqueArray>& serialized_items)
{
  mesh()->itemFamilyNetwork()->schedule([&serialized_items, buffer](IItemFamily* family) {
    auto family_serializer = std::unique_ptr<IItemFamilySerializer>{ family->policyMng()->createSerializer() }; // todo make_unique (not in 4.7.2...)
    auto& serialized_items_local_ids = serialized_items[family->name()];
    family_serializer->serializeItems(buffer, serialized_items_local_ids);
  },
                                        IItemFamilyNetwork::InverseTopologicalOrder);
  mesh()->itemFamilyNetwork()->schedule([&serialized_items, buffer](IItemFamily* family) {
    auto family_serializer = std::unique_ptr<IItemFamilySerializer>{ family->policyMng()->createSerializer() }; // todo make_unique (not in 4.7.2...)
    auto& serialized_items_local_ids = serialized_items[family->name()];
    family_serializer->serializeItemRelations(buffer, serialized_items_local_ids);
  },
                                        IItemFamilyNetwork::InverseTopologicalOrder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addParentCells(ItemVectorView& items)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  m_mesh_builder->addParentCells(items);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! AMR
void DynamicMesh::
addHChildrenCells(Cell parent_cell, Integer nb_cell, Int64ConstArrayView cells_infos, Int32ArrayView cells)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  bool allow_build_face = false /*(m_parallel_mng->commSize() == 1)*/;
  m_mesh_builder->addHChildrenCells(parent_cell, nb_cell, cells_infos, meshRank(), cells, allow_build_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addParentCellToCell(Cell child, Cell parent)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();

  m_cell_family->_addParentCellToCell(child, parent);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addChildCellToCell(Cell parent, Cell child)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();

  m_cell_family->_addChildCellToCell2(parent, child);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addParentFaceToFace(Face child, Face parent)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();

  m_face_family->_addParentFaceToFace(parent, child);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addChildFaceToFace(Face parent, Face child)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();

  m_face_family->_addChildFaceToFace(parent, child);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addParentNodeToNode(Node child, Node parent)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();

  m_node_family->_addParentNodeToNode(parent, child);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addChildNodeToNode(Node parent, Node child)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();

  m_node_family->_addChildNodeToNode(parent, child);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addFaces(Integer nb_face, Int64ConstArrayView face_infos, Int32ArrayView faces)
{
  _checkDimension();
  _checkConnectivity();
  Int32 rank = meshRank();
  if (m_use_mesh_item_family_dependencies)
    m_mesh_builder->addFaces3(nb_face, face_infos, rank, faces);
  else
    m_mesh_builder->addFaces(nb_face, face_infos, rank, faces);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addFaces(const MeshModifierAddFacesArgs& args)
{
  addFaces(args.nbFace(), args.faceInfos(), args.faceLocalIds());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addEdges(Integer nb_edge, Int64ConstArrayView edge_infos, Int32ArrayView edges)
{
  _checkDimension();
  _checkConnectivity();
  Int32 rank = meshRank();
  if (m_use_mesh_item_family_dependencies)
    m_mesh_builder->addEdges3(nb_edge, edge_infos, rank, edges);
  else
    m_mesh_builder->addEdges(nb_edge, edge_infos, rank, edges);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addNodes(Int64ConstArrayView nodes_uid, Int32ArrayView nodes)
{
  _checkDimension();
  _checkConnectivity();
  Int32 rank = meshRank();
  if (m_use_mesh_item_family_dependencies)
    m_mesh_builder->addNodes2(nodes_uid, rank, nodes);
  else
    m_mesh_builder->addNodes(nodes_uid, rank, nodes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
removeCells(Int32ConstArrayView cells_local_id, bool update_graph)
{
  ARCANE_UNUSED(update_graph);
  Trace::Setter mci(traceMng(), _className());
  if (m_use_mesh_item_family_dependencies)
    removeItems(m_cell_family, cells_local_id);
  else
    m_cell_family->internalRemoveItems(cells_local_id);

  if (m_item_family_network) {
    m_item_family_network->removeConnectedDoFsFromCells(cells_local_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
removeItems(IItemFamily* item_family, Int32ConstArrayView cells_local_id)
{
  ARCANE_UNUSED(item_family);
  ARCANE_ASSERT((itemFamilyNetwork()), ("Cannot call DynamicMesh::removeItems if no ItemFamilyNetwork available"))

  if (cells_local_id.empty())
    return;

  // Create item info (to remove items)
  ItemDataList item_data_list;
  ItemData& cell_data = item_data_list.itemData(Integer(m_cell_family->itemKind()),
                                                cells_local_id.size(), cells_local_id.size(), Int32ArrayView(),
                                                m_cell_family, (IItemFamilyModifier*)(m_cell_family), m_parallel_mng->commRank());
  Integer i(0);
  for (auto local_id : cells_local_id) {
    // TODO Find a better place in ItemData to put removed item lids (with Int32...) .
    cell_data.itemInfos()[i++] = (Int64)local_id;
  }
  itemFamilyNetwork()->schedule([&item_data_list](IItemFamily* family) {
    // send the whole ItemDataList since removed items are to be added for child families
    family->removeItems2(item_data_list);
  },
                                IItemFamilyNetwork::TopologicalOrder); // item destruction done from root to leaves
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
detachCells(Int32ConstArrayView cells_local_id)
{
  Trace::Setter mci(traceMng(), _className());
  if (m_use_mesh_item_family_dependencies)
    m_cell_family->detachCells2(cells_local_id);
  else {
    ItemInternalList cells = m_cell_family->itemsInternal();
    for (Integer i = 0, is = cells_local_id.size(); i < is; ++i)
      m_cell_family->detachCell(cells[cells_local_id[i]]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
removeDetachedCells(Int32ConstArrayView cells_local_id)
{
  Trace::Setter mci(traceMng(), _className());
  if (m_use_mesh_item_family_dependencies)
    removeItems(m_cell_family, cells_local_id);
  else {
    ItemInternalList cells = m_cell_family->itemsInternal();
    for (Integer i = 0, is = cells_local_id.size(); i < is; ++i)
      m_cell_family->removeDetachedCell(cells[cells_local_id[i]]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! AMR
void DynamicMesh::
flagCellToRefine(Int32ConstArrayView lids)
{
  Trace::Setter mci(traceMng(), _className());
  _checkAMR();
  m_mesh_refinement->flagCellToRefine(lids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
flagCellToCoarsen(Int32ConstArrayView lids)
{
  Trace::Setter mci(traceMng(), _className());
  _checkAMR();
  m_mesh_refinement->flagCellToCoarsen(lids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
refineItems()
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  _checkAMR();
  m_mesh_refinement->refineItems(true);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
coarsenItems()
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  _checkAMR();
  m_mesh_refinement->coarsenItems(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
coarsenItemsV2(bool update_parent_flag)
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  _checkAMR();
  if (m_amr_type != eMeshAMRKind::Cell) {
    ARCANE_FATAL("This method is not compatible with Cartesian Mesh Patch AMR");
  }
  m_mesh_refinement->coarsenItemsV2(update_parent_flag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DynamicMesh::
adapt()
{
  Trace::Setter mci(traceMng(), _className());
  _checkDimension();
  _checkConnectivity();
  _checkAMR();
  if (m_mesh_refinement->needUpdate())
    m_mesh_refinement->update();
  return m_mesh_refinement->refineAndCoarsenItems(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
compact()
{
  _compactItems(false, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
registerCallBack(IAMRTransportFunctor* f)
{
  Trace::Setter mci(traceMng(), _className());
  _checkAMR();
  m_mesh_refinement->registerCallBack(f);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
unRegisterCallBack(IAMRTransportFunctor* f)
{
  Trace::Setter mci(traceMng(), _className());
  _checkAMR();
  m_mesh_refinement->unRegisterCallBack(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_allocateCells2(DynamicMeshIncrementalBuilder* mib)
{
  Trace::Setter mci(traceMng(), _className());

  _finalizeMeshChanged();

  mib->printInfos();

#ifdef ARCANE_DEBUG_DYNAMIC_MESH
  OCStringStream ostr;
  _printMesh(ostr());
  info() << ostr.str();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_writeMesh(const String& base_name)
{
  StringBuilder file_name(base_name);
  file_name += "-";
  IParallelMng* pm = m_parallel_mng;
  bool is_parallel = pm->isParallel();
  Int32 rank = meshRank();
  if (is_parallel) {
    file_name += subDomain()->commonVariables().globalIteration();
    file_name += "-";
    file_name += rank;
  }
  mesh_utils::writeMeshConnectivity(this, file_name.toString());
  //TODO ability to change the service name
  auto writer(ServiceBuilder<IMeshWriter>::createReference(subDomain(), "Lima"));
  if (writer.get()) {
    String mesh_file_name = file_name.toString() + ".mli";
    writer->writeMeshToFile(this, mesh_file_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_printMesh(std::ostream& ostr)
{
  ostr << "----------- Mesh\n";
  ostr << " Nodes: " << nbNode() << '\n';
  ostr << " Edges: " << nbEdge() << '\n';
  ostr << " Faces: " << nbFace() << '\n';
  ostr << " Cells: " << nbCell() << '\n';
  mesh_utils::printItems(ostr, "Nodes", allNodes());
  mesh_utils::printItems(ostr, "Edges", allEdges());
  mesh_utils::printItems(ostr, "Faces", allFaces());
  mesh_utils::printItems(ostr, "Cells", allCells());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Saves properties before a dump.
 */
void DynamicMesh::
_saveProperties()
{
  info(4) << "DynamicMesh::_saveProperties() name=" << name()
          << " nb-ghost=" << ghostLayerMng()->nbGhostLayer();

  auto p = m_properties;
  p->setInt32("nb-ghostlayer", ghostLayerMng()->nbGhostLayer());
  p->setInt32("ghostlayer-builder-version", ghostLayerMng()->builderVersion());
  p->setInt32("part-info-part-rank", m_mesh_part_info.partRank());
  p->setInt32("part-info-nb-part", m_mesh_part_info.nbPart());
  p->setInt32("part-info-replication-rank", m_mesh_part_info.replicationRank());
  p->setInt32("part-info-nb-replication", m_mesh_part_info.nbReplication());
  p->setBool("has-itemsharedinfo-variables", true);
  p->setInt64("mesh-timestamp", m_timestamp);
  if (!m_do_not_save_need_compact)
    p->setBool("need-compact", m_need_compact);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Reloads properties from a dump.
 */
void DynamicMesh::
_loadProperties()
{
  auto p = m_properties;

  info(4) << "DynamicMesh::_readProperties() name=" << name()
          << " mesh-version=" << p->getInt32WithDefault(PROPERTY_MESH_VERSION, -1);

  {
    // Reloads info on the ghost layer manager.
    Int32 x = 0;
    if (p->get("nb-ghostlayer", x))
      ghostLayerMng()->setNbGhostLayer(x);
    if (p->get("ghostlayer-builder-version", x))
      ghostLayerMng()->setBuilderVersion(x);
    if (p->get("part-info-part-rank", x))
      m_mesh_part_info.setPartRank(x);
    if (p->get("part-info-nb-part", x))
      m_mesh_part_info.setNbPart(x);
    if (p->get("part-info-replication-rank", x))
      m_mesh_part_info.setReplicationRank(x);
    if (p->get("part-info-nb-replication", x))
      m_mesh_part_info.setNbReplication(x);
    if (!m_do_not_save_need_compact) {
      bool xb = false;
      if (p->get("need-compact", xb))
        m_need_compact = xb;
    }
    Int64 x2 = 0;
    if (p->get("mesh-timestamp", x2))
      m_timestamp = x2;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Prepares variables for a dump.
 */
void DynamicMesh::
_prepareForDump()
{
  bool want_dump = m_properties->getBool(PROPERTY_DUMP);
  info(4) << "DynamicMesh::prepareForDump() name=" << name()
          << " need_compact?=" << m_need_compact
          << " want_dump?=" << want_dump
          << " timestamp=" << m_timestamp;

  {
    eMeshEventType t = eMeshEventType::BeginPrepareDump;
    m_mesh_events.eventObservable(t).notify(MeshEventArgs(this, t));
  }

  // If the mesh is not saved, do nothing. This prevents compacting
  // and sorting the mesh, which is undesirable if the properties
  // 'sort' and 'compact' are set to 'false'.
  if (want_dump)
    _prepareForDumpReal();

  // Save properties. This must be done at the end because the call to
  // prepareForDumpReal() might modify these properties
  _saveProperties();

  {
    eMeshEventType t = eMeshEventType::EndPrepareDump;
    m_mesh_events.eventObservable(t).notify(MeshEventArgs(this, t));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Prepares variables for a dump.
 */
void DynamicMesh::
_prepareForDumpReal()
{
  if (m_need_compact) {
    // For now, the entities must be sorted and compacted
    // before a save
    _compactItems(true, true);
  }

  // Preparation of mesh/sub-mesh connections
  {
    if (m_parent_mesh) {
      ARCANE_ASSERT((m_parent_group != NULL), ("Unexpected NULL parent group"));
      m_parent_mesh_name = m_parent_mesh->name();
      m_parent_group_name = m_parent_group->name();
    }
    else {
      ARCANE_ASSERT((m_parent_group == NULL), ("Unexpected non-NULL parent group"));
      m_parent_mesh_name = String();
      m_parent_group_name = String();
    }
    const Integer n_sub_mesh = m_child_meshes.size();
    m_child_meshes_name.resize(n_sub_mesh);
    for (Integer i = 0; i < n_sub_mesh; ++i)
      m_child_meshes_name[i] = m_child_meshes[i]->name();
  }

  // Save info on entity families
  {
    Integer nb_item_family = m_item_families.count();
    m_item_families_name.resize(nb_item_family);
    m_item_families_kind.resize(nb_item_family);
    Integer index = 0;
    for (IItemFamily* family : m_item_families) {
      m_item_families_kind[index] = family->itemKind();
      m_item_families_name[index] = family->name();
      ++index;
    }
  }

  for (IItemFamily* family : m_item_families) {
    family->prepareForDump();
  }

  if (m_tied_interface_need_prepare_dump) {
    m_tied_interface_mng->prepareTiedInterfacesForDump();
    m_tied_interface_need_prepare_dump = false;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* DynamicMesh::
createItemFamily(eItemKind ik, const String& name)
{
  IItemFamily* xfamily = findItemFamily(ik, name, false, false);
  if (xfamily)
    ARCANE_FATAL("Attempting to create a family that already exists '{0}'", name);

  debug() << "Creating the entities family "
          << " name=" << name
          << " kind=" << itemKindName(ik);
  ItemFamily* family = _createNewFamily(ik, name);

  _addFamily(family);
  _buildAndInitFamily(family);

  return family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamily* DynamicMesh::
_createNewFamily(eItemKind kind, const String& name)
{
  switch (kind) {
  case IK_Node:
    return new NodeFamily(this, name);
  case IK_Edge:
    return new EdgeFamily(this, name);
  case IK_Face:
    return new FaceFamily(this, name);
  case IK_Cell:
    return new CellFamily(this, name);
  case IK_Particle:
    return new ParticleFamily(this, name);
  case IK_DoF:
    return new DoFFamily(this, name);
  case IK_Unknown:
    ARCANE_FATAL("Attempting to create an ItemFamily with an unknown item kind.");
  }
  ARCANE_FATAL("Invalid ItemKind");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilyPolicyMng* DynamicMesh::
_createFamilyPolicyMng(ItemFamily* family)
{
  eItemKind kind = family->itemKind();
  switch (kind) {
  case IK_Node:
    return createNodeFamilyPolicyMng(family);
  case IK_Edge:
    return createEdgeFamilyPolicyMng(family);
  case IK_Face:
    return createFaceFamilyPolicyMng(family);
  case IK_Cell:
    return createCellFamilyPolicyMng(family);
  case IK_Particle:
    return createParticleFamilyPolicyMng(family);
  case IK_DoF:
    return createDoFFamilyPolicyMng(family);
  case IK_Unknown:
    ARCANE_FATAL("Attempting to create an ItemFamily with an unknown item kind.");
  }
  ARCANE_FATAL("Invalid ItemKind");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_buildAndInitFamily(IItemFamily* family)
{
  family->build();
  ItemFamily* true_family = ARCANE_CHECK_POINTER(dynamic_cast<ItemFamily*>(family));
  IItemFamilyPolicyMng* policy_mng = _createFamilyPolicyMng(true_family);
  true_family->setPolicyMng(policy_mng);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_addFamily(ItemFamily* family)
{
  m_item_families.add(family);
  m_true_item_families.add(family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* DynamicMesh::
findItemFamily(eItemKind ik, const String& name, bool create_if_needed,
               bool register_modifier_if_created)
{
  for (IItemFamily* family : m_item_families)
    if (family->name() == name && family->itemKind() == ik)
      return family;
  if (create_if_needed) {
    IItemFamily* family = createItemFamily(ik, name);
    if (register_modifier_if_created) {
      IItemFamilyModifier* modifier = dynamic_cast<IItemFamilyModifier*>(family);
      if (modifier)
        m_family_modifiers.add(modifier);
    }
    return family;
  }
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* DynamicMesh::
findItemFamily(const String& name, bool throw_exception)
{
  for (IItemFamily* family : m_item_families)
    if (family->name() == name)
      return family;
  if (throw_exception)
    ARCANE_FATAL("No family with name '{0}' exist", name);
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilyModifier* DynamicMesh::
findItemFamilyModifier(eItemKind ik, const String& name)
{
  IItemFamily* family = findItemFamily(ik, name, false, false);
  if (!family)
    return nullptr;
  for (IItemFamilyModifier* modifier : m_family_modifiers)
    if (modifier->family() == family)
      return modifier;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_exchangeItems(bool do_compact)
{
  String nb_exchange_str = platform::getEnvironmentVariable("ARCANE_NB_EXCHANGE");
  // It would be necessary to calculate the default value taking into account the number
  // of exchanged cells and the number of variables and their memory usage.
  // Be careful that all procs use the same value.
  // In practice, it is better not to exceed 3 or 4 exchanges because this
  // increases the time consumed and does not reduce memory enough.
  Integer nb_exchange = 1;
  if (!nb_exchange_str.null()) {
    bool is_bad = builtInGetValue(nb_exchange, nb_exchange_str);
    if (is_bad)
      nb_exchange = 1;
  }
  String exchange_version_str = platform::getEnvironmentVariable("ARCANE_MESH_EXCHANGE_VERSION");
  Integer exchange_version = 1;
  if (!exchange_version_str.null()) {
    builtInGetValue(exchange_version, exchange_version_str);
  }

  info() << "DynamicMesh::_echangeItems() do_compact?=" << do_compact
         << " nb_exchange=" << nb_exchange << " version=" << exchange_version;

  if (nb_exchange > 1) {
    _multipleExchangeItems(nb_exchange, exchange_version, do_compact);
  }
  else
    _exchangeItemsNew();
  // Currently, the exchangeItemsNew() method only takes into account
  // one layer of ghost cells. If we request more, we must
  // add them now. This call is not optimal but allows
  // for correctly processing all cases (I hope).
  if (ghostLayerMng()->nbGhostLayer() > 1 && !m_use_mesh_item_family_dependencies) // many ghost already handled in MeshExchange with ItemFamilyNetwork
    updateGhostLayers(true);
  String check_exchange = platform::getEnvironmentVariable("ARCANE_CHECK_EXCHANGE");
  if (!check_exchange.null()) {
    m_mesh_checker->checkGhostCells();
    pwarning() << "CHECKING SYNCHRONISATION !";
    m_mesh_checker->checkVariablesSynchronization();
    m_mesh_checker->checkItemGroupsSynchronization();
  }
  if (checkLevel() >= 2)
    m_mesh_checker->checkValidMesh();
  else if (checkLevel() >= 1)
    m_mesh_checker->checkValidConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exchanges entities in multiple stages.
 *
 * There are two versions for this mechanism:
 * 1. Version 1, which is the historical version. For this algorithm,
 *    we divide the number of cells to send into \a nb_exchange parts,
 *    each part having (nb_cell / nb_exchange) cells. This algorithm
 *    allows limiting the size of messages but not the number of messages
 *    in flight.
 * 2. Version 2, which separates the list of cells to send based on
 *    the rank of each part. This is useful for limiting the number of
 *    messages sent simultaneously but will not decrease the size of the message
 *    sent to a given rank. Assuming that consecutive ranks
 *    are on the same node of a computer, we separate the exchange into
 *    \a nb_exchange using the following algorithm:
 *    - we denote 'i' as the i-th exchange (numbered from 0 to (nb_exchange-1)),
 *    - for exchange 'i', we only process the cells whose new
 *      owner modulo (nb_exchange) equals 'i'.
 *
 * We optimize slightly by only performing the optional compaction
 * once.
 *
 * TODO: optimize even better with a special function
 * instead of calling _exchangeItems();
 * TODO: instead of dividing the list of cells into \a nb_exchange
 * arbitrary parts, it would be necessary to do it by taking
 * adjacent cells to avoid having too many exchanges
 * of ghost cells if the cells are distributed randomly.
 */
void DynamicMesh::
_multipleExchangeItems(Integer nb_exchange, Integer version, bool do_compact)
{
  if (version < 1 || version > 2)
    ARCANE_FATAL("Invalid value '{0}' for version. Valid values are 1 or 2", version);

  info() << "** ** MULTIPLE EXCHANGE ITEM version=" << version << " nb_exchange=" << nb_exchange;
  UniqueArray<UniqueArray<Int32>> cells_to_exchange_new_owner(nb_exchange);
  // We must store the uid because after load balancing the localIds will change
  UniqueArray<UniqueArray<Int64>> cells_to_exchange_uid(nb_exchange);

  IItemFamily* cell_family = cellFamily();
  VariableItemInt32& cells_new_owner = cell_family->itemsNewOwner();

  Integer nb_cell = ownCells().size();
  ENUMERATE_CELL (icell, ownCells()) {
    Cell cell = *icell;
    Int32 current_owner = cell.owner();
    Int32 new_owner = cells_new_owner[icell];
    if (current_owner == new_owner)
      continue;
    Integer phase = 0;
    if (version == 2)
      phase = (new_owner % nb_exchange);
    else if (version == 1)
      phase = icell.index() / nb_cell;
    cells_to_exchange_new_owner[phase].add(new_owner);
    cells_to_exchange_uid[phase].add(cell.uniqueId().asInt64());
  }

  // Sets it as if the cell did not change owner to
  ENUMERATE_CELL (icell, ownCells()) {
    Cell cell = *icell;
    cells_new_owner[icell] = cell.owner();
  }

  // From here, cells_new_owner is identical to cell.owner()
  // for each cell.
  Int32UniqueArray uids_to_lids;
  for (Integer i = 0; i < nb_exchange; ++i) {
    Int32ConstArrayView new_owners = cells_to_exchange_new_owner[i];
    Int64ConstArrayView new_uids = cells_to_exchange_uid[i];
    Integer nb_cell = new_uids.size();
    info() << "MultipleExchange current_exchange=" << i << " nb_cell=" << nb_cell;
    uids_to_lids.resize(nb_cell);
    cell_family->itemsUniqueIdToLocalId(uids_to_lids, new_uids);
    ItemInternalList cells = cell_family->itemsInternal();
    // For each cell in the current exchange part, sets the new_owner
    // to the correct value
    for (Integer z = 0; z < nb_cell; ++z)
      cells_new_owner[cells[uids_to_lids[z]]] = new_owners[z];
    cells_new_owner.synchronize();
    mesh()->utilities()->changeOwnersFromCells();
    _exchangeItemsNew();
  }

  if (do_compact) {
    Timer::Action ts_action1(m_sub_domain, "CompactItems", true);
    bool do_sort = m_properties->getBool(PROPERTY_SORT);
    _compactItems(do_sort, true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
exchangeItems()
{
  _exchangeItems(m_properties->getBool(PROPERTY_COMPACT));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
clearItems()
{
  for (IItemFamily* family : m_item_families)
    family->clearItems();
  endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#if 0
class ItemsExchangeInfo2List
  : public UniqueArray<ItemsExchangeInfo2*>
{
public:
  ~ItemsExchangeInfo2List()
  {
    for( Integer i=0, is=size(); i<is; ++i ){
      ItemsExchangeInfo2* info = this->operator[](i);
      delete info;
    }
    clear();
  }
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_exchangeItemsNew()
{
  // The algorithm used here is not recursive with submeshes.
  // All communications are grouped and the different levels of modifications
  // will be popped here. This implementation does not allow having
  // more than one submesh level per cell.

  Trace::Setter mci(traceMng(), _className());

  if (!m_is_dynamic)
    ARCANE_FATAL("property isDynamic() has to be 'true'");

  m_need_compact = true;

  if (arcane_debug_load_balancing) {
    // TODO: do this in the MeshExchanger and by family.
    // Checks that the variables are properly synchronized
    m_node_family->itemsNewOwner().checkIfSync(10);
    m_edge_family->itemsNewOwner().checkIfSync(10);
    m_face_family->itemsNewOwner().checkIfSync(10);
    m_cell_family->itemsNewOwner().checkIfSync(10);
  }
  // TODO: Check that everyone has the same families and in the same order.

  // Cascades all meshes associated with this mesh
  typedef Collection<DynamicMesh*> DynamicMeshCollection;
  DynamicMeshCollection all_cascade_meshes = List<DynamicMesh*>();
  all_cascade_meshes.add(this);
  for (Integer i = 0; i < m_child_meshes.size(); ++i)
    all_cascade_meshes.add(m_child_meshes[i]);

  IMeshExchanger* iexchanger = m_mesh_exchange_mng->beginExchange();
  MeshExchanger* mesh_exchanger = ARCANE_CHECK_POINTER(dynamic_cast<MeshExchanger*>(iexchanger));

  // If there are no entities to exchange, the exchange stops immediately.
  if (mesh_exchanger->computeExchangeInfos()) {
    info() << "No load balance is performed";
    m_mesh_exchange_mng->endExchange();
    return;
  }

  // Performs the info exchange
  mesh_exchanger->processExchange();

  // Removes entities that should no longer be in our sub-domain.
  mesh_exchanger->removeNeededItems();

  // Readjusts the groups by removing entities that are no longer in the mesh or by
  // invalidating the calculated groups.
  // TODO: make a method in the family that does this.
  {
    auto action = [](ItemGroup& group) {
      // (HP) TODO: 'if (group.internal()->hasComputeFunctor())' does not work simply, why ?
      // Formerly: if (group.isLocalToSubDomain() || group.isOwn())
      // Redundant with ItemFamily::notifyItemsOwnerChanged calculations
      if (group.internal()->hasComputeFunctor() || group.isLocalToSubDomain())
        group.invalidate();
      else
        group.internal()->removeSuppressedItems();
    };
    for (DynamicMesh* mesh : all_cascade_meshes) {
      meshvisitor::visitGroups(mesh, action);
    }
  }

  // Removes potential ghost items remaining after updating the support group
  // which were marked by NeedRemove (which cleans up a potential inconsistent state of the
  // submesh relative to its parent)
  // This part is important because it is possible that updating the groups
  // is not enough to update the ghost parts of the submeshes
  // (unless a stronger constraint is placed on the group to also include all
  // ghost meshes of the submesh)
#if HEAD
  for (DynamicMesh* child_mesh : m_child_meshes)
    child_mesh->m_submesh_tools->removeDeadGhostCells();
#else
  for (Integer i_child_mesh = 0; i_child_mesh < m_child_meshes.size(); ++i_child_mesh)
    m_child_meshes[i_child_mesh]->m_submesh_tools->removeDeadGhostCells();
#endif

  // Allocates the entities we received from other sub-domains.
  mesh_exchanger->allocateReceivedItems();

  // We now resume a standard endUpdate cycle
  // but interleaving the submesh levels
  for (DynamicMesh* mesh : all_cascade_meshes)
    mesh->_internalEndUpdateInit(true);

  mesh_exchanger->updateItemGroups();

  // Recalculates synchronizers on groups.
  for (DynamicMesh* mesh : all_cascade_meshes)
    mesh->_computeGroupSynchronizeInfos();

  // Updates the variable values of the received entities
  mesh_exchanger->updateVariables();

  // Finalizes the modifications whose sorting and compaction
  for (DynamicMesh* mesh : all_cascade_meshes) {
    // Requests the display of info for the current mesh
    bool print_info = (mesh == this);
    mesh->_internalEndUpdateFinal(print_info);
  }

  // Finalizes the exchanges
  // For now, this is only useful for TiedInterface but it
  // should be removed.
  mesh_exchanger->finalizeExchange();

  // TODO: ensure this call in case of an exception.
  m_mesh_exchange_mng->endExchange();

  // Now, the mesh is updated but the extraordinary ghost cells
  // have been potentially removed. We replace them in the mesh.
  // Non-optimized version. Ideally, extraordinary cells
  // should be managed in MeshExchange.
  // endUpdate() must be called in all cases to ensure
  // that the variables and groups are properly sized.
  if (m_extra_ghost_cells_builder->hasBuilder() || m_extra_ghost_particles_builder->hasBuilder())
    this->endUpdate(true, false);
  else
    this->endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder)
{
  m_extra_ghost_cells_builder->addExtraGhostCellsBuilder(builder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
removeExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder)
{
  m_extra_ghost_cells_builder->removeExtraGhostCellsBuilder(builder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder)
{
  m_extra_ghost_particles_builder->addExtraGhostParticlesBuilder(builder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
removeExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder)
{
  m_extra_ghost_particles_builder->removeExtraGhostParticlesBuilder(builder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_computeExtraGhostCells()
{
  m_extra_ghost_cells_builder->computeExtraGhostCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_computeExtraGhostParticles()
{
  m_extra_ghost_particles_builder->computeExtraGhostParticles();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_removeGhostItems()
{
  const Int32 sid = meshRank();

  // Removal of ghost cells.
  // We must use an intermediate array, because deleting
  // cells invalidates the iterators on 'cells_map'.
  UniqueArray<Int32> cells_to_remove;
  cells_to_remove.reserve(1000);

  ItemInternalMap& cells_map = m_cell_family->itemsMap();
  cells_map.eachItem([&](Item cell) {
    if (cell.owner() != sid)
      cells_to_remove.add(cell.localId());
  });

  info() << "Number of cells to remove: " << cells_to_remove.size();
  m_cell_family->removeCells(cells_to_remove);

  // Readjusts the groups by removing entities that are no longer in the mesh
  _updateGroupsAfterRemove();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Updates groups following entity deletions.
 */
void DynamicMesh::
_updateGroupsAfterRemove()
{
  auto action = [&](const ItemGroup& group) { group.itemFamily()->partialEndUpdateGroup(group); };
  meshvisitor::visitGroups(this, action);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
updateGhostLayers()
{
  updateGhostLayers(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
updateGhostLayers(bool remove_old_ghost)
{
  Trace::Setter mci(traceMng(), _className());

  if (!m_is_dynamic)
    ARCANE_FATAL("property isDynamic() has to be 'true'");
  if (m_parent_mesh)
    ARCANE_FATAL("Cannot be called on submesh");

  _internalUpdateGhost(true, remove_old_ghost);
  _internalEndUpdateInit(true);
  _synchronizeGroups();
  _computeGroupSynchronizeInfos();
  _internalEndUpdateResizeVariables();
  _synchronizeVariables();
  _internalEndUpdateFinal(true);

  // Finalization of recursive submeshes
  for (Integer i = 0; i < m_child_meshes.size(); ++i) {
    m_child_meshes[i]->endUpdate(true, remove_old_ghost);
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_removeGhostChildItems()
{
  const Int32 sid = meshRank();

  // Removal of cells
  UniqueArray<Int32> cells_to_remove;
  cells_to_remove.reserve(1000);

  ItemInternalMap& cells_map = m_cell_family->itemsMap();
  Integer max_level = 0;
  cells_map.eachItem([&](impl::ItemBase cell) {
    if ((cell.owner() != sid) && (cell.level() != 0))
      max_level = math::max(cell.level(), max_level);
  });

  if (max_level == 0)
    return;

  cells_map.eachItem([&](impl::ItemBase cell) {
    if ((cell.owner() != sid) && (cell.level() == max_level)) {
      cells_to_remove.add(cell.localId());
    }
  });

  info() << "Number of cells to remove: " << cells_to_remove.size();
  m_cell_family->removeCells(cells_to_remove);

  // Readjusts the groups by removing entities that are no longer in the mesh
  _updateGroupsAfterRemove();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_removeGhostChildItems2(Array<Int64>& cells_to_coarsen)
{
  const Int32 sid = meshRank();

  cells_to_coarsen.reserve(1000);

  // Removal of cells
  UniqueArray<Cell> cells_to_remove;
  cells_to_remove.reserve(1000);

  ItemInternalMap& cells_map = m_cell_family->itemsMap();
  Integer counter = 0;
  cells_map.eachItem([&](Cell cell) {
    if (cell.owner() != sid)
      return;
    if (cell.hasFlags(ItemFlags::II_JustCoarsened)) {
      cells_to_coarsen.add(cell.uniqueId());
      for (Integer c = 0, cs = cell.nbHChildren(); c < cs; c++) {
        cells_to_remove.add(cell.hChild(c));
        counter++;
      }
    }
  });

  if (counter == 0)
    return;

  //info() << "Number of cells to remove: " << cells_to_remove.size();
  for (Integer i = 0, is = cells_to_remove.size(); i < is; ++i)
    m_cell_family->removeCell(cells_to_remove[i]);

  // Readjust groups by removing entities that are no longer in the mesh
  _updateGroupsAfterRemove();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! AMR
void DynamicMesh::
updateGhostLayerFromParent(Array<Int64>& ghost_cell_to_refine_uid,
                           Array<Int64>& ghost_cell_to_coarsen_uid, bool remove_old_ghost)
{
  Trace::Setter mci(traceMng(), _className());
  CHECKPERF(m_perf_counter.start(PerfCounter::UPGHOSTLAYER1));

  m_need_compact = true;
  //Integer current_iteration = subDomain()->commonVariables().globalIteration();
  if (!m_is_dynamic)
    ARCANE_FATAL("Property isDynamic() has to be 'true'");

  if (remove_old_ghost) {
    _removeGhostChildItems2(ghost_cell_to_coarsen_uid);
  }

  // In case of refinement/coarsening, the orientation might be invalid at a point.
  m_face_family->setCheckOrientation(false);

  m_mesh_builder->addGhostChildFromParent(ghost_cell_to_refine_uid);
  m_face_family->setCheckOrientation(true);

  // From here, all mesh entities are known. It
  // is therefore possible to compact them if necessary
  m_mesh_builder->printStats();
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPGHOSTLAYER1))

  CHECKPERF(m_perf_counter.start(PerfCounter::UPGHOSTLAYER2))
  //_finalizeMeshChanged();
  {
    ++m_timestamp;
    for (IItemFamily* family : m_item_families)
      family->endUpdate();
  }
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPGHOSTLAYER2))

  // Reallocate mesh variables because their group has changed
  // TODO: this should be done for every family
  CHECKPERF(m_perf_counter.start(PerfCounter::UPGHOSTLAYER3))
  {
    IVariableMng* vm = m_sub_domain->variableMng();
    VariableCollection used_vars(vm->usedVariables());
    used_vars.each(std::mem_fn(&IVariable::resizeFromGroup));
  }
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPGHOSTLAYER3))

  // Recalculate the necessary information for the synchronization
  // of
  // entities
  //pm->computeSynchronizeInfos();
  CHECKPERF(m_perf_counter.start(PerfCounter::UPGHOSTLAYER4))
  computeSynchronizeInfos();
  _synchronizeGroupsAndVariables();
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPGHOSTLAYER4))

  //Loop of new refine ghost
  CHECKPERF(m_perf_counter.start(PerfCounter::UPGHOSTLAYER5))
  UniqueArray<Integer> ghost_cell_to_refine_lid(ghost_cell_to_refine_uid.size());
  m_cell_family->itemsUniqueIdToLocalId(ghost_cell_to_refine_lid, ghost_cell_to_refine_uid, true);
  ItemInternalList cells = m_cell_family->itemsInternal();
  for (Integer e = 0, i_size = ghost_cell_to_refine_lid.size(); e != i_size; ++e) {
    Cell i_hParent_cell(cells[ghost_cell_to_refine_lid[e]]);
    m_mesh_refinement->populateBackFrontCellsFromParentFaces(i_hParent_cell);

    //need to populate also the new own cell connected to the new ghost
    Integer nb_face = i_hParent_cell.nbFace();
    Integer lid = i_hParent_cell.localId();
    for (Integer iface = 0; iface < nb_face; ++iface) {
      Face face = i_hParent_cell.face(iface);
      Integer nb_cell = face.nbCell();
      for (Integer icell = 0; icell < nb_cell; ++icell) {
        Cell cell = face.cell(icell);
        if ((cell.localId() != lid) && (cell.isOwn())) {
          UniqueArray<ItemInternal*> childs;
          m_face_family->familyTree(childs, cell, false);
          for (Integer i = 0, nchilds = childs.size(); i < nchilds; ++i) {
            ItemInternal* child = childs[i];
            if (child->isAncestor())
              m_mesh_refinement->populateBackFrontCellsFromParentFaces(child);
          }
        }
      }
    }
  }
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPGHOSTLAYER5))

  // Verify that the mesh conforms to the reference
  CHECKPERF(m_perf_counter.start(PerfCounter::UPGHOSTLAYER6))
  m_mesh_checker->checkMeshFromReferenceFile();

  // Compact references to avoid gaps and take
  // advantage of cache effects.
  // NOTE: this is not theoretically necessary but currently is
  // because we do not save the #m_data_index of the ItemInternal.
  //_compactItemInternalReferences();
  bool do_compact = m_properties->getBool(PROPERTY_COMPACT);
  if (do_compact) {
    bool do_sort = m_properties->getBool(PROPERTY_SORT);
    _compactItems(do_sort, do_compact);
  }
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPGHOSTLAYER6))

  if (arcane_debug_load_balancing) {
    _writeMesh("update-ghost-layer-after");
  }

  // Display the statistics of the new mesh
  {
    MeshStats ms(traceMng(), this, m_parallel_mng);
    ms.dumpStats();
    pinfo() << "Proc: " << meshRank()
            << " cellown=" << m_cell_family->allItems().own().size()
            << " cellloc=" << m_cell_family->allItems().size();
  }

  //! AMR
  CHECKPERF(m_perf_counter.start(PerfCounter::UPGHOSTLAYER7))
  if (m_is_amr_activated)
    m_mesh_checker->updateAMRFaceOrientation(ghost_cell_to_refine_uid);

  if (m_mesh_checker->checkLevel() >= 1)
    m_mesh_checker->checkValidConnectivity();
  CHECKPERF(m_perf_counter.stop(PerfCounter::UPGHOSTLAYER7))

#ifdef ACTIVATE_PERF_COUNTER
  m_perf_counter.printInfo(info().file());
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_internalUpdateGhost(bool update_ghost_layer, bool remove_old_ghost)
{
  m_need_compact = true;

  // Overrides behavior for submeshes
  if (parentMesh()) {
    if (update_ghost_layer)
      m_submesh_tools->updateGhostMesh();
  }
  else {
    if (update_ghost_layer) {
      if (remove_old_ghost) {
        _removeGhostItems();
      }
      // In case of refinement/coarsening, the orientation might be invalid at a point.
      m_face_family->setCheckOrientation(false);
      m_mesh_builder->addGhostLayers(false);
      m_face_family->setCheckOrientation(true);
      _computeExtraGhostCells();
      _computeExtraGhostParticles();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_internalEndUpdateInit(bool update_ghost_layer)
{
  // From here, all mesh entities are known. It
  // is therefore possible to compact them if necessary
  //m_mesh_builder->printStats();

  //info() << "Finalize date=" << platform::getCurrentDateTime();
  _finalizeMeshChanged();

  // Recalculate the necessary information for the synchronization
  // of
  // entities
  if (update_ghost_layer) {
    info() << "ComputeSyncInfos date=" << platform::getCurrentDateTime();
    _computeFamilySynchronizeInfos();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_internalEndUpdateResizeVariables()
{
  // Reallocate mesh variables because their group has changed
  for (IItemFamily* family : m_item_families)
    family->_internalApi()->resizeVariables(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_internalEndUpdateFinal(bool print_stat)
{
  // Verify that the mesh conforms to the reference
  m_mesh_checker->checkMeshFromReferenceFile();

  // Compact references to avoid gaps and take
  // advantage of cache effects.
  // NOTE: this is not theoretically necessary but currently is
  // because we do not save the #m_data_index of the ItemInternal.
  {
    //Timer::Action ts_action1(m_sub_domain,"CompactReferences",true);
    _compactItemInternalReferences(); // Utility to confirm

    {
      bool do_compact = m_properties->getBool(PROPERTY_COMPACT);
      info(4) << "DynamicMesh::_internalEndUpdateFinal() compact?=" << do_compact << " sort?=" << m_properties->getBool(PROPERTY_SORT);
      if (do_compact) {
        bool do_sort = m_properties->getBool(PROPERTY_SORT);
        _compactItems(do_sort, do_compact);
      }
    }
  }

  _notifyEndUpdateForFamilies();

  // Display the statistics of the new mesh
  if (print_stat) {
    if (m_properties->getBool(PROPERTY_DISPLAY_STATS)) {
      MeshStats ms(traceMng(), this, m_parallel_mng);
      ms.dumpStats();
    }
  }

  //! AMR
  if (m_is_amr_activated)
    m_mesh_checker->updateAMRFaceOrientation();

  if (m_mesh_checker->checkLevel() >= 1)
    m_mesh_checker->checkValidConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_notifyEndUpdateForFamilies()
{
  for (IItemFamily* family : m_item_families)
    family->_internalApi()->notifyEndUpdateFromMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
endUpdate()
{
  endUpdate(false, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
endUpdate(bool update_ghost_layer, bool remove_old_ghost)
{
  // The order of operations is as follows:
  // 1- Ghost layer update
  // 2- Mesh finalization (freezes items)
  //    Calculate if requested (update_ghost_layer) the synchronizers on the families
  // 3- Synchronize groups (requires updated family synchronizers)
  // 4- Calculate if requested (update_ghost_layer) the synchronizers on the groups
  // 5- Resize variables (families and groups)
  // 6- Synchronize if requested (update_ghost_layer) the variables (families and groups)
  // 7- Finalization including: compaction, sorting (if enabled), statistics, and validity check
  // 8- Recursive calls to submeshes

  Trace::Setter mci(traceMng(), _className());
  _internalUpdateGhost(update_ghost_layer, remove_old_ghost);

  _internalEndUpdateInit(update_ghost_layer);
  if (update_ghost_layer) {
    _synchronizeGroups();
    _computeGroupSynchronizeInfos();
  }
  _internalEndUpdateResizeVariables();
  if (update_ghost_layer) {
    _synchronizeVariables();
  }
  _internalEndUpdateFinal(false);

  // Recursive finalization of submeshes
  for (DynamicMesh* child_mesh : m_child_meshes)
    child_mesh->endUpdate(update_ghost_layer, remove_old_ghost);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
synchronizeGroupsAndVariables()
{
  _synchronizeGroupsAndVariables();
  // Can be tested in the case where the parent group of a submesh
  // before calling the child meshes.
  // This might require an update of the child synchronizers
  for (DynamicMesh* child_mesh : m_child_meshes)
    child_mesh->synchronizeGroupsAndVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_synchronizeGroupsAndVariables()
{
  _synchronizeGroups();
  _synchronizeVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_synchronizeGroups()
{
  {
    ItemGroupsSynchronize igs(m_node_family);
    igs.synchronize();
  }
  {
    ItemGroupsSynchronize igs(m_edge_family);
    igs.synchronize();
  }
  {
    ItemGroupsSynchronize igs(m_face_family);
    igs.synchronize();
  }
  {
    ItemGroupsSynchronize igs(m_cell_family);
    igs.synchronize();
  }
  {
    for (IItemFamily* family : m_item_families) {
      if (family->itemKind() == IK_Particle) {
        IParticleFamily* pfamily = family->toParticleFamily();
        if (pfamily && pfamily->getEnableGhostItems()) {
          ItemGroupsSynchronize igs(family);
          igs.synchronize();
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_synchronizeVariables()
{
  // We only synchronize variables on the items of the current mesh
  // - Graph items are not handled here (it was already removed from
  //   the previous version of the code)
  // - Particles and those without kind (Unknown) are not subject to synchronization
  // Synchronization is factored here by a collection of variables with the same
  // synchronizer (even for synchronizers on groups != family)
  // To preserve a consistent synchronization order, an
  // auxiliary OrderedSyncList structure is used.

  // Can we do this with a more compact structure?
  typedef UniqueArray<IVariableSynchronizer*> OrderedSyncList;
  typedef std::map<IVariableSynchronizer*, VariableCollection> SyncList;
  OrderedSyncList ordered_sync_list;
  SyncList sync_list;

  VariableCollection used_vars(subDomain()->variableMng()->usedVariables());
  for (VariableCollection::Enumerator i_var(used_vars); ++i_var;) {
    IVariable* var = *i_var;
    switch (var->itemKind()) {
    case IK_Node:
    case IK_Edge:
    case IK_Face:
    case IK_Cell:
    case IK_DoF: {
      IVariableSynchronizer* synchronizer = 0;
      if (var->isPartial())
        synchronizer = var->itemGroup().synchronizer();
      else
        synchronizer = var->itemFamily()->allItemsSynchronizer();
      IMesh* sync_mesh = synchronizer->itemGroup().mesh();
      if (sync_mesh != this)
        continue; // we only synchronize on the current mesh
      std::pair<SyncList::iterator, bool> inserter = sync_list.insert(std::make_pair(synchronizer, VariableCollection()));
      if (inserter.second) { // new synchronizer
        ordered_sync_list.add(synchronizer);
      }
      VariableCollection& collection = inserter.first->second;
      collection.add(var);
    } break;
    case IK_Particle:
    case IK_Unknown:
      break;
    }
  }

  for (Integer i_sync = 0; i_sync < ordered_sync_list.size(); ++i_sync) {
    IVariableSynchronizer* synchronizer = ordered_sync_list[i_sync];
    VariableCollection& collection = sync_list[synchronizer];
    synchronizer->synchronize(collection);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_sortInternalReferences()
{
  // TODO: this specific call must be handled by NodeFamily.
  m_node_family->sortInternalReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_finalizeMeshChanged()
{
  ++m_timestamp;
  for (IItemFamily* family : m_item_families) {
    debug() << "_finalizeMeshChanged on " << family->name() << " Family on Mesh " << name();
    family->endUpdate();
  }

  bool do_sort = m_properties->getBool(PROPERTY_SORT);
  if (do_sort)
    _sortInternalReferences();
  m_tied_interface_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_applyCompactPolicy(const String& timer_name,
                    std::function<void(IItemFamilyCompactPolicy*)> functor)
{
  Timer::Action ts_action(m_sub_domain, timer_name);
  for (IItemFamily* family : m_item_families) {
    IItemFamilyCompactPolicy* c = family->policyMng()->compactPolicy();
    if (c)
      functor(c);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_compactItemInternalReferences()
{
  _applyCompactPolicy("CompactConnectivityData", [&](IItemFamilyCompactPolicy* c) { c->compactConnectivityData(); });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_compactItems(bool do_sort, bool compact_variables_and_groups)
{
  if (do_sort)
    info(4) << "Compress and sort the mesh entities " << name() << ".";
  else
    info(4) << "Compress the mesh entities " << name() << ".";

  IMeshCompacter* compacter = m_mesh_compact_mng->beginCompact();

  try {
    compacter->setSorted(do_sort);
    compacter->_setCompactVariablesAndGroups(compact_variables_and_groups);

    compacter->doAllActions();
  }
  catch (...) {
    m_mesh_compact_mng->endCompact();
    throw;
  }
  m_mesh_compact_mng->endCompact();

  if (do_sort) {
    Timer::Action ts_action(m_sub_domain, "CompactItemSortReferences");
    // TODO: put this in the family policy (apparently only useful
    // for node families)
    _sortInternalReferences();
  }

  m_need_compact = false;

  // Consider compaction as a mesh evolution because this
  // changes the structures associated with the entities and their connectivities
  ++m_timestamp;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
setEstimatedCells(Integer nb_cell0)
{
  Real factor = 1.0;
  if (m_parallel_mng->isParallel())
    factor = 1.2; // Considers 20% ghost entities.
  // The following estimations corresponding to an evaluation for a cube of side N=16
  Integer nb_node = Convert::toInteger(nb_cell0 * 1.2 * factor); //     (N+1)^3/N^3 => 1.2
  Integer nb_edge = Convert::toInteger(nb_cell0 * 6.8 * factor); // 6*(N+1)^2*N/N^3 => 6.8
  Integer nb_face = Convert::toInteger(nb_cell0 * 3.4 * factor); // 3*N^2*(N+1)/N^3 => 3.19
  Integer nb_cell = Convert::toInteger(nb_cell0 * 1.0 * factor); // trivial         => 1.0
  info() << "Estimating the number of entities:"
         << " Node=" << nb_node
         << " Edge=" << nb_edge
         << " Face=" << nb_face
         << " Cell=" << nb_cell;
  m_node_family->preAllocate(nb_node);
  m_edge_family->preAllocate(nb_edge);
  m_face_family->preAllocate(nb_face);
  m_cell_family->preAllocate(nb_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Reloads the mesh from protected variables.
 */
void DynamicMesh::
_readFromDump()
{
  _loadProperties();

  // Do nothing on a mesh that has not yet been allocated.
  if (m_mesh_dimension() < 0)
    return;

  // Mesh/sub-mesh connection
  {
    IMeshMng* mm = meshMng();
    if (!m_parent_mesh_name.value().null()) {
      m_parent_mesh = mm->findMeshHandle(m_parent_mesh_name.value()).mesh();
    }
    if (!m_parent_group_name.value().null()) {
      ARCANE_ASSERT((m_parent_mesh != NULL), ("Unexpected NULL Mesh"));
      m_parent_group = m_parent_mesh->findGroup(m_parent_group_name.value()).internal();
    }
    const Integer n_sub_mesh = m_child_meshes_name.size();
    m_child_meshes.resize(n_sub_mesh);
    for (Integer i = 0; i < n_sub_mesh; ++i) {
      IMesh* child_mesh = mm->findMeshHandle(m_child_meshes_name[i]).mesh();
      DynamicMesh* dynamic_child_mesh = dynamic_cast<DynamicMesh*>(child_mesh);
      if (dynamic_child_mesh == 0)
        ARCANE_FATAL("Cannot associate sub mesh from a different concrete type");
      m_child_meshes[i] = dynamic_child_mesh;
    }
  }

  {
    Integer nb_item_family = m_item_families_name.size();
    for (Integer i = 0; i < nb_item_family; ++i) {
      info(5) << "Found family: I=" << i
              << " name=" << m_item_families_name[i]
              << " kind=" << (eItemKind)m_item_families_kind[i];
    }
  }

  // Relit les infos sur les familles d'entités
  {
    Integer nb_item_family = m_item_families_name.size();
    for (Integer i = 0; i < nb_item_family; ++i) {
      findItemFamily((eItemKind)m_item_families_kind[i], m_item_families_name[i], true, false);
    }
  }

  // GG: Update of connectivity choices
  // This must be done here because the families might be created at the beginning of this
  // method and thus not account for connectivity.
  if (!m_is_sub_connectivity_set)
    _setSubConnectivity();

  for (IItemFamily* family : m_item_families)
    family->readFromDump();

  // After reading, the families must be notified of the potential
  // change in entities.
  _notifyEndUpdateForFamilies();

  //TODO: should not be done here.
  m_item_internal_list.nodes = m_node_family->itemsInternal();
  m_item_internal_list.edges = m_edge_family->itemsInternal();
  m_item_internal_list.faces = m_face_family->itemsInternal();
  m_item_internal_list.cells = m_cell_family->itemsInternal();
  m_item_internal_list.mesh = this;

  m_tied_interface_mng->readTiedInterfacesFromDump();

  m_mesh_builder->readFromDump();
  //_writeMesh("after-read-dump-"+name());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DynamicMesh::
hasTiedInterface()
{
  return m_tied_interface_mng->hasTiedInterface();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterfaceCollection DynamicMesh::
tiedInterfaces()
{
  return m_tied_interface_mng->tiedInterfaces();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<TiedInterface*> DynamicMesh::
trueTiedInterfaces()
{
  return m_tied_interface_mng->trueTiedInterfaces();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
computeTiedInterfaces(const XmlNode& mesh_node)
{
  m_tied_interface_mng->computeTiedInterfaces(mesh_node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
setDimension(Integer dim)
{
  _setDimension(dim);
  _checkConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_setDimension(Integer dim)
{
  if (m_is_allocated)
    ARCANE_FATAL("DynamicMesh::setDimension(): mesh is already allocated");
  info() << "Mesh name=" << name() << " set dimension = " << dim;
  m_mesh_dimension = dim;
  const bool is_non_manifold = meshKind().isNonManifold();
  // Force the fact of not re-numbering faces and edges
  // in the case of a non-manifold mesh
  if (is_non_manifold) {
    info() << "Force no-renumbering of edge and face uid because we are using non manifold mesh";
    m_mesh_unique_id_mng->setFaceBuilderVersion(0);
    m_mesh_unique_id_mng->setEdgeBuilderVersion(0);
  }
  bool v = m_mesh_unique_id_mng->isUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId();
  // If the mesh is non-manifold, then it is mandatory to use
  // generation from uniqueId() from nodes to guarantee
  // the consistency of the created entities.
  // This constraint may eventually be removed when this type
  // of mesh is no longer experimental.
  if (!v && is_non_manifold) {
    v = true;
    info() << "Force using edge and face uid generation from nodes because we are using non manifold mesh";
  }
  if (m_mesh_builder) {
    auto* adder = m_mesh_builder->oneMeshItemAdder();
    if (adder)
      adder->setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(v);
  }
  // In 3D, with non-manifold meshes, it is mandatory to create edges.
  // They will be used instead of faces for 2D cells.
  if (dim == 3 && is_non_manifold) {
    Connectivity c(m_mesh_connectivity);
    if (!c.hasConnectivity(Connectivity::CT_HasEdge)) {
      c.enableConnectivity(Connectivity::CT_HasEdge);
      info() << "Force creating edges for 3D non-manifold mesh";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_checkDimension() const
{
  if (m_mesh_dimension() < 0)
    ARCANE_FATAL("dimension not set. setDimension() must be called before allocating cells");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_checkConnectivity()
{
  Connectivity c(m_mesh_connectivity);
  if (!c.isFrozen()) {
    c.freeze(this);
    debug() << "Mesh " << name() << " connectivity : " << Connectivity::Printer(m_mesh_connectivity());
    _setSubConnectivity();
    _updateItemFamilyDependencies(m_mesh_connectivity);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// AMR
void DynamicMesh::
_checkAMR() const
{
  if (!m_is_amr_activated)
    ARCANE_FATAL("DynamicMesh::_checkAMR(): amr activator not set.\t"
                 "amr='true' must be set in the .arc file");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_setSubConnectivity()
{
  m_mesh_builder->setConnectivity(m_mesh_connectivity());
  m_node_family->setConnectivity(m_mesh_connectivity());
  m_edge_family->setConnectivity(m_mesh_connectivity());
  m_face_family->setConnectivity(m_mesh_connectivity());
  m_cell_family->setConnectivity(m_mesh_connectivity());
  m_is_sub_connectivity_set = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_writeCells(const String& filename)
{
  CellGroup cells(m_cell_family->allItems());
  std::ofstream ofile(filename.localstr());
  ENUMERATE_CELL (icell, cells) {
    Cell cell = *icell;
    ofile << "CELL: uid=" << cell.uniqueId() << " isown="
          << cell.isOwn() << " owner=" << cell.owner() << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableNodeReal3& DynamicMesh::
nodesCoordinates()
{
  return m_node_family->nodesCoordinates();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedVariableNodeReal3 DynamicMesh::
sharedNodesCoordinates()
{
  if (parentMesh())
    return SharedVariableNodeReal3(nodeFamily(), parentMesh()->toPrimaryMesh()->nodesCoordinates());
  else
    return SharedVariableNodeReal3(nodeFamily(), nodesCoordinates());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_setOwnersFromCells()
{
  // We assume we know the new owners of the cells, which
  // are found in cells_owner. We must now determine the new owners of the nodes and
  // faces. Until we have an algorithm that better balances
  // the messages, we apply the following:
  // - each sub-domain is responsible for determining the new
  // owner of the nodes and faces belonging to it.
  // - for nodes and edges, the new owner is the new owner of the cell
  // connected to this node whose uniqueId() is the smallest.
  // - for faces, the new owner is the new owner
  // of the cell behind this face if it is an internal face, and of the connected cell if it is a boundary face.
  // - for dual nodes, the new owner is the new owner
  // of the cell connected to the dual element
  // - for links, the new owner is the new owner
  // of the cell connected to the first dual node, i.e., the owner
  // of the first dual node of the link

  VariableItemInt32& nodes_owner(nodeFamily()->itemsNewOwner());
  VariableItemInt32& edges_owner(edgeFamily()->itemsNewOwner());
  VariableItemInt32& faces_owner(faceFamily()->itemsNewOwner());

  const Integer sid = subDomain()->subDomainId();

  // Owner assignment tool for items
  if (m_new_item_owner_builder == NULL)
    m_new_item_owner_builder = new NewItemOwnerBuilder();

  // Determines the new owners of the nodes
  {
    ENUMERATE_NODE (i_node, ownNodes()) {
      Node node = *i_node;
      nodes_owner[node] = m_new_item_owner_builder->ownerOfItem(node);
    }
    nodes_owner.synchronize();
  }

  ENUMERATE_NODE (i_node, allNodes()) {
    Node node = *i_node;
    node.mutableItemBase().setOwner(nodes_owner[node], sid);
  }

  // Determines the new owners of the edges
  {
    ENUMERATE_EDGE (i_edge, ownEdges()) {
      Edge edge = *i_edge;
      edges_owner[edge] = m_new_item_owner_builder->ownerOfItem(edge);
    }
    edges_owner.synchronize();
  }

  ENUMERATE_EDGE (i_edge, allEdges()) {
    Edge edge = *i_edge;
    edge.mutableItemBase().setOwner(edges_owner[edge], sid);
  }

  // Determines the new owners of the faces
  {
    ENUMERATE_FACE (i_face, ownFaces()) {
      Face face = *i_face;
      faces_owner[face] = m_new_item_owner_builder->ownerOfItem(face);
    }
    faces_owner.synchronize();
  }

  ENUMERATE_FACE (i_face, allFaces()) {
    Face face = *i_face;
    face.mutableItemBase().setOwner(faces_owner[face], sid);
  }

  nodeFamily()->notifyItemsOwnerChanged();
  edgeFamily()->notifyItemsOwnerChanged();
  faceFamily()->notifyItemsOwnerChanged();
  _computeFamilySynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshUtilities* DynamicMesh::
utilities()
{
  if (!m_mesh_utilities)
    m_mesh_utilities = new UnstructuredMeshUtilities(this);
  return m_mesh_utilities;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup DynamicMesh::
outerFaces()
{
  return m_cell_family->allItems().outerFaceGroup();
}

//! AMR
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Group of all active cells
CellGroup DynamicMesh::
allActiveCells()
{
  return m_cell_family->allItems().activeCellGroup();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Group of all active and domain-specific cells
CellGroup DynamicMesh::
ownActiveCells()
{
  return m_cell_family->allItems().ownActiveCellGroup();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Group of all cells of level \p level
CellGroup DynamicMesh::
allLevelCells(const Integer& level)
{
  return m_cell_family->allItems().levelCellGroup(level);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Group of all own cells of level \p level
CellGroup DynamicMesh::
ownLevelCells(const Integer& level)
{
  return m_cell_family->allItems().ownLevelCellGroup(level);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Group of all active faces
FaceGroup DynamicMesh::
allActiveFaces()
{
  return m_cell_family->allItems().activeFaceGroup();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Group of all active faces specific to the sub-domain.
FaceGroup DynamicMesh::
ownActiveFaces()
{
  return m_cell_family->allItems().ownActiveFaceGroup();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Group of all internal active faces
FaceGroup DynamicMesh::
innerActiveFaces()
{
  return m_cell_family->allItems().innerActiveFaceGroup();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Group of all active faces on the boundary.
FaceGroup DynamicMesh::
outerActiveFaces()
{
  return m_cell_family->allItems().outerActiveFaceGroup();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
defineParentForBuild(IMesh* mesh, ItemGroup group)
{
  Trace::Setter mci(traceMng(), _className());
  if (!mesh)
    ARCANE_FATAL("Cannot set NULL parent mesh to mesh '{0}'", name());

  if (mesh != group.mesh())
    ARCANE_FATAL("Cannot set inconsistant mesh/group parents to mesh '{0}'", name());

  if (m_parent_mesh) {
    if (m_parent_mesh != mesh)
      ARCANE_FATAL("Mesh '{0}' already has parent mesh '{1}'", name(), m_parent_mesh->name());
    if (m_parent_group != group.internal())
      ARCANE_FATAL("Mesh '{0}' already has parent group '{1}'", name(), m_parent_group->name());
  }

  m_parent_mesh = mesh;
  m_parent_group = group.internal();

  Integer dimension_shift = 0;
  if (group.itemKind() == IK_Face) {
    dimension_shift = 1;
  }
  else if (group.itemKind() == IK_Cell) {
    dimension_shift = 0;
  }
  else {
    ARCANE_FATAL("Only SubMesh on FaceGroup or CellGoup is allowed");
  }

  _setDimension(mesh->dimension() - dimension_shift);

  for (IItemFamily* family : m_item_families) {
    const eItemKind kind = family->itemKind();
    // Uniquement sur les items constructifs d'un maillage
    if (kind == IK_Node || kind == IK_Edge || kind == IK_Face || kind == IK_Cell) {
      const eItemKind parent_kind = MeshToMeshTransposer::kindTranspose(kind, this, mesh);
      if (parent_kind != IK_Unknown) {
        family->setParentFamily(mesh->itemFamily(parent_kind));
      } // else : pas de transposition
    }
    else {
      // do nothing. Another idea ?
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshPartitionConstraintMng* DynamicMesh::
partitionConstraintMng()
{
  return m_partition_constraint_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* DynamicMesh::
parentMesh() const
{
  return m_parent_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup DynamicMesh::
parentGroup() const
{
  return ItemGroup(m_parent_group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
addChildMesh(IMesh* sub_mesh)
{
  DynamicMesh* dynamic_child_mesh = dynamic_cast<DynamicMesh*>(sub_mesh);
  if (!dynamic_child_mesh)
    ARCANE_FATAL("Cannot associate sub mesh from a different concrete type");
  for (Integer i = 0; i < m_child_meshes.size(); ++i)
    if (m_child_meshes[i] == dynamic_child_mesh)
      return;
  m_child_meshes.add(dynamic_child_mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshCollection DynamicMesh::
childMeshes() const
{
  IMeshCollection collection = List<IMesh*>();
  for (Integer i = 0; i < m_child_meshes.size(); ++i) {
    collection.add(m_child_meshes[i]);
  }
  return collection;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshChecker* DynamicMesh::
checker() const
{
  return m_mesh_checker;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DynamicMesh::
isPrimaryMesh() const
{
  return (this->parentMesh() == nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* DynamicMesh::
toPrimaryMesh()
{
  if (!isPrimaryMesh())
    throw BadCastException(A_FUNCINFO, "Mesh is not a primary mesh");
  return this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer DynamicMesh::
nbNode()
{
  return m_node_family->nbItem();
}
Integer DynamicMesh::
nbEdge()
{
  return m_edge_family->nbItem();
}
Integer DynamicMesh::
nbFace()
{
  return m_face_family->nbItem();
}
Integer DynamicMesh::
nbCell()
{
  return m_cell_family->nbItem();
}

IItemFamily* DynamicMesh::
nodeFamily()
{
  return m_node_family;
}

IItemFamily* DynamicMesh::
edgeFamily()
{
  return m_edge_family;
}

IItemFamily* DynamicMesh::
faceFamily()
{
  return m_face_family;
}

IItemFamily* DynamicMesh::
cellFamily()
{
  return m_cell_family;
}

DynamicMeshKindInfos::ItemInternalMap& DynamicMesh::
nodesMap()
{
  return m_node_family->itemsMap();
}

DynamicMeshKindInfos::ItemInternalMap& DynamicMesh::
edgesMap()
{
  return m_edge_family->itemsMap();
}

DynamicMeshKindInfos::ItemInternalMap& DynamicMesh::
facesMap()
{
  return m_face_family->itemsMap();
}

DynamicMeshKindInfos::ItemInternalMap& DynamicMesh::
cellsMap()
{
  return m_cell_family->itemsMap();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
mergeMeshes(ConstArrayView<IMesh*> meshes)
{
  UniqueArray<DynamicMesh*> true_meshes;
  for (IMesh* mesh : meshes) {
    DynamicMesh* true_mesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(mesh));
    true_meshes.add(true_mesh);
  }
  DynamicMeshMerger merger(this);
  merger.mergeMeshes(true_meshes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_printConnectivityPolicy()
{
  info() << "Connectivity policy=" << (int)m_connectivity_policy;

  if (m_connectivity_policy != InternalConnectivityPolicy::NewOnly)
    ARCANE_FATAL("Invalid value '{0}' for InternalConnectivityPolicy. Only '{1}' is allowed",
                 (int)m_connectivity_policy, (int)InternalConnectivityPolicy::NewOnly);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
setMeshPartInfo(const MeshPartInfo& mpi)
{
  m_mesh_part_info = mpi;
  // TODO: notify the families
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
_updateItemFamilyDependencies(VariableScalarInteger connectivity)
{
  if (!m_item_family_network)
    return;
  Connectivity c(connectivity);
  for (const auto& con : m_item_family_network->getConnectivities()) {
    if (c.hasConnectivity(c.kindsToConnectivity(con->sourceFamily()->itemKind(), con->targetFamily()->itemKind()))) {
      m_item_family_network->setIsStored(con);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshInternal* DynamicMesh::
_internalApi()
{
  return m_internal_api.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshModifierInternal* DynamicMesh::
_modifierInternalApi()
{
  return m_internal_api.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMesh::
computeSynchronizeInfos()
{
  _computeSynchronizeInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT DynamicMeshFactoryBase
: public AbstractService
, public IMeshFactory
{
 public:

  DynamicMeshFactoryBase(const ServiceBuildInfo& sbi, bool is_amr)
  : AbstractService(sbi)
  , m_is_amr(is_amr)
  {}

 public:

  void build() override {}
  IPrimaryMesh* createMesh(IMeshMng* mm, const MeshBuildInfo& build_info) override
  {
    MeshBuildInfo mbi(build_info);
    MeshKind mk(mbi.meshKind());
    // If AMR is requested but not indicated in MeshPart,
    // we add it.
    if (m_is_amr && mk.meshAMRKind() == eMeshAMRKind::None)
      mk.setMeshAMRKind(eMeshAMRKind::Cell);
    mbi.addMeshKind(mk);
    ISubDomain* sd = mm->variableMng()->_internalApi()->internalSubDomain();
    bool is_submesh = !mbi.parentGroup().null();
    if (is_submesh && m_is_amr)
      ARCANE_FATAL("Submesh cannot be refined with AMR.");
    return new DynamicMesh(sd, mbi, is_submesh);
  }

 private:

  bool m_is_amr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT DynamicMeshFactory
: public DynamicMeshFactoryBase
{
 public:

  explicit DynamicMeshFactory(const ServiceBuildInfo& sbi)
  : DynamicMeshFactoryBase(sbi, false)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT DynamicAMRMeshFactory
: public DynamicMeshFactoryBase
{
 public:

  explicit DynamicAMRMeshFactory(const ServiceBuildInfo& sbi)
  : DynamicMeshFactoryBase(sbi, true)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(DynamicMeshFactory,
                        ServiceProperty("ArcaneDynamicMeshFactory", ST_Application),
                        ARCANE_SERVICE_INTERFACE(IMeshFactory));

ARCANE_REGISTER_SERVICE(DynamicAMRMeshFactory,
                        ServiceProperty("ArcaneDynamicAMRMeshFactory", ST_Application),
                        ARCANE_SERVICE_INTERFACE(IMeshFactory));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
