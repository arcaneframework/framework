// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCasePartitioner.cc                                    (C) 2000-2025 */
/*                                                                           */
/* External mesh partitioning service.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/BasicService.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ServiceFinder2.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/IInitialPartitioner.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IDirectExecution.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/ITimeStats.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IMeshPartitionConstraintMng.h"
#include "arcane/core/ExternalPartitionConstraint.h"
#include "arcane/core/internal/MshMeshGenerationInfo.h"

#include "arcane/std/ArcaneCasePartitioner_axl.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneCasePartitioner;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneInitialPartitioner
: public IInitialPartitioner
{
 public:

  struct TrueOwnerInfo
  {
    VariableCellInt32* m_true_cells_owner = nullptr;
    VariableNodeInt32* m_true_nodes_owner = nullptr;
  };

 public:

  ArcaneInitialPartitioner(ArcaneCasePartitioner* mt, ISubDomain* sd)
  : m_sub_domain(sd)
  , m_main(mt)
  {
  }
  void build() override {}
  void partitionAndDistributeMeshes(ConstArrayView<IMesh*> meshes) override;

 private:

  //! Groups meshes associated with constraints on the same process
  void _mergeConstraints(ConstArrayView<IMesh*> meshes);

  //! Prints statistics on the partitioning
  void _printStats(Integer nb_part, IMesh* mesh, VariableCellInt32& new_owners);

 public:

  ISubDomain* m_sub_domain = nullptr;
  ArcaneCasePartitioner* m_main = nullptr;
  //! Stores for each mesh a variable indicating which partition each mesh belongs to.
  UniqueArray<TrueOwnerInfo> m_part_indexes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief External mesh partitioning service.
 */
class ArcaneCasePartitioner
: public ArcaneArcaneCasePartitionerObject
{
 public:
 public:

  explicit ArcaneCasePartitioner(const ServiceBuildInfo& sbi);
  ~ArcaneCasePartitioner() override;

 public:

  void build() override {}
  void execute() override;
  void setParallelMng(IParallelMng*) override {}
  bool isActive() const override { return true; }

 private:

  //! Opens the Correspondence file (only on proc 0)
  void _initCorrespondance(Int32 my_rank);

  //! Writes the Correspondence file
  void _writeCorrespondance(Int32 rank, Int64Array& nodesUniqueId, Int64Array& cellsUniqueId);

  //! Closes the Correspondence file (only on proc 0)
  void _finalizeCorrespondance(Int32 my_rank);

 private:

  std::ofstream m_sortiesCorrespondance;

  ArcaneInitialPartitioner* m_init_part = nullptr;

  void _partitionMesh(Int32 nb_part);
  void _computeGroups(IItemFamily* current_family, IItemFamily* new_family);
  void _addGhostLayers(CellGroup current_all_cells, Array<Cell>& cells_selected_for_new_mesh,
                       Integer nb_layer, Integer maxLocalIdCell, Integer maxLocalIdNode);
  void _addGhostGroups(IMesh* new_mesh, Array<Cell>& cells_selected_for_new_mesh,
                       VariableCellInt32& true_cells_owner, VariableNodeInt32& true_nodes_owner,
                       Int32Array& new_cells_local_id, Integer id_loc);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneInitialPartitioner::
_mergeConstraints(ConstArrayView<IMesh*> meshes)
{
  Integer nb_mesh = meshes.size();
  if (nb_mesh != 1)
    ARCANE_FATAL("Can not partition multiple meshes");

  IMesh* mesh = meshes[0];
  ISubDomain* sd = m_sub_domain;
  ITraceMng* tm = sd->traceMng();

  tm->info() << " _regroupeContraintes: nbMailles = " << meshes[0]->nbCell() << ", nbMaillesLocales = " << meshes[0]->ownCells().size();

  Integer nb_contraintes = m_main->options()->constraints.size();
  tm->info() << "Number of constraints = " << nb_contraintes;
  if (nb_contraintes == 0)
    return;

  IItemFamily* current_cell_family = mesh->cellFamily();
  VariableItemInt32& cells_new_owner(current_cell_family->itemsNewOwner());
  ENUMERATE_CELL (icell, current_cell_family->allItems()) {
    cells_new_owner[icell] = (*icell).owner();
  }

  sd->timeStats()->dumpTimeAndMemoryUsage(sd->parallelMng());
  IMeshPartitionConstraint* c = new ExternalPartitionConstraint(mesh, m_main->options()->constraints);
  mesh->partitionConstraintMng()->addConstraint(c);
  mesh->partitionConstraintMng()->computeAndApplyConstraints();

  cells_new_owner.synchronize();
  mesh->utilities()->changeOwnersFromCells();
  mesh->modifier()->setDynamic(true);
  bool compact = mesh->properties()->getBool("compact");
  mesh->properties()->setBool("compact", true);
  mesh->toPrimaryMesh()->exchangeItems();
  mesh->properties()->setBool("compact", compact);
#if 0
#ifdef ARCANE_DEBUG
  ScopedPtrT<IMeshWriter> mesh_writer;
  FactoryT<IMeshWriter> mesh_writer_factory(sd->serviceMng());
  mesh_writer = mesh_writer_factory.createInstance("Lima",true);
  IParallelMng* pm = sd->parallelMng();
  Int32 my_rank = pm->commRank();

  StringBuilder filename = "cut_mesh_after_";
  filename += my_rank;
  filename += ".mli2";
  mesh_writer->writeMeshToFile(mesh,filename);
#endif
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneInitialPartitioner::
partitionAndDistributeMeshes(ConstArrayView<IMesh*> meshes)
{
  String lib_name = m_main->options()->library();
  ISubDomain* sd = m_sub_domain;
  IParallelMng* pm = sd->parallelMng();
  Int32 nb_rank = pm->commSize();
  //Int32 my_rank = pm->commRank();
  ServiceBuilder<IMeshPartitioner> service_builder(sd);
  auto mesh_partitioner(service_builder.createReference(lib_name, SB_AllowNull));
  ITraceMng* tm = sd->traceMng();
  tm->info() << "DoInitialPartition. Service=" << lib_name;

  if (!mesh_partitioner.get())
    ARCANE_FATAL("can not found service named '{0}' for initial mesh partitioning", lib_name);

  _mergeConstraints(meshes);

  Integer nb_mesh = meshes.size();
  if (nb_mesh != 1)
    ARCANE_FATAL("Can not partition multiple meshes");

  m_part_indexes.resize(nb_mesh);
  Int32 nb_part = m_main->options()->nbCutPart();
  if (nb_part == 0)
    nb_part = nb_rank;
  tm->info() << "NbPart = " << nb_part << " nb_mesh=" << nb_mesh;

  for (Integer i = 0; i < nb_mesh; ++i) {
    IMesh* mesh = meshes[i];
    ARCANE_CHECK_POINTER(mesh);
    VariableCellInt32* p_true_cells_owner = new VariableCellInt32(VariableBuildInfo(mesh, "TrueCellsOwner"));
    VariableNodeInt32* p_true_nodes_owner = new VariableNodeInt32(VariableBuildInfo(mesh, "TrueNodesOwner"));
    m_part_indexes[i].m_true_cells_owner = p_true_cells_owner;
    m_part_indexes[i].m_true_nodes_owner = p_true_nodes_owner;
    VariableCellInt32& true_cells_owner = *p_true_cells_owner;
    VariableNodeInt32& true_nodes_owner = *p_true_nodes_owner;
    IItemFamily* current_cell_family = mesh->cellFamily();
    IItemFamily* current_node_family = mesh->nodeFamily();
    VariableItemInt32& cells_new_owner(current_cell_family->itemsNewOwner());
    VariableItemInt32& nodes_new_owner(current_node_family->itemsNewOwner());
    bool is_dynamic = mesh->isDynamic();
    mesh->modifier()->setDynamic(true);
    // First partitioning (optional) to provide an initial correct result
    //mesh_partitioner->partitionMesh(mesh);
    //mesh->exchangeItems(false);

    // Final partitioning
    {
      sd->timeStats()->dumpTimeAndMemoryUsage(pm);
      Timer t(sd, "InitPartTimer", Timer::TimerReal);
      {
        Timer::Sentry ts(&t);
        mesh_partitioner->partitionMesh(mesh, nb_part);
      }
      tm->info() << "Partitioning time t=" << t.lastActivationTime();
      sd->timeStats()->dumpTimeAndMemoryUsage(pm);
    }
    ENUMERATE_CELL (icell, current_cell_family->allItems()) {
      Int32 new_owner = cells_new_owner[icell];
      true_cells_owner[icell] = new_owner;
      cells_new_owner[icell] = new_owner % nb_rank;
    }
    ENUMERATE_NODE (inode, current_node_family->allItems()) {
      true_nodes_owner[inode] = nodes_new_owner[inode];
    }
    _printStats(nb_part, mesh, true_cells_owner);
    mesh->utilities()->changeOwnersFromCells();
    //mesh->modifier()->setDynamic(true);
    //PRIMARYMESH_CAST(mesh)->exchangeItems();
    bool compact = mesh->properties()->getBool("compact");
    mesh->properties()->setBool("compact", true);
    mesh->toPrimaryMesh()->exchangeItems();
    mesh->modifier()->setDynamic(is_dynamic);
    mesh->properties()->setBool("compact", compact);
  }

  // Adding a second layer of meshes
  // We should no longer call exchangeItems with the two mesh layers
  IMesh* mesh = meshes[0];
  if (m_main->options()->nbGhostLayer() == 2)
    mesh->updateGhostLayers(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneInitialPartitioner::
_printStats(Integer nb_part, IMesh* mesh, VariableCellInt32& new_owners)
{
  Int64UniqueArray nb_cells(nb_part, 0);
  ENUMERATE_CELL (icell, mesh->ownCells()) {
    Int32 new_owner = new_owners[icell];
    ++nb_cells[new_owner];
  }
  IParallelMng* pm = mesh->parallelMng();
  pm->reduce(Parallel::ReduceSum, nb_cells);
  ITraceMng* tm = m_sub_domain->traceMng();
  tm->info() << " -- Partitioning statistics --";
  tm->info() << "   Part              NbCell";
  for (Integer i = 0; i < nb_part; ++i) {
    tm->info() << Trace::Width(6) << i << Trace::Width(18) << nb_cells[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCasePartitioner::
ArcaneCasePartitioner(const ServiceBuildInfo& sb)
: ArcaneArcaneCasePartitionerObject(sb)
{
  m_init_part = new ArcaneInitialPartitioner(this, sb.subDomain());
  info() << "** ** SET INITIAL PARTITIONER 2";
  sb.subDomain()->setInitialPartitioner(m_init_part);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCasePartitioner::
~ArcaneCasePartitioner()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
execute()
{
  Int32 nb_part = options()->nbCutPart();
  info() << "ArcaneCasePartitioner::execute() nb_part=" << nb_part;
  if (nb_part != 0) {
    subDomain()->timeStats()->dumpTimeAndMemoryUsage(subDomain()->parallelMng());
    _partitionMesh(nb_part);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
_partitionMesh(Int32 nb_part)
{
  //String lib_name = m_main->options()->library(); //"Metis";
  ISubDomain* sd = subDomain();
  IMesh* current_mesh = mesh();
  IParallelMng* pm = sd->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();

  //FactoryT<IMeshWriter> mesh_writer_factory(sd->serviceMng());
  String mesh_writer_name = options()->writerServiceName();
  if (mesh_writer_name.empty())
    pfatal() << "No service selected to write the mesh";
  ServiceBuilder<IMeshWriter> sb(sd);
  auto mesh_writer = sb.createReference(mesh_writer_name, SB_Collective);

  String pattern = options()->meshFileNamePattern();
  info() << "Mesh file pattern=" << pattern;

  // Partitions the mesh.
  // In return, \a cells_new_owner contains the number of the partition to which
  // each mesh will belong. To save the file, all
  // meshes of a partition must be on the same subdomain. For this,
  // we store the partition number in \a true_cells_owner, then
  // we exchange the mesh.
  VariableCellInt32 true_cells_owner(*m_init_part->m_part_indexes[0].m_true_cells_owner);
  VariableNodeInt32 true_nodes_owner(*m_init_part->m_part_indexes[0].m_true_nodes_owner);
  IItemFamily* current_cell_family = mesh()->cellFamily();
  //VariableItemInt32& cells_new_owner(current_cell_family->itemsNewOwner());
  CellGroup current_all_cells = current_cell_family->allItems();
  Integer total_current_nb_cell = pm->reduce(Parallel::ReduceSum, current_all_cells.own().size());
  info() << "TOTAL_NB_CELL=" << total_current_nb_cell;

  IPrimaryMesh* new_mesh = nullptr;
  {
    IMeshFactoryMng* mfm = sd->meshMng()->meshFactoryMng();
    MeshBuildInfo build_info("SubMesh");
    build_info.addMeshKind(current_mesh->meshKind());
    build_info.addParallelMng(makeRef(pm->sequentialParallelMng()));
    new_mesh = mfm->createMesh(build_info);
  }

  new_mesh->setDimension(mesh()->dimension());
  // To optimize, there is no need to sort or compact the entities.
  new_mesh->properties()->setBool("compact", false);
  new_mesh->properties()->setBool("sort", false);
  new_mesh->modifier()->setDynamic(true);
  new_mesh->allocateCells(0, Int64ConstArrayView(), true);

  // If the original mesh has MSH generation information,
  // we copy it to the new mesh.
  // TODO: look into how to do this automatically, for example by adding
  // a method to clone the mesh.
  impl::MshMeshGenerationInfo* new_msh_mesh_info = nullptr;
  auto* msh_mesh_info = impl::MshMeshGenerationInfo::getReference(mesh(), false);
  if (msh_mesh_info) {
    new_msh_mesh_info = impl::MshMeshGenerationInfo::getReference(new_mesh, true);
    *new_msh_mesh_info = *msh_mesh_info;
  }

  Int32 saved_nb_cell = 0;
  Int32 min_nb_cell = total_current_nb_cell;
  Int32 max_nb_cell = 0;

  if (options()->createCorrespondances())
    _initCorrespondance(my_rank);

  // Searches for the maximum IDs once.
  Integer maxLocalIdCell = mesh()->cellFamily()->maxLocalId();
  Integer maxLocalIdNode = mesh()->nodeFamily()->maxLocalId();

  // Forces the owner of the entities to subdomain 0 because
  // new_mesh uses a sequential parallelMng().
  for (IItemFamily* family : mesh()->itemFamilies()) {
    ENUMERATE_ITEM (iitem, family->allItems()) {
      iitem->mutableItemBase().setOwner(0, 0);
    }
  }
  // For each part to process, create a mesh
  // containing the entities of that part
  // and save it
  info() << "NbPart=" << nb_part << " my_rank=" << my_rank;
  for (Integer i = 0; i < nb_part; ++i) {
    if ((i % nb_rank) != my_rank) {
      if (my_rank == 0 && options()->createCorrespondances()) {

        info() << "Receive on master to build correspondence file on sub-domain " << i
               << " sent from processor " << i % nb_rank;
        Int32UniqueArray taillesTab(2);
        Int64UniqueArray nodesUniqueId;
        Int64UniqueArray cellsUniqueId;

        pm->recv(taillesTab, i % nb_rank);
        nodesUniqueId.resize(taillesTab[0]);
        cellsUniqueId.resize(taillesTab[1]);
        pm->recv(nodesUniqueId, i % nb_rank);
        pm->recv(cellsUniqueId, i % nb_rank);
        _writeCorrespondance(i, nodesUniqueId, cellsUniqueId);
      }
      continue;
    }

    new_mesh->destroyGroups();
    new_mesh->modifier()->clearItems();
    new_mesh->modifier()->endUpdate();
    UniqueArray<Cell> cells_selected_for_new_mesh;
    ENUMERATE_CELL (icell, current_all_cells.own()) {
      if (true_cells_owner[icell] == i) {
        Cell cell = *icell;
        cells_selected_for_new_mesh.add(cell);
        //info() << "ADD CELL " << ItemPrinter(cell);
      }
    }

    // select ghost layers additionally if necessary
    _addGhostLayers(current_all_cells, cells_selected_for_new_mesh, options()->nbGhostLayer(), maxLocalIdCell, maxLocalIdNode);

    Int32UniqueArray cells_local_id;
    Int64UniqueArray cells_unique_id;
    for (Integer j = 0, js = cells_selected_for_new_mesh.size(); j < js; ++j) {
      Cell cell = cells_selected_for_new_mesh[j];
      cells_local_id.add(cell.localId());
      cells_unique_id.add(static_cast<Int64>(cell.uniqueId()));
    }

    Integer nb_cell_to_copy = cells_local_id.size();
    SerializeBuffer buffer;
    current_mesh->serializeCells(&buffer, cells_local_id);
    info() << "NB_CELL_TO_SERIALIZE=" << nb_cell_to_copy;
    new_mesh->modifier()->addCells(&buffer);
    new_mesh->modifier()->endUpdate();
    // To update coordinates
    //new_mesh->nodeFamily()->endUpdate();
    ItemInternalList new_cells = new_mesh->itemsInternal(IK_Cell);
    ItemInternalList current_cells = current_mesh->itemsInternal(IK_Cell);
    VariableNodeReal3& new_coordinates(new_mesh->nodesCoordinates());
    VariableNodeReal3& current_coordinates(current_mesh->toPrimaryMesh()->nodesCoordinates());
    Int32UniqueArray new_cells_local_id(nb_cell_to_copy);
    new_mesh->cellFamily()->itemsUniqueIdToLocalId(new_cells_local_id, cells_unique_id);
    for (Integer zid = 0; zid < nb_cell_to_copy; ++zid) {
      Cell current_cell = current_cells[cells_local_id[zid]];
      Cell new_cell = new_cells[new_cells_local_id[zid]];
      if (current_cell.uniqueId() != new_cell.uniqueId())
        fatal() << "Inconsistent unique ids";
      Integer nb_node = current_cell.nbNode();
      //info() << "Current=" << ItemPrinter(current_cell)
      //       << " new=" << ItemPrinter(new_cell)
      //       << " nb_node=" << nb_node;
      for (Integer z2 = 0; z2 < nb_node; ++z2) {
        Real3 coord = current_coordinates[current_cell.node(z2)];
        //         info() << "Node=" << ItemPrinter(new_cell.node(z2)) << " coord=" << coord
        //                << " orig_node=" << ItemPrinter(current_cell.node(z2));
        new_coordinates[new_cell.node(z2)] = coord;
        // Position the final owner of the node
        new_cell.node(z2).mutableItemBase().setOwner(true_nodes_owner[current_cell.node(z2)], 0);
      }
    }
    // Now, we must copy the groups
    {
      _computeGroups(current_mesh->nodeFamily(), new_mesh->nodeFamily());
      _computeGroups(current_mesh->edgeFamily(), new_mesh->edgeFamily());
      _computeGroups(current_mesh->faceFamily(), new_mesh->faceFamily());
      _computeGroups(current_mesh->cellFamily(), new_mesh->cellFamily());

      if (options()->nbGhostLayer() > 0)
        _addGhostGroups(new_mesh, cells_selected_for_new_mesh, true_cells_owner, true_nodes_owner, new_cells_local_id, i);
    }
    Integer new_nb_cell = new_mesh->nbCell();
    info() << "NB_NEW_CELL=" << new_nb_cell;
    min_nb_cell = math::min(min_nb_cell, new_nb_cell);
    max_nb_cell = math::max(max_nb_cell, new_nb_cell);
    saved_nb_cell += new_nb_cell;
    String filename;
    if (pattern.empty()) {
      StringBuilder sfilename = "cut_mesh_";
      sfilename += i;
      sfilename += ".mli2";
      filename = sfilename;
    }
    else {
      // ATTENTION potential overflow if pattern is too long.
      // Also check if there is a %d. Eventually, use String::format()
      char buf[4096];
      if (pattern.length() > 128) {
        pfatal() << "Pattern too long (max=128)";
      }
      sprintf(buf, pattern.localstr(), i);
      filename = String(StringView(buf));
    }
    {
      info() << "Writing mesh file filename='" << filename << "'";
      bool is_bad = mesh_writer->writeMeshToFile(new_mesh, filename);
      if (is_bad)
        ARCANE_FATAL("Can not write mesh file '{0}'", filename);
    }

    // Correspondence File
    if (options()->createCorrespondances()) {
      info() << "Participation to build correspondence file on sub-domain " << i;

      Int32UniqueArray taillesTab;
      taillesTab.add(new_mesh->nodeFamily()->nbItem());
      taillesTab.add(new_mesh->cellFamily()->nbItem());
      Int64UniqueArray nodesUniqueId(taillesTab[0]);
      Int64UniqueArray cellsUniqueId(taillesTab[1]);

      NodeInfoListView nodes(new_mesh->nodeFamily());
      for (int j = 0; j < taillesTab[0]; ++j) {
        Node node = nodes[j];
        nodesUniqueId[j] = node.uniqueId();
      }

      CellInfoListView cells(new_mesh->cellFamily());
      for (int j = 0; j < taillesTab[1]; ++j) {
        Cell cell = cells[j];
        cellsUniqueId[j] = cell.uniqueId();
      }

      if (my_rank != 0) {
        pm->send(taillesTab, 0);
        pm->send(nodesUniqueId, 0);
        pm->send(cellsUniqueId, 0);
      }
      else {
        _writeCorrespondance(i, nodesUniqueId, cellsUniqueId);
      }
    }
  } // end i<nb_part

  Integer total_new_nb_cell = pm->reduce(Parallel::ReduceSum, saved_nb_cell);
  Integer total_min_nb_cell = pm->reduce(Parallel::ReduceMin, min_nb_cell);
  Integer total_max_nb_cell = pm->reduce(Parallel::ReduceMax, max_nb_cell);
  info() << "TOTAL_NEW_NB_CELL=" << total_new_nb_cell
         << " min=" << total_min_nb_cell
         << " max=" << total_max_nb_cell
         << " computed_average=" << (total_current_nb_cell / nb_part);

  subDomain()->timeStats()->dumpTimeAndMemoryUsage(pm);

  if (options()->createCorrespondances())
    _finalizeCorrespondance(my_rank);

  if (options()->nbGhostLayer() == 0)
    if (total_new_nb_cell != total_current_nb_cell)
      pfatal() << "Bad number of saved cells current=" << total_current_nb_cell
               << " saved=" << total_new_nb_cell;

  pinfo() << "Total Memory Used : " << platform::getMemoryUsed();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
_initCorrespondance(Int32 my_rank)
{
  info() << " _initCorrespondance(" << my_rank << ")";

  if (my_rank)
    return;

  m_sortiesCorrespondance.open("Correspondances");

  if (m_sortiesCorrespondance.fail()) {
    pfatal() << "Unable to write to file 'Correspondances' ";
  }

  m_sortiesCorrespondance << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\" ?>\n";
  m_sortiesCorrespondance << "<!-- Correspondence file generated by Arcane/Decoupe3D V2 -->\n";
  m_sortiesCorrespondance << "\n<cpus>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
_writeCorrespondance(Int32 rank, Int64Array& nodesUniqueId, Int64Array& cellsUniqueId)
{
  info() << " _writeCorrespondance(" << rank << ", nodesUniqueId.size() = "
         << nodesUniqueId.size() << ", cellsUniqueId.size() = " << cellsUniqueId.size() << ")";

  m_sortiesCorrespondance << "  <cpu id=\"" << rank << "\">" << "\n"
                          << "    <noeuds>" << "\n"
                          << "    ";
  for (Integer i = 0; i < nodesUniqueId.size(); ++i)
    m_sortiesCorrespondance << nodesUniqueId[i] << " ";

  m_sortiesCorrespondance << "\n"
                          << "    </noeuds>"
                          << "\n"
                          << "    <mailles>" << "\n"
                          << "      ";
  for (Integer i = 0; i < cellsUniqueId.size(); ++i)
    m_sortiesCorrespondance << cellsUniqueId[i] << " ";
  m_sortiesCorrespondance << "\n"
                          << "    </mailles>" << "\n"
                          << "  </cpu>" << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCasePartitioner::
_finalizeCorrespondance(Int32 my_rank)
{
  if (my_rank)
    return;

  m_sortiesCorrespondance << "</cpus>\n";
  m_sortiesCorrespondance.close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recopy the groups of the current family into the new one.
 *
 * The principle is as follows:
 * 1. For each entity type, determine the list of localId()
 * of entities of that type in the original mesh.
 * 2. From this list, build an array indicating
 * for each localId() of the original mesh its localId() in the new
 * mesh (or NULL_ITEM_LOCAL_ID if the entity is absent).
 * 3. Iterate through the original groups and build
 * for each the list of entities to add to the new mesh.
 */
void ArcaneCasePartitioner::
_computeGroups(IItemFamily* current_family, IItemFamily* new_family)
{
  info() << "Compute groups family=" << current_family->name();

  ItemGroup new_all_items = new_family->allItems();
  Integer nb_new_item = new_all_items.size();

  Int64UniqueArray new_items_uid(nb_new_item);
  Int32UniqueArray new_items_lid(nb_new_item);
  {
    Integer index = 0;
    ENUMERATE_ITEM (iitem, new_all_items) {
      new_items_uid[index] = (*iitem).uniqueId();
      new_items_lid[index] = iitem.itemLocalId();
      ++index;
    }
  }
  Int32UniqueArray items_lid(nb_new_item);
  // Determine the localId() in the original mesh of the entities
  current_family->itemsUniqueIdToLocalId(items_lid, new_items_uid);

  Int32UniqueArray items_current_to_new_local_id(current_family->maxLocalId());
  items_current_to_new_local_id.fill(NULL_ITEM_LOCAL_ID);
  for (Integer i = 0; i < nb_new_item; ++i)
    items_current_to_new_local_id[items_lid[i]] = new_items_lid[i];

  Int32UniqueArray create_local_ids;
  for (ItemGroupCollection::Enumerator igroup(current_family->groups()); ++igroup;) {
    ItemGroup group = *igroup;
    if (group.isOwn())
      continue;
    if (group.isAllItems())
      continue;
    create_local_ids.clear();
    ENUMERATE_ITEM (iitem, group) {
      Int32 current_uid = iitem.itemLocalId();
      Int32 new_lid = items_current_to_new_local_id[current_uid];
      if (new_lid != NULL_ITEM_LOCAL_ID)
        create_local_ids.add(new_lid);
    }
    new_family->createGroup(group.name(), create_local_ids, true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Adds to the mesh array the desired number of mesh layers
 */
void ArcaneCasePartitioner::
_addGhostLayers(CellGroup current_all_cells, Array<Cell>& cells_selected_for_new_mesh,
                Integer nbCouches, Integer maxLocalIdCell, Integer maxLocalIdNode)
{
  if (nbCouches == 0)
    return;

  Int32UniqueArray filtre_lid_cell(maxLocalIdCell);
  filtre_lid_cell.fill(0);
  Int32UniqueArray filtre_lid_node(maxLocalIdNode);
  filtre_lid_node.fill(0);

  // mark the already selected cells
  for (Integer j = 0, js = cells_selected_for_new_mesh.size(); j < js; ++j) {
    Cell cell = cells_selected_for_new_mesh[j];
    filtre_lid_cell[cell.localId()] = 1;
  }

  // search for all nodes associated with selected cells a connected cell
  // to this same node that is not selected
  for (Integer j = 0, js = cells_selected_for_new_mesh.size(); j < js; ++j) {
    Cell cell = cells_selected_for_new_mesh[j];

    NodeVectorView nodes = cell.nodes();
    for (Integer k = 0, ks = nodes.size(); k < ks; ++k) {
      Node node = nodes[k];
      if (filtre_lid_node[node.localId()] == 0) {
        // cells connected by a node
        CellVectorView cells_vois = node.cells();

        for (Integer i = 0, is = cells_vois.size(); i < is; ++i) {
          Cell cell_vois = cells_vois[i];
          if (filtre_lid_cell[cell_vois.localId()] == 0) {
            // add the cell that has not yet been seen
            cells_selected_for_new_mesh.add(cell_vois);

            filtre_lid_cell[cell_vois.localId()] = 1;
          }
        }
        filtre_lid_node[node.localId()] = 1;
      }
    }
  }

  // for the second layer (if needed) it is simpler to do it recursively
  _addGhostLayers(current_all_cells, cells_selected_for_new_mesh, nbCouches - 1, maxLocalIdCell, maxLocalIdNode);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Adds the TOUT, LOCAL, and MF_* mesh groups based on neighbor groups
 *        Also adds the LOCALN node group (but not the NF_*)
 */
void ArcaneCasePartitioner::
_addGhostGroups(IMesh* new_mesh, Array<Cell>& cells_selected_for_new_mesh, VariableCellInt32& true_cells_owner,
                VariableNodeInt32& true_nodes_owner,
                Int32Array& new_cells_local_id, Integer id_loc)
{
  info() << "ArcaneCasePartitioner::_addGhostGroups (id_loc = " << id_loc << ")";
  // we must determine the existing neighbor groups
  // we use a "map" to store the different sub-domains that appear and the number of cells in them
  std::map<Integer, Integer> dom_vois;
  for (Integer j = 0, js = cells_selected_for_new_mesh.size(); j < js; ++j) {
    Cell cell = cells_selected_for_new_mesh[j];
    dom_vois[true_cells_owner[cell]] += 1;
  }

  // we use a second map to list the cells according to the destination domain
  std::map<Integer, SharedArray<Int32>> map_groupes;
  for (std::map<Integer, Integer>::const_iterator iter = dom_vois.begin(); iter != dom_vois.end(); ++iter) {
    Integer no_sous_dom = iter->first;
    Integer nb_mailles_sous_dom = iter->second;

    // memory reservation for the different lists
    Int32Array& tab = map_groupes[no_sous_dom];
    tab.reserve(nb_mailles_sous_dom);
  }

  for (Integer j = 0, js = cells_selected_for_new_mesh.size(); j < js; ++j) {
    Cell cell = cells_selected_for_new_mesh[j];
    Integer no_sous_dom = true_cells_owner[cell];

    // filling the lists by sub-domain
    Int32Array& liste_lid = map_groupes[no_sous_dom];
    liste_lid.add(new_cells_local_id[j]);
  }

  // creation (if necessary) of the different groups and adding the cells to them
  for (std::map<Integer, SharedArray<Int32>>::iterator iter = map_groupes.begin(); iter != map_groupes.end(); ++iter) {
    Integer no_sous_dom = iter->first;
    Int32Array& liste_lid = iter->second;

    ItemGroup groupe_loc;
    if (no_sous_dom == id_loc)
      groupe_loc = new_mesh->cellFamily()->findGroup("LOCAL", true);
    else {
      String nom_mf("MF_");
      nom_mf = nom_mf + no_sous_dom;
      groupe_loc = new_mesh->cellFamily()->findGroup(nom_mf, true);
    }

    groupe_loc.addItems(liste_lid, false);
  }

  // Create the LOCALN group: local nodes
  {
    // TODO: Optimize the way this group is built
    Int32UniqueArray liste_lid;
    Integer nbnodes = new_mesh->nodeFamily()->nbItem();
    liste_lid.reserve(nbnodes);
    NodeInfoListView nodes(new_mesh->nodeFamily());
    for (int j = 0; j < nbnodes; ++j) {
      Node node = nodes[j];
      if (true_nodes_owner[node] == id_loc)
        liste_lid.add(node.localId());
    }

    ItemGroup groupe_loc = new_mesh->nodeFamily()->findGroup("LOCALN", true);
    groupe_loc.addItems(liste_lid, false);
  }

  // the group with all cells
  ItemGroup groupe_glob = new_mesh->cellFamily()->findGroup("TOUT", true);

  groupe_glob.addItems(new_cells_local_id, false);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANECASEPARTITIONER(ArcaneCasePartitioner, ArcaneCasePartitioner);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
