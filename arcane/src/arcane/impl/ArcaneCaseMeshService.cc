// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCaseMeshService.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Arcane Service managing a dataset mesh.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"

#include "arcane/core/ServiceFactory.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/ICaseMeshService.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMeshPartitionerBase.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/IMeshSubdivider.h"
#include "arcane/core/IMeshUniqueIdMng.h"
#include "arcane/core/internal/StringVariableReplace.h"

#include "arcane/impl/ArcaneCaseMeshService_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arcane Service for meshing the dataset.
 */
class ArcaneCaseMeshService
: public ArcaneArcaneCaseMeshServiceObject
{
 public:

  explicit ArcaneCaseMeshService(const ServiceBuildInfo& sbi);

 public:

  void createMesh(const String& default_name) override;
  void allocateMeshItems() override;
  void partitionMesh() override;
  void applyAdditionalOperations() override;

 private:

  ISubDomain* m_sub_domain = nullptr;
  IPrimaryMesh* m_mesh = nullptr;
  IMeshBuilder* m_mesh_builder = nullptr;
  Ref<IMeshBuilder> m_mesh_builder_ref;
  String m_mesh_file_name;
  String m_partitioner_name;

 private:

  void _fillReadInfo(CaseMeshReaderReadInfo& read_info);
  Ref<IMeshBuilder> _createBuilderFromFile(const CaseMeshReaderReadInfo& read_info);
  void _initializeVariables();
  void _doInitialPartition();
  void _doInitialPartition2(const String& name);
  void _setGhostLayerInfos();
  void _checkMeshCreationAndAllocation(bool is_check_allocated);
  void _setUniqueIdNumberingVersion();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCaseMeshService::
ArcaneCaseMeshService(const ServiceBuildInfo& sbi)
: ArcaneArcaneCaseMeshServiceObject(sbi)
, m_sub_domain(sbi.subDomain())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
createMesh(const String& default_name)
{
  if (m_mesh)
    ARCANE_FATAL("Mesh is already created");

  info() << "Creating mesh from 'ArcaneCaseMeshService'";
  MeshBuildInfo build_info(default_name);
  ISubDomain* sd = m_sub_domain;
  String filename = options()->filename();
  bool has_filename = options()->filename.isPresent();
  bool has_generator = options()->generator.isPresent();

  // Either the file name or the generator must be specified, but not both
  if ((has_filename && has_generator) || (!has_filename && !has_generator))
    ARCANE_FATAL("In '{0}': one and one only of <{1}> or <{2}> has to be specified",
                 options()->configList()->xpathFullName(),
                 options()->generator.rootTagName(),
                 options()->filename.name());
  if (has_filename) {
    m_mesh_file_name = options()->filename();
    if (m_mesh_file_name.empty())
      ARCANE_FATAL("Invalid filename '{0}' in option '{1}'",
                   m_mesh_file_name, options()->filename.xpathFullName());
    CaseMeshReaderReadInfo read_info;
    _fillReadInfo(read_info);
    auto& specific_reader = options()->specificReader;
    if (specific_reader.isPresent()) {
      m_mesh_builder_ref = specific_reader()->createBuilder(read_info);
      if (m_mesh_builder_ref.isNull())
        ARCANE_FATAL("No 'IMeshBuilder' created by specific reader");
    }
    else {
      m_mesh_builder_ref = _createBuilderFromFile(read_info);
    }
    m_mesh_builder = m_mesh_builder_ref.get();
  }
  else if (has_generator)
    m_mesh_builder = options()->generator();
  else
    ARCANE_FATAL("Invalid operation");

  ARCANE_CHECK_POINTER(m_mesh_builder);

  // Indicates the management of cell dimensions
  eMeshCellDimensionKind mesh_dim_kind = options()->cellDimensionKind();
  if (mesh_dim_kind != build_info.meshKind().meshDimensionKind()) {
    MeshKind mesh_kind = build_info.meshKind();
    mesh_kind.setMeshDimensionKind(mesh_dim_kind);
    build_info.addMeshKind(mesh_kind);
  }

  m_mesh_builder->fillMeshBuildInfo(build_info);
  // The generator can force the use of partitioning
  if (build_info.isNeedPartitioning())
    m_partitioner_name = options()->partitioner();

  // Positions fields not filled with default values.
  if (build_info.factoryName().empty())
    build_info.addFactoryName("ArcaneDynamicMeshFactory");
  if (build_info.parallelMngRef().isNull())
    build_info.addParallelMng(makeRef(sd->parallelMng()));
  IPrimaryMesh* pm = sd->meshMng()->meshFactoryMng()->createMesh(build_info);
  m_mesh = pm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
allocateMeshItems()
{
  _checkMeshCreationAndAllocation(false);

  ARCANE_CHECK_POINTER(m_mesh_builder);

  _setGhostLayerInfos();
  _setUniqueIdNumberingVersion();

  m_mesh_builder->allocateMeshItems(m_mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
partitionMesh()
{
  _checkMeshCreationAndAllocation(true);

  if (m_mesh->meshPartInfo().nbPart() > 1)
    if (!m_partitioner_name.empty())
      _doInitialPartition();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
applyAdditionalOperations()
{
  _checkMeshCreationAndAllocation(true);

  IMeshSubdivider* subdivider = options()->subdivider();
  if (subdivider)
    subdivider->subdivideMesh(m_mesh);

  _initializeVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
_checkMeshCreationAndAllocation(bool is_check_allocated)
{
  if (!m_mesh)
    ARCANE_FATAL("Mesh is not created. You should call createMesh() before");
  if (is_check_allocated && !m_mesh->isAllocated())
    ARCANE_FATAL("Mesh is not allocated. You should call initializeMesh() before");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
_fillReadInfo(CaseMeshReaderReadInfo& read_info)
{
  // Searches for the file extension.
  String file_extension;
  {
    std::string_view fview = m_mesh_file_name.toStdStringView();
    std::size_t extension_pos = fview.find_last_of('.');
    if (extension_pos != std::string_view::npos) {
      fview.remove_prefix(extension_pos + 1);
      file_extension = fview;
    }
    read_info.setFormat(file_extension);
  }

  // Mesh file name to use for reading
  // Normally it is the mesh name in the dataset
  // unless an external partitioner is used, in which case
  // it is 'CPU%05d.$file_extension'.
  String mesh_file_name = m_mesh_file_name;

  String partitioner_name = options()->partitioner();
  bool use_internal_partitioner = partitioner_name != "External";
  if (use_internal_partitioner) {
    m_partitioner_name = partitioner_name;
  }
  else {
    info() << "Using external partitioner";
    int mesh_rank = m_sub_domain->parallelMng()->commRank();
    char buf[128];
    sprintf(buf, "CPU%05d.%s", mesh_rank, file_extension.localstr());
    mesh_file_name = String(std::string_view(buf));
  }

  read_info.setFileName(mesh_file_name);

  info() << "Mesh filename=" << mesh_file_name
         << " extension=" << read_info.format() << " partitioner=" << partitioner_name;

  read_info.setParallelRead(use_internal_partitioner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMeshBuilder> ArcaneCaseMeshService::
_createBuilderFromFile(const CaseMeshReaderReadInfo& read_info)
{
  // Constructs potential mesh reading services. These are
  // those that implement ICaseMeshReader. An instance
  // of each service is constructed and the createBuilder() method is called.
  // As soon as one of these methods returns a non-null reference, it is used
  // to generate the mesh.
  ServiceBuilder<ICaseMeshReader> builder(m_sub_domain);
  UniqueArray<Ref<ICaseMeshReader>> mesh_readers(builder.createAllInstances());

  for (auto& mesh_reader_ref : mesh_readers) {
    ICaseMeshReader* mesh_reader = mesh_reader_ref.get();
    Ref<IMeshBuilder> builder = mesh_reader->createBuilder(read_info);
    if (!builder.isNull())
      return builder;
  }

  // No service found for this file format.
  // Displays the list of available services and throws a fatal error.
  StringUniqueArray valid_names;
  builder.getServicesNames(valid_names);
  String available_readers = String::join(", ", valid_names);
  ARCANE_FATAL("The mesh reader required for format '{0}' is not available."
               "The following reader services are available: {1}",
               read_info.format(), available_readers);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
_doInitialPartition()
{
  // No longer uses the test partitioning service to ensure
  // that with ParMetis there are no empty partitions, as this is now
  // normally supported.
  const bool use_partitioner_tester = false;
  String test_service = "MeshPartitionerTester";
  if (use_partitioner_tester) {
    Int64 nb_cell = m_mesh->nbCell();
    Int64 min_nb_cell = m_mesh->parallelMng()->reduce(Parallel::ReduceMin, nb_cell);
    info() << "Min nb cell=" << min_nb_cell;
    if (min_nb_cell == 0)
      _doInitialPartition2(test_service);
    else
      info() << "Mesh name=" << m_mesh->name() << " have cells. Do not use " << test_service;
  }
  else
    info() << "No basic partition first needed";
  _doInitialPartition2(m_partitioner_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
_doInitialPartition2(const String& partitioner_name)
{
  info() << "Doing initial partitioning service=" << partitioner_name;
  // NOTE: This service only uses partitioners that implement
  // IMeshPartitionerBase and not those (historical) that only implement
  // IMeshPartitioner.
  ServiceBuilder<IMeshPartitionerBase> sbuilder(m_sub_domain);
  auto mesh_partitioner = sbuilder.createReference(partitioner_name, m_mesh);

  IMesh* mesh = m_mesh;
  bool is_dynamic = mesh->isDynamic();
  mesh->modifier()->setDynamic(true);
  mesh->utilities()->partitionAndExchangeMeshWithReplication(mesh_partitioner.get(), true);
  mesh->modifier()->setDynamic(is_dynamic);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
_initializeVariables()
{
  IVariableMng* vm = m_sub_domain->variableMng();
  const auto& vars_opt = options()->initialization().variable;
  UniqueArray<String> errors;
  for (Integer i = 0, n = vars_opt.size(); i < n; ++i) {
    const auto& o = vars_opt[i];
    String var_name = o.name;
    String group_name = o.group;
    String value = o.value;
    info() << "Initialize variable=" << var_name << " group=" << group_name << " value=" << value;
    IVariable* var = vm->findMeshVariable(m_mesh, var_name);
    if (!var) {
      errors.add(String::format("No variable named '{0}' exists", var_name));
      continue;
    }

    // Allocate the variable if necessary.
    if (!var->isUsed())
      var->setUsed(true);
    IItemFamily* var_family = var->itemFamily();
    if (!var_family) {
      errors.add(String::format("Variable '{0}' has no family", var->fullName()));
      continue;
    }

    ItemGroup group = var_family->findGroup(group_name);
    if (group.null()) {
      errors.add(String::format("No group named '{0}' exists in family '{1}'",
                                group_name, var_family->name()));
      continue;
    }

    bool ret = var->initialize(group, value);
    if (ret) {
      errors.add(String::format("Bad value '{0}' for initializing variable '{1}'",
                                value, var->fullName()));
      continue;
    }
  }
  if (!errors.empty()) {
    for (String s : errors)
      pinfo() << "ERROR: " << s;
    ARCANE_FATAL("Variable initialization failed for option '{0}'",
                 vars_opt.xpathFullName());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
_setGhostLayerInfos()
{
  IGhostLayerMng* gm = m_mesh->ghostLayerMng();
  if (!gm)
    return;

  // Positions the info on the ghost cells.
  // TODO This is done to remain compatible with the historical mode but
  // it should be possible to manage this differently (via a service for example)
  Integer nb_ghost_layer = options()->nbGhostLayer();
  if (nb_ghost_layer >= 0) {
    info() << "Set number of ghost layers to '" << nb_ghost_layer << "' from caseoption";
    gm->setNbGhostLayer(nb_ghost_layer);
  }

  Integer builder_version = options()->ghostLayerBuilderVersion();
  if (builder_version >= 0) {
    info() << "Set ghostlayer builder version to '" << builder_version << "' from caseoption";
    gm->setBuilderVersion(builder_version);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCaseMeshService::
_setUniqueIdNumberingVersion()
{
  // NOTE: currently (12/2024) the 'PolyedralMesh' implementation raises a
  // exception if meshUniqueIdMng() is called. We only do this if
  // the option is present.
  if (options()->faceNumberingVersion.isPresent()) {
    Int32 v = options()->faceNumberingVersion.value();
    info() << "Set face uniqueId numbering version to '" << v << "' from caseoption";
    IMeshUniqueIdMng* mum = m_mesh->meshUniqueIdMng();
    mum->setFaceBuilderVersion(v);
  }

  if (options()->edgeNumberingVersion.isPresent()) {
    Int32 v = options()->edgeNumberingVersion.value();
    info() << "Set edge uniqueId numbering version to '" << v << "' from caseoption";
    IMeshUniqueIdMng* mum = m_mesh->meshUniqueIdMng();
    mum->setEdgeBuilderVersion(v);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANECASEMESHSERVICE(ArcaneCaseMeshService,
                                              ArcaneCaseMeshService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
