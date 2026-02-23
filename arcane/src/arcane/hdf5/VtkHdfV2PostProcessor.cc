// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkHdfV2PostProcessor.cc                                    (C) 2000-2026 */
/*                                                                           */
/* Pos-traitement au format VTK HDF.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/PostProcessorWriterBase.h"
#include "arcane/core/Directory.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IDataWriter.h"
#include "arcane/core/IData.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/internal/VtkCellTypes.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshEnvironment.h"

#include "arcane/hdf5/Hdf5Utils.h"
#include "arcane/hdf5/VtkHdfV2PostProcessor_axl.h"

#include <map>

// Ce format est décrit sur la page web suivante :
//
// https://kitware.github.io/vtk-examples/site/VTKFileFormats/#hdf-file-formats
//
// Le format 2.0 avec le support intégré de l'évolution temporelle n'est
// disponible que dans la branche master de VTK à partir d'avril 2023.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Ajouter test de vérifcation des valeurs sauvegardées

// TODO: Regarder la sauvegarde des uniqueId() (via vtkOriginalCellIds)

// TODO: Regarder comment éviter de sauver le maillage à chaque itération s'il
//       ne change pas.

// TODO: Regarder la compression

// TODO: gérer les variables 2D

// TODO: hors HDF5, faire un mécanisme qui regroupe plusieurs parties
// du maillage en une seule. Cela permettra de réduire le nombre de mailles
// fantômes et d'utiliser MPI/IO en mode hybride.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Hdf5Utils;
using namespace Materials;

namespace
{
  template <typename T> Span<const T>
  asConstSpan(const T* v)
  {
    return Span<const T>(v, 1);
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkHdfV2DataWriter
: public TraceAccessor
, public IDataWriter
{
 public:

  /*!
   * \brief Classe pour conserver un couple (hdf_group,nom_du_dataset).
   *
   * Les instances de cette classe utilisent une référence sur un groupe HDF5
   * et ce dernier doit donc vivre plus longtemps que l'instance.
   */
  struct DatasetGroupAndName
  {
   public:

    DatasetGroupAndName(HGroup& group_, const String& name_)
    : group(group_)
    , name(name_)
    {}

   public:

    HGroup& group;
    String name;
  };

  /*!
   * \brief Classe pour conserver les information d'un offset.
   *
   * Il s'agit d'un couple (hdf_group,nom_du_dataset).
   *
   * Le groupe peut être nul auquel cas il s'agit d'un offset qui est
   * uniquement calculé et qui ne sera pas sauvegardé.
   *
   * Les instances de cette classe utilisent une référence sur un groupe HDF5
   * et ce dernier doit donc vivre plus longtemps que l'instance.
   */
  struct DatasetInfo
  {
    DatasetInfo() = default;
    explicit DatasetInfo(const String& name)
    : m_name(name)
    {}
    DatasetInfo(HGroup& _group, const String& name)
    : m_group(&_group)
    , m_name(name)
    {}
    bool isNull() const { return m_name.null(); }

    HGroup* group() const { return m_group; }
    const String& name() const { return m_name; }
    //! Valeur de l'offset. (-1) si on écrit à la fin du tableau
    Int64 offset() const { return m_offset; }
    void setOffset(Int64 v) { m_offset = v; }
    friend bool operator<(const DatasetInfo& s1, const DatasetInfo& s2)
    {
      return (s1.m_name < s2.m_name);
    }

   private:

    HGroup* m_group = nullptr;
    String m_name;
    Int64 m_offset = -1;
  };

  //! Informations sur l'offset de la partie à écrire associée à un rang
  struct WritePartInfo
  {
   public:

    void setTotalSize(Int64 v) { m_total_size = v; }
    void setSize(Int64 v) { m_size = v; }
    void setOffset(Int64 v) { m_offset = v; }

    Int64 totalSize() const { return m_total_size; }
    Int64 size() const { return m_size; }
    Int64 offset() const { return m_offset; }

   private:

    //! Nombre d'éléments sur tous les rangs
    Int64 m_total_size = 0;
    //! Nombre d'éléments de mon rang
    Int64 m_size = 0;
    //! Offset de mon rang
    Int64 m_offset = -1;
  };

  //! Informations collectives sur un ItemGroup;
  struct ItemGroupCollectiveInfo
  {
   public:

    explicit ItemGroupCollectiveInfo(const ItemGroup& g)
    : m_item_group(g)
    {}

   public:

    void setWritePartInfo(const WritePartInfo& part_info) { m_write_part_info = part_info; }
    const WritePartInfo& writePartInfo() const { return m_write_part_info; }

   public:

    //! Groupe associé
    ItemGroup m_item_group;
    //! Informations sur l'écriture.
    WritePartInfo m_write_part_info;
  };

  /*!
   * \brief Conserve les infos sur les données à sauver et l'offset associé.
   */
  struct DataInfo
  {
   public:

    DataInfo(const DatasetGroupAndName& dname, const DatasetInfo& dataset_info)
    : dataset(dname)
    , m_dataset_info(dataset_info)
    {
    }
    DataInfo(const DatasetGroupAndName& dname, const DatasetInfo& dataset_info,
             ItemGroupCollectiveInfo* group_info)
    : dataset(dname)
    , m_dataset_info(dataset_info)
    , m_group_info(group_info)
    {
    }

   public:

    DatasetInfo datasetInfo() const { return m_dataset_info; }

   public:

    DatasetGroupAndName dataset;
    DatasetInfo m_dataset_info;
    ItemGroupCollectiveInfo* m_group_info = nullptr;
  };

 public:

  VtkHdfV2DataWriter(IMesh* mesh, const ItemGroupCollection& groups, bool is_collective_io);

 public:

  void beginWrite(const VariableCollection& vars) override;
  void endWrite() override;
  void setMetaData(const String& meta_data) override;
  void write(IVariable* var, IData* data) override;

 public:

  void setTimes(RealConstArrayView times) { m_times = times; }
  void setDirectoryName(const String& dir_name) { m_directory_name = dir_name; }
  void setMaxWriteSize(Int64 v) { m_max_write_size = v; }

 private:

  //! Maillage associé
  IMesh* m_mesh = nullptr;

  //! Gestionnaire de matériaux associé (peut-être nul)
  IMeshMaterialMng* m_material_mng = nullptr;

  //! Liste des groupes à sauver
  ItemGroupCollection m_groups;

  //! Liste des temps
  UniqueArray<Real> m_times;

  //! Nom du fichier HDF courant
  String m_full_filename;

  //! Répertoire de sortie.
  String m_directory_name;

  //! Identifiant HDF du fichier
  HFile m_file_id;

  HGroup m_top_group;
  HGroup m_cell_data_group;
  HGroup m_node_data_group;

  HGroup m_steps_group;
  HGroup m_point_data_offsets_group;
  HGroup m_cell_data_offsets_group;
  HGroup m_field_data_offsets_group;

  bool m_is_parallel = false;
  bool m_is_master_io = false;
  bool m_is_collective_io = false;
  bool m_is_first_call = false;
  bool m_is_writer = false;

  DatasetInfo m_cell_offset_info;
  DatasetInfo m_point_offset_info;
  DatasetInfo m_connectivity_offset_info;
  DatasetInfo m_offset_for_cell_offset_info;
  DatasetInfo m_part_offset_info;
  DatasetInfo m_time_offset_info;
  std::map<DatasetInfo, Int64> m_offset_info_list;

  StandardTypes m_standard_types{ false };

  ItemGroupCollectiveInfo m_all_cells_info;
  ItemGroupCollectiveInfo m_all_nodes_info;
  UniqueArray<Ref<ItemGroupCollectiveInfo>> m_materials_groups;

  /*!
   * \brief Taille maximale (en kilo-octet) pour une écriture.
   *
   * Si l'écriture dépasse cette taille, elle est scindée en plusieurs écriture.
   * Cela peut être nécessaire avec MPI-IO pour les gros volumes.
   */
  Int64 m_max_write_size = 0;

 private:

  void _addInt64ArrayAttribute(Hid& hid, const char* name, Span<const Int64> values);
  void _addStringAttribute(Hid& hid, const char* name, const String& value);

  template <typename DataType> void
  _writeDataSet1D(const DataInfo& data_info, Span<const DataType> values);
  template <typename DataType> void
  _writeDataSet1DUsingCollectiveIO(const DataInfo& data_info, Span<const DataType> values);
  template <typename DataType> void
  _writeDataSet1DCollective(const DataInfo& data_info, Span<const DataType> values);
  template <typename DataType> void
  _writeDataSet2D(const DataInfo& data_info, Span2<const DataType> values);
  template <typename DataType> void
  _writeDataSet2DUsingCollectiveIO(const DataInfo& data_info, Span2<const DataType> values);
  template <typename DataType> void
  _writeDataSet2DCollective(const DataInfo& data_info, Span2<const DataType> values);
  template <typename DataType> void
  _writeBasicTypeDataset(const DataInfo& data_info, IData* data);
  void _writeReal3Dataset(const DataInfo& data_info, IData* data);
  void _writeReal2Dataset(const DataInfo& data_info, IData* data);

  String _getFileName()
  {
    StringBuilder sb(m_mesh->name());
    sb += ".hdf";
    return sb.toString();
  }
  template <typename DataType> void
  _writeDataSetGeneric(const DataInfo& data_info, Int32 nb_dim,
                       Int64 dim1_size, Int64 dim2_size, const DataType* values_data,
                       bool is_collective);
  void _writeDataSetGeneric(const DataInfo& data_info, Int32 nb_dim,
                            Int64 dim1_size, Int64 dim2_size, ConstMemoryView values_data,
                            const hid_t hdf_datatype_type, bool is_collective);
  void _addInt64Attribute(Hid& hid, const char* name, Int64 value);
  Int64 _readInt64Attribute(Hid& hid, const char* name);
  void _openOrCreateGroups();
  void _closeGroups();
  void _readAndSetOffset(DatasetInfo& offset_info, Int32 wanted_step);
  void _initializeOffsets();
  void _initializeItemGroupCollectiveInfos(ItemGroupCollectiveInfo& group_info);
  WritePartInfo _computeWritePartInfo(Int64 local_size);
  void _writeConstituentsGroups();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VtkHdfV2DataWriter::
VtkHdfV2DataWriter(IMesh* mesh, const ItemGroupCollection& groups, bool is_collective_io)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_groups(groups)
, m_is_collective_io(is_collective_io)
, m_all_cells_info(mesh->allCells())
, m_all_nodes_info(mesh->allNodes())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
beginWrite(const VariableCollection& vars)
{
  ARCANE_UNUSED(vars);

  // Récupère le gestionnaire de matériaux s'il existe
  m_material_mng = IMeshMaterialMng::getReference(m_mesh, false);

  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 nb_rank = pm->commSize();
  m_is_parallel = nb_rank > 1;
  m_is_master_io = pm->isMasterIO();

  Int32 time_index = m_times.size();
  const bool is_first_call = (time_index < 2);
  m_is_first_call = is_first_call;
  if (is_first_call)
    info() << "WARNING: L'implémentation au format 'VtkHdfV2' est expérimentale";

  String filename = _getFileName();

  Directory dir(m_directory_name);

  m_full_filename = dir.file(filename);
  info(4) << "VtkHdfV2DataWriter::beginWrite() file=" << m_full_filename;

  HInit();

  // Il est possible d'utiliser le mode collectif de HDF5 via MPI-IO dans les cas suivants :
  // * Hdf5 a été compilé avec MPI,
  // * on est en mode MPI pure (ni mode mémoire partagé, ni mode hybride).
  m_is_collective_io = m_is_collective_io && (pm->isParallel() && HInit::hasParallelHdf5());
  if (pm->isHybridImplementation() || pm->isThreadImplementation())
    m_is_collective_io = false;

  if (is_first_call) {
    info() << "VtkHdfV2DataWriter: using collective MPI/IO ?=" << m_is_collective_io;
    info() << "VtkHdfV2DataWriter: max_write_size (kB) =" << m_max_write_size;
    info() << "VtkHdfV2DataWriter: has_material?=" << (m_material_mng != nullptr);
  }

  // Vrai si on doit participer aux écritures
  // Si on utilise MPI/IO avec HDF5, il faut tout de même que tous
  // les rangs fassent toutes les opérations d'écriture pour garantir
  // la cohérence des méta-données.
  m_is_writer = m_is_master_io || m_is_collective_io;

  // Indique qu'on utilise MPI/IO si demandé
  HProperty plist_id;
  if (m_is_collective_io)
    plist_id.createFilePropertyMPIIO(pm);

  if (is_first_call && m_is_master_io)
    dir.createDirectory();

  if (m_is_collective_io)
    pm->barrier();

  if (m_is_writer) {
    m_standard_types.initialize();

    if (is_first_call)
      m_file_id.openTruncate(m_full_filename, plist_id.id());
    else
      m_file_id.openAppend(m_full_filename, plist_id.id());

    _openOrCreateGroups();

    if (is_first_call) {
      std::array<Int64, 2> version = { 2, 0 };
      _addInt64ArrayAttribute(m_top_group, "Version", version);
      _addStringAttribute(m_top_group, "Type", "UnstructuredGrid");
    }
  }

  // Initialise les informations collectives sur les groupes de mailles et noeuds
  _initializeItemGroupCollectiveInfos(m_all_cells_info);
  _initializeItemGroupCollectiveInfos(m_all_nodes_info);

  CellGroup all_cells = m_mesh->allCells();
  NodeGroup all_nodes = m_mesh->allNodes();

  const Int32 nb_cell = all_cells.size();
  const Int32 nb_node = all_nodes.size();

  Int32 total_nb_connected_node = 0;
  ENUMERATE_ (Cell, icell, all_cells) {
    Cell cell = *icell;
    total_nb_connected_node += cell.nodeIds().size();
  }

  // Pour les offsets, la taille du tableau est égal
  // au nombre de mailles plus 1.
  UniqueArray<Int64> cells_connectivity(total_nb_connected_node);
  UniqueArray<Int64> cells_offset(nb_cell + 1);
  UniqueArray<unsigned char> cells_ghost_type(nb_cell);
  UniqueArray<unsigned char> cells_type(nb_cell);
  UniqueArray<Int64> cells_uid(nb_cell);
  cells_offset[0] = 0;
  {
    Int32 connected_node_index = 0;
    ENUMERATE_CELL (icell, all_cells) {
      Int32 index = icell.index();
      Cell cell = *icell;

      cells_uid[index] = cell.uniqueId();

      Byte ghost_type = 0;
      bool is_ghost = !cell.isOwn();
      if (is_ghost)
        ghost_type = VtkUtils::CellGhostTypes::DUPLICATECELL;
      cells_ghost_type[index] = ghost_type;

      unsigned char vtk_type = VtkUtils::arcaneToVtkCellType(cell.type());
      cells_type[index] = vtk_type;
      for (NodeLocalId node : cell.nodeIds()) {
        cells_connectivity[connected_node_index] = node;
        ++connected_node_index;
      }
      cells_offset[index + 1] = connected_node_index;
    }
  }

  _initializeOffsets();

  // TODO: faire un offset pour cet objet (ou regarder comment le calculer automatiquement
  _writeDataSet1DCollective<Int64>({ { m_top_group, "Offsets" }, m_offset_for_cell_offset_info }, cells_offset);

  _writeDataSet1DCollective<Int64>({ { m_top_group, "Connectivity" }, m_connectivity_offset_info },
                                   cells_connectivity);
  _writeDataSet1DCollective<unsigned char>({ { m_top_group, "Types" }, m_cell_offset_info }, cells_type);

  {
    Int64 nb_cell_int64 = nb_cell;
    _writeDataSet1DCollective<Int64>({ { m_top_group, "NumberOfCells" }, m_part_offset_info },
                                     asConstSpan(&nb_cell_int64));
    Int64 nb_node_int64 = nb_node;
    _writeDataSet1DCollective<Int64>({ { m_top_group, "NumberOfPoints" }, m_part_offset_info },
                                     asConstSpan(&nb_node_int64));
    Int64 number_of_connectivity_ids = cells_connectivity.size();
    _writeDataSet1DCollective<Int64>({ { m_top_group, "NumberOfConnectivityIds" }, m_part_offset_info },
                                     asConstSpan(&number_of_connectivity_ids));
  }

  // Sauve les uniqueIds, les types et les coordonnées des noeuds.
  {
    UniqueArray<Int64> nodes_uid(nb_node);
    UniqueArray<unsigned char> nodes_ghost_type(nb_node);
    VariableNodeReal3& nodes_coordinates(m_mesh->nodesCoordinates());
    UniqueArray2<Real> points;
    points.resize(nb_node, 3);
    ENUMERATE_ (Node, inode, all_nodes) {
      Int32 index = inode.index();
      Node node = *inode;

      nodes_uid[index] = node.uniqueId();

      Byte ghost_type = 0;
      bool is_ghost = !node.isOwn();
      if (is_ghost)
        ghost_type = VtkUtils::PointGhostTypes::DUPLICATEPOINT;
      nodes_ghost_type[index] = ghost_type;

      Real3 pos = nodes_coordinates[inode];
      points[index][0] = pos.x;
      points[index][1] = pos.y;
      points[index][2] = pos.z;
    }

    // Sauve l'uniqueId de chaque nœud dans le dataset "GlobalNodeId".
    _writeDataSet1DCollective<Int64>({ { m_node_data_group, "GlobalNodeId" }, m_cell_offset_info }, nodes_uid);

    // Sauve les informations sur le type de nœud (réel ou fantôme).
    _writeDataSet1DCollective<unsigned char>({ { m_node_data_group, "vtkGhostType" }, m_cell_offset_info }, nodes_ghost_type);

    // Sauve les coordonnées des noeuds.
    _writeDataSet2DCollective<Real>({ { m_top_group, "Points" }, m_point_offset_info }, points);
  }

  // Sauve les informations sur le type de maille (réel ou fantôme)
  _writeDataSet1DCollective<unsigned char>({ { m_cell_data_group, "vtkGhostType" }, m_cell_offset_info }, cells_ghost_type);

  // Sauve l'uniqueId de chaque maille dans le dataset "GlobalCellId".
  // L'utilisation du dataset "vtkOriginalCellIds" ne fonctionne pas dans Paraview.
  _writeDataSet1DCollective<Int64>({ { m_cell_data_group, "GlobalCellId" }, m_cell_offset_info }, cells_uid);

  if (m_is_writer) {
    // Liste des temps.
    Real current_time = m_times[time_index - 1];
    _writeDataSet1D<Real>({ { m_steps_group, "Values" }, m_time_offset_info }, asConstSpan(&current_time));

    // Offset de la partie.
    Int64 comm_size = pm->commSize();
    Int64 part_offset = (time_index - 1) * comm_size;
    _writeDataSet1D<Int64>({ { m_steps_group, "PartOffsets" }, m_time_offset_info }, asConstSpan(&part_offset));

    // Nombre de temps
    _addInt64Attribute(m_steps_group, "NSteps", time_index);
  }

  _writeConstituentsGroups();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_writeConstituentsGroups()
{
  if (!m_material_mng)
    return;

  // Remplit les informations des groupes liés aux constituents
  // NOTE : Pour l'instant, on ne traite que les milieux.
  for (IMeshEnvironment* env : m_material_mng->environments()) {
    CellGroup cells = env->cells();
    Ref<ItemGroupCollectiveInfo> group_info_ref = createRef<ItemGroupCollectiveInfo>(cells);
    m_materials_groups.add(group_info_ref);
    ItemGroupCollectiveInfo& group_info = *group_info_ref.get();
    _initializeItemGroupCollectiveInfos(group_info);
    ConstArrayView<Int32> groups_ids = cells.view().localIds();
    DatasetGroupAndName dataset_group_name(m_top_group, String("Constituent_") + cells.name());
    if (m_is_first_call)
      info() << "Writing infos for group '" << cells.name() << "'";
    _writeDataSet1DCollective<Int32>({ dataset_group_name, m_cell_offset_info }, groups_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule l'offset de notre partie et le nombre total d'éléments.
 */
VtkHdfV2DataWriter::WritePartInfo VtkHdfV2DataWriter::
_computeWritePartInfo(Int64 local_size)
{
  // TODO: regarder pour utiliser un scan.
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();

  UniqueArray<Int64> ranks_size(nb_rank);
  ArrayView<Int64> all_sizes(ranks_size);
  Int64 dim1_size = local_size;
  pm->allGather(ConstArrayView<Int64>(1, &dim1_size), all_sizes);

  Int64 total_size = 0;
  for (Integer i = 0; i < nb_rank; ++i)
    total_size += all_sizes[i];

  Int64 my_index = 0;
  for (Integer i = 0; i < my_rank; ++i)
    my_index += all_sizes[i];

  WritePartInfo part_info;
  part_info.setTotalSize(total_size);
  part_info.setSize(local_size);
  part_info.setOffset(my_index);
  return part_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_initializeItemGroupCollectiveInfos(ItemGroupCollectiveInfo& group_info)
{
  Int64 dim1_size = group_info.m_item_group.size();
  group_info.setWritePartInfo(_computeWritePartInfo(dim1_size));
}

namespace
{
  std::pair<Int64, Int64> _getInterval(Int64 index, Int64 nb_interval, Int64 total_size)
  {
    Int64 n = total_size;
    Int64 isize = n / nb_interval;
    Int64 ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index + 1) == nb_interval)
      isize = n - ibegin;
    return { ibegin, isize };
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écrit une donnée 1D ou 2D.
 *
 * Pour chaque temps ajouté, la donnée est écrite à la fin des valeurs précédentes
 * sauf en cas de retour arrière où l'offset est dans data_info.
 */
void VtkHdfV2DataWriter::
_writeDataSetGeneric(const DataInfo& data_info, Int32 nb_dim,
                     Int64 dim1_size, Int64 dim2_size,
                     ConstMemoryView values_data,
                     const hid_t hdf_type, bool is_collective)
{
  if (nb_dim == 1)
    dim2_size = 1;

  HGroup& group = data_info.dataset.group;
  const String& name = data_info.dataset.name;

  // Si positif ou nul, indique l'offset d'écriture.
  // Sinon, on écrit à la fin du dataset actuel.
  Int64 wanted_offset = data_info.datasetInfo().offset();

  static constexpr int MAX_DIM = 2;
  HDataset dataset;

  // En cas d'opération collective, local_dims et global_dims sont
  // différents sur la première dimension. La deuxième dimension est toujours
  // identique pour local_dims et global_dims et ne doit pas être modifiée durant
  // tout le calcul.

  // Dimensions du dataset que le rang courant va écrire.
  FixedArray<hsize_t, MAX_DIM> local_dims;
  local_dims[0] = dim1_size;
  local_dims[1] = dim2_size;

  // Dimensions cumulées de tous les rangs pour l'écriture.
  FixedArray<hsize_t, MAX_DIM> global_dims;

  // Dimensions maximales du DataSet
  // Pour la deuxième dimension, on suppose qu'elle est constante au cours du temps.
  FixedArray<hsize_t, MAX_DIM> max_dims;
  max_dims[0] = H5S_UNLIMITED;
  max_dims[1] = dim2_size;

  herr_t herror = 0;
  Int64 write_offset = 0;

  Int64 my_index = 0;
  Int64 global_dim1_size = dim1_size;
  Int32 nb_participating_rank = 1;

  if (is_collective) {
    nb_participating_rank = m_mesh->parallelMng()->commSize();
    WritePartInfo part_info;
    if (data_info.m_group_info) {
      // Si la donnée est associée à un groupe, alors les informations
      // sur l'offset ont déjà été calculées
      part_info = data_info.m_group_info->writePartInfo();
    }
    else {
      part_info = _computeWritePartInfo(dim1_size);
    }
    global_dim1_size = part_info.totalSize();
    my_index = part_info.offset();
  }

  HProperty write_plist_id;
  if (is_collective)
    write_plist_id.createDatasetTransfertCollectiveMPIIO();

  HSpace file_space;
  FixedArray<hsize_t, MAX_DIM> hyperslab_offsets;

  if (m_is_first_call) {
    // TODO: regarder comment mieux calculer le chunk
    FixedArray<hsize_t, MAX_DIM> chunk_dims;
    global_dims[0] = global_dim1_size;
    global_dims[1] = dim2_size;
    // Il est important que tout le monde ait la même taille de chunk.
    Int64 chunk_size = global_dim1_size / nb_participating_rank;
    if (chunk_size < 1024)
      chunk_size = 1024;
    const Int64 max_chunk_size = 1024 * 1024 * 10;
    chunk_size = math::min(chunk_size, max_chunk_size);
    chunk_dims[0] = chunk_size;
    chunk_dims[1] = dim2_size;
    info() << "CHUNK nb_dim=" << nb_dim
           << " global_dim1_size=" << global_dim1_size
           << " chunk0=" << chunk_dims[0]
           << " chunk1=" << chunk_dims[1]
           << " name=" << name;
    file_space.createSimple(nb_dim, global_dims.data(), max_dims.data());
    HProperty plist_id;
    plist_id.create(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id.id(), nb_dim, chunk_dims.data());
    dataset.create(group, name.localstr(), hdf_type, file_space, HProperty{}, plist_id, HProperty{});

    if (is_collective) {
      hyperslab_offsets[0] = my_index;
      hyperslab_offsets[1] = 0;
    }
  }
  else {
    // Agrandit la première dimension du dataset.
    // On va ajouter 'global_dim1_size' à cette dimension.
    dataset.open(group, name.localstr());
    file_space = dataset.getSpace();
    int nb_dimension = file_space.nbDimension();
    if (nb_dimension != nb_dim)
      ARCANE_THROW(IOException, "Bad dimension '{0}' for dataset '{1}' (should be 1)",
                   nb_dimension, name);
    // TODO: Vérifier que la deuxième dimension est la même que celle sauvée.
    FixedArray<hsize_t, MAX_DIM> original_dims;
    file_space.getDimensions(original_dims.data(), nullptr);
    hsize_t offset0 = original_dims[0];
    // Si on a un offset positif issu de DatasetInfo alors on le prend.
    // Cela signifie qu'on a fait un retour arrière.
    if (wanted_offset >= 0) {
      offset0 = wanted_offset;
      info() << "Forcing offset to " << wanted_offset;
    }
    global_dims[0] = offset0 + global_dim1_size;
    global_dims[1] = dim2_size;
    write_offset = offset0;
    // Agrandit le dataset.
    // ATTENTION cela invalide file_space. Il faut donc le relire juste après.
    if ((herror = dataset.setExtent(global_dims.data())) < 0)
      ARCANE_THROW(IOException, "Can not extent dataset '{0}' (err={1})", name, herror);
    file_space = dataset.getSpace();

    hyperslab_offsets[0] = offset0 + my_index;
    hyperslab_offsets[1] = 0;
    info(4) << "APPEND nb_dim=" << nb_dim
            << " dim0=" << global_dims[0]
            << " count0=" << local_dims[0]
            << " offsets0=" << hyperslab_offsets[0] << " name=" << name;
  }

  Int64 nb_write_byte = global_dim1_size * dim2_size * values_data.datatypeSize();

  // Effectue l'écriture en plusieurs parties si demandé.
  // Cela n'est possible que pour l'écriture collective.
  Int64 nb_interval = 1;
  if (is_collective && m_max_write_size > 0) {
    nb_interval = 1 + nb_write_byte / (m_max_write_size * 1024);
  }
  info(4) << "WRITE global_size=" << nb_write_byte << " max_size=" << m_max_write_size << " nb_interval=" << nb_interval;

  for (Int64 i = 0; i < nb_interval; ++i) {
    auto [index, nb_element] = _getInterval(i, nb_interval, dim1_size);
    // Sélectionne la partie de la donnée à écrire
    FixedArray<hsize_t, 2> dims;
    dims[0] = nb_element;
    dims[1] = dim2_size;
    FixedArray<hsize_t, 2> offsets;
    offsets[0] = hyperslab_offsets[0] + index;
    offsets[1] = 0;
    if ((herror = H5Sselect_hyperslab(file_space.id(), H5S_SELECT_SET, offsets.data(), nullptr, dims.data(), nullptr)) < 0)
      ARCANE_THROW(IOException, "Can not select hyperslab '{0}' (err={1})", name, herror);

    HSpace memory_space;
    memory_space.createSimple(nb_dim, dims.data());
    Int64 data_offset = index * values_data.datatypeSize() * dim2_size;
    // Effectue l'écriture
    if ((herror = dataset.write(hdf_type, values_data.data() + data_offset, memory_space, file_space, write_plist_id)) < 0)
      ARCANE_THROW(IOException, "Can not write dataset '{0}' (err={1})", name, herror);

    if (dataset.isBad())
      ARCANE_THROW(IOException, "Can not write dataset '{0}'", name);
  }

  if (!data_info.datasetInfo().isNull())
    m_offset_info_list.insert(std::make_pair(data_info.datasetInfo(), write_offset));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSetGeneric(const DataInfo& data_info, Int32 nb_dim,
                     Int64 dim1_size, Int64 dim2_size, const DataType* values_data,
                     bool is_collective)
{
  const hid_t hdf_type = m_standard_types.nativeType(DataType{});
  ConstMemoryView mem_view = makeConstMemoryView(values_data, sizeof(DataType), dim1_size * dim2_size);
  _writeDataSetGeneric(data_info, nb_dim, dim1_size, dim2_size, mem_view, hdf_type, is_collective);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet1D(const DataInfo& data_info, Span<const DataType> values)
{
  _writeDataSetGeneric(data_info, 1, values.size(), 1, values.data(), false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet1DUsingCollectiveIO(const DataInfo& data_info, Span<const DataType> values)
{
  _writeDataSetGeneric(data_info, 1, values.size(), 1, values.data(), true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet1DCollective(const DataInfo& data_info, Span<const DataType> values)
{
  if (!m_is_parallel)
    return _writeDataSet1D(data_info, values);
  if (m_is_collective_io)
    return _writeDataSet1DUsingCollectiveIO(data_info, values);
  UniqueArray<DataType> all_values;
  IParallelMng* pm = m_mesh->parallelMng();
  pm->gatherVariable(values.smallView(), all_values, pm->masterIORank());
  if (m_is_master_io)
    _writeDataSet1D<DataType>(data_info, all_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet2D(const DataInfo& data_info, Span2<const DataType> values)
{
  _writeDataSetGeneric(data_info, 2, values.dim1Size(), values.dim2Size(), values.data(), false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet2DUsingCollectiveIO(const DataInfo& data_info, Span2<const DataType> values)
{
  _writeDataSetGeneric(data_info, 2, values.dim1Size(), values.dim2Size(), values.data(), true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet2DCollective(const DataInfo& data_info, Span2<const DataType> values)
{
  if (!m_is_parallel)
    return _writeDataSet2D(data_info, values);
  if (m_is_collective_io)
    return _writeDataSet2DUsingCollectiveIO(data_info, values);

  Int64 dim2_size = values.dim2Size();
  UniqueArray<DataType> all_values;
  IParallelMng* pm = m_mesh->parallelMng();
  Span<const DataType> values_1d(values.data(), values.totalNbElement());
  pm->gatherVariable(values_1d.smallView(), all_values, pm->masterIORank());
  if (m_is_master_io) {
    Int64 dim1_size = all_values.size();
    if (dim2_size != 0)
      dim1_size = dim1_size / dim2_size;
    Span2<const DataType> span2(all_values.data(), dim1_size, dim2_size);
    return _writeDataSet2D<DataType>(data_info, span2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_addInt64ArrayAttribute(Hid& hid, const char* name, Span<const Int64> values)
{
  hsize_t len = values.size();
  hid_t aid = H5Screate_simple(1, &len, nullptr);
  hid_t attr = H5Acreate2(hid.id(), name, H5T_NATIVE_INT64, aid, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0)
    ARCANE_FATAL("Can not create attribute '{0}'", name);
  int ret = H5Awrite(attr, H5T_NATIVE_INT64, values.data());
  if (ret < 0)
    ARCANE_FATAL("Can not write attribute '{0}'", name);
  H5Aclose(attr);
  H5Sclose(aid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_addInt64Attribute(Hid& hid, const char* name, Int64 value)
{
  HSpace aid(H5Screate(H5S_SCALAR));
  HAttribute attr;
  if (m_is_first_call)
    attr.create(hid, name, H5T_NATIVE_INT64, aid);
  else
    attr.open(hid, name);
  if (attr.isBad())
    ARCANE_FATAL("Can not create attribute '{0}'", name);
  herr_t ret = attr.write(H5T_NATIVE_INT64, &value);
  if (ret < 0)
    ARCANE_FATAL("Can not write attribute '{0}'", name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 VtkHdfV2DataWriter::
_readInt64Attribute(Hid& hid, const char* name)
{
  HAttribute attr;
  attr.open(hid, name);
  if (attr.isBad())
    ARCANE_FATAL("Can not open attribute '{0}'", name);
  Int64 value;
  herr_t ret = attr.read(H5T_NATIVE_INT64, &value);
  if (ret < 0)
    ARCANE_FATAL("Can not read attribute '{0}'", name);
  return value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_addStringAttribute(Hid& hid, const char* name, const String& value)
{
  hid_t aid = H5Screate(H5S_SCALAR);
  hid_t attr_type = H5Tcopy(H5T_C_S1);
  H5Tset_size(attr_type, value.length());
  hid_t attr = H5Acreate2(hid.id(), name, attr_type, aid, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0)
    ARCANE_FATAL("Can not create attribute {0}", name);
  int ret = H5Awrite(attr, attr_type, value.localstr());
  ret = H5Tclose(attr_type);
  if (ret < 0)
    ARCANE_FATAL("Can not write attribute '{0}'", name);
  H5Aclose(attr);
  H5Sclose(aid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
endWrite()
{
  // Sauvegarde les offsets enregistrés

  if (m_is_writer) {
    for (const auto& i : m_offset_info_list) {
      Int64 offset = i.second;
      const DatasetInfo& offset_info = i.first;
      HGroup* hdf_group = offset_info.group();
      //info() << "OFFSET_INFO name=" << offset_info.name() << " offset=" << offset;
      if (hdf_group)
        _writeDataSet1D<Int64>({ { *hdf_group, offset_info.name() }, m_time_offset_info }, asConstSpan(&offset));
    }
  }
  _closeGroups();
  m_file_id.close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_openOrCreateGroups()
{
  // Tout groupe ouvert ici doit être fermé dans closeGroups().
  m_top_group.openOrCreate(m_file_id, "VTKHDF");
  m_cell_data_group.openOrCreate(m_top_group, "CellData");
  m_node_data_group.openOrCreate(m_top_group, "PointData");
  m_steps_group.openOrCreate(m_top_group, "Steps");
  m_point_data_offsets_group.openOrCreate(m_steps_group, "PointDataOffsets");
  m_cell_data_offsets_group.openOrCreate(m_steps_group, "CellDataOffsets");
  m_field_data_offsets_group.openOrCreate(m_steps_group, "FieldDataOffsets");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_closeGroups()
{
  m_cell_data_group.close();
  m_node_data_group.close();
  m_point_data_offsets_group.close();
  m_cell_data_offsets_group.close();
  m_field_data_offsets_group.close();
  m_steps_group.close();
  m_top_group.close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
setMetaData(const String& meta_data)
{
  ARCANE_UNUSED(meta_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
write(IVariable* var, IData* data)
{
  info(4) << "Write VtkHdfV2 var=" << var->name();

  eItemKind item_kind = var->itemKind();

  if (var->dimension() != 1)
    ARCANE_FATAL("Only export of scalar item variable is implemented (name={0})", var->name());
  if (var->isPartial())
    ARCANE_FATAL("Export of partial variable is not implemented");

  HGroup* group = nullptr;
  DatasetInfo offset_info;
  ItemGroupCollectiveInfo* group_info = nullptr;
  switch (item_kind) {
  case IK_Cell:
    group = &m_cell_data_group;
    offset_info = m_cell_offset_info;
    group_info = &m_all_cells_info;
    break;
  case IK_Node:
    group = &m_node_data_group;
    offset_info = m_point_offset_info;
    group_info = &m_all_nodes_info;
    break;
  default:
    ARCANE_FATAL("Only export of 'Cell' or 'Node' variable is implemented (name={0})", var->name());
  }

  ARCANE_CHECK_POINTER(group);

  DataInfo data_info(DatasetGroupAndName{ *group, var->name() }, offset_info, group_info);
  eDataType data_type = var->dataType();
  switch (data_type) {
  case DT_Real:
    _writeBasicTypeDataset<Real>(data_info, data);
    break;
  case DT_Int64:
    _writeBasicTypeDataset<Int64>(data_info, data);
    break;
  case DT_Int32:
    _writeBasicTypeDataset<Int32>(data_info, data);
    break;
  case DT_Real3:
    _writeReal3Dataset(data_info, data);
    break;
  case DT_Real2:
    _writeReal2Dataset(data_info, data);
    break;
  default:
    warning() << String::format("Export for datatype '{0}' is not supported (var_name={1})", data_type, var->name());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeBasicTypeDataset(const DataInfo& data_info, IData* data)
{
  auto* true_data = dynamic_cast<IArrayDataT<DataType>*>(data);
  ARCANE_CHECK_POINTER(true_data);
  _writeDataSet1DCollective(data_info, Span<const DataType>(true_data->view()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_writeReal3Dataset(const DataInfo& data_info, IData* data)
{
  auto* true_data = dynamic_cast<IArrayDataT<Real3>*>(data);
  ARCANE_CHECK_POINTER(true_data);
  SmallSpan<const Real3> values(true_data->view());
  Int32 nb_value = values.size();
  // TODO: optimiser cela sans passer par un tableau temporaire
  UniqueArray2<Real> scalar_values;
  scalar_values.resize(nb_value, 3);
  for (Int32 i = 0; i < nb_value; ++i) {
    Real3 v = values[i];
    scalar_values[i][0] = v.x;
    scalar_values[i][1] = v.y;
    scalar_values[i][2] = v.z;
  }
  _writeDataSet2DCollective<Real>(data_info, scalar_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_writeReal2Dataset(const DataInfo& data_info, IData* data)
{
  // Converti en un tableau de 3 composantes dont la dernière vaudra 0.
  auto* true_data = dynamic_cast<IArrayDataT<Real2>*>(data);
  ARCANE_CHECK_POINTER(true_data);
  SmallSpan<const Real2> values(true_data->view());
  Int32 nb_value = values.size();
  UniqueArray2<Real> scalar_values;
  scalar_values.resize(nb_value, 3);
  for (Int32 i = 0; i < nb_value; ++i) {
    Real2 v = values[i];
    scalar_values[i][0] = v.x;
    scalar_values[i][1] = v.y;
    scalar_values[i][2] = 0.0;
  }
  _writeDataSet2DCollective<Real>(data_info, scalar_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_readAndSetOffset(DatasetInfo& offset_info, Int32 wanted_step)
{
  HGroup* hgroup = offset_info.group();
  ARCANE_CHECK_POINTER(hgroup);
  StandardArrayT<Int64> a(hgroup->id(), offset_info.name());
  UniqueArray<Int64> values;
  a.directRead(m_standard_types, values);
  Int64 offset_value = values[wanted_step];
  offset_info.setOffset(offset_value);
  info() << "VALUES name=" << offset_info.name() << " values=" << values
         << " wanted_step=" << wanted_step << " v=" << offset_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_initializeOffsets()
{
  // Il y a 5 valeurs d'offset utilisées :
  //
  // - offset sur le nombre de mailles (CellOffsets). Cet offset a pour nombre d'éléments
  //   le nombre de temps sauvés et est augmenté à chaque sortie du nombre de mailles. Cet offset
  //   est aussi utilisé pour les variables aux mailles
  // - offset sur le nombre de noeuds (PointOffsets). Il est équivalent à 'CellOffsets' mais
  //   pour les noeuds.
  // - offset pour "NumberOfCells", "NumberOfPoints" et "NumberOfConnectivityIds". Pour chacun
  //   de ces champs il y a NbPart valeurs par temps, avec 'NbPart' le nombre de parties (donc
  //   le nombre de sous-domaines si on ne fait pas de regroupement). Il y a ainsi au total
  //   NbPart * NbTimeStep dans ce champ d'offset.
  // - offset pour le champ "Connectivity" qui s'appelle "ConnectivityIdOffsets".
  //   Cet offset a pour nombre d'éléments le nombre de temps sauvés.
  // - offset pour le champ "Offsets". "Offset" contient pour chaque maille l'offset dans
  //   "Connectivity" de la connectivité des noeuds de la maille. Cet offset n'est pas sauvés,
  //   mais comme ce champ à un nombre de valeurs égal au nombre de mailles plus un il est possible
  //   de le déduire de "CellOffsets" (il vaut "CellOffsets" plus l'index du temps courant).

  m_cell_offset_info = DatasetInfo(m_steps_group, "CellOffsets");
  m_point_offset_info = DatasetInfo(m_steps_group, "PointOffsets");
  m_connectivity_offset_info = DatasetInfo(m_steps_group, "ConnectivityIdOffsets");
  // Ces trois offsets ne sont pas sauvegardés dans le format VTK
  m_offset_for_cell_offset_info = DatasetInfo("_OffsetForCellOffsetInfo");
  m_part_offset_info = DatasetInfo("_PartOffsetInfo");
  m_time_offset_info = DatasetInfo("_TimeOffsetInfo");

  // Regarde si on n'a pas fait de retour-arrière.
  // C'est le cas si le nombre de temps sauvés est supérieur au nombre
  // de valeurs de \a m_times.
  if (m_is_writer && !m_is_first_call) {
    IParallelMng* pm = m_mesh->parallelMng();
    const Int32 nb_rank = pm->commSize();
    Int64 nb_current_step = _readInt64Attribute(m_steps_group, "NSteps");
    Int32 time_index = m_times.size();
    info(4) << "NB_STEP=" << nb_current_step << " time_index=" << time_index
            << " current_time=" << m_times.back();
    const bool debug_times = false;
    if (debug_times) {
      StandardArrayT<Real> a1(m_steps_group.id(), "Values");
      UniqueArray<Real> times;
      a1.directRead(m_standard_types, times);
      info() << "TIMES=" << times;
    }
    if ((nb_current_step + 1) != time_index) {
      info() << "[VtkHdf] go_backward detected";
      Int32 wanted_step = time_index - 1;
      // Signifie qu'on a fait un retour arrière.
      // Dans ce cas, il faut relire les offsets
      _readAndSetOffset(m_cell_offset_info, wanted_step);
      _readAndSetOffset(m_point_offset_info, wanted_step);
      _readAndSetOffset(m_connectivity_offset_info, wanted_step);
      m_part_offset_info.setOffset(wanted_step * nb_rank);
      m_time_offset_info.setOffset(wanted_step);
      m_offset_for_cell_offset_info.setOffset(m_cell_offset_info.offset() + wanted_step * nb_rank);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Post-traitement au format VtkHdf V2.
 */
class VtkHdfV2PostProcessor
: public ArcaneVtkHdfV2PostProcessorObject
{
 public:

  explicit VtkHdfV2PostProcessor(const ServiceBuildInfo& sbi)
  : ArcaneVtkHdfV2PostProcessorObject(sbi)
  {
  }

  IDataWriter* dataWriter() override { return m_writer.get(); }
  void notifyBeginWrite() override
  {
    bool use_collective_io = true;
    Int64 max_write_size = 0;
    if (options()) {
      use_collective_io = options()->useCollectiveWrite();
      max_write_size = options()->maxWriteSize();
    }
    auto w = std::make_unique<VtkHdfV2DataWriter>(mesh(), groups(), use_collective_io);
    w->setMaxWriteSize(max_write_size);
    w->setTimes(times());
    Directory dir(baseDirectoryName());
    w->setDirectoryName(dir.file("vtkhdfv2"));
    m_writer = std::move(w);
  }
  void notifyEndWrite() override
  {
    m_writer = nullptr;
  }
  void close() override {}

 private:

  std::unique_ptr<IDataWriter> m_writer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_VTKHDFV2POSTPROCESSOR(VtkHdfV2PostProcessor,
                                              VtkHdfV2PostProcessor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
