// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkHdfV2PostProcessor.cc                                    (C) 2000-2026 */
/*                                                                           */
/* Post-processing in VTK HDF format.                                        */
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
#include "arcane/core/internal/IParallelMngInternal.h"
#include "arcane/core/internal/VtkCellTypes.h"
#include "arcane/core/internal/GatherGroup.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshEnvironment.h"

#include "arcane/hdf5/Hdf5Utils.h"
#include "arcane/hdf5/VtkHdfV2PostProcessor_axl.h"

#include <map>

// This format is described on the following web page:
//
// https://kitware.github.io/vtk-examples/site/VTKFileFormats/#hdf-file-formats
//
// Format 2.0 with integrated temporal evolution support is only available
// in the VTK master branch starting from April 2023.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Add verification test for saved values

// TODO: Look into saving uniqueId() (via vtkOriginalCellIds)

// TODO: Look into how to avoid saving the mesh at every iteration if it
//       does not change.

// TODO: Look into compression

// TODO: handle 2D variables

// TODO: outside of HDF5, implement a mechanism that groups several parts
// of the cell into one. This will allow reducing the number of ghost cells
// and using MPI/IO in hybrid mode.

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
   * \brief Class to store a pair (hdf_group, dataset_name).
   *
   * Instances of this class use a reference to an HDF5 group
   * and thus this group must live longer than the instance.
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
   * \brief Class to store offset information.
   *
   * This is a pair (hdf_group, dataset_name).
   *
   * The group can be null in which case it is an offset that is
   * only calculated and will not be saved.
   *
   * Instances of this class use a reference to an HDF5 group
   * and thus this group must live longer than the instance.
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
    //! Offset value. (-1) if writing to the end of the array
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

  //! Offset information for the part to write associated with a rank
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

    //! Number of elements across all ranks
    Int64 m_total_size = 0;
    //! Number of elements on my rank
    Int64 m_size = 0;
    //! Offset of my rank
    Int64 m_offset = -1;
  };

  //! Collective information for an ItemGroup;
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

    //! Associated group
    ItemGroup m_item_group;
    //! Writing information.
    WritePartInfo m_write_part_info;
  };

  /*!
   * \brief Stores info about the data to be saved and the associated offset.
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

  //! Associated mesh
  IMesh* m_mesh = nullptr;

  //! Associated material manager (may be null)
  IMeshMaterialMng* m_material_mng = nullptr;

  //! List of groups to save
  ItemGroupCollection m_groups;

  //! List of times
  UniqueArray<Real> m_times;

  //! Current HDF filename
  String m_full_filename;

  //! Output directory.
  String m_directory_name;

  //! HDF file identifier
  HFile m_file_id;

  HGroup m_top_group;
  HGroup m_cell_data_group;
  HGroup m_node_data_group;

  HGroup m_steps_group;
  HGroup m_point_data_offsets_group;
  HGroup m_cell_data_offsets_group;
  HGroup m_field_data_offsets_group;

  bool m_is_parallel = false;
  bool m_is_collective_io = false;
  bool m_is_first_call = false;
  bool m_is_writer = false;
  Int32 m_writer = 0;

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

  GatherGroupInfo m_all_cells_gather_group_info;
  GatherGroupInfo m_all_nodes_gather_group_info;
  UniqueArray<Ref<GatherGroupInfo>> m_gather_info_materials_groups;

  /*!
   * \brief Maximum size (in kilobytes) for a write operation.
   *
   * If the write exceeds this size, it is split into multiple writes.
   * This may be necessary with MPI-IO for large volumes.
   */
  Int64 m_max_write_size = 0;

 private:

  void _addInt64ArrayAttribute(Hid& hid, const char* name, Span<const Int64> values);
  void _addStringAttribute(Hid& hid, const char* name, const String& value);

  template <typename DataType> void
  _writeDataSet1D(const DataInfo& data_info, GatherGroupInfo* gather_info, Span<const DataType> values);
  template <typename DataType> void
  _writeDataSet1DUsingCollectiveIO(const DataInfo& data_info, GatherGroupInfo* gather_info, Span<const DataType> values);
  template <typename DataType> void
  _writeDataSet1DCollective(const DataInfo& data_info, GatherGroupInfo* gather_info, Span<const DataType> values);
  template <typename DataType> void
  _writeDataSet2D(const DataInfo& data_info, GatherGroupInfo* gather_info, Span2<const DataType> values);
  template <typename DataType> void
  _writeDataSet2DUsingCollectiveIO(const DataInfo& data_info, GatherGroupInfo* gather_info, Span2<const DataType> values);
  template <typename DataType> void
  _writeDataSet2DCollective(const DataInfo& data_info, GatherGroupInfo* gather_info, Span2<const DataType> values);
  template <typename DataType> void
  _writeBasicTypeDataset(const DataInfo& data_info, GatherGroupInfo* gather_info, IData* data);
  void _writeReal3Dataset(const DataInfo& data_info, GatherGroupInfo* gather_info, IData* data);
  void _writeReal2Dataset(const DataInfo& data_info, GatherGroupInfo* gather_info, IData* data);

  String _getFileName()
  {
    StringBuilder sb(m_mesh->name());
    sb += ".hdf";
    return sb.toString();
  }
  template <typename DataType> void
  _writeDataSetGeneric(const DataInfo& data_info, GatherGroupInfo* gather_info, Int32 nb_dim,
                       Int64 dim1_size, Int64 dim2_size, const DataType* values_data,
                       bool is_collective);
  void _writeDataSetGeneric(const DataInfo& data_info, GatherGroupInfo* gather_info, Int32 nb_dim,
                            Int64 dim1_size, Int64 dim2_size, ConstMemoryView values_data,
                            const hid_t hdf_datatype_type, bool is_collective);
  void _addInt64Attribute(Hid& hid, const char* name, Int64 value);
  Int64 _readInt64Attribute(Hid& hid, const char* name);
  void _openOrCreateGroups();
  void _closeGroups();
  void _readAndSetOffset(DatasetInfo& offset_info, Int32 wanted_step);
  void _initializeOffsets();
  void _initializeItemGroupCollectiveInfos(ItemGroupCollectiveInfo& group_info, GatherGroupInfo& gather_info);
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
, m_all_cells_gather_group_info(mesh->parallelMng(), is_collective_io)
, m_all_nodes_gather_group_info(mesh->parallelMng(), is_collective_io)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
beginWrite(const VariableCollection& vars)
{
  ARCANE_UNUSED(vars);

  // Retrieve the material manager if it exists
  m_material_mng = IMeshMaterialMng::getReference(m_mesh, false);

  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 nb_rank = pm->commSize();
  m_is_parallel = nb_rank > 1;

  Int32 time_index = m_times.size();
  const bool is_first_call = (time_index < 2);
  m_is_first_call = is_first_call;
  if (is_first_call)
    info() << "WARNING: The 'VtkHdfV2' implementation is experimental";

  String filename = _getFileName();

  Directory dir(m_directory_name);

  m_full_filename = dir.file(filename);
  info(4) << "VtkHdfV2DataWriter::beginWrite() file=" << m_full_filename;

  HInit();
  HInit::useMutex(pm->isThreadImplementation(), pm);

  // It is possible to use the collective mode of HDF5 via MPI-IO in the following cases:
  // * Hdf5 was compiled with MPI,
  // * we are in pure MPI mode (neither shared memory mode nor hybrid mode).
  m_is_collective_io = m_is_collective_io && (pm->isParallel() && HInit::hasParallelHdf5());
  if (pm->isThreadImplementation() && !pm->isHybridImplementation())
    m_is_collective_io = false;

  if (is_first_call) {
    info() << "VtkHdfV2DataWriter: using collective MPI/IO ?=" << m_is_collective_io;
    info() << "VtkHdfV2DataWriter: max_write_size (kB) =" << m_max_write_size;
    info() << "VtkHdfV2DataWriter: has_material?=" << (m_material_mng != nullptr);
  }

  bool is_master_io = pm->isMasterIO();

  // True if we must participate in the writes
  // If we use MPI/IO with HDF5, all
  // ranks must perform all write operations to guarantee
  // metadata consistency.
  if (m_is_collective_io) {
    m_writer = pm->_internalApi()->masterParallelIORank();
    m_is_writer = (m_writer == pm->commRank());
  }
  else {
    m_writer = pm->masterIORank();
    m_is_writer = is_master_io;
  }

  // Indicates that we are using MPI/IO if requested
  HProperty plist_id;
  if (m_is_collective_io && m_is_writer)
    plist_id.createFilePropertyMPIIO(pm);

  // Even with MPI-IO, only one proc must create the directory.
  if (is_first_call && is_master_io)
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

  // Initializes collective information on cell and node groups
  _initializeItemGroupCollectiveInfos(m_all_cells_info, m_all_cells_gather_group_info);
  _initializeItemGroupCollectiveInfos(m_all_nodes_info, m_all_nodes_gather_group_info);

  CellGroup all_cells = m_mesh->allCells();
  NodeGroup all_nodes = m_mesh->allNodes();

  const Int32 nb_cell = all_cells.size();
  const Int32 nb_node = all_nodes.size();

  Int32 total_nb_connected_node = 0;
  ENUMERATE_ (Cell, icell, all_cells) {
    Cell cell = *icell;
    total_nb_connected_node += cell.nodeIds().size();
  }

  // For the offsets, the array size is equal
  // to the number of cells plus 1.
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

  // ggi is a GatherGroupInfo for arrays that are not of size nb_cell or nb_node.
  Ref<GatherGroupInfo> ggi_ref = createRef<GatherGroupInfo>(m_mesh->parallelMng(), m_is_collective_io);
  GatherGroupInfo* ggi = ggi_ref.get();

  // TODO: create an offset for this object (or look into how to calculate it automatically
  _writeDataSet1DCollective<Int64>({ { m_top_group, "Offsets" }, m_offset_for_cell_offset_info }, ggi, cells_offset);
  ggi->setNeedRecompute();

  _writeDataSet1DCollective<Int64>({ { m_top_group, "Connectivity" }, m_connectivity_offset_info }, ggi, cells_connectivity);
  ggi->setNeedRecompute();

  _writeDataSet1DCollective<unsigned char>({ { m_top_group, "Types" }, m_cell_offset_info }, &m_all_cells_gather_group_info, cells_type);

  {
    Int64 nb_cell_int64 = nb_cell;
    _writeDataSet1DCollective<Int64>({ { m_top_group, "NumberOfCells" }, m_part_offset_info }, ggi,
                                     asConstSpan(&nb_cell_int64));

    Int64 nb_node_int64 = nb_node;
    _writeDataSet1DCollective<Int64>({ { m_top_group, "NumberOfPoints" }, m_part_offset_info }, ggi,
                                     asConstSpan(&nb_node_int64));

    Int64 number_of_connectivity_ids = cells_connectivity.size();
    _writeDataSet1DCollective<Int64>({ { m_top_group, "NumberOfConnectivityIds" }, m_part_offset_info }, ggi,
                                     asConstSpan(&number_of_connectivity_ids));
  }
  ggi->setNeedRecompute();

  // Saves the unique IDs, types, and coordinates of the nodes.
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

    // Saves the unique ID of each node in the dataset "GlobalNodeId".
    _writeDataSet1DCollective<Int64>({ { m_node_data_group, "GlobalIds" }, m_cell_offset_info }, &m_all_nodes_gather_group_info, nodes_uid);

    // Saves the information about the node type (real or ghost).
    _writeDataSet1DCollective<unsigned char>({ { m_node_data_group, "vtkGhostType" }, m_cell_offset_info }, &m_all_nodes_gather_group_info, nodes_ghost_type);

    // Saves the coordinates of the nodes.
    _writeDataSet2DCollective<Real>({ { m_top_group, "Points" }, m_point_offset_info }, ggi, points);
    ggi->setNeedRecompute();
  }

  // Saves the information about the cell type (real or ghost)
  _writeDataSet1DCollective<unsigned char>({ { m_cell_data_group, "vtkGhostType" }, m_cell_offset_info }, &m_all_cells_gather_group_info, cells_ghost_type);

  // Saves the unique ID of each cell in the dataset "GlobalCellId".
  // Using the dataset "vtkOriginalCellIds" does not work in Paraview.
  _writeDataSet1DCollective<Int64>({ { m_cell_data_group, "GlobalIds" }, m_cell_offset_info }, &m_all_cells_gather_group_info, cells_uid);

  if (m_is_writer) {
    // List of times.
    Real current_time = m_times[time_index - 1];
    // TODO: Replace ggi with nullptr in non-collective mode?
    _writeDataSet1D<Real>({ { m_steps_group, "Values" }, m_time_offset_info }, ggi, asConstSpan(&current_time));

    // Part offset.
    Int64 comm_size = pm->commSize();
    Int64 part_offset = (time_index - 1) * comm_size;
    _writeDataSet1D<Int64>({ { m_steps_group, "PartOffsets" }, m_time_offset_info }, ggi, asConstSpan(&part_offset));

    // Number of times
    _addInt64Attribute(m_steps_group, "NSteps", time_index);
  }
  //ggi->needRecompute();

  _writeConstituentsGroups();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_writeConstituentsGroups()
{
  if (!m_material_mng)
    return;

  // Fills the information for groups related to constituents
  // NOTE: For now, we only process the media.
  for (IMeshEnvironment* env : m_material_mng->environments()) {
    CellGroup cells = env->cells();

    Ref<ItemGroupCollectiveInfo> group_info_ref = createRef<ItemGroupCollectiveInfo>(cells);
    m_materials_groups.add(group_info_ref);

    Ref<GatherGroupInfo> gather_info_ref = createRef<GatherGroupInfo>(m_material_mng->mesh()->parallelMng(), m_is_collective_io);
    m_gather_info_materials_groups.add(gather_info_ref);

    ItemGroupCollectiveInfo& group_info = *group_info_ref.get();
    GatherGroupInfo& gather_info = *gather_info_ref.get();
    _initializeItemGroupCollectiveInfos(group_info, gather_info);

    ConstArrayView<Int32> groups_ids = cells.view().localIds();
    DatasetGroupAndName dataset_group_name(m_top_group, String("Constituent_") + cells.name());
    if (m_is_first_call)
      info() << "Writing infos for group '" << cells.name() << "'";
    _writeDataSet1DCollective<Int32>({ dataset_group_name, m_cell_offset_info, group_info_ref.get() }, gather_info_ref.get(), groups_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Computes the offset of our part and the total number of elements.
 */
VtkHdfV2DataWriter::WritePartInfo VtkHdfV2DataWriter::
_computeWritePartInfo(Int64 local_size)
{
  // TODO: look into using a scan.
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
_initializeItemGroupCollectiveInfos(ItemGroupCollectiveInfo& group_info, GatherGroupInfo& gather_info)
{
  Int32 dim1_size = group_info.m_item_group.size();

  gather_info.setCollectiveIO(m_is_collective_io);
  gather_info.computeSize(dim1_size);

  Int32 computed_nb_elem = gather_info.nbElemOutput();

  group_info.setWritePartInfo(_computeWritePartInfo(computed_nb_elem));
}

namespace
{
  std::pair<Int64, Int64> _getInterval(Int64 index, Int64 nb_interval, Int64 total_size)
  {
    Int64 n = total_size;
    Int64 isize = n / nb_interval;
    Int64 ibegin = index * isize;
    // For the last interval, take the remaining elements
    if ((index + 1) == nb_interval)
      isize = n - ibegin;
    return { ibegin, isize };
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Writes a 1D or 2D data.
 *
 * For each time added, the data is written at the end of previous values
 * unless rolling back, in which case the offset is in data_info.
 */
void VtkHdfV2DataWriter::
_writeDataSetGeneric(const DataInfo& data_info, GatherGroupInfo* gather_info, Int32 nb_dim,
                     Int64 dim1_size, Int64 dim2_size,
                     ConstMemoryView values_data,
                     const hid_t hdf_type, bool is_collective)
{
  ARCANE_CHECK_POINTER(gather_info);
  if (nb_dim == 1)
    dim2_size = 1;

  HGroup& group = data_info.dataset.group;
  const String& name = data_info.dataset.name;

  // If positive or zero, indicates the write offset.
  // Otherwise, we write to the end of the current dataset.
  Int64 wanted_offset = data_info.datasetInfo().offset();

  static constexpr int MAX_DIM = 2;
  HDataset dataset;

  // In case of collective operation, local_dims and global_dims are
  // different on the first dimension. The second dimension is always
  // identical for local_dims and global_dims and should not be modified during
  // the entire calculation.

  // Dimensions of the dataset that the current rank will write.
  FixedArray<hsize_t, MAX_DIM> local_dims;
  local_dims[0] = dim1_size;
  local_dims[1] = dim2_size;

  // Cumulative dimensions of all ranks for writing.
  FixedArray<hsize_t, MAX_DIM> global_dims;

  // Maximum dimensions of the DataSet
  // For the second dimension, we assume it is constant over time.
  FixedArray<hsize_t, MAX_DIM> max_dims;
  max_dims[0] = H5S_UNLIMITED;
  max_dims[1] = dim2_size;

  herr_t herror = 0;
  Int64 write_offset = 0;

  Int64 my_index = 0;
  Int64 global_dim1_size = dim1_size;
  Int32 nb_participating_rank = 1;

  if (is_collective) {
    nb_participating_rank = gather_info->nbWriterGlobal();
    WritePartInfo part_info;
    if (data_info.m_group_info) {
      // If the data is associated with a group, then the information
      // about the offset has already been calculated
      part_info = data_info.m_group_info->writePartInfo();
    }
    else {
      part_info = _computeWritePartInfo(dim1_size);
    }
    global_dim1_size = part_info.totalSize();
    my_index = part_info.offset();
  }

  // The only collective operation was _computeWritePartInfo().
  if (!m_is_writer) {
    return;
  }

  HProperty write_plist_id;
  if constexpr (HInit::hasParallelHdf5()) {
    if (is_collective)
      write_plist_id.createDatasetTransfertCollectiveMPIIO();
    else
      write_plist_id.createDatasetTransfertIndependentMPIIO();
  }

  HSpace file_space;
  FixedArray<hsize_t, MAX_DIM> hyperslab_offsets;

  if (m_is_first_call) {
    // TODO: look into how to better calculate the chunk
    FixedArray<hsize_t, MAX_DIM> chunk_dims;
    global_dims[0] = global_dim1_size;
    global_dims[1] = dim2_size;
    // It is important that everyone has the same chunk size.
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
    // Expands the first dimension of the dataset.
    // We are going to add 'global_dim1_size' to this dimension.
    dataset.open(group, name.localstr());
    file_space = dataset.getSpace();
    int nb_dimension = file_space.nbDimension();
    if (nb_dimension != nb_dim)
      ARCANE_THROW(IOException, "Bad dimension '{0}' for dataset '{1}' (should be 1)",
                   nb_dimension, name);
    // TODO: Check that the second dimension is the same as the one saved.
    FixedArray<hsize_t, MAX_DIM> original_dims;
    file_space.getDimensions(original_dims.data(), nullptr);
    hsize_t offset0 = original_dims[0];
    // If we have a positive offset from DatasetInfo, we take it.
    // This means we have performed a rollback.
    if (wanted_offset >= 0) {
      offset0 = wanted_offset;
      info() << "Forcing offset to " << wanted_offset;
    }
    global_dims[0] = offset0 + global_dim1_size;
    global_dims[1] = dim2_size;
    write_offset = offset0;
    // Expands the dataset.
    // WARNING this invalidates file_space. It must therefore be re-read immediately after.
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

  // Performs the writing in multiple parts if requested.
  // This is only possible for collective writing.
  Int64 nb_interval = 1;
  if (is_collective && m_max_write_size > 0) {
    nb_interval = 1 + nb_write_byte / (m_max_write_size * 1024);
  }
  info(4) << "WRITE global_size=" << nb_write_byte << " max_size=" << m_max_write_size << " nb_interval=" << nb_interval;

  for (Int64 i = 0; i < nb_interval; ++i) {
    auto [index, nb_element] = _getInterval(i, nb_interval, dim1_size);
    // Selects the part of the data to write
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
    // Performs the writing
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
_writeDataSetGeneric(const DataInfo& data_info, GatherGroupInfo* gather_info, Int32 nb_dim,
                     Int64 dim1_size, Int64 dim2_size, const DataType* values_data,
                     bool is_collective)
{
  const hid_t hdf_type = m_standard_types.nativeType(DataType{});
  ConstMemoryView mem_view = makeConstMemoryView(values_data, sizeof(DataType), dim1_size * dim2_size);
  _writeDataSetGeneric(data_info, gather_info, nb_dim, dim1_size, dim2_size, mem_view, hdf_type, is_collective);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet1D(const DataInfo& data_info, GatherGroupInfo* gather_info, Span<const DataType> values)
{
  _writeDataSetGeneric(data_info, gather_info, 1, values.size(), 1, values.data(), false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet1DUsingCollectiveIO(const DataInfo& data_info, GatherGroupInfo* gather_info, Span<const DataType> values)
{
  _writeDataSetGeneric(data_info, gather_info, 1, values.size(), 1, values.data(), true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet1DCollective(const DataInfo& data_info, GatherGroupInfo* gather_info, Span<const DataType> values)
{
  ARCANE_CHECK_POINTER(gather_info);

  GatherGroup gg;

  gather_info->computeSizeT(values);
  gg.setGatherGroupInfo(gather_info);

  if (gg.isNeedGather()) {
    UniqueArray<DataType> all_values;
    gg.gatherToMasterIOT(values, all_values);

    if (m_is_collective_io)
      _writeDataSet1DUsingCollectiveIO(data_info, gather_info, all_values.constSpan());
    else
      _writeDataSet1D(data_info, gather_info, all_values.constSpan());
  }
  else {
    if (m_is_collective_io)
      _writeDataSet1DUsingCollectiveIO(data_info, gather_info, values);
    else
      _writeDataSet1D(data_info, gather_info, values);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet2D(const DataInfo& data_info, GatherGroupInfo* gather_info, Span2<const DataType> values)
{
  _writeDataSetGeneric(data_info, gather_info, 2, values.dim1Size(), values.dim2Size(), values.data(), false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet2DUsingCollectiveIO(const DataInfo& data_info, GatherGroupInfo* gather_info, Span2<const DataType> values)
{
  _writeDataSetGeneric(data_info, gather_info, 2, values.dim1Size(), values.dim2Size(), values.data(), true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeDataSet2DCollective(const DataInfo& data_info, GatherGroupInfo* gather_info, Span2<const DataType> values)
{
  ARCANE_CHECK_POINTER(gather_info);

  GatherGroup gg;

  gather_info->computeSizeT(values);
  gg.setGatherGroupInfo(gather_info);

  if (gg.isNeedGather()) {
    UniqueArray2<DataType> all_values;
    gg.gatherToMasterIOT(values, all_values);

    if (m_is_collective_io)
      _writeDataSet2DUsingCollectiveIO(data_info, gather_info, all_values.constSpan());
    else
      _writeDataSet2D(data_info, gather_info, all_values.constSpan());
  }
  else {
    if (m_is_collective_io)
      _writeDataSet2DUsingCollectiveIO(data_info, gather_info, values);
    else
      _writeDataSet2D(data_info, gather_info, values);
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
  // Save the recorded offsets

  if (m_is_writer) {
    for (const auto& i : m_offset_info_list) {
      Int64 offset = i.second;
      const DatasetInfo& offset_info = i.first;
      HGroup* hdf_group = offset_info.group();
      //info() << "OFFSET_INFO name=" << offset_info.name() << " offset=" << offset;
      if (hdf_group) {
        Ref<GatherGroupInfo> ggi = createRef<GatherGroupInfo>(m_mesh->parallelMng(), m_is_collective_io);
        _writeDataSet1D<Int64>({ { *hdf_group, offset_info.name() }, m_time_offset_info }, ggi.get(), asConstSpan(&offset));
      }
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
  // Any group opened here must be closed in closeGroups().
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
  GatherGroupInfo* gather_info = nullptr;
  switch (item_kind) {
  case IK_Cell:
    group = &m_cell_data_group;
    offset_info = m_cell_offset_info;
    group_info = &m_all_cells_info;
    gather_info = &m_all_cells_gather_group_info;
    break;
  case IK_Node:
    group = &m_node_data_group;
    offset_info = m_point_offset_info;
    group_info = &m_all_nodes_info;
    gather_info = &m_all_nodes_gather_group_info;
    break;
  default:
    ARCANE_FATAL("Only export of 'Cell' or 'Node' variable is implemented (name={0})", var->name());
  }

  ARCANE_CHECK_POINTER(group);

  DataInfo data_info(DatasetGroupAndName{ *group, var->name() }, offset_info, group_info);
  eDataType data_type = var->dataType();
  switch (data_type) {
  case DT_Real:
    _writeBasicTypeDataset<Real>(data_info, gather_info, data);
    break;
  case DT_Int64:
    _writeBasicTypeDataset<Int64>(data_info, gather_info, data);
    break;
  case DT_Int32:
    _writeBasicTypeDataset<Int32>(data_info, gather_info, data);
    break;
  case DT_Real3:
    _writeReal3Dataset(data_info, gather_info, data);
    break;
  case DT_Real2:
    _writeReal2Dataset(data_info, gather_info, data);
    break;
  default:
    warning() << String::format("Export for datatype '{0}' is not supported (var_name={1})", data_type, var->name());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VtkHdfV2DataWriter::
_writeBasicTypeDataset(const DataInfo& data_info, GatherGroupInfo* gather_info, IData* data)
{
  auto* true_data = dynamic_cast<IArrayDataT<DataType>*>(data);
  ARCANE_CHECK_POINTER(true_data);
  _writeDataSet1DCollective(data_info, gather_info, Span<const DataType>(true_data->view()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_writeReal3Dataset(const DataInfo& data_info, GatherGroupInfo* gather_info, IData* data)
{
  auto* true_data = dynamic_cast<IArrayDataT<Real3>*>(data);
  ARCANE_CHECK_POINTER(true_data);
  SmallSpan<const Real3> values(true_data->view());
  Int32 nb_value = values.size();
  // TODO: optimize this without passing through a temporary array
  UniqueArray2<Real> scalar_values;
  scalar_values.resize(nb_value, 3);
  for (Int32 i = 0; i < nb_value; ++i) {
    Real3 v = values[i];
    scalar_values[i][0] = v.x;
    scalar_values[i][1] = v.y;
    scalar_values[i][2] = v.z;
  }
  _writeDataSet2DCollective<Real>(data_info, gather_info, scalar_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_writeReal2Dataset(const DataInfo& data_info, GatherGroupInfo* gather_info, IData* data)
{
  // Convert to an array of 3 components where the last one will be 0.
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
  _writeDataSet2DCollective<Real>(data_info, gather_info, scalar_values);
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
  // There are 5 offset values used:
  //
  // - offset on the number of cells (CellOffsets). This offset has a number of elements
  //   equal to the number of saved time steps and is increased at each number of cells output. This offset
  //   is also used for cell variables
  // - offset on the number of nodes (PointOffsets). It is equivalent to 'CellOffsets' but
  //   for nodes.
  // - offset for "NumberOfCells", "NumberOfPoints" and "NumberOfConnectivityIds". For each
  //   of these fields there are NbPart values per time step, with 'NbPart' being the number of parts (and
  //   thus the number of sub-domains if no grouping is done). There are thus a total of
  //   NbPart * NbTimeStep in this offset field.
  // - offset for the "Connectivity" field, which is called "ConnectivityIdOffsets".
  //   This offset has a number of elements equal to the number of saved time steps.
  // - offset for the "Offsets" field. "Offset" contains for each cell the offset in
  //   "Connectivity" of the cell node connectivity. This offset is not saved,
  //   but since this field has a number of values equal to the number of cells plus one it is possible
  //   to deduce it from "CellOffsets" (it equals "CellOffsets" plus the current time index).

  m_cell_offset_info = DatasetInfo(m_steps_group, "CellOffsets");
  m_point_offset_info = DatasetInfo(m_steps_group, "PointOffsets");
  m_connectivity_offset_info = DatasetInfo(m_steps_group, "ConnectivityIdOffsets");
  // These three offsets are not saved in the VTK format
  m_offset_for_cell_offset_info = DatasetInfo("_OffsetForCellOffsetInfo");
  m_part_offset_info = DatasetInfo("_PartOffsetInfo");
  m_time_offset_info = DatasetInfo("_TimeOffsetInfo");

  // Check if we haven't done a backward run.
  // This is the case if the number of saved time steps is greater than the number
  // of values in \a m_times.
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
      // This means a backward run has been performed.
      // In this case, the offsets must be reread
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
 * \brief Post-processing in VtkHdf V2 format.
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
