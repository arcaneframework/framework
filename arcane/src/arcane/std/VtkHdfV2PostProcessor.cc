// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkHdfV2PostProcessor.cc                                    (C) 2000-2023 */
/*                                                                           */
/* Pos-traitement au format VTK HDF.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/IOException.h"

#include "arcane/core/PostProcessorWriterBase.h"
#include "arcane/core/Directory.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IDataWriter.h"
#include "arcane/core/IData.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMesh.h"

#include "arcane/std/Hdf5Utils.h"
#include "arcane/std/VtkHdfV2PostProcessor_axl.h"
#include "arcane/std/internal/VtkCellTypes.h"

#include <map>

// Ce format est décrit sur la page web suivante:
//
// https://kitware.github.io/vtk-examples/site/VTKFileFormats/#hdf-file-formats
//
// Le format 2.0 avec le support intégrée de l'évolution temporelle n'est
// disponible que dans la branche master de VTK à partir d'avril 2023.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Regarder la sauvegarde des uniqueId() (via vtkOriginalCellIds)

// TODO: Regarder comment éviter de sauver le maillage à chaque itération s'il
//       ne change pas.

// TODO: Regarder la compression

// TODO: gérer les variables 2D

// TODO: gérer les retour arrière : il faut réduire la taille des dataset.

// TODO: hors HDF5, faire un mécanisme qui regroupe plusieurs parties
// du maillage en une seule. Cela permettra de réduire le nombre de mailles
// fantômes et d'utiliser MPI/IO en mode hybride.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Hdf5Utils;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkHdfV2DataWriter
: public TraceAccessor
, public IDataWriter
{
 public:

  VtkHdfV2DataWriter(IMesh* mesh, ItemGroupCollection groups);

 public:

  void beginWrite(const VariableCollection& vars) override;
  void endWrite() override;
  void setMetaData(const String& meta_data) override;
  void write(IVariable* var, IData* data) override;

 public:

  void setTimes(RealConstArrayView times) { m_times = times; }
  void setDirectoryName(const String& dir_name) { m_directory_name = dir_name; }

 private:

  IMesh* m_mesh;

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

  std::map<String, Int64> m_variable_offset;

 private:

  void _addInt64ArrayAttribute(Hid& hid, const char* name, Span<const Int64> values);
  void _addStringAttribute(Hid& hid, const char* name, const String& value);

  template <typename DataType> Int64
  _writeDataSet1D(HGroup& group, const String& name, Span<const DataType> values);
  template <typename DataType> Int64
  _writeDataSet1DUsingCollectiveIO(HGroup& group, const String& name, Span<const DataType> values);
  template <typename DataType> Int64
  _writeDataSet1DCollective(HGroup& group, const String& name, Span<const DataType> values);
  template <typename DataType> Int64
  _writeDataSet2D(HGroup& group, const String& name, Span2<const DataType> values);
  template <typename DataType> Int64
  _writeDataSet2DUsingCollectiveIO(HGroup& group, const String& name, Span2<const DataType> values);
  template <typename DataType> Int64
  _writeDataSet2DCollective(HGroup& group, const String& name, Span2<const DataType> values);
  template <typename DataType> Int64
  _writeBasicTypeDataset(HGroup& group, IVariable* var, IData* data);
  Int64 _writeReal3Dataset(HGroup& group, IVariable* var, IData* data);
  Int64 _writeReal2Dataset(HGroup& group, IVariable* var, IData* data);

  String _getFileNameForTimeIndex(Int32)
  {
    StringBuilder sb(m_mesh->name());
    sb += ".hdf";
    return sb.toString();
  }
  template <typename DataType> Int64
  _writeDataSetGeneric(HGroup& group, const String& name, Int32 nb_dim,
                       Int64 dim1_size, Int64 dim2_size, const DataType* values_data,
                       bool is_collective);
  void _addInt64ttribute(Hid& hid, const char* name, Int64 value);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VtkHdfV2DataWriter::
VtkHdfV2DataWriter(IMesh* mesh, ItemGroupCollection groups)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_groups(groups)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
beginWrite(const VariableCollection& vars)
{
  ARCANE_UNUSED(vars);

  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 nb_rank = pm->commSize();
  m_is_parallel = nb_rank > 1;
  m_is_master_io = pm->isMasterIO();

  Int32 time_index = m_times.size();
  const bool is_first_call = (time_index < 2);
  m_is_first_call = is_first_call;
  if (is_first_call)
    pwarning() << "L'implémentation au format 'VtkHdfV2' est expérimentale";

  String filename = _getFileNameForTimeIndex(time_index);

  Directory dir(m_directory_name);

  m_full_filename = dir.file(filename);
  info(4) << "VtkHdfV2DataWriter::beginWrite() file=" << m_full_filename;

  HInit();

  // Il est possible d'utiliser le mode collectif de HDF5 via MPI-IO dans les cas suivants:
  // - Hdf5 a été compilé avec MPI
  // - on est en mode MPI pure (ni mode mémoire partagé, ni mode hybride)
  m_is_collective_io = pm->isParallel() && HInit::hasParallelHdf5();
  if (pm->isHybridImplementation() || pm->isThreadImplementation())
    m_is_collective_io = false;

  if (is_first_call)
    info() << "VtkHdfV2DataWriter: using collective MPI/IO ?=" << m_is_collective_io;

  // Vrai si on doit participer aux écritures
  m_is_writer = m_is_master_io || m_is_collective_io;

  HProperty plist_id;
  if (m_is_collective_io)
    plist_id.createFilePropertyMPIIO(pm);

  if (is_first_call) {
    if (m_is_master_io) {
      dir.createDirectory();
    }
  }

  if (m_is_collective_io)
    pm->barrier();

  HGroup top_group;

  if (m_is_writer) {
    if (is_first_call) {
      m_file_id.openTruncate(m_full_filename, plist_id.id());

      top_group.create(m_file_id, "VTKHDF");

      std::array<Int64, 2> version = { 2, 0 };
      _addInt64ArrayAttribute(top_group, "Version", version);
      _addStringAttribute(top_group, "Type", "UnstructuredGrid");

      m_cell_data_group.create(top_group, "CellData");
      m_node_data_group.create(top_group, "PointData");
      m_steps_group.create(top_group, "Steps");
      m_point_data_offsets_group.create(m_steps_group, "PointDataOffsets");
      m_cell_data_offsets_group.create(m_steps_group, "CellDataOffsets");
      m_field_data_offsets_group.create(m_steps_group, "FieldDataOffsets");
    }
    else {
      m_file_id.openAppend(m_full_filename, plist_id.id());
      top_group.open(m_file_id, "VTKHDF");
      m_cell_data_group.open(top_group, "CellData");
      m_node_data_group.open(top_group, "PointData");

      m_steps_group.open(top_group, "Steps");
      m_point_data_offsets_group.open(m_steps_group, "PointDataOffsets");
      m_cell_data_offsets_group.open(m_steps_group, "CellDataOffsets");
      m_field_data_offsets_group.open(m_steps_group, "FieldDataOffsets");
    }
  }

  CellGroup all_cells = m_mesh->allCells();
  NodeGroup all_nodes = m_mesh->allNodes();

  const Int32 nb_cell = all_cells.size();
  const Int32 nb_node = all_nodes.size();

  Int32 total_nb_connected_node = 0;
  {
    ENUMERATE_CELL (icell, all_cells) {
      Cell cell = *icell;
      total_nb_connected_node += cell.nodeIds().size();
    }
  }

  // Pour les offsets, la taille du tableau est égal
  // au nombre de mailles plus 1.
  UniqueArray<Int64> cells_connectivity(total_nb_connected_node);
  UniqueArray<Int64> cells_offset(nb_cell + 1);
  UniqueArray<unsigned char> cells_type(nb_cell);
  UniqueArray<unsigned char> cells_ghost_type(nb_cell);
  cells_offset[0] = 0;
  {
    Int32 connected_node_index = 0;
    ENUMERATE_CELL (icell, all_cells) {
      Int32 index = icell.index();
      Cell cell = *icell;
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

  {
    _writeDataSet1DCollective<Int64>(top_group, "Offsets", cells_offset);
  }
  {
    Int64 offset = _writeDataSet1DCollective<Int64>(top_group, "Connectivity", cells_connectivity);
    if (m_is_writer)
      _writeDataSet1D<Int64>(m_steps_group, "ConnectivityIdOffsets", Span<const Int64>(&offset, 1));
  }
  {
    Int64 offset = _writeDataSet1DCollective<unsigned char>(top_group, "Types", cells_type);
    if (m_is_writer)
      _writeDataSet1D<Int64>(m_steps_group, "CellOffsets", Span<const Int64>(&offset, 1));
  }

  UniqueArray<Int64> nb_cell_by_ranks(1);
  nb_cell_by_ranks[0] = nb_cell;
  _writeDataSet1DCollective<Int64>(top_group, "NumberOfCells", nb_cell_by_ranks);

  UniqueArray<Int64> nb_node_by_ranks(1);
  nb_node_by_ranks[0] = nb_node;
  _writeDataSet1DCollective<Int64>(top_group, "NumberOfPoints", nb_node_by_ranks);

  UniqueArray<Int64> number_of_connectivity_ids(1);
  number_of_connectivity_ids[0] = cells_connectivity.size();
  _writeDataSet1DCollective<Int64>(top_group, "NumberOfConnectivityIds", number_of_connectivity_ids);

  VariableNodeReal3& nodes_coordinates(m_mesh->nodesCoordinates());
  UniqueArray2<Real> points;
  points.resize(nb_node, 3);
  ENUMERATE_NODE (inode, all_nodes) {
    Int32 index = inode.index();
    Real3 pos = nodes_coordinates[inode];
    points[index][0] = pos.x;
    points[index][1] = pos.y;
    points[index][2] = pos.z;
  }
  {
    Int64 offset = _writeDataSet2DCollective<Real>(top_group, "Points", points);
    if (m_is_writer)
      _writeDataSet1D<Int64>(m_steps_group, "PointOffsets", Span<const Int64>(&offset, 1));
  }

  _writeDataSet1DCollective<unsigned char>(m_cell_data_group, "vtkGhostType", cells_ghost_type);

  if (m_is_writer) {
    {
      // Liste des temps.
      Real current_time = m_times[time_index - 1];
      _writeDataSet1D<Real>(m_steps_group, "Values", Span<const Real>(&current_time, 1));
    }

    // Nombre de temps
    _addInt64ttribute(m_steps_group, "NSteps", time_index);

    {
      // Offset de la partie.
      Int64 part_offset = (time_index - 1) * pm->commSize();
      _writeDataSet1D<Int64>(m_steps_group, "PartOffsets", Span<const Int64>(&part_offset, 1));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  template <typename DataType> class HDFTraits;

  template <> class HDFTraits<Int64>
  {
   public:

    static hid_t hdfType() { return H5T_NATIVE_INT64; }
  };

  template <> class HDFTraits<Int32>
  {
   public:

    static hid_t hdfType() { return H5T_NATIVE_INT32; }
  };

  template <> class HDFTraits<double>
  {
   public:

    static hid_t hdfType() { return H5T_NATIVE_DOUBLE; }
  };

  template <> class HDFTraits<unsigned char>
  {
   public:

    static hid_t hdfType() { return H5T_NATIVE_UINT8; }
  };

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecrit une donnée 1D ou 2D.
 *
 * Pour chaque temps ajouté, la donnée est écrite à la fin des valeurs précédentes.
 *
 * Retourne l'offset d'écriture de la première dimension.
 * Cela est nécessaire pour le format VTK pour indiquer où commence les
 * valeurs du temps courant.
 */
template <typename DataType> Int64 VtkHdfV2DataWriter::
_writeDataSetGeneric(HGroup& group, const String& name, Int32 nb_dim,
                     Int64 dim1_size, Int64 dim2_size, const DataType* values_data,
                     bool is_collective)
{
  static constexpr int MAX_DIM = 2;
  HDataset dataset;
  // En cas d'opération collective, local_dims et global_dims sont
  // différents sur la première dimension. La deuxième dimension est toujours
  // la même.

  // Dimensions du dataset que le rang courant va écrire.
  hsize_t local_dims[MAX_DIM];
  local_dims[0] = dim1_size;
  local_dims[1] = dim2_size;

  // Dimensions cumulées de tous les rangs pour l'écriture.
  hsize_t global_dims[MAX_DIM];

  // Dimensions maximales du DataSet
  // Pour la deuxième dimension, on suppose qu'elle est constante au cours du temps.
  hsize_t max_dims[MAX_DIM];
  max_dims[0] = H5S_UNLIMITED;
  max_dims[1] = dim2_size;

  const hid_t hdf_type = HDFTraits<DataType>::hdfType();
  herr_t herror = 0;
  Int64 write_offset = 0;

  Int64 my_index = 0;
  Int64 global_dim1_size = dim1_size;
  Int32 nb_participating_rank = 1;
  if (is_collective) {
    // En mode collectif il faut récupérer les index de chaque rang.
    // TODO: pour les variables, ces indices ne dépendent que du groupe associé
    // et on peut donc conserver l'information pour éviter le gather à chaque fois.
    IParallelMng* pm = m_mesh->parallelMng();
    nb_participating_rank = pm->commSize();
    Int32 my_rank = pm->commRank();

    UniqueArray<Int64> all_sizes(nb_participating_rank);
    pm->allGather(ConstArrayView<Int64>(1, &dim1_size), all_sizes);

    global_dim1_size = 0;
    for (Integer i = 0; i < nb_participating_rank; ++i)
      global_dim1_size += all_sizes[i];
    my_index = 0;
    for (Integer i = 0; i < my_rank; ++i)
      my_index += all_sizes[i];
  }

  HProperty write_plist_id;
  if (is_collective)
    write_plist_id.createDatasetTransfertCollectiveMPIIO();

  HSpace memory_space;
  memory_space.createSimple(nb_dim, local_dims);

  HSpace file_space;

  if (m_is_first_call) {
    // TODO: regarder comment mieux calculer le chunk
    hsize_t chunk_dims[MAX_DIM];
    global_dims[0] = global_dim1_size;
    global_dims[1] = dim2_size;
    // Il est important que tout le monde ait la même taille de chunk.
    Int64 chunk_size = global_dim1_size / nb_participating_rank;
    if (chunk_size < 1024)
      chunk_size = 1024;
    chunk_dims[0] = chunk_size;
    chunk_dims[1] = dim2_size;
    info(4) << "CHUNK nb_dim=" << nb_dim
            << " chunk0=" << chunk_dims[0]
            << " name=" << name
            << " nb_byte=" << global_dim1_size * sizeof(DataType);
    file_space.createSimple(nb_dim, global_dims, max_dims);
    HProperty plist_id;
    plist_id.create(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id.id(), nb_dim, chunk_dims);

    dataset.create(group, name.localstr(), hdf_type, file_space, HProperty{}, plist_id, HProperty{});

    if (is_collective) {
      hsize_t offset[MAX_DIM];
      offset[0] = my_index;
      offset[1] = 0;
      if ((herror = H5Sselect_hyperslab(file_space.id(), H5S_SELECT_SET, offset, nullptr, local_dims, nullptr)) < 0)
        ARCANE_THROW(IOException, "Can not select hyperslab '{0}' (err={1})", name, herror);
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

    hsize_t original_dims[MAX_DIM];
    file_space.getDimensions(original_dims, nullptr);
    hsize_t offset0 = original_dims[0];
    global_dims[0] = offset0 + global_dim1_size;
    global_dims[1] = dim2_size;
    write_offset = offset0;
    // Agrandit le dataset.
    // ATTENTION cela invalide file_space. Il faut donc le relire juste après.
    if ((herror = dataset.setExtent(global_dims)) < 0)
      ARCANE_THROW(IOException, "Can not extent dataset '{0}' (err={1})", name, herror);
    file_space = dataset.getSpace();

    hsize_t offsets[MAX_DIM];
    offsets[0] = offset0 + my_index;
    offsets[1] = 0;
    info(4) << "APPEND nb_dim=" << nb_dim
            << " dim0=" << global_dims[0]
            << " count0=" << local_dims[0]
            << " offsets0=" << offsets[0] << " name=" << name;

    if ((herror = H5Sselect_hyperslab(file_space.id(), H5S_SELECT_SET, offsets, nullptr, local_dims, nullptr)) < 0)
      ARCANE_THROW(IOException, "Can not select hyperslab '{0}' (err={1})", name, herror);
  }

  // Effectue l'écriture
  if ((herror = dataset.write(hdf_type, values_data, memory_space, file_space, write_plist_id)) < 0)
    ARCANE_THROW(IOException, "Can not write dataset '{0}' (err={1})", name, herror);

  if (dataset.isBad())
    ARCANE_THROW(IOException, "Can not write dataset '{0}'", name);

  return write_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> Int64 VtkHdfV2DataWriter::
_writeDataSet1D(HGroup& group, const String& name, Span<const DataType> values)
{
  return _writeDataSetGeneric(group, name, 1, values.size(), 1, values.data(), false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> Int64 VtkHdfV2DataWriter::
_writeDataSet1DUsingCollectiveIO(HGroup& group, const String& name, Span<const DataType> values)
{
  return _writeDataSetGeneric(group, name, 1, values.size(), 1, values.data(), true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> Int64 VtkHdfV2DataWriter::
_writeDataSet1DCollective(HGroup& group, const String& name, Span<const DataType> values)
{
  if (!m_is_parallel)
    return _writeDataSet1D(group, name, values);
  if (m_is_collective_io)
    return _writeDataSet1DUsingCollectiveIO(group, name, values);
  UniqueArray<DataType> all_values;
  IParallelMng* pm = m_mesh->parallelMng();
  pm->gatherVariable(values.smallView(), all_values, pm->masterIORank());
  if (m_is_master_io)
    return _writeDataSet1D<DataType>(group, name, all_values);
  return (-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> Int64 VtkHdfV2DataWriter::
_writeDataSet2D(HGroup& group, const String& name, Span2<const DataType> values)
{
  return _writeDataSetGeneric(group, name, 2, values.dim1Size(), values.dim2Size(), values.data(), false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> Int64 VtkHdfV2DataWriter::
_writeDataSet2DUsingCollectiveIO(HGroup& group, const String& name, Span2<const DataType> values)
{
  return _writeDataSetGeneric(group, name, 2, values.dim1Size(), values.dim2Size(), values.data(), true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> Int64 VtkHdfV2DataWriter::
_writeDataSet2DCollective(HGroup& group, const String& name, Span2<const DataType> values)
{
  if (!m_is_parallel)
    return _writeDataSet2D(group, name, values);
  if (m_is_collective_io)
    return _writeDataSet2DUsingCollectiveIO(group, name, values);

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
    return _writeDataSet2D<DataType>(group, name, span2);
  }
  return (-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfV2DataWriter::
_addInt64ArrayAttribute(Hid& hid, const char* name, Span<const Int64> values)
{
  hsize_t len = values.size();
  hid_t aid = H5Screate_simple(1, &len, 0);
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
_addInt64ttribute(Hid& hid, const char* name, Int64 value)
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
  m_cell_data_group.close();
  m_node_data_group.close();
  m_point_data_offsets_group.close();
  m_cell_data_offsets_group.close();
  m_field_data_offsets_group.close();
  m_steps_group.close();
  m_file_id.close();
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
  return;
  info(4) << "Write VtkHdfV2 var=" << var->name();

  eItemKind item_kind = var->itemKind();

  if (var->dimension() != 1)
    ARCANE_FATAL("Only export of scalar item variable is implemented (name={0})", var->name());

  HGroup* group = nullptr;
  switch (item_kind) {
  case IK_Cell:
    group = &m_cell_data_group;
    break;
  case IK_Node:
    group = &m_node_data_group;
    break;
  default:
    ARCANE_FATAL("Only export of 'Cell' or 'Node' variable is implemented (name={0})", var->name());
  }
  ARCANE_CHECK_POINTER(group);

  eDataType data_type = var->dataType();
  switch (data_type) {
  case DT_Real:
    _writeBasicTypeDataset<Real>(*group, var, data);
    break;
  case DT_Int64:
    _writeBasicTypeDataset<Int64>(*group, var, data);
    break;
  case DT_Int32:
    _writeBasicTypeDataset<Int32>(*group, var, data);
    break;
  case DT_Real3:
    _writeReal3Dataset(*group, var, data);
    break;
  case DT_Real2:
    _writeReal2Dataset(*group, var, data);
    break;
  default:
    warning() << String::format("Export for datatype '{0}' is not supported (var_name={1})", data_type, var->name());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> Int64 VtkHdfV2DataWriter::
_writeBasicTypeDataset(HGroup& group, IVariable* var, IData* data)
{
  auto* true_data = dynamic_cast<IArrayDataT<DataType>*>(data);
  ARCANE_CHECK_POINTER(true_data);
  return _writeDataSet1DCollective(group, var->name(), Span<const DataType>(true_data->view()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 VtkHdfV2DataWriter::
_writeReal3Dataset(HGroup& group, IVariable* var, IData* data)
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
  return _writeDataSet2DCollective<Real>(group, var->name(), scalar_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 VtkHdfV2DataWriter::
_writeReal2Dataset(HGroup& group, IVariable* var, IData* data)
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
  return _writeDataSet2DCollective<Real>(group, var->name(), scalar_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Post-traitement au format Ensight Hdf.
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
    auto w = std::make_unique<VtkHdfV2DataWriter>(mesh(), groups());
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

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VtkHdfV2PostProcessor,
                                   IPostProcessorWriter,
                                   VtkHdfV2PostProcessor);

ARCANE_REGISTER_SERVICE_VTKHDFV2POSTPROCESSOR(VtkHdfV2PostProcessor,
                                              VtkHdfV2PostProcessor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
