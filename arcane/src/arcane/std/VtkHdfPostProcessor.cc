// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkHdfPostProcessor.cc                                      (C) 2000-2023 */
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

#include "arcane/core/PostProcessorWriterBase.h"
#include "arcane/core/Directory.h"

#include "arcane/std/Hdf5Utils.h"
#include "arcane/std/VtkHdfPostProcessor_axl.h"

#include "arcane/FactoryService.h"
#include "arcane/IDataWriter.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IData.h"
#include "arcane/ISerializedData.h"
#include "arcane/IItemFamily.h"
#include "arcane/VariableCollection.h"

#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Hdf5Utils;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkHdfDataWriter
: public TraceAccessor
, public IDataWriter
{
 public:

  VtkHdfDataWriter(IMesh* mesh, ItemGroupCollection groups);

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
  String m_filename;

  //! Répertoire de sortie.
  String m_directory_name;

  //! Identifiant HDF du fichier
  HFile m_file_id;

  HGroup m_cell_data_group;
  HGroup m_node_data_group;

 private:

  void _addRealAttribute(Hid& hid, const char* name, double value);
  void _addRealArrayAttribute(Hid& hid, const char* name, Span<const Real> values);
  void _addIntegerAttribute(Hid& hid, const char* name, int value);
  void _addInt64ArrayAttribute(Hid& hid, const char* name, Span<const Int64> values);
  void _addStringAttribute(Hid& hid, const char* name, const String& value);

  template <typename DataType> void
  _writeDataSet1D(HGroup& group, const String& name, Span<const DataType> values);
  template <typename DataType> void
  _writeDataSet1D(HGroup& group, const String& name, Array<DataType>& values)
  {
    _writeDataSet1D(group, name, values.constSpan());
  }
  template <typename DataType> void
  _writeDataSet2D(HGroup& group, const String& name, Span2<const DataType> values);
  template <typename DataType> void
  _writeDataSet2D(HGroup& group, const String& name, Array2<DataType>& values)
  {
    _writeDataSet2D(group, name, values.constSpan());
  }
  template <typename DataType> void
  _writeBasicTypeDataset(HGroup& group, IVariable* var, IData* data);
  void _writeReal3Dataset(HGroup& group, IVariable* var, IData* data);
  void _writeReal2Dataset(HGroup& group, IVariable* var, IData* data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VtkHdfDataWriter::
VtkHdfDataWriter(IMesh* mesh, ItemGroupCollection groups)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_groups(groups)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
beginWrite(const VariableCollection& vars)
{
  ARCANE_UNUSED(vars);
  warning() << "L'implémentation du format 'VtkHdf' n'est pas encore opérationnelle";

  Int32 time_index = m_times.size();

  StringBuilder sb("vtk_hdf_");
  sb += time_index;
  sb += ".hdf";
  m_filename = sb.toString();

  String dir_name = m_directory_name;
  Directory dir(dir_name);
  String full_path = dir.file(m_filename);
  info() << "ENSIGHT HDF BEGIN WRITE file=" << full_path;

  H5open();

  m_file_id.openTruncate(full_path);
  HGroup top_group;
  top_group.create(m_file_id, "VTKHDF");

  m_cell_data_group.create(top_group, "CellData");
  m_node_data_group.create(top_group, "PointData");

  std::array<Int64, 2> version = { 1, 0 };
  _addInt64ArrayAttribute(top_group, "Version", version);

  _addStringAttribute(top_group, "Type", "UnstructuredGrid");

  std::array<Int64, 2> nb_rank_array = { 1, 0 };
  Span<const Int64> ranks{ nb_rank_array };

  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 nb_rank = pm->commSize();
  if (nb_rank != 1)
    ARCANE_FATAL("Only sequential output is allowed");

  CellGroup all_cells = m_mesh->allCells();
  NodeGroup all_nodes = m_mesh->allNodes();

  const Int32 nb_cell = all_cells.size();
  const Int32 nb_node = all_nodes.size();

  // Pour les connectivités, la taille du tableau est égal
  // au nombre de mailes plus 1.
  UniqueArray<Int64> cells_connectivity;
  UniqueArray<Int64> cells_offset;
  UniqueArray<unsigned char> cells_type;
  const int VTK_HEXAHEDRON = 12;
  cells_offset.add(0);
  ENUMERATE_CELL (icell, all_cells) {
    Cell cell = *icell;
    cells_type.add(VTK_HEXAHEDRON);
    for (NodeLocalId node : cell.nodeIds())
      cells_connectivity.add(node);
    cells_offset.add(cells_connectivity.size());
  }

  _writeDataSet1D(top_group, "Offsets", cells_offset);
  _writeDataSet1D(top_group, "Connectivity", cells_connectivity);
  _writeDataSet1D(top_group, "Types", cells_type);

  UniqueArray<Int64> nb_cell_by_ranks(nb_rank);
  nb_cell_by_ranks[0] = nb_cell;
  _writeDataSet1D(top_group, "NumberOfCells", nb_cell_by_ranks);

  UniqueArray<Int64> nb_node_by_ranks(nb_rank);
  nb_node_by_ranks[0] = nb_node;
  _writeDataSet1D(top_group, "NumberOfPoints", nb_node_by_ranks);

  UniqueArray<Int64> number_of_connectivity_ids(nb_rank);
  number_of_connectivity_ids[0] = cells_connectivity.size();
  _writeDataSet1D(top_group, "NumberOfConnectivityIds", number_of_connectivity_ids);

  VariableNodeReal3& nodes_coordinates(m_mesh->nodesCoordinates());
  UniqueArray2<Real> points;
  points.resize(nb_node, 3);
  ENUMERATE_NODE (inode, all_nodes) {
    //Node node = *inode;
    Int32 index = inode.index();
    Real3 pos = nodes_coordinates[inode];
    points[index][0] = pos.x;
    points[index][1] = pos.y;
    points[index][2] = pos.z;
  }
  _writeDataSet2D(top_group, "Points", points);
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

template <typename DataType> void VtkHdfDataWriter::
_writeDataSet1D(HGroup& group, const String& name, Span<const DataType> values)
{
  hsize_t dims[1];
  dims[0] = values.size();
  HSpace hspace;
  hspace.createSimple(1, dims);
  HDataset dataset;
  const hid_t hdf_type = HDFTraits<DataType>::hdfType();
  dataset.create(group, name.localstr(), hdf_type, hspace, H5P_DEFAULT);
  dataset.write(hdf_type, values.data());
}

template <typename DataType> void VtkHdfDataWriter::
_writeDataSet2D(HGroup& group, const String& name, Span2<const DataType> values)
{
  hsize_t dims[2];
  dims[0] = values.dim1Size();
  dims[1] = values.dim2Size();
  HSpace hspace;
  hspace.createSimple(2, dims);
  HDataset dataset;
  const hid_t hdf_type = HDFTraits<DataType>::hdfType();
  dataset.create(group, name.localstr(), hdf_type, hspace, H5P_DEFAULT);
  dataset.write(hdf_type, values.data());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
_addIntegerAttribute(Hid& hid, const char* name, int value)
{
  hid_t aid = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(hid.id(), name, H5T_NATIVE_INT, aid, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0)
    throw FatalErrorException(A_FUNCINFO, String("Can not create attribute ") + name);
  int ret = H5Awrite(attr, H5T_NATIVE_INT, &value);
  ret = H5Sclose(aid);
  ret = H5Aclose(attr);
  if (ret < 0)
    throw FatalErrorException(A_FUNCINFO, String("Can not write attribute ") + name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
_addRealAttribute(Hid& hid, const char* name, double value)
{
  hid_t aid = H5Screate(H5S_SCALAR);
  hid_t attr = H5Acreate2(hid.id(), name, H5T_NATIVE_FLOAT, aid, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0)
    throw FatalErrorException(String("Can not create attribute ") + name);
  int ret = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
  ret = H5Sclose(aid);
  ret = H5Aclose(attr);
  if (ret < 0)
    throw FatalErrorException(A_FUNCINFO, String("Can not write attribute ") + name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
_addInt64ArrayAttribute(Hid& hid, const char* name, Span<const Int64> values)
{
  hsize_t len = values.size();
  hid_t aid = H5Screate_simple(1, &len, 0);
  hid_t attr = H5Acreate2(hid.id(), name, H5T_NATIVE_INT64, aid, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0)
    ARCANE_FATAL("Can not create attribute '{0}'", name);
  int ret = H5Awrite(attr, H5T_NATIVE_INT64, values.data());
  ret = H5Sclose(aid);
  ret = H5Aclose(attr);
  if (ret < 0)
    ARCANE_FATAL("Can not write attribute '{0}'", name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
_addRealArrayAttribute(Hid& hid, const char* name, Span<const Real> values)
{
  hsize_t len = values.size();
  hid_t aid = H5Screate_simple(1, &len, 0);
  hid_t attr = H5Acreate2(hid.id(), name, H5T_NATIVE_FLOAT, aid, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0)
    throw FatalErrorException(String("Can not create attribute ") + name);
  int ret = H5Awrite(attr, H5T_NATIVE_DOUBLE, values.data());
  ret = H5Sclose(aid);
  ret = H5Aclose(attr);
  if (ret < 0)
    throw FatalErrorException(A_FUNCINFO, String("Can not write attribute ") + name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
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
  ret = H5Sclose(aid);
  ret = H5Aclose(attr);
  if (ret < 0)
    ARCANE_FATAL("Can not write attribute '{0}'", name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
endWrite()
{
  m_file_id.close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
setMetaData(const String& meta_data)
{
  ARCANE_UNUSED(meta_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
write(IVariable* var, IData* data)
{
  info() << "SAVE var=" << var->name();

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
    _writeBasicTypeDataset<Real>(*group, var, data);
    break;
  case DT_Int32:
    _writeBasicTypeDataset<Real>(*group, var, data);
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

template <typename DataType> void VtkHdfDataWriter::
_writeBasicTypeDataset(HGroup& group, IVariable* var, IData* data)
{
  auto* true_data = dynamic_cast<IArrayDataT<DataType>*>(data);
  ARCANE_CHECK_POINTER(true_data);
  _writeDataSet1D(group, var->name(), Span<const DataType>(true_data->view()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
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
  _writeDataSet2D(group, var->name(), scalar_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkHdfDataWriter::
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
  _writeDataSet2D(group, var->name(), scalar_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Post-traitement au format Ensight Hdf.
 */
class VtkHdfPostProcessor
: public ArcaneVtkHdfPostProcessorObject
{
 public:

  explicit VtkHdfPostProcessor(const ServiceBuildInfo& sbi)
  : ArcaneVtkHdfPostProcessorObject(sbi)
  {
  }

  IDataWriter* dataWriter() override { return m_writer.get(); }
  void notifyBeginWrite() override
  {
    auto w = std::make_unique<VtkHdfDataWriter>(mesh(), groups());
    w->setTimes(times());
    w->setDirectoryName(baseDirectoryName());
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

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VtkHdfPostProcessor,
                                   IPostProcessorWriter,
                                   VtkHdfPostProcessor);

ARCANE_REGISTER_SERVICE_VTKHDFPOSTPROCESSOR(VtkHdfPostProcessor,
                                            VtkHdfPostProcessor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
