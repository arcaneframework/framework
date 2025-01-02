// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* KdiPostProcessor.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Pos-traitement avec l'outil KDI.                                          */
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

#include "arcane/std/KdiPostProcessor_axl.h"

#include "arcane/core/internal/VtkCellTypes.h"
#include "arcane/std/internal/Kdi.h"

// Timers. Pas actif pour l'instant
#define tic(a)
#define tac(a)

// Pour pouvoir fonctionner il faut que KDI soit installé et les variables
// suivantes soient positionnées
// export PYTHONPATH=/path/to/kdi/pykdi/src
// export KDI_DICTIONARY_PATH=/path/to/kdi/pykdi/src/pykdi/dictionary

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class KdiDataWriter
: public TraceAccessor
, public IDataWriter
{
 public:

  KdiDataWriter(IMesh* mesh, ItemGroupCollection groups);

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

  bool m_is_master_io = false;

  //! Nom du fichier HDF courant
  std::string m_kdi_full_filename;
  //! Identifiant KDI de la sortie
  KDIBase* m_kdi_base = nullptr;
  //! Identifiant Chunk de la sortie (vtps, ipart) et (vtps, partless)
  KDIChunk* m_kdi_chunk = nullptr;
  KDIChunk* m_kdi_chunk_partless = nullptr;

 private:

  template <typename DataType> PyArrayObject*
  _numpyDataSet1D(Span<const DataType> values);
  template <typename DataType> PyArrayObject*
  _numpyDataSet1D(IData* data);
  template <typename DataType> PyArrayObject*
  _numpyDataSet2D(Span2<const DataType> values);

  PyArrayObject* _numpyDataSetReal3D(IData* data);
  PyArrayObject* _numpyDataSetReal2D(IData* data);

  String _getFileNameForTimeIndex(Int32 index)
  {
    StringBuilder sb(m_mesh->name());
    if (index >= 0) {
      sb += "_";
      sb += index;
    }
    sb += ".hdf";
    return sb.toString();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

KdiDataWriter::
KdiDataWriter(IMesh* mesh, ItemGroupCollection groups)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_groups(groups)
{
  static bool python_initialize = false;
  if (!python_initialize) {
    // https://stackoverflow.com/questions/52828873/how-does-import-array-in-numpy-c-api-work
    // auto start = chrono::steady_clock::now();
    // OLD clock_t begin = clock();
    tic("initialize");
    Py_Initialize();
    import_array1();
    // auto end = chrono::steady_clock::now();
    // cout << "Elapsed time in milliseconds: "
    //     << chrono::duration_cast<chrono::milliseconds>(end - start).count()
    //     << " ms" << endl;
    // OLD clock_t end = clock();
    // OLD calcule le temps écoulé en trouvant la différence (end - begin) et
    // OLD divisant la différence par CLOCKS_PER_SEC pour convertir en secondes
    // OLD double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    // OLD printf("The elapsed time is %f seconds", time_spent);
    tac("initialize");

    python_initialize = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KdiDataWriter::
beginWrite(const VariableCollection& vars)
{
  tic("beginWrite");

  ARCANE_UNUSED(vars);

  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 nb_rank = pm->commSize();

  m_is_master_io = true;

  Int32 time_index = m_times.size();
  const bool is_first_call = (time_index < 2);
  if (is_first_call)
    pwarning() << "L'implémentation au format 'Kdi' est expérimentale";

  String filename = _getFileNameForTimeIndex(time_index);

  Directory dir(m_directory_name);

  m_full_filename = dir.file(filename);
  info(4) << "KdiDataWriter::beginWrite() file=" << m_full_filename;

  if (time_index <= 1) {
    if (m_is_master_io) {
      dir.createDirectory();
    }
  }

  info() << "KDI begin";
  info() << "KDI createBase full_filename=" << m_full_filename;
  std::string full_filename = m_full_filename.localstr();
  m_kdi_full_filename = full_filename.substr(0, full_filename.find_last_of("/\\") + 1) + "Mesh_kdi";
  info() << "KDI createBase kdi_full_filename=" << m_kdi_full_filename;
  if (!m_kdi_base) {
    info() << "KDI createBase m_kdi_base=nullptr";
    //indique si un fichier est lisible (et donc si il existe)
    std::string filename = m_kdi_full_filename + ".json";
    ifstream fichier(filename.c_str());
    if (!fichier.fail()) {
      info() << "KDI createBase " << m_kdi_full_filename << " exist";
      tic("loadVTKHDF");
      m_kdi_base = loadVTKHDF(m_kdi_full_filename);
      tac("loadVTKHDF");
      info() << "KDI createBase in file";
    }
    else {
      if (nb_rank != 1) {
        ARCANE_FATAL("Today, not support parallel execution!");
      }
      info() << "KDI createBase " << m_kdi_full_filename << " not exist";
      tic("createBase");
      m_kdi_base = createBase(nb_rank, true); // true : active les traces
      tac("createBase");
      info() << "KDI createBase in memory";
    }
  }
  else {
    info() << "KDI already createBase";
  }
  info() << "KDI update";
  m_kdi_base->update("UnstructuredGrid", "/mymesh");
  info() << "KDI chunk";
  double float_step = m_times[time_index - 1];
  Int32 my_rank = pm->commRank();
  info() << "KDI chunk " << float_step << " " << my_rank;
  m_kdi_chunk = m_kdi_base->chunk(float_step, my_rank);
  m_kdi_chunk_partless = m_kdi_base->chunk(float_step);

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

  // Pour les connectivités, la taille du tableau est égal
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

      cells_uid[index] = icell->uniqueId();

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
    info() << "KDI chunk set connectivity begin";
    info() << "KDI DTYPE C++ connectivity";
    PyArrayObject* parray = _numpyDataSet1D<Int64>(cells_connectivity);
    m_kdi_chunk->set("/mymesh/cells/connectivity", parray);
    info() << "KDI chunk set connectivity end";
  }
  {
    info() << "KDI chunk set cells types begin";
    // PyArrayObject* parray = _numpyDataSet1D<unsigned char>(cells_type);
    info() << "KDI DTYPE C++ cells types";
    PyArrayObject* parray = _numpyDataSet1D<unsigned char>(cells_type);
    m_kdi_chunk->set("/mymesh/cells/types", parray);
    info() << "KDI chunk set cells types end";
  }

  // Sauve les uniqueIds, les types et les coordonnées des noeuds.
  {
    UniqueArray<Int64> nodes_uid(nb_node);
    UniqueArray<unsigned char> nodes_ghost_type(nb_node);
    VariableNodeReal3& nodes_coordinates(m_mesh->nodesCoordinates());
    UniqueArray2<Real> points;
    points.resize(nb_node, 3);
    ENUMERATE_NODE (inode, all_nodes) {
      Int32 index = inode.index();
      Node node = *inode;

      nodes_uid[index] = inode->uniqueId();

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

    {
      info() << "KDI chunk set connectivity begin";
      PyArrayObject* parray = _numpyDataSet2D<Real>(points);
      m_kdi_chunk->set("/mymesh/points/cartesianCoordinates", parray);
      info() << "KDI chunk set connectivity end";
    }
    // Fields points
    {
      const std::string namefield = m_kdi_base->update_fields("/mymesh/points/fields", "GlobalNodeId");
      assert(namefield == "/mymesh/points/fields/GlobalNodeId");
      info() << "KDI chunk set GlobalNodeId begin";
      info() << "KDI DTYPE C++ GlobalNodeId";
      PyArrayObject* parray = _numpyDataSet1D<Int64>(nodes_uid);
      m_kdi_chunk->set(namefield, parray);
      info() << "KDI chunk set GlobalNodeId end";
    }
    {
      const std::string namefield = m_kdi_base->update_fields("/mymesh/points/fields", "vtkGhostType");
      assert(namefield == "/mymesh/points/fields/vtkGhostType");
      info() << "KDI chunk set node vtkGhostType begin";
      info() << "KDI DTYPE C++ node vtkGhostType";
      PyArrayObject* parray = _numpyDataSet1D<unsigned char>(nodes_ghost_type);
      m_kdi_chunk->set(namefield, parray);
      info() << "KDI chunk set node vtkGhostType end";
    }
  }

  // Fields cells
  {
    const std::string namefield = m_kdi_base->update_fields("/mymesh/cells/fields", "GlobalCellId");
    assert(namefield == "/mymesh/cells/fields/GlobalCellId");
    info() << "KDI chunk set GlobalCellId begin";
    info() << "KDI DTYPE C++ cell GlobalCellId";
    PyArrayObject* parray = _numpyDataSet1D<Int64>(cells_uid);
    m_kdi_chunk->set(namefield, parray);
    info() << "KDI chunk set GlobalCellId end";
  }
  {
    const std::string namefield = m_kdi_base->update_fields("/mymesh/cells/fields", "vtkGhostType");
    assert(namefield == "/mymesh/cells/fields/vtkGhostType");
    info() << "KDI chunk set cell vtkGhostType begin";
    info() << "KDI DTYPE C++ cell vtkGhostType";
    PyArrayObject* parray = _numpyDataSet1D<unsigned char>(cells_ghost_type);
    m_kdi_chunk->set(namefield, parray);
    info() << "KDI chunk set cell vtkGhostType end";
  }
  tac("beginWrite");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// https://omz-software.com/pythonista/numpy/reference/c-api.dtype.html
// https://numpy.org/doc/stable/reference/arrays.dtypes.html
namespace
{
  template <typename DataType> class KDITraits;

  template <> class KDITraits<Int64>
  {
   public:

    static int kdiType()
    {
      std::cout << "KDI DTYPE C++ PyArray_LONG" << std::endl;
      return NPY_INT64;
    }
  };

  template <> class KDITraits<Int32>
  {
   public:

    static int kdiType()
    {
      std::cout << "KDI DTYPE C++ PyArray_INT" << std::endl;
      return NPY_INT32;
    }
  };

  template <> class KDITraits<double>
  {
   public:

    static int kdiType()
    {
      std::cout << "KDI DTYPE C++ PyArray_DOUBLE" << std::endl;
      return NPY_FLOAT64;
    }
  };

  template <> class KDITraits<unsigned char>
  {
   public:

    static int kdiType()
    {
      std::cout << "KDI DTYPE C++ PyArray_UINT8" << std::endl;
      return NPY_UINT8;
    }
  };

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// https://github.com/BorgwardtLab/GraphKernels/blob/master/src/GKextCPy/eigen.i
template <typename DataType> PyArrayObject* KdiDataWriter::
_numpyDataSet1D(Span<const DataType> values)
{
  info() << "KDI _numpyDataSet1D begin";
  bool trace{ true };
  npy_intp dims[]{ values.size() };
  const int py_type = KDITraits<DataType>::kdiType();
  info() << "KDI DTYPE C++ py_type " << py_type;
  PyArrayObject* vec_array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, py_type));
  DataType* vec_array_pointer = static_cast<DataType*>(PyArray_DATA(vec_array));
  KTRACE(trace, vec_array_pointer);
  std::copy(values.data(), values.data() + values.size(), vec_array_pointer);
  info() << "KDI _numpyDataSet1D end";
  return vec_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// https://github.com/BorgwardtLab/GraphKernels/blob/master/src/GKextCPy/eigen.i
template <typename DataType> PyArrayObject* KdiDataWriter::
_numpyDataSet2D(Span2<const DataType> values)
{
  info() << "KDI _numpyDataSet2D begin";
  bool trace{ true };
  npy_intp dims[2] = { values.dim1Size(), values.dim2Size() };
  const int py_type = KDITraits<DataType>::kdiType();
  PyArrayObject* vec_array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, dims, py_type));
  DataType* vec_array_pointer = static_cast<DataType*>(PyArray_DATA(vec_array));
  KTRACE(trace, vec_array_pointer);
  std::copy(values.data(), values.data() + values.dim1Size() * values.dim2Size(), vec_array_pointer);
  info() << "KDI _numpyDataSet2D end";
  return vec_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KdiDataWriter::
endWrite()
{
  tic("endWrite");

  tic("saveVTKHDF");
  m_kdi_chunk_partless->saveVTKHDF(m_kdi_full_filename);
  tac("saveVTKHDF");
  // m_kdi_base = loadVTKHDF(m_kdi_full_filename);
  // N'est pas utile puisqu'on detruit cet objet apres chaque ecriture

  // Ecrit le fichier contenant les temps (à partir de la version 5.5 de paraview)
  // https://www.paraview.org/Wiki/ParaView_Release_Notes#JSON_based_new_meta_file_format_for_series_added
  //
  // Exemple:
  // {
  //   "file-series-version" : "1.0",
  //   "files" : [
  //     { "name" : "foo1.vtk", "time" : 0 },
  //     { "name" : "foo2.vtk", "time" : 5.5 },
  //     { "name" : "foo3.vtk", "time" : 11.2 }
  //   ]
  // }

  if (!m_is_master_io)
    return;

  tac("endWrite");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KdiDataWriter::
setMetaData(const String& meta_data)
{
  ARCANE_UNUSED(meta_data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void KdiDataWriter::
write(IVariable* var, IData* data)
{
  tic("write");

  info(4) << "Write Kdi var=" << var->name();

  eItemKind item_kind = var->itemKind();

  if (var->dimension() != 1)
    ARCANE_FATAL("Only export of scalar item variable is implemented (name={0})", var->name());

  {
    info() << "KDI chunk set " << var->name() << " begin";
    std::string namefield;
    switch (item_kind) {
    case IK_Cell:
      namefield = m_kdi_base->update_fields("/mymesh/cells/fields", var->name().localstr());
      break;
    case IK_Node:
      namefield = m_kdi_base->update_fields("/mymesh/points/fields", var->name().localstr());
      break;
    default:
      ARCANE_FATAL("Only export of 'Cell' or 'Node' variable is implemented (name={0})", var->name());
    }

    info() << "KDI DTYPE C++ var->name() " << var->name();

    eDataType data_type = var->dataType();
    PyArrayObject* parray = nullptr;
    switch (data_type) {
    case DT_Real:
      parray = _numpyDataSet1D<Real>(data);
      break;
    case DT_Int64:
      parray = _numpyDataSet1D<Int64>(data);
      break;
    case DT_Int32:
      parray = _numpyDataSet1D<Int32>(data);
      break;
    case DT_Real3:
      parray = _numpyDataSetReal3D(data);
      break;
    case DT_Real2:
      parray = _numpyDataSetReal2D(data);
      break;
    default:
      warning() << String::format("Export for datatype '{0}' is not supported (var_name={1})", data_type, var->name());
    }

    m_kdi_chunk->set(namefield, parray);
    info() << "KDI chunk set " << var->name() << " end";
  }

  tac("write");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// https://github.com/BorgwardtLab/GraphKernels/blob/master/src/GKextCPy/eigen.i
template <typename DataType> PyArrayObject* KdiDataWriter::
_numpyDataSet1D(IData* data)
{
  auto* true_data = dynamic_cast<IArrayDataT<DataType>*>(data);
  ARCANE_CHECK_POINTER(true_data);
  info() << "KDI _numpyDataSet1D var/data begin";
  PyArrayObject* vec_array = _numpyDataSet1D(Span<const DataType>(true_data->view()));
  info() << "KDI _numpyDataSet1D var/data end";
  return vec_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// https://github.com/BorgwardtLab/GraphKernels/blob/master/src/GKextCPy/eigen.i
PyArrayObject* KdiDataWriter::
_numpyDataSetReal3D(IData* data)
{
  info() << "KDI _numpyDataSetReal3D var/data begin";
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

  PyArrayObject* vec_array = _numpyDataSet2D<Real>(scalar_values);
  info() << "KDI _numpyDataSetReal3D var/data end";
  return vec_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// https://github.com/BorgwardtLab/GraphKernels/blob/master/src/GKextCPy/eigen.i
PyArrayObject* KdiDataWriter::
_numpyDataSetReal2D(IData* data)
{
  info() << "KDI _numpyDataSetReal3D var/data begin";
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

  PyArrayObject* vec_array = _numpyDataSet2D<Real>(scalar_values);
  info() << "KDI _numpyDataSetReal3D var/data end";
  return vec_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Post-traitement utilisant Kdi.
 */
class KdiPostProcessor
: public ArcaneKdiPostProcessorObject
{
 public:

  explicit KdiPostProcessor(const ServiceBuildInfo& sbi)
  : ArcaneKdiPostProcessorObject(sbi)
  {
  }

  IDataWriter* dataWriter() override { return m_writer.get(); }
  void notifyBeginWrite() override
  {
    auto w = std::make_unique<KdiDataWriter>(mesh(), groups());
    w->setTimes(times());
    Directory dir(baseDirectoryName());
    w->setDirectoryName(dir.file("kdi"));
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

ARCANE_REGISTER_SERVICE_KDIPOSTPROCESSOR(KdiPostProcessor,
                                         KdiPostProcessor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
