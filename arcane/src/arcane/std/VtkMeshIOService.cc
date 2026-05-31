// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkMeshIOService.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Reading/Writing a mesh in legacy VTK format.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/BasicService.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariableAccessor.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/UnstructuredMeshAllocateBuildInfo.h"

#include "arcane/core/internal/IVariableMngInternal.h"
#include "arcane/core/internal/VtkCellTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkFile;
using namespace Arcane::VtkUtils;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mesh file reader for legacy VTK format.
 *
 * This is a preliminary version that only supports
 * DATASETs of type STRUCTURED_GRID or UNSTRUCTURED_GRID. Furthermore,
 * the reader and writer have only been partially tested.
 *
 * The VTK file header must be:
 * # vtk DataFile Version X.X
 * Where X.X is the VTK file version (support for VTK files <= 4.2).
 *
 * It is possible to specify a set of variables in the file.
 * In this case, their values are read simultaneously with the mesh
 * and are used to initialize the variables. Currently, only cell values
 * are supported.
 *
 * Since VTK does not support the concept of a group, it is possible
 * to specify a group as a variable (CELL_DATA).
 * By convention, if the variable starts with the string 'GROUP_', then
 * it is a group. The variable must be declared as follows:
 * \begincode
 * CELL_DATA %n
 * SCALARS GROUP_%m int 1
 * LOOKUP_TABLE default
 * \endcode
 * with %n being the number of cells, and %m being the group name.
 * A cell belongs to the group if the data value is
 * different from 0.
 *
 * Currently, point groups CANNOT be specified.
 *
 * To specify face groups, a VTK file is required
 * additional, identical to the original file but containing the
 * description of the faces instead of the cells. By convention, if the
 * currently read file is named 'toto.vtk', the file describing the
 * faces will be 'toto.vtkfaces.vtk'. This file is optional.
 */
class VtkMeshIOService
: public TraceAccessor
{
 public:

  explicit VtkMeshIOService(ITraceMng* tm)
  : TraceAccessor(tm)
  {}

 public:

  void build() {}

 public:

  enum eMeshType
  {
    VTK_MT_Unknown,
    VTK_MT_StructuredGrid,
    VTK_MT_UnstructuredGrid
  };

  class VtkMesh
  {
   public:
  };

  class VtkStructuredGrid
  : public VtkMesh
  {
   public:

    int m_nb_x;
    int m_nb_y;
    int m_nb_z;
  };

 public:

  bool readMesh(IPrimaryMesh* mesh, const String& file_name, const String& dir_name, bool use_internal_partition);

 private:

  bool _readStructuredGrid(IPrimaryMesh* mesh, VtkFile&, bool use_internal_partition);
  bool _readUnstructuredGrid(IPrimaryMesh* mesh, VtkFile& vtk_file, bool use_internal_partition);
  void _readCellVariable(IMesh* mesh, VtkFile& vtk_file, const String& name_str, Integer nb_cell);
  void _readItemGroup(IMesh* mesh, VtkFile& vtk_file, const String& name_str, Integer nb_item,
                      eItemKind ik, ConstArrayView<Int32> local_id);
  void _readNodeGroup(IMesh* mesh, VtkFile& vtk_file, const String& name, Integer nb_item);
  void _createFaceGroup(IMesh* mesh, const String& name, Int32ConstArrayView faces_lid);
  bool _readData(IMesh* mesh, VtkFile& vtk_file, bool use_internal_partition, eItemKind cell_kind,
                 Int32ConstArrayView local_id, Integer nb_node);
  void _readNodesUnstructuredGrid(IMesh* mesh, VtkFile& vtk_file, Array<Real3>& node_coords);
  void _readCellsUnstructuredGrid(IMesh* mesh, VtkFile& vtk_file,
                                  Array<Int32>& cells_nb_node,
                                  Array<ItemTypeId>& cells_type,
                                  Array<Int64>& cells_connectivity);
  void _readFacesMesh(IMesh* mesh, const String& file_name,
                      const String& dir_name, bool use_internal_partition);
  bool _readMetadata(IMesh* mesh, VtkFile& vtk_file);

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkFile
{
 public:

  static const int BUFSIZE = 10000;

 public:

  explicit VtkFile(std::istream* stream)
  : m_stream(stream)
  , m_is_init(false)
  , m_need_reread_current_line(false)
  , m_is_eof(false)
  , m_is_binary_file(false)
  , m_buf{}
  {}

  const char* getCurrentLine();
  bool isEmptyNextLine();
  const char* getNextLine();

  void checkString(const String& current_value, const String& expected_value);
  void checkString(const String& current_value,
                   const String& expected_value1,
                   const String& expected_value2);

  static bool isEqualString(const String& current_value, const String& expected_value);

  void reReadSameLine() { m_need_reread_current_line = true; }

  bool isEof() { return m_is_eof; }

  template <class T>
  void getBinary(T& type);
  float getFloat();
  double getDouble();
  int getInt();

  void setIsBinaryFile(bool new_val) { m_is_binary_file = new_val; }

 private:

  //! The stream.
  std::istream* m_stream = nullptr;

  //! Has at least one line been read.
  bool m_is_init;

  //! Should the same line be reread.
  bool m_need_reread_current_line;

  //! Is the end of the file reached.
  bool m_is_eof;

  //! Is this a file containing binary data.
  bool m_is_binary_file;

  //! The buffer containing the read line.
  char m_buf[BUFSIZE];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows returning the line present in the buffer.
 *
 * \return the buffer containing the last line read
 */
const char* VtkFile::
getCurrentLine()
{
  if (!m_is_init)
    getNextLine();
  return m_buf;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows checking if the next line is empty.
 *
 * At the end of this method, the buffer will contain the next non-empty line. The boolean m_need_reread_current_line will allow getNextLine
 * to return this line which has not been read.
 *
 * \return true if there is an empty line, false otherwise
 */
bool VtkFile::
isEmptyNextLine()
{
  m_is_init = true;

  // We want getNextLine to read a new line.
  // (we set it to false in case this method is
  // called multiple times in a row).
  m_need_reread_current_line = false;

  // If we reached the end of the file during the previous call to this method or
  // getNextLine, we throw an error.
  if (m_is_eof) {
    ARCANE_THROW(IOException, "Unexpected EndOfFile");
  }

  if (!m_stream->good())
    ARCANE_THROW(IOException, "Error when reading stream");

  // getline stops (by default) at the char '\n' and does not include it in the buf
  // but replaces it with '\0'.
  m_stream->getline(m_buf, sizeof(m_buf) - 1);

  // If we reach the end of the file, we return true (to indicate yes, there is an empty line,
  // for the caller to handle).
  if (m_stream->eof()) {
    m_is_eof = true;
    return true;
  }

  // On Windows, an empty line starts with \r.
  // getline replaces \n with \0, whether on Windows or Linux.
  if (m_buf[0] == '\r' || m_buf[0] == '\0') {
    getNextLine();

    // We ask that the next call to getNextLine returns the line
    // that was just buffered.
    m_need_reread_current_line = true;
    return true;
  }
  else {
    bool is_comment = true;

    // We remove the comment, if there is one, by replacing '#' with '\0'.
    for (int i = 0; i < BUFSIZE && m_buf[i] != '\0'; ++i) {
      if (!isspace(m_buf[i]) && m_buf[i] != '#' && is_comment) {
        is_comment = false;
      }
      if (m_buf[i] == '#') {
        m_buf[i] = '\0';
        break;
      }
    }

    // If it is not a comment, we just remove the final '\r' (if windows).
    if (!is_comment) {
      // Remove the final '\r'
      for (int i = 0; i < BUFSIZE && m_buf[i] != '\0'; ++i) {
        if (m_buf[i] == '\r') {
          m_buf[i] = '\0';
          break;
        }
      }
    }

    // If it was a comment, we search for the next "valid" line
    // by calling getNextLine.
    else {
      getNextLine();
    }
  }
  m_need_reread_current_line = true;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows retrieving the next line from the file.
 *
 * \return the buffer containing the last line read
 */
const char* VtkFile::
getNextLine()
{
  m_is_init = true;

  // We return the current buffer, if it has not been used.
  if (m_need_reread_current_line) {
    m_need_reread_current_line = false;
    return getCurrentLine();
  }

  // If we reached the end of the file during the previous call to this method or
  // isEmptyNextLine, we throw an error.
  if (m_is_eof) {
    ARCANE_THROW(IOException, "Unexpected EndOfFile");
  }

  while (m_stream->good()) {
    // getline stops (by default) at the char '\n' and does not include it in the buf but replaces it with '\0'.
    m_stream->getline(m_buf, sizeof(m_buf) - 1);

    // If we reach the end of the file, we return the buffer with \0 at the beginning (it is up to the caller to call
    // isEof() to know if the file is finished or not).
    if (m_stream->eof()) {
      m_is_eof = true;
      m_buf[0] = '\0';
      return m_buf;
    }

    bool is_comment = true;

    // On Windows, an empty line starts with \r.
    // getline replaces \n with \0, whether on Windows or Linux.
    if (m_buf[0] == '\0' || m_buf[0] == '\r')
      continue;

    // We remove the comment, if there is one, by replacing '#' with '\0'.
    for (int i = 0; i < BUFSIZE && m_buf[i] != '\0'; ++i) {
      if (!isspace(m_buf[i]) && m_buf[i] != '#' && is_comment) {
        is_comment = false;
      }
      if (m_buf[i] == '#') {
        m_buf[i] = '\0';
        break;
      }
    }

    // If it is not a comment, we just remove the final '\r' (if windows).
    if (!is_comment) {
      // Remove the final '\r'
      for (int i = 0; i < BUFSIZE && m_buf[i] != '\0'; ++i) {
        if (m_buf[i] == '\r') {
          m_buf[i] = '\0';
          break;
        }
      }
      return m_buf;
    }
  }
  ARCANE_THROW(IOException, "Error when reading stream");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows retrieving the following float.
 *
 * \return the retrieved float
 */
float VtkFile::
getFloat()
{
  float v = 0.;
  if (m_is_binary_file) {
    getBinary(v);
    return v;
  }
  (*m_stream) >> ws >> v;

  if (m_stream->good())
    return v;

  ARCANE_THROW(IOException, "Can not read 'Float'");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows retrieving the next double.
 *
 * \return the retrieved double
 */
double VtkFile::
getDouble()
{
  double v = 0.;
  if (m_is_binary_file) {
    getBinary(v);
    return v;
  }
  (*m_stream) >> ws >> v;

  if (m_stream->good())
    return v;

  ARCANE_THROW(IOException, "Can not read 'Double'");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows retrieving the next integer.
 *
 * \return the retrieved integer
 */
int VtkFile::
getInt()
{
  int v = 0;
  if (m_is_binary_file) {
    getBinary(v);
    return v;
  }
  (*m_stream) >> ws >> v;

  if (m_stream->good())
    return v;

  ARCANE_THROW(IOException, "Can not read 'int'");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows retrieving the next binary number.
 *
 * \param value The reference to the variable to be filled (the type of value tells us the number of bytes to read).
 */
template <class T>
void VtkFile::
getBinary(T& value)
{
  constexpr size_t sizeofT = sizeof(T);

  // The VTK file is in big endian and current CPUs are in little endian.
  Byte big_endian[sizeofT];
  Byte little_endian[sizeofT];

  // We read the next 'sizeofT' bytes and put them into big_endian.
  m_stream->read((char*)big_endian, sizeofT);

  // We transform big_endian into little_endian.
  for (size_t i = 0; i < sizeofT; i++) {
    little_endian[sizeofT - 1 - i] = big_endian[i];
  }

  // We 'cast' the byte array into type 'T'.
  T* conv = new (little_endian) T;
  value = *conv;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows checking if expected_value == current_value.
 *
 * Allows checking if expected_value matches current_value.
 * An exception is thrown otherwise.
 *
 * \param current_value the reference value
 * \param expected_value the value to compare
 */
void VtkFile::
checkString(const String& current_value, const String& expected_value)
{
  String current_value_low = current_value.lower();
  String expected_value_low = expected_value.lower();

  if (current_value_low != expected_value_low) {
    ARCANE_THROW(IOException, "Bad string. Expecting '{0}', buf found '{1}'",
                 expected_value, current_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allows checking if expected_value1 or expected_value2 == current_value.
 *
 * Allows checking if expected_value1 or expected_value2 matches current_value.
 * An exception is thrown otherwise.
 *
 * \param current_value the reference value
 * \param expected_value1 the first value to compare
 * \param expected_value2 the second value to compare
 */
void VtkFile::
checkString(const String& current_value, const String& expected_value1, const String& expected_value2)
{
  String current_value_low = current_value.lower();
  String expected_value1_low = expected_value1.lower();
  String expected_value2_low = expected_value2.lower();

  if (current_value_low != expected_value1_low && current_value_low != expected_value2_low) {
    ARCANE_THROW(IOException, "Bad string. Expecting '{0}' or '{1}, buf found '{2}'",
                 expected_value1, expected_value2, current_value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allows checking if expected_value == current_value.
 *
 * Allows checking if expected_value matches current_value.
 *
 * \param current_value the reference value
 * \param expected_value the value to compare
 * \return true if the values are equal, false otherwise
 */
bool VtkFile::
isEqualString(const String& current_value, const String& expected_value)
{
  String current_value_low = current_value.lower();
  String expected_value_low = expected_value.lower();
  return (current_value_low == expected_value_low);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows starting the reading of a vtk file.
 *
 * \param mesh The mesh to be filled
 * \param file_name The name of the vtk file (with extension)
 * \param dir_name The file path
 * \param use_internal_partition Should the internal partitioner be used or not
 * \return false if everything went well, true otherwise
 */
bool VtkMeshIOService::
readMesh(IPrimaryMesh* mesh, const String& file_name, const String& dir_name, bool use_internal_partition)
{
  std::ifstream ifile(file_name.localstr(), std::ifstream::binary);

  if (!ifile) {
    error() << "Unable to read file '" << file_name << "'";
    return true;
  }

  debug() << "Fichier ouvert : " << file_name.localstr();

  VtkFile vtk_file(&ifile);
  const char* buf = 0;

  // Reading the description
  // Reading title.
  String title = vtk_file.getNextLine();

  info() << "Titre du fichier VTK : " << title.localstr();

  // Reading format.
  String format = vtk_file.getNextLine();

  debug() << "Format du fichier VTK : " << format.localstr();

  if (VtkFile::isEqualString(format, "BINARY")) {
    vtk_file.setIsBinaryFile(true);
  }

  eMeshType mesh_type = VTK_MT_Unknown;

  // Reading the mesh type
  // TODO: in parallel, with use_internal_partition true, only processor 0
  // reads the data. In this case, it is unnecessary for others to open the file.
  {
    buf = vtk_file.getNextLine();

    std::istringstream mesh_type_line(buf);
    std::string dataset_str;
    std::string mesh_type_str;

    mesh_type_line >> ws >> dataset_str >> ws >> mesh_type_str;

    vtk_file.checkString(dataset_str, "DATASET");

    if (VtkFile::isEqualString(mesh_type_str, "STRUCTURED_GRID")) {
      mesh_type = VTK_MT_StructuredGrid;
    }

    if (VtkFile::isEqualString(mesh_type_str, "UNSTRUCTURED_GRID")) {
      mesh_type = VTK_MT_UnstructuredGrid;
    }

    if (mesh_type == VTK_MT_Unknown) {
      error() << "Support exists only for 'STRUCTURED_GRID' and 'UNSTRUCTURED_GRID' formats (format=" << mesh_type_str << "')";
      return true;
    }
  }
  debug() << "Lecture en-tête OK";

  bool ret = true;
  switch (mesh_type) {
  case VTK_MT_StructuredGrid:
    ret = _readStructuredGrid(mesh, vtk_file, use_internal_partition);
    break;

  case VTK_MT_UnstructuredGrid:
    ret = _readUnstructuredGrid(mesh, vtk_file, use_internal_partition);
    debug() << "Lecture _readUnstructuredGrid OK";
    if (!ret) {
      // Tries to read the faces file if it exists
      _readFacesMesh(mesh, file_name + "faces.vtk", dir_name, use_internal_partition);
      debug() << "Lecture _readFacesMesh OK";
    }
    break;

  case VTK_MT_Unknown:
    break;
  }
  /*while ( (buf=vtk_file.getNextLine()) != 0 ){
    info() << " STR " << buf;
    }*/

  ifile.close();
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows reading a vtk file containing a STRUCTURED_GRID.
 *
 * \param mesh The mesh to be filled
 * \param vtk_file Reference to a VtkFile object
 * \param use_internal_partition Should the internal partitioner be used or not
 * \return false if everything went well, true otherwise
 */
bool VtkMeshIOService::
_readStructuredGrid(IPrimaryMesh* mesh, VtkFile& vtk_file, bool use_internal_partition)
{
  // Reading the number of points: DIMENSIONS nx ny nz
  const char* buf = nullptr;
  Integer nb_node_x = 0;
  Integer nb_node_y = 0;
  Integer nb_node_z = 0;
  {
    buf = vtk_file.getNextLine();
    std::istringstream iline(buf);
    std::string dimension_str;
    iline >> ws >> dimension_str >> ws >> nb_node_x >> ws >> nb_node_y >> ws >> nb_node_z;

    if (!iline) {
      error() << "Syntax error while reading grid dimensions";
      return true;
    }

    vtk_file.checkString(dimension_str, "DIMENSIONS");
    if (nb_node_x <= 1 || nb_node_y <= 1 || nb_node_z <= 1) {
      error() << "Invalid dimensions: x=" << nb_node_x << " y=" << nb_node_y << " z=" << nb_node_z;
      return true;
    }
  }
  info() << " Infos: " << nb_node_x << " " << nb_node_y << " " << nb_node_z;
  Integer nb_node = nb_node_x * nb_node_y * nb_node_z;

  // Reading the number of points: POINTS nb float
  std::string float_str;
  {
    buf = vtk_file.getNextLine();
    std::istringstream iline(buf);
    std::string points_str;
    Integer nb_node_read = 0;
    iline >> ws >> points_str >> ws >> nb_node_read >> ws >> float_str;
    if (!iline) {
      error() << "Syntax error while reading grid dimensions";
      return true;
    }
    vtk_file.checkString(points_str, "POINTS");
    if (nb_node_read != nb_node) {
      error() << "Number of invalid nodes: expected=" << nb_node << " found=" << nb_node_read;
      return true;
    }
  }

  Int32 rank = mesh->parallelMng()->commRank();

  Integer nb_cell_x = nb_node_x - 1;
  Integer nb_cell_y = nb_node_y - 1;
  Integer nb_cell_z = nb_node_z - 1;

  if (use_internal_partition && rank != 0) {
    nb_node_x = 0;
    nb_node_y = 0;
    nb_node_z = 0;
    nb_cell_x = 0;
    nb_cell_y = 0;
    nb_cell_z = 0;
  }

  const Integer nb_node_yz = nb_node_y * nb_node_z;
  const Integer nb_node_xy = nb_node_x * nb_node_y;

  Integer nb_cell = nb_cell_x * nb_cell_y * nb_cell_z;
  UniqueArray<Int32> cells_local_id(nb_cell);

  // Mesh creation
  {
    UniqueArray<Integer> nodes_unique_id(nb_node);

    info() << " NODE YZ = " << nb_node_yz;
    // Node creation
    //Integer nb_node_local_id = 0;
    {
      Integer node_local_id = 0;
      for (Integer x = 0; x < nb_node_x; ++x) {
        for (Integer z = 0; z < nb_node_z; ++z) {
          for (Integer y = 0; y < nb_node_y; ++y) {

            Integer node_unique_id = y + (z)*nb_node_y + x * nb_node_y * nb_node_z;

            nodes_unique_id[node_local_id] = node_unique_id;

            ++node_local_id;
          }
        }
      }
      //nb_node_local_id = node_local_id;
      //warning() << " NB NODE LOCAL ID=" << node_local_id;
    }

    // Cell creation

    // Info for cell creation
    // per cell: 1 for its unique id,
    //           1 for its type,
    //           8 for each node
    UniqueArray<Int64> cells_infos(nb_cell * 10);

    {
      Integer cell_local_id = 0;
      Integer cells_infos_index = 0;

      // Normally should not happen because the values of nb_node_x and
      // nb_node_y are tested during reading.
      if (nb_node_xy == 0)
        ARCANE_FATAL("Null value for nb_node_xy");

      //Integer index = 0;
      for (Integer z = 0; z < nb_cell_z; ++z) {
        for (Integer y = 0; y < nb_cell_y; ++y) {
          for (Integer x = 0; x < nb_cell_x; ++x) {
            Integer current_cell_nb_node = 8;

            //Integer cell_unique_id = y + (z)*nb_cell_y + x*nb_cell_y*nb_cell_z;
            Int64 cell_unique_id = x + y * nb_cell_x + z * nb_cell_x * nb_cell_y;

            cells_infos[cells_infos_index] = IT_Hexaedron8;
            ++cells_infos_index;

            cells_infos[cells_infos_index] = cell_unique_id;
            ++cells_infos_index;

            //Integer base_id = y + z*nb_node_y + x*nb_node_yz;
            Integer base_id = x + y * nb_node_x + z * nb_node_xy;
            cells_infos[cells_infos_index + 0] = nodes_unique_id[base_id];
            cells_infos[cells_infos_index + 1] = nodes_unique_id[base_id + 1];
            cells_infos[cells_infos_index + 2] = nodes_unique_id[base_id + nb_node_x + 1];
            cells_infos[cells_infos_index + 3] = nodes_unique_id[base_id + nb_node_x + 0];
            cells_infos[cells_infos_index + 4] = nodes_unique_id[base_id + nb_node_xy];
            cells_infos[cells_infos_index + 5] = nodes_unique_id[base_id + nb_node_xy + 1];
            cells_infos[cells_infos_index + 6] = nodes_unique_id[base_id + nb_node_xy + nb_node_x + 1];
            cells_infos[cells_infos_index + 7] = nodes_unique_id[base_id + nb_node_xy + nb_node_x + 0];
            cells_infos_index += current_cell_nb_node;
            cells_local_id[cell_local_id] = cell_local_id;
            ++cell_local_id;
          }
        }
      }
    }

    mesh->setDimension(3);
    mesh->allocateCells(nb_cell, cells_infos, false);
    mesh->endAllocate();

    // Positioning the coordinates
    {
      UniqueArray<Real3> coords(nb_node);

      if (vtk_file.isEqualString(float_str, "int")) {
        for (Integer z = 0; z < nb_node_z; ++z) {
          for (Integer y = 0; y < nb_node_y; ++y) {
            for (Integer x = 0; x < nb_node_x; ++x) {
              Real nx = vtk_file.getInt();
              Real ny = vtk_file.getInt();
              Real nz = vtk_file.getInt();
              Integer node_unique_id = x + y * nb_node_x + z * nb_node_xy;
              coords[node_unique_id] = Real3(nx, ny, nz);
            }
          }
        }
      }
      else if (vtk_file.isEqualString(float_str, "float")) {
        for (Integer z = 0; z < nb_node_z; ++z) {
          for (Integer y = 0; y < nb_node_y; ++y) {
            for (Integer x = 0; x < nb_node_x; ++x) {
              Real nx = vtk_file.getFloat();
              Real ny = vtk_file.getFloat();
              Real nz = vtk_file.getFloat();
              Integer node_unique_id = x + y * nb_node_x + z * nb_node_xy;
              coords[node_unique_id] = Real3(nx, ny, nz);
            }
          }
        }
      }
      else if (vtk_file.isEqualString(float_str, "double")) {
        for (Integer z = 0; z < nb_node_z; ++z) {
          for (Integer y = 0; y < nb_node_y; ++y) {
            for (Integer x = 0; x < nb_node_x; ++x) {
              Real nx = vtk_file.getDouble();
              Real ny = vtk_file.getDouble();
              Real nz = vtk_file.getDouble();
              Integer node_unique_id = x + y * nb_node_x + z * nb_node_xy;
              coords[node_unique_id] = Real3(nx, ny, nz);
            }
          }
        }
      }
      else {
        ARCANE_THROW(IOException, "Invalid type name");
      }

      VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
      ENUMERATE_NODE (inode, mesh->allNodes()) {
        Node node = *inode;
        nodes_coord_var[inode] = coords[node.uniqueId().asInt32()];
      }
    }
  }

  // Created the face groups for the sides of the cuboid
  {
    Int32UniqueArray xmin_surface_lid;
    Int32UniqueArray xmax_surface_lid;
    Int32UniqueArray ymin_surface_lid;
    Int32UniqueArray ymax_surface_lid;
    Int32UniqueArray zmin_surface_lid;
    Int32UniqueArray zmax_surface_lid;

    ENUMERATE_FACE (iface, mesh->allFaces()) {
      Face face = *iface;
      Integer face_local_id = face.localId();
      bool is_xmin = true;
      bool is_xmax = true;
      bool is_ymin = true;
      bool is_ymax = true;
      bool is_zmin = true;
      bool is_zmax = true;
      for (Node node : face.nodes()) {
        Int64 node_unique_id = node.uniqueId().asInt64();
        Int64 node_z = node_unique_id / nb_node_xy;
        Int64 node_y = (node_unique_id - node_z * nb_node_xy) / nb_node_x;
        Int64 node_x = node_unique_id - node_z * nb_node_xy - node_y * nb_node_x;
        if (node_x != 0)
          is_xmin = false;
        if (node_x != (nb_node_x - 1))
          is_xmax = false;
        if (node_y != 0)
          is_ymin = false;
        if (node_y != (nb_node_y - 1))
          is_ymax = false;
        if (node_z != 0)
          is_zmin = false;
        if (node_z != (nb_node_z - 1))
          is_zmax = false;
      }
      if (is_xmin)
        xmin_surface_lid.add(face_local_id);
      if (is_xmax)
        xmax_surface_lid.add(face_local_id);
      if (is_ymin)
        ymin_surface_lid.add(face_local_id);
      if (is_ymax)
        ymax_surface_lid.add(face_local_id);
      if (is_zmin)
        zmin_surface_lid.add(face_local_id);
      if (is_zmax)
        zmax_surface_lid.add(face_local_id);
    }
    _createFaceGroup(mesh, "XMIN", xmin_surface_lid);
    _createFaceGroup(mesh, "XMAX", xmax_surface_lid);
    _createFaceGroup(mesh, "YMIN", ymin_surface_lid);
    _createFaceGroup(mesh, "YMAX", ymax_surface_lid);
    _createFaceGroup(mesh, "ZMIN", zmin_surface_lid);
    _createFaceGroup(mesh, "ZMAX", zmax_surface_lid);
  }

  _readMetadata(mesh, vtk_file);

  // Now, check if there is data associated with the file
  bool r = _readData(mesh, vtk_file, use_internal_partition, IK_Cell, cells_local_id, nb_node);
  if (r)
    return r;

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read nodes and their coordinates.
 *
 * \param mesh The mesh to fill
 * \param vtk_file Reference to a VtkFile object
 * \param node_coords The array to fill with node coordinates
 */
void VtkMeshIOService::
_readNodesUnstructuredGrid(IMesh* mesh, VtkFile& vtk_file, Array<Real3>& node_coords)
{
  ARCANE_UNUSED(mesh);

  const char* buf = vtk_file.getNextLine();
  std::istringstream iline(buf);
  std::string points_str;
  std::string data_type_str;
  Integer nb_node = 0;

  iline >> ws >> points_str >> ws >> nb_node >> ws >> data_type_str;

  if (!iline)
    ARCANE_THROW(IOException, "Syntax error while reading number of nodes");

  vtk_file.checkString(points_str, "POINTS");

  if (nb_node < 0)
    ARCANE_THROW(IOException, "Invalid number of nodes: n={0}", nb_node);

  info() << "VTK file : number of nodes = " << nb_node;

  // Read coordinates
  node_coords.resize(nb_node);
  {
    if (vtk_file.isEqualString(data_type_str, "int")) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real nx = vtk_file.getInt();
        Real ny = vtk_file.getInt();
        Real nz = vtk_file.getInt();
        node_coords[i] = Real3(nx, ny, nz);
      }
    }
    else if (vtk_file.isEqualString(data_type_str, "float")) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real nx = vtk_file.getFloat();
        Real ny = vtk_file.getFloat();
        Real nz = vtk_file.getFloat();
        node_coords[i] = Real3(nx, ny, nz);
      }
    }
    else if (vtk_file.isEqualString(data_type_str, "double")) {
      for (Integer i = 0; i < nb_node; ++i) {
        Real nx = vtk_file.getDouble();
        Real ny = vtk_file.getDouble();
        Real nz = vtk_file.getDouble();
        node_coords[i] = Real3(nx, ny, nz);
      }
    }
    else {
      ARCANE_THROW(IOException, "Invalid type name");
    }
  }
  _readMetadata(mesh, vtk_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read meshes and their connectivity.
 *
 * Returns by filling \a cells_nb_node, \a cells_type, and \a cells_connectivity.
 *
 * \param mesh The mesh to fill
 * \param vtk_file Reference to a VtkFile object
 * \param cells_nb_node Number of nodes per cell
 * \param cells_type Type of each cell
 * \param cells_connectivity Connectivity between cells
 */
void VtkMeshIOService::
_readCellsUnstructuredGrid(IMesh* mesh, VtkFile& vtk_file,
                           Array<Int32>& cells_nb_node,
                           Array<ItemTypeId>& cells_type,
                           Array<Int64>& cells_connectivity)
{
  ARCANE_UNUSED(mesh);

  const char* buf = vtk_file.getNextLine();

  std::istringstream iline(buf);
  std::string cells_str;
  Int32 i64_nb_cell = 0;
  Int32 i64_nb_cell_node = 0;

  iline >> ws >> cells_str >> ws >> i64_nb_cell >> ws >> i64_nb_cell_node;

  if (!iline)
    ARCANE_THROW(IOException, "Syntax error while reading cells");

  vtk_file.checkString(cells_str, "CELLS");

  info() << "VTK file : nb_cell = " << i64_nb_cell << " nb_cell_node=" << i64_nb_cell_node;

  if (i64_nb_cell < 0 || i64_nb_cell_node < 0) {
    ARCANE_THROW(IOException, "Invalid dimensions: nb_cell={0} nb_cell_node={1}",
                 i64_nb_cell, i64_nb_cell_node);
  }

  Int32 nb_cell = CheckedConvert::toInt32(i64_nb_cell);
  Int32 nb_cell_node = CheckedConvert::toInt32(i64_nb_cell_node);

  cells_nb_node.resize(nb_cell);
  cells_type.resize(nb_cell);
  cells_connectivity.resize(nb_cell_node);

  {
    Integer connectivity_index = 0;
    for (Integer i = 0; i < nb_cell; ++i) {
      Integer n = vtk_file.getInt();
      cells_nb_node[i] = n;
      for (Integer j = 0; j < n; ++j) {
        Integer id = vtk_file.getInt();
        cells_connectivity[connectivity_index] = id;
        ++connectivity_index;
      }
    }
  }

  _readMetadata(mesh, vtk_file);

  // Read cell types
  {
    buf = vtk_file.getNextLine();
    std::istringstream iline(buf);
    std::string cell_types_str;
    Integer nb_cell_type;
    iline >> ws >> cell_types_str >> ws >> nb_cell_type;

    if (!iline) {
      ARCANE_THROW(IOException, "Syntax error while reading cell types");
    }

    vtk_file.checkString(cell_types_str, "CELL_TYPES");
    if (nb_cell_type != nb_cell) {
      ARCANE_THROW(IOException, "Inconsistency in number of CELL_TYPES: v={0} nb_cell={1}",
                   nb_cell_type, nb_cell);
    }
  }

  for (Integer i = 0; i < nb_cell; ++i) {
    Integer vtk_ct = vtk_file.getInt();
    Int16 it = vtkToArcaneCellType(vtk_ct, cells_nb_node[i]);
    cells_type[i] = ItemTypeId{ it };
  }
  _readMetadata(mesh, vtk_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Read metadata.
 *
 * \param mesh The mesh to fill
 * \param vtk_file Reference to a VtkFile object
 * \return false if everything went well, true otherwise
 */
bool VtkMeshIOService::
_readMetadata(IMesh* mesh, VtkFile& vtk_file)
{
  ARCANE_UNUSED(mesh);

  //const char* func_name = "VtkMeshIOService::_readMetadata()";

  if (vtk_file.isEof())
    return false;
  String meta = vtk_file.getNextLine();

  // METADATA ?
  if (!vtk_file.isEqualString(meta, "METADATA")) {
    // If there is no METADATA, we ask that the line read be reread next time.
    vtk_file.reReadSameLine();
    return false;
  }

  // As long as there is no empty line, we read.
  while (!vtk_file.isEmptyNextLine() && !vtk_file.isEof()) {
  }
  return false;

  // // If we need to do something with METADATA someday, here is untested code.
  // std::string trash;
  // Real trash_real;
  // const char* buf = vtk_file.getNextLine();

  // // INFORMATION or COMPONENT_NAMES
  // std::istringstream iline(buf);

  // String name_str;
  // String data_type_str;
  // Integer nb_info = 0;
  // Integer size_vector = 0;

  // iline >> ws >> name_str;

  // if(vtk_file.isEqualString(name_str, "INFORMATION"))
  // {
  //   iline >> ws >> nb_info;
  //   for( Integer i = 0; i < nb_info; i++)
  //   {
  //     buf = vtk_file.getNextLine();
  //     std::istringstream iline(buf);

  //     // NAME [key name] LOCATION [key location (e.g. class name)]
  //     iline >> ws >> trash >> ws >> data_type_str >> ws >> trash >> ws >> name_str;

  //     buf = vtk_file.getNextLine();
  //     std::istringstream iline(buf);

  //     if(vtk_file.isEqualString(data_type_str, "StringVector"))
  //     {
  //       iline >> ws >> trash >> ws >> size_vector;
  //       for(Integer j = 0; j < size_vector; j++)
  //       {
  //         trash = vtk_file.getNextLine();
  //       }
  //     }

  //     // else if(vtk_file.isEqualString(data_type_str, "Double")
  //     // || vtk_file.isEqualString(data_type_str, "IdType")
  //     // || vtk_file.isEqualString(data_type_str, "Integer")
  //     // || vtk_file.isEqualString(data_type_str, "UnsignedLong")
  //     // || vtk_file.isEqualString(data_type_str, "String"))
  //     // {
  //     //   iline >> ws >> trash >> ws >> trash_real;
  //     // }

  //     // else if(vtk_file.isEqualString(data_type_str, "DoubleVector")
  //     //      || vtk_file.isEqualString(data_type_str, "IntegerVector")
  //     //      || vtk_file.isEqualString(data_type_str, "StringVector"))
  //     // {
  //     //   iline >> ws >> trash >> ws >> size_vector;
  //     //   for(Integer j = 0; j < size_vector; j++)
  //     //   {
  //     //     iline >> ws >> trash_real;
  //     //   }
  //     // }
  //   }
  // }

  // else if(vtk_file.isEqualString(name_str, "COMPONENT_NAMES"))
  // {
  //   while(!vtk_file.isEmptyNextLine())
  //   {
  //     trash = vtk_file.getCurrentLine();
  //   }
  // }

  // else
  // {
  //   throw IOException(func_name,"Syntax error after METADATA tag");
  // }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows reading a vtk file containing an UNSTRUCTURED_GRID.
 *
 * \param mesh The mesh to fill
 * \param vtk_file Reference to a VtkFile object
 * \param use_internal_partition Should we use the internal partitioner or not
 * \return false if everything went well, true otherwise
 */
bool VtkMeshIOService::
_readUnstructuredGrid(IPrimaryMesh* mesh, VtkFile& vtk_file, bool use_internal_partition)
{
  Integer nb_node = 0;
  Integer nb_cell = 0;
  Integer nb_cell_node = 0;
  Int32 sid = mesh->parallelMng()->commRank();
  UniqueArray<Real3> node_coords;
  UniqueArray<Int32> cells_local_id;

  // If we use the internal partitioner, only the subdomain reads the mesh
  bool need_read = true;

  if (use_internal_partition)
    need_read = (sid == 0);

  std::array<Int64, 4> nb_cell_by_dimension = {};
  Int32 mesh_dimension = -1;
  ItemTypeMng* itm = mesh->itemTypeMng();
  UnstructuredMeshAllocateBuildInfo mesh_build_info(mesh);

  if (need_read) {
    // Read the first part of the file (after header).
    _readNodesUnstructuredGrid(mesh, vtk_file, node_coords);
    debug() << "Lecture _readNodesUnstructuredGrid OK";
    nb_node = node_coords.size();

    // Read mesh info
    // Read connectivity
    UniqueArray<Int32> cells_nb_node;
    UniqueArray<Int64> cells_connectivity;
    UniqueArray<ItemTypeId> cells_type;
    _readCellsUnstructuredGrid(mesh, vtk_file, cells_nb_node, cells_type, cells_connectivity);
    debug() << "Reading _readCellsUnstructuredGrid OK";

    nb_cell = cells_nb_node.size();
    nb_cell_node = cells_connectivity.size();
    cells_local_id.resize(nb_cell);

    // Mesh creation
    mesh_build_info.preAllocate(nb_cell, nb_cell_node);

    {
      Int32 connectivity_index = 0;
      for (Integer i = 0; i < nb_cell; ++i) {
        Int32 current_cell_nb_node = cells_nb_node[i];
        Int64 cell_unique_id = i;

        cells_local_id[i] = i;

        Int16 cell_dim = itm->typeFromId(cells_type[i])->dimension();
        if (cell_dim >= 0 && cell_dim <= 3)
          ++nb_cell_by_dimension[cell_dim];

        auto cell_nodes = cells_connectivity.subView(connectivity_index, current_cell_nb_node);
        mesh_build_info.addCell(cells_type[i], cell_unique_id, cell_nodes);
        connectivity_index += current_cell_nb_node;
      }
    }
    // Check that there are no meshes of different dimensions
    Int32 nb_different_dim = 0;
    for (Int32 i = 0; i < 4; ++i)
      if (nb_cell_by_dimension[i] != 0) {
        ++nb_different_dim;
        mesh_dimension = i;
      }
    if (nb_different_dim > 1)
      ARCANE_FATAL("The mesh contains cells of different dimension. nb0={0} nb1={1} nb2={2} nb3={3}",
                   nb_cell_by_dimension[0], nb_cell_by_dimension[1], nb_cell_by_dimension[2], nb_cell_by_dimension[3]);
  }

  // Sets the mesh dimension.
  {
    Integer wanted_dimension = mesh_dimension;
    wanted_dimension = mesh->parallelMng()->reduce(Parallel::ReduceMax, wanted_dimension);
    //mesh->setDimension(wanted_dimension);
    mesh_build_info.setMeshDimension(wanted_dimension);
  }

  mesh_build_info.allocateMesh();

  // Positions the coordinates
  {
    VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
    ENUMERATE_NODE (inode, mesh->allNodes()) {
      Node node = *inode;
      nodes_coord_var[inode] = node_coords[node.uniqueId().asInt32()];
    }
  }

  // Now, check if there is data associated with the file
  bool r = _readData(mesh, vtk_file, use_internal_partition, IK_Cell, cells_local_id, nb_node);
  debug() << "Reading _readData OK";

  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows reading the truc.vtkfaces.vtk file (if it exists).
 *
 * \param mesh The mesh to fill
 * \param file_name The vtk file name (with extension)
 * \param dir_name The file path
 * \param use_internal_partition Should the internal partitioner be used or not
 */
void VtkMeshIOService::
_readFacesMesh(IMesh* mesh, const String& file_name, const String& dir_name,
               bool use_internal_partition)
{
  ARCANE_UNUSED(dir_name);

  std::ifstream ifile(file_name.localstr(), std::ifstream::binary);
  if (!ifile) {
    info() << "No face descriptor file found '" << file_name << "'";
    return;
  }

  VtkFile vtk_file(&ifile);
  const char* buf = 0;

  // Reading the description
  String title = vtk_file.getNextLine();
  info() << "Reading VTK file '" << file_name << "'";
  info() << "Title of VTK file: " << title;

  String format = vtk_file.getNextLine();
  if (VtkFile::isEqualString(format, "BINARY")) {
    vtk_file.setIsBinaryFile(true);
  }

  eMeshType mesh_type = VTK_MT_Unknown;
  // Reading the mesh type
  // TODO: in parallel, with use_internal_partition true, only processor 0
  // reads the data. In this case, it is unnecessary for others to open the file.
  {
    buf = vtk_file.getNextLine();
    std::istringstream mesh_type_line(buf);
    std::string dataset_str;
    std::string mesh_type_str;
    mesh_type_line >> ws >> dataset_str >> ws >> mesh_type_str;
    vtk_file.checkString(dataset_str, "DATASET");

    if (VtkFile::isEqualString(mesh_type_str, "UNSTRUCTURED_GRID")) {
      mesh_type = VTK_MT_UnstructuredGrid;
    }

    if (mesh_type == VTK_MT_Unknown) {
      error() << "Face descriptor file type must be 'UNSTRUCTURED_GRID' (format=" << mesh_type_str << "')";
      return;
    }
  }

  {
    IParallelMng* pm = mesh->parallelMng();
    Integer nb_face = 0;
    Int32 sid = pm->commRank();

    UniqueArray<Int32> faces_local_id;

    // If we use the internal partitioner, only the subdomain reads the mesh
    bool need_read = true;
    if (use_internal_partition)
      need_read = (sid == 0);

    if (need_read) {
      {
        // Reads nodes, but does not keep their coordinates because it is
        // not necessary.
        UniqueArray<Real3> node_coords;
        _readNodesUnstructuredGrid(mesh, vtk_file, node_coords);
        //nb_node = node_coords.size();
      }

      // Reading face info
      // Reading connectivity
      UniqueArray<Integer> faces_nb_node;
      UniqueArray<Int64> faces_connectivity;
      UniqueArray<ItemTypeId> faces_type;
      _readCellsUnstructuredGrid(mesh, vtk_file, faces_nb_node, faces_type, faces_connectivity);
      nb_face = faces_nb_node.size();
      //nb_face_node = faces_connectivity.size();

      // We must retrieve the localId() of the faces from the connectivity
      faces_local_id.resize(nb_face);
      {
        IMeshUtilities* mu = mesh->utilities();
        mu->getFacesLocalIdFromConnectivity(faces_type, faces_connectivity, faces_local_id);
      }
    }

    // Now, check if there is data associated with the files
    _readData(mesh, vtk_file, use_internal_partition, IK_Face, faces_local_id, 0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows reading supplementary data (POINT_DATA / CELL_DATA).
 *
 * \param mesh The mesh to fill
 * \param file_name The vtk file name (with extension)
 * \param use_internal_partition Should the internal partitioner be used or not
 * \param cell_kind Type of mesh cells
 * \param local_id Array containing the local_id of the cells
 * \param nb_node Number of nodes
 * \return false if everything went well, true otherwise
 */
bool VtkMeshIOService::
_readData(IMesh* mesh, VtkFile& vtk_file, bool use_internal_partition,
          eItemKind cell_kind, Int32ConstArrayView local_id, Integer nb_node)
{
  // Only the master subdomain reads the values. However, the other
  // subdomains must know the list of variables and groups created.
  // If a data item is named 'GROUP_*', it is considered a
  // group

  IParallelMng* pm = mesh->parallelMng();
  IVariableMng* variable_mng = mesh->variableMng();

  Int32 sid = pm->commRank();

  // If there is no data, return immediately.
  {
    Byte has_data = 1;
    if ((sid == 0) && vtk_file.isEof())
      has_data = 0;

    ByteArrayView bb(1, &has_data);
    pm->broadcast(bb, 0);
    // No data.
    if (!has_data)
      return false;
  }

  OStringStream created_infos_str;
  created_infos_str() << "<?xml version='1.0' ?>\n";
  created_infos_str() << "<infos>";

  Integer nb_cell_kind = mesh->nbItem(cell_kind);
  const char* buf = 0;

  if (sid == 0) {
    bool reading_node = false;
    bool reading_cell = false;
    while (((buf = vtk_file.getNextLine()) != 0) && !vtk_file.isEof()) {
      debug() << "Read line";
      std::istringstream iline(buf);
      std::string data_str;
      iline >> data_str;

      // If we have a "CELL_DATA" block.
      if (VtkFile::isEqualString(data_str, "CELL_DATA")) {
        Integer nb_item = 0;
        iline >> ws >> nb_item;
        reading_node = false;
        reading_cell = true;
        if (nb_item != nb_cell_kind)
          error() << "Size expected = " << nb_cell_kind << " found = " << nb_item;
      }

      // If we have a "POINT_DATA" block.
      else if (VtkFile::isEqualString(data_str, "POINT_DATA")) {
        Integer nb_item = 0;
        iline >> ws >> nb_item;
        reading_node = true;
        reading_cell = false;
        if (nb_item != nb_node)
          error() << "Size expected = " << nb_node << " found = " << nb_item;
      }

      // If we have a "FIELD" block.
      else if (VtkFile::isEqualString(data_str, "FIELD")) {
        std::string name_str;
        int nb_fields;

        iline >> ws >> name_str >> ws >> nb_fields;

        Integer nb_item = 0;
        std::string type_str;
        std::string s_name_str;
        int nb_component = 1;
        bool is_group = false;

        for (Integer i = 0; i < nb_fields; i++) {
          buf = vtk_file.getNextLine();
          std::istringstream iline(buf);
          iline >> ws >> s_name_str >> ws >> nb_component >> ws >> nb_item >> ws >> type_str;

          if (nb_item != nb_cell_kind && reading_cell && !reading_node)
            error() << "Size expected = " << nb_cell_kind << " found = " << nb_item;

          if (nb_item != nb_node && !reading_cell && reading_node)
            error() << "Size expected = " << nb_node << " found = " << nb_item;

          String name_str = s_name_str;
          String cstr = name_str.substring(0, 6);

          if (cstr == "GROUP_") {
            is_group = true;
            String new_name = name_str.substring(6);
            debug() << "** ** ** GROUP ! name=" << new_name;
            name_str = new_name;
          }

          if (is_group) {
            if (!VtkFile::isEqualString(type_str, "int")) {
              error() << "Group type must be 'int', found=" << type_str;
              return true;
            }

            if (reading_node) {
              created_infos_str() << "<node-group name='" << name_str << "'/>";
              _readNodeGroup(mesh, vtk_file, name_str, nb_node);
            }

            if (reading_cell) {
              created_infos_str() << "<cell-group name='" << name_str << "'/>";
              _readItemGroup(mesh, vtk_file, name_str, nb_cell_kind, cell_kind, local_id);
            }
          }

          // TODO : See an example if possible.
          else {
            if (!VtkFile::isEqualString(type_str, "float") && !VtkFile::isEqualString(type_str, "double")) {
              error() << "Expecting 'float' or 'double' data type, found=" << type_str;
              return true;
            }

            if (reading_node) {
              fatal() << "Unable to read POINT_DATA: feature not implemented";
            }
            if (reading_cell) {
              created_infos_str() << "<cell-variable name='" << name_str << "'/>";

              if (cell_kind != IK_Cell)
                throw IOException("Unable to read face variables: feature not supported");

              _readCellVariable(mesh, vtk_file, name_str, nb_cell_kind);
            }
          }
        }
      }

      else {
        // Reading the values (CELL_DATA or POINT_DATA block)
        if (reading_node || reading_cell) {
          std::string type_str;
          std::string s_name_str;
          //String name_str;
          bool is_group = false;
          int nb_component = 1;

          iline >> ws >> s_name_str >> ws >> type_str >> ws >> nb_component;
          debug() << "** ** ** READNAME: name=" << s_name_str << " type=" << type_str;

          String name_str = s_name_str;
          String cstr = name_str.substring(0, 6);

          if (cstr == "GROUP_") {
            is_group = true;
            String new_name = name_str.substring(6);
            info() << "** ** ** GROUP ! name=" << new_name;
            name_str = new_name;
          }

          if (!VtkFile::isEqualString(data_str, "SCALARS")) {
            error() << "Expecting 'SCALARS' data type, found=" << data_str;
            return true;
          }

          if (is_group) {
            if (!VtkFile::isEqualString(type_str, "int")) {
              error() << "Group type must be 'int', found=" << type_str;
              return true;
            }

            // To read LOOKUP_TABLE
            buf = vtk_file.getNextLine();

            if (reading_node) {
              created_infos_str() << "<node-group name='" << name_str << "'/>";
              _readNodeGroup(mesh, vtk_file, name_str, nb_node);
            }

            if (reading_cell) {
              created_infos_str() << "<cell-group name='" << name_str << "'/>";
              _readItemGroup(mesh, vtk_file, name_str, nb_cell_kind, cell_kind, local_id);
            }
          }
          else {
            if (!VtkFile::isEqualString(type_str, "float") && !VtkFile::isEqualString(type_str, "double")) {
              error() << "Expecting 'float' or 'double' data type, found=" << type_str;
              return true;
            }

            // To read LOOKUP_TABLE
            /*buf = */ vtk_file.getNextLine();
            if (reading_node) {
              fatal() << "Unable to read POINT_DATA: feature not implemented";
            }
            if (reading_cell) {
              created_infos_str() << "<cell-variable name='" << name_str << "'/>";

              if (cell_kind != IK_Cell)
                throw IOException("Unable to read face variables: feature not supported");

              _readCellVariable(mesh, vtk_file, name_str, nb_cell_kind);
            }
          }
        }
        else {
          error() << "Expecting value CELL_DATA or POINT_DATA, found='" << data_str << "'";
          return true;
        }
      }
    }
  }
  created_infos_str() << "</infos>";
  if (use_internal_partition) {
    ByteUniqueArray bytes;
    if (sid == 0) {
      String str = created_infos_str.str();
      ByteConstArrayView bv = str.utf8();
      Integer len = bv.size();
      bytes.resize(len + 1);
      bytes.copy(bv);
    }

    pm->broadcastMemoryBuffer(bytes, 0);

    if (sid != 0) {
      String str = String::fromUtf8(bytes);
      info() << "FOUND STR=" << bytes.size() << " " << str;
      ScopedPtrT<IXmlDocumentHolder> doc(IXmlDocumentHolder::loadFromBuffer(bytes, "InternalBuffer", traceMng()));
      XmlNode doc_node = doc->documentNode();

      // Reading variables
      {
        XmlNodeList vars = doc_node.documentElement().children("cell-variable");
        for (XmlNode xnode : vars) {
          String name = xnode.attrValue("name");
          info() << "Building variable: " << name;
          VariableCellReal* var = new VariableCellReal(VariableBuildInfo(mesh, name));
          variable_mng->_internalApi()->addAutoDestroyVariable(var);
        }
      }
      // Reading cell groups
      {
        XmlNodeList vars = doc_node.documentElement().children("cell-group");
        IItemFamily* cell_family = mesh->itemFamily(cell_kind);
        for (XmlNode xnode : vars) {
          String name = xnode.attrValue("name");
          info() << "Building group: " << name;
          cell_family->createGroup(name);
        }
      }

      // Reading node groups
      {
        XmlNodeList vars = doc_node.documentElement().children("node-group");
        IItemFamily* node_family = mesh->nodeFamily();
        for (XmlNode xnode : vars) {
          String name = xnode.attrValue("name");
          info() << "Create node group: " << name;
          node_family->createGroup(name);
        }
      }
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows creating a face group named "name" composed of
 * faces having the IDs included in "faces_lid".
 *
 * \param mesh The mesh to fill
 * \param name The name of the group to create
 * \param faces_lid The IDs of the faces to include in the group
 */
void VtkMeshIOService::
_createFaceGroup(IMesh* mesh, const String& name, Int32ConstArrayView faces_lid)
{
  info() << "Building face group '" << name << "'"
         << " size=" << faces_lid.size();

  mesh->faceFamily()->createGroup(name, faces_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows creating a cell variable from the information in the vtk file.
 *
 * \param mesh The mesh to fill
 * \param vtk_file Reference to a VtkFile object
 * \param var_name The name of the variable to create
 * \param nb_cell The number of cells
 */
void VtkMeshIOService::
_readCellVariable(IMesh* mesh, VtkFile& vtk_file, const String& var_name, Integer nb_cell)
{
  //TODO Perform the correct conversion from uniqueId() to localId()
  info() << "Reading values for variable: " << var_name << " n=" << nb_cell;
  auto* var = new VariableCellReal(VariableBuildInfo(mesh, var_name));
  mesh->variableMng()->_internalApi()->addAutoDestroyVariable(var);
  RealArrayView values(var->asArray());
  for (Integer i = 0; i < nb_cell; ++i) {
    Real v = vtk_file.getDouble();
    values[i] = v;
  }
  _readMetadata(mesh, vtk_file);
  info() << "Variable build finished: " << vtk_file.isEof();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows creating an item group.
 *
 * \param mesh The mesh to fill
 * \param vtk_file Reference to a VtkFile object
 * \param name The name of the group to create
 * \param nb_item Number of items to read and include in the group
 * \param ik Type of items read
 * \param local_id Array containing the local_ids of the cells
 */
void VtkMeshIOService::
_readItemGroup(IMesh* mesh, VtkFile& vtk_file, const String& name, Integer nb_item,
               eItemKind ik, Int32ConstArrayView local_id)
{
  IItemFamily* item_family = mesh->itemFamily(ik);
  info() << "Reading group info for group: " << name;

  Int32UniqueArray ids;
  for (Integer i = 0; i < nb_item; ++i) {
    Integer v = vtk_file.getInt();
    if (v != 0)
      ids.add(local_id[i]);
  }
  info() << "Building group: " << name << " nb_element=" << ids.size();

  item_family->createGroup(name, ids);

  _readMetadata(mesh, vtk_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Allows creating a node group.
 *
 * \param mesh The mesh to fill
 * \param vtk_file Reference to a VtkFile object
 * \param name The name of the group to create
 * \param nb_item Number of items to read and include in the group
 */
void VtkMeshIOService::
_readNodeGroup(IMesh* mesh, VtkFile& vtk_file, const String& name, Integer nb_item)
{
  IItemFamily* item_family = mesh->itemFamily(IK_Node);
  info() << "Reading node group info for group: " << name;

  Int32UniqueArray ids;
  for (Integer i = 0; i < nb_item; ++i) {
    Integer v = vtk_file.getInt();
    if (v != 0)
      ids.add(i);
  }
  info() << "Creating group: " << name << " nb_element=" << ids.size();

  item_family->createGroup(name, ids);

  _readMetadata(mesh, vtk_file);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkLegacyMeshWriter
: public BasicService
, public IMeshWriter
{
 public:

  explicit VtkLegacyMeshWriter(const ServiceBuildInfo& sbi)
  : BasicService(sbi)
  {}

 public:

  void build() override {}

 public:

  bool writeMeshToFile(IMesh* mesh, const String& file_name) override;

 private:

  void _writeMeshToFile(IMesh* mesh, const String& file_name, eItemKind cell_kind);
  void _saveGroups(IItemFamily* family, std::ostream& ofile);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(VtkLegacyMeshWriter,
                        ServiceProperty("VtkLegacyMeshWriter", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshWriter));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Writing the mesh to VTK.
 *
 * To save both mesh and face information along with their corresponding groups,
 * two files are created. The first contains connectivity and cell groups, the
 * second contains the same information but for faces.
 *
 * Only connectivity and groups are saved. Variables are not saved.
 *
 * The DATASET type is always UNSTRUCTURED_GRID, even if the
 * mesh is structured.
 */
bool VtkLegacyMeshWriter::
writeMeshToFile(IMesh* mesh, const String& file_name)
{
  String fname = file_name;
  // Append the '.vtk' extension if it is not present.
  if (!fname.endsWith(".vtk"))
    fname = fname + ".vtk";
  _writeMeshToFile(mesh, fname, IK_Cell);
  _writeMeshToFile(mesh, fname + "faces.vtk", IK_Face);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Writes the mesh in VTK format.
 *
 * \a cell_kind indicates the type of entities to save as meshes.
 * This can be IK_Cell or IK_Face.
 */
void VtkLegacyMeshWriter::
_writeMeshToFile(IMesh* mesh, const String& file_name, eItemKind cell_kind)
{
  std::ofstream ofile(file_name.localstr());
  ofile.precision(FloatInfo<Real>::maxDigit());
  if (!ofile)
    throw IOException("VtkMeshIOService::writeMeshToFile(): Unable to open file");
  ofile << "# vtk DataFile Version 2.0\n";
  ofile << "Maillage Arcane\n";
  ofile << "ASCII\n";
  ofile << "DATASET UNSTRUCTURED_GRID\n";
  UniqueArray<Integer> nodes_local_id_to_current(mesh->itemFamily(IK_Node)->maxLocalId());
  nodes_local_id_to_current.fill(NULL_ITEM_ID);

  Integer nb_node = mesh->nbNode();
  IItemFamily* cell_kind_family = mesh->itemFamily(cell_kind);
  Integer nb_cell_kind = cell_kind_family->nbItem();

  // Save nodes
  {
    ofile << "POINTS " << nb_node << " double\n";
    VariableNodeReal3& coords(mesh->toPrimaryMesh()->nodesCoordinates());
    Integer node_index = 0;
    ENUMERATE_NODE (inode, mesh->allNodes()) {
      Node node = *inode;
      nodes_local_id_to_current[node.localId()] = node_index;
      Real3 xyz = coords[inode];
      ofile << xyz.x << ' ' << xyz.y << ' ' << xyz.z << '\n';
      ++node_index;
    }
  }

  // Save cells or faces
  {
    Integer nb_node_cell_kind = nb_cell_kind;
    ENUMERATE_ITEMWITHNODES(iitem, cell_kind_family->allItems())
    {
      nb_node_cell_kind += (*iitem).nbNode();
    }
    ofile << "CELLS " << nb_cell_kind << ' ' << nb_node_cell_kind << "\n";
    ENUMERATE_ITEMWITHNODES(iitem, cell_kind_family->allItems())
    {
      ItemWithNodes item = *iitem;
      Integer item_nb_node = item.nbNode();
      ofile << item_nb_node;
      for (NodeLocalId node_id : item.nodes()) {
        ofile << ' ' << nodes_local_id_to_current[node_id];
      }
      ofile << '\n';
    }
    // The type must be consistent with vtkCellType.h
    ofile << "CELL_TYPES " << nb_cell_kind << "\n";
    ENUMERATE_ (ItemWithNodes, iitem, cell_kind_family->allItems()) {
      const ItemTypeInfo* iti = iitem->typeInfo();
      // Check if the type is a polygon for VTK (dimension 2 and more than 4 nodes)
      int type = VTK_BAD_ARCANE_TYPE;
      if (iti->isPolygon())
        type = VTK_POLYGON;
      else
        type = arcaneToVtkCellType(iti);
      ofile << type << '\n';
    }
  }

  // If we are in the cell mesh, save node groups.
  if (cell_kind == IK_Cell) {
    ofile << "POINT_DATA " << nb_node << "\n";
    _saveGroups(mesh->itemFamily(IK_Node), ofile);
  }

  // Save cell groups
  ofile << "CELL_DATA " << nb_cell_kind << "\n";
  _saveGroups(mesh->itemFamily(cell_kind), ofile);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VtkLegacyMeshWriter::
_saveGroups(IItemFamily* family, std::ostream& ofile)
{
  info() << "Saving groups for family name=" << family->name();
  UniqueArray<char> in_group_list(family->maxLocalId());
  for (ItemGroupCollection::Enumerator igroup(family->groups()); ++igroup;) {
    ItemGroup group = *igroup;
    // No need to save the group of all entities
    if (group == family->allItems())
      continue;
    //HACK: to be removed
    if (group.name() == "OuterFaces")
      continue;
    ofile << "SCALARS GROUP_" << group.name() << " int 1\n";
    ofile << "LOOKUP_TABLE default\n";
    in_group_list.fill('0');
    ENUMERATE_ITEM (iitem, group) {
      in_group_list[(*iitem).localId()] = '1';
    }
    ENUMERATE_ITEM (iitem, family->allItems()) {
      ofile << in_group_list[(*iitem).localId()] << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkMeshReader
: public AbstractService
, public IMeshReader
{
 public:

  explicit VtkMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}

 public:

  bool allowExtension(const String& str) override { return str == "vtk"; }
  eReturnType readMeshFromFile(IPrimaryMesh* mesh, const XmlNode& mesh_node, const String& file_name,
                               const String& dir_name, bool use_internal_partition) override

  {
    ARCANE_UNUSED(mesh_node);
    VtkMeshIOService vtk_service(traceMng());
    bool ret = vtk_service.readMesh(mesh, file_name, dir_name, use_internal_partition);
    if (ret)
      return RTError;

    return RTOk;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(VtkMeshReader, IMeshReader, VtkMeshIO);

ARCANE_REGISTER_SERVICE(VtkMeshReader,
                        ServiceProperty("VtkLegacyMeshReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkLegacyCaseMeshReader
: public AbstractService
, public ICaseMeshReader
{
 public:

  class Builder
  : public IMeshBuilder
  {
   public:

    explicit Builder(ITraceMng* tm, const CaseMeshReaderReadInfo& read_info)
    : m_trace_mng(tm)
    , m_read_info(read_info)
    {}

   public:

    void fillMeshBuildInfo(MeshBuildInfo& build_info) override
    {
      ARCANE_UNUSED(build_info);
    }
    void allocateMeshItems(IPrimaryMesh* pm) override
    {
      VtkMeshIOService vtk_service(m_trace_mng);
      String fname = m_read_info.fileName();
      m_trace_mng->info() << "VtkLegacy Reader (ICaseMeshReader) file_name=" << fname;
      bool ret = vtk_service.readMesh(pm, fname, m_read_info.directoryName(), m_read_info.isParallelRead());
      if (ret)
        ARCANE_FATAL("Can not read VTK File");
    }

   private:

    ITraceMng* m_trace_mng;
    CaseMeshReaderReadInfo m_read_info;
  };

 public:

  explicit VtkLegacyCaseMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}

 public:

  Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const override
  {
    IMeshBuilder* builder = nullptr;
    if (read_info.format() == "vtk")
      builder = new Builder(traceMng(), read_info);
    return makeRef(builder);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(VtkLegacyCaseMeshReader,
                        ServiceProperty("VtkLegacyCaseMeshReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ICaseMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
