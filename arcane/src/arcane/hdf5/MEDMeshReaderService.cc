// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MEDMeshReaderService.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Reading a mesh in MED format.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/FixedArray.h"

#include "arcane/core/IMeshReader.h"
#include "arcane/core/BasicService.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/NodesOfItemReorderer.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/ItemPrinter.h"

#include <med.h>
#define MESGERR 1
#include <med_utils.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief MED format mesh reader.
 *
 * First version of a MED reader handling only 2D, 3D, and
 * unstructured meshes.
 */
class MEDMeshReader
: public TraceAccessor
{
 public:

  /*!
   * \brief Information for mapping MED types to Arcane types for entities.
   *
   * \a indirection() is non-null if the MED connectivity differs from the
   * Arcane connectivity, which is the case for 2D and 3D entities.
   */
  class MEDToArcaneItemInfo
  {
   public:

    MEDToArcaneItemInfo(int dimension, int nb_node, med_int med_type,
                        ItemTypeId arcane_type, const Int32* indirection)
    : m_dimension(dimension)
    , m_nb_node(nb_node)
    , m_med_type(med_type)
    , m_arcane_type(arcane_type)
    , m_indirection(indirection)
    {}

   public:

    int dimension() const { return m_dimension; }
    int nbNode() const { return m_nb_node; }
    med_int medType() const { return m_med_type; }
    Int16 arcaneType() const { return m_arcane_type; }
    const Int32* indirection() const { return m_indirection; }

   private:

    int m_dimension = -1;
    int m_nb_node = -1;
    med_int m_med_type = {};
    ItemTypeId m_arcane_type = ITI_NullType;
    const Int32* m_indirection = nullptr;
  };

  //! Information about a MED entity family
  class MEDFamilyInfo
  {
   public:

    explicit MEDFamilyInfo(Int32 family_id)
    : m_family_id(family_id)
    {}

   public:

    //! Family ID for MED
    Int32 m_family_id = 0;
    //! Index in the Arcane group list.
    Int32 m_index = -1;
  };

  /*!
   * \brief List of groups and the entities belonging to them.
   *
   * For each group, we can provide either the list of uniqueId()
   * of the entities inside, or the list of localId().
   * The first case is used by meshes and the second
   * by faces and nodes
   */
  class MEDGroupInfo
  {
   public:

    explicit MEDGroupInfo(Int32 index)
    : m_index(index)
    {}

   public:

    //! Index of the group in the group list
    Int32 m_index = -1;
    //! Associated group names
    UniqueArray<String> m_names;
    //! List of uniqueId() of the group's entities.
    UniqueArray<Int64> m_unique_ids;
    //! List of localId() of the group's entities.
    UniqueArray<Int32> m_local_ids;
  };

 public:

  explicit MEDMeshReader(ITraceMng* tm)
  : TraceAccessor(tm)
  {
    _initMEDToArcaneTypes();
  }

 public:

  [[nodiscard]] IMeshReader::eReturnType
  readMesh(IPrimaryMesh* mesh, const String& file_name);

 private:

  IMeshReader::eReturnType _readMesh(IPrimaryMesh* mesh, const String& filename);

 private:

  // Structure to automatically close open MED files
  struct AutoCloseMED
  {
    explicit AutoCloseMED(med_idt id)
    : fid(id)
    {}
    ~AutoCloseMED()
    {
      if (fid >= 0)
        ::MEDfileClose(fid);
    }

    med_idt fid;
  };

  //! Mesh currently being read
  IPrimaryMesh* m_mesh = nullptr;
  //! Conversion table between MED and Arcane types
  UniqueArray<MEDToArcaneItemInfo> m_med_to_arcane_types;
  //! Table of indices in \a m_med_to_arcane_type for each geotype
  std::unordered_map<med_int, Int32> m_med_geotype_to_arcane_type_index;
  //! List of families
  std::unordered_map<Int32, MEDFamilyInfo> m_med_families_map;
  //! List of group information
  UniqueArray<MEDGroupInfo> m_med_groups;
  //! List of 'geotypes' present in the mesh
  UniqueArray<med_int> m_med_geotypes_in_mesh;

 private:

  Int32 _readItems(med_idt fid, const char* meshnane, const MEDToArcaneItemInfo& iinfo,
                   Array<Int16>& polygon_nb_nodes, Array<med_int>& connectivity, Array<med_int>& family_values);
  void _initMEDToArcaneTypes();
  void _addTypeInfo(int dimension, int nb_node, med_int med_type, ItemTypeId arcane_type)
  {
    _addTypeInfo(dimension, nb_node, med_type, arcane_type, nullptr);
  }
  void _addTypeInfo(int dimension, int nb_node, med_int med_type, ItemTypeId arcane_type,
                    const Int32* indirection)
  {
    MEDToArcaneItemInfo t(dimension, nb_node, med_type, arcane_type, indirection);
    Int32 index = m_med_to_arcane_types.size();
    m_med_to_arcane_types.add(t);
    m_med_geotype_to_arcane_type_index.insert(std::make_pair(med_type, index));
  }
  void _readAndCreateCells(IPrimaryMesh* mesh, Int32 mesh_dimension, med_idt fid, const char* meshname);
  void _readFaces(IPrimaryMesh* mesh, Int32 mesh_dimension, med_idt fid, const char* meshname);

  [[nodiscard]] IMeshReader::eReturnType
  _readNodesCoordinates(IPrimaryMesh* mesh, Int64 nb_node, Int32 spacedim,
                        med_idt fid, const char* meshname);
  void _readFamilies(med_idt fid, const char* meshname);
  void _readAvailableTypes(med_idt fid, const char* meshname);
  void _clearItemsInGroups()
  {
    for (MEDGroupInfo& g : m_med_groups) {
      g.m_unique_ids.clear();
      g.m_local_ids.clear();
    }
  }
  void _broadcastGroups(ConstArrayView<String> names, IItemFamily* family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  // MED numbering conventions are different from those used in Arcane.
  // These arrays allow for renumbering.
  const Int32 Hexaedron8_indirection[] = { 1, 0, 3, 2, 5, 4, 7, 6 };
  const Int32 Hexaedron20_indirection[] = { 1, 8, 10, 3, 9, 2, 0, 11, 5, 14, 18, 7, 6, 4, 16, 15, 13, 12, 17, 19 };
  const Int32 Pyramid5_indirection[] = { 1, 0, 3, 2, 4 };
  const Int32 Quad4_indirection[] = { 1, 0, 3, 2 };
  const Int32 Quad8_indirection[] = { 1, 0, 3, 2, 4, 7, 6, 5 };
  const Int32 Triangle3_indirection[] = { 1, 0, 2 };
  // Not used for now. To be tested.
  const Int32 Tetraedron4_indirection[] = { 1, 0, 2, 3 };
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MEDMeshReader::
_initMEDToArcaneTypes()
{
  m_med_to_arcane_types.clear();

  // TODO: check the connectivity correspondence between
  // Arcane and MED for quadrilateral elements
  // 1D Types
  _addTypeInfo(1, 2, MED_SEG2, ITI_Line2);
  _addTypeInfo(1, 3, MED_SEG3, ITI_Line3); // Not supported
  _addTypeInfo(1, 4, MED_SEG4, ITI_NullType); // Not supported

  // 2D Types.
  _addTypeInfo(2, 3, MED_TRIA3, ITI_Triangle3, Triangle3_indirection);
  _addTypeInfo(2, 4, MED_QUAD4, ITI_Quad4, Quad4_indirection);
  _addTypeInfo(2, 6, MED_TRIA6, ITI_NullType); // Not supported
  _addTypeInfo(2, 7, MED_TRIA7, ITI_NullType); // Not supported
  _addTypeInfo(2, 8, MED_QUAD8, ITI_Quad8, Quad8_indirection);
  _addTypeInfo(2, 9, MED_QUAD9, ITI_NullType); // Not supported

  // 3D Types
  _addTypeInfo(3, 4, MED_TETRA4, ITI_Tetraedron4);
  _addTypeInfo(3, 5, MED_PYRA5, ITI_Pyramid5, Pyramid5_indirection);
  _addTypeInfo(3, 6, MED_PENTA6, ITI_Pentaedron6);
  _addTypeInfo(3, 8, MED_HEXA8, ITI_Hexaedron8);
  _addTypeInfo(3, 10, MED_TETRA10, ITI_Tetraedron10);
  _addTypeInfo(3, 12, MED_OCTA12, ITI_Octaedron12);
  _addTypeInfo(3, 13, MED_PYRA13, ITI_NullType); // Not supported
  _addTypeInfo(3, 15, MED_PENTA15, ITI_NullType); // Not supported
  _addTypeInfo(3, 18, MED_PENTA18, ITI_NullType); // Not supported
  _addTypeInfo(3, 20, MED_HEXA20, ITI_Hexaedron20);
  _addTypeInfo(3, 27, MED_HEXA27, ITI_NullType); // Not supported

  // Meshes whose geometry has variable connectivity.
  // For now, we do not support any of these types in Arcane.
  // We still process these elements to display an error if they are
  // present in the mesh. By setting the node count to (0), we signal to _readItems()
  // that we cannot process these elements.

  _addTypeInfo(2, 0, MED_POLYGON, ITI_GenericPolygon);
  _addTypeInfo(2, 0, MED_POLYGON2, ITI_NullType);
  _addTypeInfo(3, 0, MED_POLYHEDRON, ITI_NullType);

  // Meshes whose geometry is dynamic (model discovery in the file)
  // TODO: check how to process them
  //#define MED_STRUCT_GEO_INTERNAL 600
  //#define MED_STRUCT_GEO_SUP_INTERNAL 700
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType MEDMeshReader::
readMesh(IPrimaryMesh* mesh, const String& file_name)
{
  info() << "Trying to read MED File name=" << file_name;
  m_mesh = mesh;
  return _readMesh(mesh, file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType MEDMeshReader::
_readMesh(IPrimaryMesh* mesh, const String& filename)
{
  const med_idt fid = MEDfileOpen(filename.localstr(), MED_ACC_RDONLY);
  if (fid < 0) {
    MESSAGE("ERROR: can not open MED file ");
    error() << "ERROR: can not open MED file '" << filename << "'";
    return IMeshReader::RTError;
  }
  // To guarantee file closure.
  AutoCloseMED auto_close_med(fid);

  int nb_mesh = MEDnMesh(fid);
  if (nb_mesh < 0) {
    error() << "Error reading number of meshes";
    return IMeshReader::RTError;
  }
  info() << "MED: nb_mesh=" << nb_mesh;
  if (nb_mesh == 0) {
    error() << "No mesh is present";
    return IMeshReader::RTError;
  }

  // The mesh we read is always the first one
  int mesh_index = 1;

  // Get the space dimension. This is necessary to dimension axisname and unitname
  int nb_axis = MEDmeshnAxis(fid, mesh_index);
  if (nb_axis < 0) {
    error() << "Can not read number of axis (MEDmeshnAxis)";
    return IMeshReader::RTError;
  }
  info() << "MED: nb_axis=" << nb_axis;

  UniqueArray<char> axisname(MED_SNAME_SIZE * nb_axis + 1, '\0');
  UniqueArray<char> unitname(MED_SNAME_SIZE * nb_axis + 1, '\0');

  char meshname[MED_NAME_SIZE + 1];
  meshname[0] = '\0';
  char meshdescription[MED_COMMENT_SIZE + 1];
  meshdescription[0] = '\0';
  char dtunit[MED_SNAME_SIZE + 1];
  dtunit[0] = '\0';
  med_int spacedim = 0;
  med_int meshdim = 0;
  med_mesh_type meshtype = MED_UNDEF_MESH_TYPE;
  med_sorting_type sortingtype = MED_SORT_UNDEF;
  med_int nstep = 0;
  med_axis_type axistype = MED_UNDEF_AXIS_TYPE;
  int err = 0;
  err = MEDmeshInfo(fid, mesh_index, meshname, &spacedim, &meshdim, &meshtype, meshdescription,
                    dtunit, &sortingtype, &nstep, &axistype, axisname.data(), unitname.data());
  if (err < 0) {
    error() << "Can not read mesh info (MEDmeshInfo) r=" << err;
    return IMeshReader::RTError;
  }
  if (meshtype != MED_UNSTRUCTURED_MESH) {
    error() << "Arcane handle only MED unstructured mesh (MED_UNSTRUCTURED_MESH) type=" << meshtype;
    return IMeshReader::RTError;
  }
  Integer mesh_dimension = meshdim;
  if (mesh_dimension != 2 && mesh_dimension != 3)
    ARCANE_FATAL("MED reader handles only 2D or 3D meshes");

  info() << "MED: name=" << meshname;
  info() << "MED: description=" << meshdescription;
  info() << "MED: spacedim=" << spacedim;
  info() << "MED: meshdim=" << meshdim;
  info() << "MED: dtunit=" << dtunit;
  info() << "MED: meshtype=" << meshtype;
  info() << "MED: sortingtype=" << sortingtype;
  info() << "MED: axistype=" << axistype;
  info() << "MED: nstep=" << nstep;

  Int64 nb_node = 0;
  // Reading the number of nodes.
  {
    med_bool coordinatechangement;
    med_bool geotransformation;
    // TODO: process information such as coordinatechangement
    // and geotransformation if needed
    med_int med_nb_node = MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT, MED_NODE, MED_NO_GEOTYPE,
                                         MED_COORDINATE, MED_NO_CMODE, &coordinatechangement,
                                         &geotransformation);
    if (med_nb_node < 0) {
      error() << "Can not read number of nodes (MEDmeshnEntity) err=" << med_nb_node;
      return IMeshReader::RTError;
    }
    nb_node = med_nb_node;
  }
  info() << "MED: nb_node=" << nb_node;

  mesh->setDimension(mesh_dimension);

  // MED meshes can contain polygons.
  // We therefore build the corresponding types.
  // (NOTE: all subdomains must do this)
  mesh->itemTypeMng()->buildPolygonTypes();

  IParallelMng* pm = mesh->parallelMng();
  bool is_parallel = pm->isParallel();
  Int32 rank = mesh->meshPartInfo().partRank();
  // In parallel, only rank 0 reads the mesh
  bool is_read_items = !(is_parallel && rank != 0);
  if (is_read_items) {
    _readAvailableTypes(fid, meshname);
    _readFamilies(fid, meshname);
    _readAndCreateCells(mesh, mesh_dimension, fid, meshname);
  }
  // The IPrimaryMesh::endAllocate() method is collective, so everyone
  // must call it even if ranks other than rank 0
  // do not have meshes.
  mesh->endAllocate();

  // List of names of created mesh groups
  // It will be used to transfer the list of groups to all ranks.
  UniqueArray<String> cell_group_names;
  IItemFamily* cell_family = mesh->cellFamily();
  if (is_read_items) {
    // Now that all meshes have been created, we create the corresponding groups
    // To do this, we iterate through all instances of 'm_med_groups' and if one has entities
    // then they are meshes to be added to a group.
    // ATTENTION ATTENTION:
    // NOTE: The groups must be common to all ranks. They must be broadcasted
    UniqueArray<Int32> cell_local_ids;
    for (const MEDGroupInfo& g : m_med_groups) {
      Int32 nb_cell_in_group = g.m_unique_ids.size();
      cell_local_ids.resize(nb_cell_in_group);
      cell_family->itemsUniqueIdToLocalId(cell_local_ids, g.m_unique_ids);
      for (const String& name : g.m_names) {
        info() << "Group=" << name << " index=" << g.m_index << " nb_item=" << nb_cell_in_group;
        CellGroup cell_group = cell_family->findGroup(name, true);
        cell_group.addItems(cell_local_ids);
        cell_group_names.add(name);
      }
    }
  }
  _broadcastGroups(cell_group_names, cell_family);

  // Reading the faces
  if (is_read_items) {
    // Since the face numbering is not necessarily correct for all
    // entity types (especially for order 2), we add an option to
    // not read the faces.
    bool is_face_group_disabled = false;
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_MED_DISABLE_FACEGROUP", true))
      is_face_group_disabled = (v.value());
    if (!is_face_group_disabled)
      _readFaces(mesh, mesh_dimension, fid, meshname);
  }

  UniqueArray<String> face_group_names;
  IItemFamily* face_family = mesh->faceFamily();
  // Now add the faces to the groups.
  if (is_read_items) {
    for (const MEDGroupInfo& g : m_med_groups) {
      Int32 nb_face_in_group = g.m_local_ids.size();
      info() << "Check Group index=" << g.m_index << " nb_item=" << nb_face_in_group;
      if (nb_face_in_group == 0)
        continue;
      for (const String& name : g.m_names) {
        info() << "FaceGroup=" << name << " index=" << g.m_index << " nb_item=" << nb_face_in_group;
        FaceGroup face_group = face_family->findGroup(name, true);
        face_group.addItems(g.m_local_ids);
        face_group_names.add(name);
      }
    }
  }
  _broadcastGroups(face_group_names, face_family);

  if (is_read_items) {
    // Reading the coordinates
    return _readNodesCoordinates(mesh, nb_node, spacedim, fid, meshname);
  }
  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retrieves the list of geometric types present in the mesh.
 */
void MEDMeshReader::
_readAvailableTypes(med_idt fid, const char* meshname)
{
  // Retrieves the number of geometric types
  med_bool coordinatechangement;
  med_bool geotransformation;
  med_int nb_geo = MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, MED_GEO_ALL,
                                  MED_CONNECTIVITY, MED_NODAL, &coordinatechangement,
                                  &geotransformation);
  if (nb_geo < 0)
    ARCANE_FATAL("Can not read number of geometric entities nb_geo={0}", nb_geo);
  info() << "MED: nb_geotype = " << nb_geo;

  // Loop through the present types
  for (med_int it = 1; it <= nb_geo; it++) {

    med_geometry_type geotype = MED_GEO_ALL;
    FixedArray<char, MED_NAME_SIZE + 1> geotype_name;

    /* get geometry type */
    med_int type_ret = MEDmeshEntityInfo(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, it,
                                         geotype_name.data(), &geotype);
    if (type_ret < 0)
      ARCANE_FATAL("Can not read informations for geotype index={0} ret={1}", it, type_ret);
    /* how many cells of type geotype ? */
    med_int nb_item = MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, geotype,
                                     MED_CONNECTIVITY, MED_NODAL, &coordinatechangement,
                                     &geotransformation);
    if (nb_item < 0)
      ARCANE_FATAL("Can not read number of items for geotype={0} name={1} ret={2}",
                   geotype, geotype_name.data(), nb_item);
    info() << "MED: type=" << geotype << " '" << geotype_name.data() << "' nb_item=" << nb_item;
    m_med_geotypes_in_mesh.add(geotype);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MEDMeshReader::
_readAndCreateCells(IPrimaryMesh* mesh, Int32 mesh_dimension, med_idt fid, const char* meshname)
{
  _clearItemsInGroups();

  // As a matter of principle, there is no uniqueId() for entities in MED (TODO: to verify)
  // So we number the meshes starting from zero and increment for each
  // mesh created.
  Int64 cell_unique_id = 0;

  UniqueArray<Int16> polygon_nb_nodes;
  UniqueArray<med_int> med_connectivity;
  UniqueArray<med_int> med_family_values;

  ItemTypeMng* itm = mesh->itemTypeMng();
  // Allocates meshes type by type.
  // Iterates through the available types and processes those that match the dimension
  // of the mesh.
  for (med_int geotype : m_med_geotypes_in_mesh) {
    Int32 index_in_list = m_med_geotype_to_arcane_type_index[geotype];
    const MEDToArcaneItemInfo& iinfo = m_med_to_arcane_types[index_in_list];

    Int32 item_dimension = iinfo.dimension();
    // We only process entities of the mesh dimension.
    if (item_dimension != mesh_dimension)
      continue;
    Int32 nb_item = _readItems(fid, meshname, iinfo, polygon_nb_nodes, med_connectivity, med_family_values);
    if (nb_item == 0)
      continue;
    Int16 arcane_type = iinfo.arcaneType();
    Int32 nb_item_node = iinfo.nbNode();
    Int32 nb_family_values = med_family_values.size();
    if (arcane_type == IT_NullType) {
      // Indicates a type supported by MED but not by Arcane
      ARCANE_FATAL("MED type '{0}' is not supported by Arcane", iinfo.medType());
    }
    Int64 cells_infos_index = 0;
    Int64 med_connectivity_index = 0;
    const bool is_polygon = (iinfo.medType() == MED_POLYGON);

    UniqueArray<Int64> cells_infos;
    if (is_polygon)
      cells_infos.resize(2 * nb_item + med_connectivity.size());
    else
      cells_infos.resize((2 + nb_item_node) * nb_item);

    info() << "CELL_INFOS size=" << cells_infos.size() << " nb_item=" << nb_item
           << " type=" << arcane_type;

    const Int32* indirection = iinfo.indirection();
    for (Int32 i = 0; i < nb_item; ++i) {
      Int64 current_cell_unique_id = cell_unique_id;
      ++cell_unique_id;
      if (is_polygon) {
        nb_item_node = polygon_nb_nodes[i];
        arcane_type = itm->getPolygonType(static_cast<Int16>(nb_item_node));
        cells_infos[cells_infos_index] = arcane_type;
        ++cells_infos_index;
        cells_infos[cells_infos_index] = current_cell_unique_id;
        ++cells_infos_index;
        Span<Int64> cinfo_span(cells_infos.span().subspan(cells_infos_index, nb_item_node));
        Span<med_int> med_cinfo_span(med_connectivity.span().subspan(med_connectivity_index, nb_item_node));
        for (Integer k = 0; k < nb_item_node; ++k) {
          cinfo_span[k] = med_cinfo_span[k];
        }
      }
      else {
        cells_infos[cells_infos_index] = arcane_type;
        ++cells_infos_index;

        cells_infos[cells_infos_index] = current_cell_unique_id;
        ++cells_infos_index;
        Span<Int64> cinfo_span(cells_infos.span().subspan(cells_infos_index, nb_item_node));
        Span<med_int> med_cinfo_span(med_connectivity.span().subspan(med_connectivity_index, nb_item_node));
        if (indirection) {
          for (Integer k = 0; k < nb_item_node; ++k) {
            cinfo_span[k] = med_cinfo_span[indirection[k]];
          }
        }
        else {
          for (Integer k = 0; k < nb_item_node; ++k)
            cinfo_span[k] = med_cinfo_span[k];
        }
      }
      if (i < nb_family_values) {
        // There is a family associated with the entity
        med_int f = med_family_values[i];
        auto x = m_med_families_map.find(f);
        if (x == m_med_families_map.end()) {
          ARCANE_FATAL("Can not find family id '{0}' for cell '{1}' of geotype '{2}'",
                       f, i, iinfo.medType());
        }
        m_med_groups[x->second.m_index].m_unique_ids.add(current_cell_unique_id);
      }

      med_connectivity_index += nb_item_node;
      cells_infos_index += nb_item_node;
    }
    mesh->allocateCells(nb_item, cells_infos, false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Reads the faces.
 *
 * There is no need to explicitly create the faces because this is done
 * automatically in Arcane. We therefore use the MED faces only
 * to add the faces into the corresponding groups in the mesh file.
 */
void MEDMeshReader::
_readFaces(IPrimaryMesh* mesh, Int32 mesh_dimension, med_idt fid, const char* meshname)
{
  _clearItemsInGroups();
  ItemTypeMng* itm = mesh->itemTypeMng();
  NodesOfItemReorderer nodes_reorderer(itm);

  IItemFamily* node_family = mesh->nodeFamily();
  NodeInfoListView mesh_nodes(node_family);

  UniqueArray<Int16> polygon_nb_nodes;
  UniqueArray<med_int> med_connectivity;
  UniqueArray<med_int> med_family_values;
  // Iterates through the available types and processes those that correspond to the dimension
  // of the mesh minus 1.
  for (med_int geotype : m_med_geotypes_in_mesh) {
    Int32 index_in_list = m_med_geotype_to_arcane_type_index[geotype];
    const MEDToArcaneItemInfo& iinfo = m_med_to_arcane_types[index_in_list];

    Int32 item_dimension = iinfo.dimension();
    // We only process entities of the mesh dimension.
    if (item_dimension != (mesh_dimension - 1))
      continue;
    ItemTypeInfo* iti = itm->typeFromId(iinfo.arcaneType());
    info() << "Reading faces geotype=" << geotype << " arcane_type=" << iinfo.arcaneType()
           << " " << iti->typeName();

    Int32 nb_item = _readItems(fid, meshname, iinfo, polygon_nb_nodes, med_connectivity, med_family_values);
    if (nb_item == 0)
      continue;
    ItemTypeId arcane_type(iinfo.arcaneType());
    Int32 nb_item_node = iinfo.nbNode();
    Int32 nb_family_values = med_family_values.size();
    if (arcane_type == IT_NullType) {
      // Indicates a type supported by MED but not by Arcane
      ARCANE_FATAL("MED type '{0}' is not supported by Arcane", iinfo.medType());
    }

    SmallArray<Int64> orig_nodes_id(nb_item_node);
    info() << "FACES_INFOS nb_item=" << nb_item << " type=" << arcane_type
           << " nb_family_values=" << nb_family_values;

    const Int32* indirection = iinfo.indirection();
    Int64 med_connectivity_index = 0;

    for (Int32 i = 0; i < nb_item; ++i) {
      ArrayView<Int64> cinfo_span(orig_nodes_id);
      Span<med_int> med_cinfo_span(med_connectivity.span().subspan(med_connectivity_index, nb_item_node));
      if (indirection) {
        for (Integer k = 0; k < nb_item_node; ++k) {
          cinfo_span[k] = med_cinfo_span[indirection[k]];
        }
      }
      else {
        for (Integer k = 0; k < nb_item_node; ++k)
          cinfo_span[k] = med_cinfo_span[k];
      }
      med_connectivity_index += nb_item_node;
      // Search for the face in the mesh starting from the sorted uniqueIds of its nodes
      nodes_reorderer.reorder(arcane_type, cinfo_span);
      ConstArrayView<Int64> ordered_nodes = nodes_reorderer.sortedNodes();
      //info() << "OrigMedNodes=" << med_cinfo_span;
      //info() << "OrigNodes=" << orig_nodes_id;
      //info() << "Nodes=" << ordered_nodes;
      Node first_node(MeshUtils::findOneItem(node_family, ordered_nodes[0]));
      if (first_node.null())
        ARCANE_FATAL("Can not find node uid={0} for face index '{1}'", ordered_nodes[0], i);
      Face face = MeshUtils::getFaceFromNodesUniqueId(first_node, ordered_nodes);
      if (face.null()) {
        info() << "ERROR: Can not find face in mesh i=" << i << " nodes=" << ordered_nodes;
        info() << "List of faces for node=" << ItemPrinter(first_node);
        for (Face subface : first_node.faces()) {
          info() << "Face=" << ItemPrinter(subface);
          for (Node subnode : subface.nodes()) {
            info() << "  Node=" << ItemPrinter(subnode);
          }
        }
        ARCANE_FATAL("Can not find face with nodes=", ordered_nodes);
      }
      //info() << "Face=" << ItemPrinter(face);

      // Add the face to the corresponding groups
      if (i < nb_family_values) {
        // There is a family associated with the entity
        med_int f = med_family_values[i];
        auto x = m_med_families_map.find(f);
        if (x == m_med_families_map.end()) {
          ARCANE_FATAL("Can not find family id '{0}' for face '{1}' of geotype '{2}'",
                       f, i, iinfo.medType());
        }
        //info() << "Add face to group_index=" << x->second.m_index;
        m_med_groups[x->second.m_index].m_local_ids.add(face.localId());
      }
    }
    info() << "END_READING_ITEMS";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType MEDMeshReader::
_readNodesCoordinates(IPrimaryMesh* mesh, Int64 nb_node, Int32 spacedim,
                      med_idt fid, const char* meshname)
{
  const bool do_verbose = false;
  // Reads the node coordinates and positions the coordinates in Arcane

  // Connectivity in MED starts at 1 and in Arcane at 0.
  // The first node therefore has the value for uniqueId()
  UniqueArray<Real3> nodes_coordinates(nb_node + 1);
  {
    UniqueArray<med_float> coordinates(nb_node * spacedim);
    int err = MEDmeshNodeCoordinateRd(fid, meshname, MED_NO_DT, MED_NO_IT, MED_FULL_INTERLACE,
                                      coordinates.data());
    if (err < 0) {
      error() << "Can not read nodes coordinates err=" << err;
      return IMeshReader::RTError;
    }

    if (spacedim == 3) {
      for (Int64 i = 0; i < nb_node; ++i) {
        Real3 xyz(coordinates[i * 3], coordinates[(i * 3) + 1], coordinates[(i * 3) + 2]);
        if (do_verbose)
          info() << "I=" << i << " XYZ=" << xyz;
        nodes_coordinates[i + 1] = xyz;
      }
    }
    else if (spacedim == 2) {
      for (Int64 i = 0; i < nb_node; ++i) {
        Real3 xyz(coordinates[i * 2], coordinates[(i * 2) + 1], 0.0);
        if (do_verbose)
          info() << "I=" << i << " XYZ=" << xyz;
        nodes_coordinates[i + 1] = xyz;
      }
    }
    else
      ARCANE_THROW(NotImplementedException, "spacedim!=2 && spacedim!=3");
  }

  // Positions the coordinates
  {
    VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
    ENUMERATE_NODE (inode, mesh->allNodes()) {
      Node node = *inode;
      nodes_coord_var[inode] = nodes_coordinates[node.uniqueId()];
    }
  }
  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Reads information about entities of a given type.
 *
 * Reads information about entities whose type is given by \a iinfo.
 * The entities are meshes in the MED sense, i.e., Edge, Face, or Cell.
 * Returns the number of entities read.
 * \a connectivity will contain the connectivities for the entities read and
 * \a family_values the array for each entity of the family it belongs to. Note
 * that \a family_values may be empty if there is no family associated with the
 * entities.
 *
 * If the type is MED_POLYGON, then \a polygon_nb_nodes will contain the number
 * of nodes for each polygon.
 */
Int32 MEDMeshReader::
_readItems(med_idt fid, const char* meshname, const MEDToArcaneItemInfo& iinfo,
           Array<Int16>& polygon_nb_nodes, Array<med_int>& connectivity,
           Array<med_int>& family_values)
{
  constexpr bool is_verbose = false;

  connectivity.clear();
  family_values.clear();

  int med_item_type = iinfo.medType();
  med_bool coordinatechangement = {};
  med_bool geotransformation = {};
  med_int nb_med_item = 0;
  if (iinfo.medType() == MED_POLYGON) {
    // For polygons, a specific call is needed for the number of indices.
    // This number corresponds to the number of entities plus one.
    med_int nb_index = ::MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, med_item_type,
                                        MED_INDEX_NODE, MED_NODAL, &coordinatechangement,
                                        &geotransformation);
    if (nb_index < 0)
      ARCANE_FATAL("Can not read MED med_item_type '{0}' error={1}", med_item_type, nb_index);

    info() << "MED: Reading items";
    info() << "MED: type=" << med_item_type << " nb_index=" << nb_index;
    if (nb_index < 1)
      return 0;
    nb_med_item = nb_index - 1;
    polygon_nb_nodes.resize(nb_med_item);
    // how many nodes for the polygon connectivity ?
    med_int nb_connectivity = MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT,
                                             MED_CELL, MED_POLYGON, MED_CONNECTIVITY, MED_NODAL,
                                             &coordinatechangement, &geotransformation);
    if (nb_connectivity < 0)
      ARCANE_FATAL("Can not get connectivity size for MED_POLYGON err={0}", nb_connectivity);

    // The table \a indexes contains for each mesh the index of its first
    // node in the connectivity. The number of nodes of the i-th entity
    // is therefore equal to (indexes[i+1]-indexes[i]).
    UniqueArray<med_int> indexes(nb_index);
    connectivity.resize(nb_connectivity);
    info() << "Reading polygons nb_connectivity=" << nb_connectivity;
    int r = MEDmeshPolygonRd(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, MED_NODAL,
                             indexes.data(), connectivity.data());
    if (r < 0)
      ARCANE_FATAL("Can not read connectivity for MED_POLYGON err={0}", r);
    info() << "INDEXES=" << indexes;
    for (Int32 i = 0; i < nb_med_item; ++i)
      polygon_nb_nodes[i] = static_cast<Int16>(indexes[i + 1] - indexes[i]);
  }
  else {
    nb_med_item = ::MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, med_item_type,
                                   MED_CONNECTIVITY, MED_NODAL, &coordinatechangement,
                                   &geotransformation);
    if (nb_med_item < 0)
      ARCANE_FATAL("Can not read MED med_item_type '{0}' error={1}", med_item_type, nb_med_item);

    info() << "MED: Reading items";
    info() << "MED: type=" << med_item_type << " nb_item=" << nb_med_item;
    if (nb_med_item == 0)
      return 0;

    Int64 nb_node = iinfo.nbNode();
    if (nb_node == 0)
      // Indicates an element that we do not know how to process.
      ARCANE_THROW(NotImplementedException, "Reading items with MED type '{0}'", med_item_type);

    connectivity.resize(nb_node * nb_med_item);
    int err = MEDmeshElementConnectivityRd(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL,
                                           med_item_type, MED_NODAL, MED_FULL_INTERLACE,
                                           connectivity.data());
    if (err < 0)
      ARCANE_FATAL("Can not read connectivity MED med_item_type '{0}' error={1}",
                   med_item_type, err);
  }
  if (is_verbose)
    info() << "CON: " << connectivity;
  {
    med_int nb_med_family = MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT,
                                           MED_CELL, med_item_type, MED_FAMILY_NUMBER, MED_NODAL,
                                           &coordinatechangement, &geotransformation);
    info() << "nb_family=" << nb_med_family;
    if (nb_med_family < 0)
      ARCANE_FATAL("Can not read family size for type med_item_type={0} error={1}", med_item_type, nb_med_family);
    if (nb_med_family > 0) {
      family_values.resize(nb_med_family);
      int r = MEDmeshEntityFamilyNumberRd(fid, meshname, MED_NO_DT, MED_NO_IT,
                                          MED_CELL, med_item_type, family_values.data());
      if (r < 0)
        ARCANE_FATAL("Can not read family values for type med_item_type={0} error={1}", med_item_type, nb_med_family);
      if (is_verbose)
        info() << "FAM: " << family_values;
    }
  }
  return nb_med_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MEDMeshReader::
_readFamilies(med_idt fid, const char* meshname)
{
  FixedArray<char, MED_NAME_SIZE + 1> familyname;

  info() << "Read families";

  // Retrieves the number of families
  med_int nb_family = MEDnFamily(fid, meshname);
  if (nb_family < 0)
    ARCANE_FATAL("Can not read number of families (error={0})", nb_family);

  info() << "MED: nb_family= " << nb_family;
  for (med_int i = 0; i < nb_family; i++) {
    info() << "MED: Read family i=" << i;

    med_int nb_group = MEDnFamilyGroup(fid, meshname, i + 1);
    if (nb_group < 0)
      ARCANE_FATAL("Can not read number of groups for family index={0}", i);
    info() << "MED: family index=" << i << " nb_group=" << nb_group;

    // Reads the family groups
    // Even if there are no groups associated with the family, we continue
    // the processing because entities may reference families without groups.

    // In MED, groups have a fixed maximum size MED_LNAME_SIZE
    UniqueArray<char> all_group_names(MED_LNAME_SIZE * nb_group + 1);
    med_int family_number = 0;
    if (MEDfamilyInfo(fid, meshname, i + 1, familyname.data(), &family_number, all_group_names.data()) < 0)
      ARCANE_FATAL("Can not read group names from family index={0}", i);

    MEDFamilyInfo med_family(family_number);
    Int32 group_index = m_med_groups.size();
    med_family.m_index = group_index;
    MEDGroupInfo med_group(group_index);

    // Retrieves the names of the family groups
    for (Int32 z = 0; z < nb_group; ++z) {
      //info() << " groupname=" << group_names << " number=" << familynumber;
      SmallSpan<char> med_group_name = all_group_names.smallSpan().subSpan(MED_LNAME_SIZE * z, MED_LNAME_SIZE);
      // Groups in MED may contain characters not supported by Arcane.
      // We remove them.
      SmallArray<Byte, MED_LNAME_SIZE + 1> valid_name;
      Int32 pos = 0;
      for (; pos < MED_LNAME_SIZE; ++pos) {
        char c = med_group_name[pos];
        if (c == '\0')
          break;
        if (c == ' ' || c == '_')
          continue;
        valid_name.add(static_cast<Byte>(c));
      }
      String name(valid_name.view());
      med_group.m_names.add(name);
      info() << "Family id=" << family_number << " group='" << name << "'";
    }

    m_med_families_map.insert(std::make_pair(family_number, med_family));
    m_med_groups.add(med_group);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Broadcast the groups of \a group_names for the family \a family.
 *
 * The list of groups \a group_names is only used for rank 0.
 */
void MEDMeshReader::
_broadcastGroups(ConstArrayView<String> group_names, IItemFamily* family)
{
  IParallelMng* pm = m_mesh->parallelMng();

  Int32 rank = pm->commRank();
  // Ensures that all ranks know the groups
  if (rank == 0) {
    Int32 nb_group = group_names.size();
    pm->broadcast(ArrayView<Int32>(1, &nb_group), 0);
    for (String name : group_names)
      pm->broadcastString(name, 0);
  }
  else {
    Int32 nb_group = 0;
    pm->broadcast(ArrayView<Int32>(1, &nb_group), 0);
    String current_group_name;
    for (Int32 i = 0; i < nb_group; ++i) {
      pm->broadcastString(current_group_name, 0);
      CellGroup cell_group = family->findGroup(current_group_name, true);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service for reading a mesh in MED format.
 */
class MEDMeshReaderService
: public BasicService
, public IMeshReader
{
 public:

  explicit MEDMeshReaderService(const ServiceBuildInfo& sbi)
  : BasicService(sbi)
  {}

 public:

  void build() override {}
  bool allowExtension(const String& str) override
  {
    return str == "med";
  }
  eReturnType readMeshFromFile(IPrimaryMesh* mesh,
                               [[maybe_unused]] const XmlNode& mesh_element,
                               const String& file_name,
                               const String& dir_name,
                               [[maybe_unused]] bool use_internal_partition) override
  {
    ARCANE_UNUSED(dir_name);
    MEDMeshReader reader(traceMng());
    return reader.readMesh(mesh, file_name);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MEDMeshReaderService,
                        ServiceProperty("MEDMeshReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service for reading a mesh in MED format from the dataset.
 */
class MEDCaseMeshReader
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
      MEDMeshReader reader(m_trace_mng);
      String fname = m_read_info.fileName();
      m_trace_mng->info() << "MED Reader (ICaseMeshReader) file_name=" << fname;
      IMeshReader::eReturnType ret = reader.readMesh(pm, fname);
      if (ret != IMeshReader::RTOk)
        ARCANE_FATAL("Can not read MED File");
    }

   private:

    ITraceMng* m_trace_mng;
    CaseMeshReaderReadInfo m_read_info;
  };

 public:

  explicit MEDCaseMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}

 public:

  Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const override
  {
    IMeshBuilder* builder = nullptr;
    if (read_info.format() == "med")
      builder = new Builder(traceMng(), read_info);
    return makeRef(builder);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MEDCaseMeshReader,
                        ServiceProperty("MEDCaseMeshReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ICaseMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
