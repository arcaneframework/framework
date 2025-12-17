// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MEDMeshReaderService.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Lecture d'un maillage au format MED.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"

#include "arcane/IMeshReader.h"
#include "arcane/BasicService.h"
#include "arcane/ServiceFactory.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/ICaseMeshReader.h"
#include "arcane/IMeshBuilder.h"
#include "arcane/IParallelMng.h"
#include "arcane/MeshPartInfo.h"

#include <med.h>
#define MESGERR 1
#include <med_utils.h>
#include <string.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur de maillages au format MED.
 *
 * Première version d'un lecteur MED gérant uniquement les maillages 2D, 3D et
 * non structurés.
 */
class MEDMeshReader
: public TraceAccessor
{
 public:

  /*!
   * \brief Informations pour passer des types MED aux types Arcane pour les entités.
   *
   * \a indirection() est non nul si la connectivité MED est différente de la
   * connectivité Arcane, ce qui est le cas pour les entités 2D et 3D.
   */
  struct MEDToArcaneItemInfo
  {
   public:

    MEDToArcaneItemInfo(int dimension, int nb_node, med_int med_type,
                        Integer arcane_type, const Int32* indirection)
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
    Integer arcaneType() const { return m_arcane_type; }
    const Int32* indirection() const { return m_indirection; }

   private:

    int m_dimension;
    int m_nb_node;
    med_int m_med_type;
    Integer m_arcane_type;
    const Int32* m_indirection;
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

  // Structure pour fermer automatiquement les fichiers MED ouverts
  struct AutoCloseMED
  {
    AutoCloseMED(med_idt id)
    : fid(id)
    {}
    ~AutoCloseMED()
    {
      if (fid >= 0)
        ::MEDfileClose(fid);
    }

    med_idt fid;
  };
  UniqueArray<MEDToArcaneItemInfo> m_med_to_arcane_types;

 private:

  Int32 _readItems(med_idt fid, const char* meshnane, const MEDToArcaneItemInfo& iinfo,
                   Array<med_int>& connectivity);
  void _initMEDToArcaneTypes();
  void _addTypeInfo(int dimension, int nb_node, med_int med_type, Integer arcane_type)
  {
    MEDToArcaneItemInfo t(dimension, nb_node, med_type, arcane_type, nullptr);
    m_med_to_arcane_types.add(t);
  }
  void _addTypeInfo(int dimension, int nb_node, med_int med_type, Integer arcane_type,
                    const Int32* indirection)
  {
    MEDToArcaneItemInfo t(dimension, nb_node, med_type, arcane_type, indirection);
    m_med_to_arcane_types.add(t);
  }
  void _readAndAllocateCells(IPrimaryMesh* mesh, Int32 mesh_dimension, med_idt fid, const char* meshname);

  [[nodiscard]] IMeshReader::eReturnType
  _readNodesCoordinates(IPrimaryMesh* mesh, Int64 nb_node, Int32 spacedim,
                        med_idt fid, const char* meshname);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  const Int32 Hexaedron8_indirection[] = { 1, 0, 3, 2, 5, 4, 7, 6 };
  const Int32 Hexaedron20_indirection[] = { 1, 8, 10, 3,   9, 2, 0, 11,   5, 14, 18, 7,   6, 4, 16, 15,  13, 12, 17, 19 };
  const Int32 Pyramid5_indirection[] = { 1, 0, 3, 2, 4 };
  const Int32 Quad4_indirection[] = { 1, 0, 3, 2 };
  const Int32 Triangle3_indirection[] = { 1, 0, 2 };
  // PAS utilisé pour l'instant. A tester.
  const Int32 Tetraedron4_indirection[] = { 1, 0, 2, 3 };
} // namespace

void MEDMeshReader::
_initMEDToArcaneTypes()
{
  m_med_to_arcane_types.clear();

  // TODO: regarder la correspondance de connectivité entre
  // Arcane et MED pour les éléments quadratiques
  // Types 1D
  _addTypeInfo(1, 2, MED_SEG2, IT_Line2);
  _addTypeInfo(1, 3, MED_SEG3, IT_Line3); // Non supporté
  _addTypeInfo(1, 4, MED_SEG4, IT_NullType); // Non supporté

  // Types 2D.
  _addTypeInfo(2, 3, MED_TRIA3, IT_Triangle3, Triangle3_indirection);
  _addTypeInfo(2, 4, MED_QUAD4, IT_Quad4, Quad4_indirection);
  _addTypeInfo(2, 6, MED_TRIA6, IT_NullType); // Non supporté
  _addTypeInfo(2, 7, MED_TRIA7, IT_NullType); // Non supporté
  _addTypeInfo(2, 8, MED_QUAD8, IT_Quad8); // Non supporté
  _addTypeInfo(2, 9, MED_QUAD9, IT_NullType); // Non supporté

  // Types 3D
  _addTypeInfo(3, 4, MED_TETRA4, IT_Tetraedron4);
  _addTypeInfo(3, 5, MED_PYRA5, IT_Pyramid5, Pyramid5_indirection);
  _addTypeInfo(3, 6, MED_PENTA6, IT_Pentaedron6);
  _addTypeInfo(3, 8, MED_HEXA8, IT_Hexaedron8, Hexaedron8_indirection);
  _addTypeInfo(3, 10, MED_TETRA10, IT_Tetraedron10);
  _addTypeInfo(3, 12, MED_OCTA12, IT_Octaedron12);
  _addTypeInfo(3, 13, MED_PYRA13, IT_NullType); // Non supporté
  _addTypeInfo(3, 15, MED_PENTA15, IT_NullType); // Non supporté
  _addTypeInfo(3, 18, MED_PENTA18, IT_NullType); // Non supporté
  _addTypeInfo(3, 20, MED_HEXA20, IT_Hexaedron20, Hexaedron20_indirection);
  _addTypeInfo(3, 27, MED_HEXA27, IT_NullType); // Non supporté

  // Mailles dont la géométrie à une connectivité variable.
  // Pour l'instant on ne supporte aucun de ces types dans Arcane.
  // On traite quand même ces éléments pour afficher une erreur s'ils sont
  // présents dans le maillage. En mettant la valeur (0) pour le nombre
  // de noeuds on signale à _readItems() qu'on ne sait pas traiter ces éléments.
  ///
  _addTypeInfo(2, 0, MED_POLYGON, IT_NullType);
  _addTypeInfo(2, 0, MED_POLYGON2, IT_NullType);
  _addTypeInfo(3, 0, MED_POLYHEDRON, IT_NullType);

  // Mailles dont la géométrie est dynamique (découverte du modèle dans le fichier)
  // TODO: regarder comment les traiter
  //#define MED_STRUCT_GEO_INTERNAL 600
  //#define MED_STRUCT_GEO_SUP_INTERNAL 700
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType MEDMeshReader::
readMesh(IPrimaryMesh* mesh, const String& file_name)
{
  info() << "Trying to read MED File name=" << file_name;
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
  // Pour garantir la fermeture du fichier.
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

  // Le maillage qu'on lit est toujours le premier
  int mesh_index = 1;

  // Récupère la dimension d'espace. Cela est nécessaire pour dimensionner axisname eet unitname
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
  // Lecture du nombre de noeuds.
  {
    med_bool coordinatechangement;
    med_bool geotransformation;
    // TODO: traiter les informations telles que coordinatechangement
    // et geotransformation si besoin
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

  IParallelMng* pm = mesh->parallelMng();
  bool is_parallel = pm->isParallel();
  Int32 rank = mesh->meshPartInfo().partRank();
  // En parallèle, seul le rang 0 lit le maillage
  bool is_read_items = !(is_parallel && rank != 0);
  if (is_read_items) {
    _readAndAllocateCells(mesh, mesh_dimension, fid, meshname);
    mesh->endAllocate();
    return _readNodesCoordinates(mesh, nb_node, spacedim, fid, meshname);
  }
  // Appelle la méthode d'allocation avec zéro mailles.
  // Cela est nécessaire car IPrimaryMesh::allocateCells() est collective.
  mesh->allocateCells(0, {}, true);

  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MEDMeshReader::
_readAndAllocateCells(IPrimaryMesh* mesh, Int32 mesh_dimension, med_idt fid, const char* meshname)
{
  Int64 cell_unique_id = 0;
  // Alloue les entités types par type.
  for (const auto& iinfo : m_med_to_arcane_types) {
    Integer item_dimension = iinfo.dimension();
    // On ne traite que les entités de la dimension du maillage.
    if (item_dimension != mesh_dimension)
      continue;
    UniqueArray<med_int> med_connectivity;
    Int32 nb_item = _readItems(fid, meshname, iinfo, med_connectivity);
    if (nb_item == 0)
      continue;
    Integer arcane_type = iinfo.arcaneType();
    Integer nb_item_node = iinfo.nbNode();
    if (arcane_type == IT_NullType) {
      // Indique un type supporté par MED mais pas par Arcane
      ARCANE_FATAL("MED type '{0}' is not supported by Arcane", iinfo.medType());
    }
    Int64 cells_infos_index = 0;
    Int64 med_connectivity_index = 0;
    UniqueArray<Int64> cells_infos((2 + nb_item_node) * nb_item);
    info() << "CELL_INFOS size=" << cells_infos.size() << " nb_item=" << nb_item
           << " type=" << arcane_type;
    const Int32* indirection = iinfo.indirection();
    for (Int32 i = 0; i < nb_item; ++i) {
      cells_infos[cells_infos_index] = arcane_type;
      ++cells_infos_index;
      cells_infos[cells_infos_index] = cell_unique_id;
      ++cells_infos_index;
      ++cell_unique_id;
      // La connectivité dans MED commence à 1 et Arcane à 0.
      // Il faut donc retrancher 1 de la connectivité donnée par MED.
      Span<Int64> cinfo_span(cells_infos.span().subspan(cells_infos_index, nb_item_node));
      Span<med_int> med_cinfo_span(med_connectivity.span().subspan(med_connectivity_index, nb_item_node));
      if (indirection) {
        for (Integer k = 0; k < nb_item_node; ++k) {
          cinfo_span[k] = med_cinfo_span[indirection[k]] - 1;
        }
      }
      else {
        for (Integer k = 0; k < nb_item_node; ++k)
          cinfo_span[k] = med_cinfo_span[k] - 1;
      }
      med_connectivity_index += nb_item_node;
      cells_infos_index += nb_item_node;

    }
    mesh->allocateCells(nb_item, cells_infos, false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType MEDMeshReader::
_readNodesCoordinates(IPrimaryMesh* mesh, Int64 nb_node, Int32 spacedim,
                      med_idt fid, const char* meshname)
{
  const bool do_verbose = false;
  // Lit les coordonnées des noeuds et positionne les coordonnées dans Arcane
  UniqueArray<Real3> nodes_coordinates(nb_node);
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
        nodes_coordinates[i] = xyz;
      }
    }
    else if (spacedim == 2) {
      for (Int64 i = 0; i < nb_node; ++i) {
        Real3 xyz(coordinates[i * 2], coordinates[(i * 2) + 1], 0.0);
        if (do_verbose)
          info() << "I=" << i << " XYZ=" << xyz;
        nodes_coordinates[i] = xyz;
      }
    }
    else
      ARCANE_THROW(NotImplementedException, "spacedim!=2 && spacedim!=3");
  }

  // Positionne les coordonnées
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

Int32 MEDMeshReader::
_readItems(med_idt fid, const char* meshname, const MEDToArcaneItemInfo& iinfo,
           Array<med_int>& connectivity)
{
  int med_item_type = iinfo.medType();
  med_bool coordinatechangement;
  med_bool geotransformation;
  med_int nb_med_item = ::MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, med_item_type,
                                         MED_CONNECTIVITY, MED_NODAL, &coordinatechangement,
                                         &geotransformation);
  if (nb_med_item < 0) {
    ARCANE_FATAL("Can not read MED med_item_type '{0}' error={1}", med_item_type, nb_med_item);
  }
  info() << "MED: type=" << med_item_type << " nb_item=" << nb_med_item;
  if (nb_med_item == 0)
    return 0;
  Int64 nb_node = iinfo.nbNode();
  if (nb_node == 0)
    // Indique un élément qu'on ne sais pas traiter.
    ARCANE_THROW(NotImplementedException, "Reading items with MED type '{0}'", med_item_type);

  connectivity.resize(nb_node * nb_med_item);
  int err = MEDmeshElementConnectivityRd(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL,
                                         med_item_type, MED_NODAL, MED_FULL_INTERLACE,
                                         connectivity.data());
  if (err < 0) {
    ARCANE_FATAL("Can not read connectivity MED med_item_type '{0}' error={1}",
                 med_item_type, err);
  }
  info() << "CON: " << connectivity;
  return nb_med_item;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
                               const XmlNode& mesh_element,
                               const String& file_name,
                               const String& dir_name,
                               bool use_internal_partition) override
  {
    ARCANE_UNUSED(mesh_element);
    ARCANE_UNUSED(dir_name);
    ARCANE_UNUSED(use_internal_partition);
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
