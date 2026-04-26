// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MEDMeshReaderService.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Lecture d'un maillage au format MED.                                      */
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

  //! Information sur une famille d'entité MED
  class MEDFamilyInfo
  {
   public:

    explicit MEDFamilyInfo(Int32 family_id)
    : m_family_id(family_id)
    {}

   public:

    //! Id de la famille pour MED
    Int32 m_family_id = 0;
    //! Index dans la liste des groupes Arcane.
    Int32 m_index = -1;
  };

  //! Liste des groupes et des entités leur appartenant
  class MEDGroupInfo
  {
   public:

    explicit MEDGroupInfo(Int32 index)
    : m_index(index)
    {}

   public:

    //! Index du groupe dans la liste des groupes
    Int32 m_index = -1;
    //! Nom des groupes associés
    UniqueArray<String> m_names;
    //! Liste des uniqueId() des entités du groupe.
    UniqueArray<Int64> m_unique_ids;
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

  //! Tableau de conversion entre les type MED et Arcane
  UniqueArray<MEDToArcaneItemInfo> m_med_to_arcane_types;
  //! Table des index dans \a m_med_to_arcane_type de chaque geotype
  std::unordered_map<med_int, Int32> m_med_geotype_to_arcane_type_index;
  //! Liste des familles
  std::unordered_map<Int32, MEDFamilyInfo> m_med_families_map;
  //! Liste des informations sur les groupes
  UniqueArray<MEDGroupInfo> m_med_groups;
  //! Liste des 'geotype' présents dans le maillage
  UniqueArray<med_int> m_med_geotypes_in_mesh;

 private:

  Int32 _readItems(med_idt fid, const char* meshnane, const MEDToArcaneItemInfo& iinfo,
                   Array<med_int>& connectivity, Array<med_int>& family_values);
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

  [[nodiscard]] IMeshReader::eReturnType
  _readNodesCoordinates(IPrimaryMesh* mesh, Int64 nb_node, Int32 spacedim,
                        med_idt fid, const char* meshname);
  void _readFamilies(med_idt fid, const char* meshname);
  void _readAvailableTypes(med_idt fid, const char* meshname);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  // Les conventions de numérotations de MED sont différentes de celles
  // utilisées dans Arcane. Ces tableaux permettent d'effectuer la renumérotation.
  const Int32 Hexaedron8_indirection[] = { 1, 0, 3, 2, 5, 4, 7, 6 };
  const Int32 Hexaedron20_indirection[] = { 1, 8, 10, 3, 9, 2, 0, 11, 5, 14, 18, 7, 6, 4, 16, 15, 13, 12, 17, 19 };
  const Int32 Pyramid5_indirection[] = { 1, 0, 3, 2, 4 };
  const Int32 Quad4_indirection[] = { 1, 0, 3, 2 };
  const Int32 Triangle3_indirection[] = { 1, 0, 2 };
  // PAS utilisé pour l'instant. A tester.
  const Int32 Tetraedron4_indirection[] = { 1, 0, 2, 3 };
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MEDMeshReader::
_initMEDToArcaneTypes()
{
  m_med_to_arcane_types.clear();

  // TODO: regarder la correspondance de connectivité entre
  // Arcane et MED pour les éléments quadratiques
  // Types 1D
  _addTypeInfo(1, 2, MED_SEG2, ITI_Line2);
  _addTypeInfo(1, 3, MED_SEG3, ITI_Line3); // Non supporté
  _addTypeInfo(1, 4, MED_SEG4, ITI_NullType); // Non supporté

  // Types 2D.
  _addTypeInfo(2, 3, MED_TRIA3, ITI_Triangle3, Triangle3_indirection);
  _addTypeInfo(2, 4, MED_QUAD4, ITI_Quad4, Quad4_indirection);
  _addTypeInfo(2, 6, MED_TRIA6, ITI_NullType); // Non supporté
  _addTypeInfo(2, 7, MED_TRIA7, ITI_NullType); // Non supporté
  _addTypeInfo(2, 8, MED_QUAD8, ITI_Quad8); // Non supporté
  _addTypeInfo(2, 9, MED_QUAD9, ITI_NullType); // Non supporté

  // Types 3D
  _addTypeInfo(3, 4, MED_TETRA4, ITI_Tetraedron4);
  _addTypeInfo(3, 5, MED_PYRA5, ITI_Pyramid5, Pyramid5_indirection);
  _addTypeInfo(3, 6, MED_PENTA6, ITI_Pentaedron6);
  _addTypeInfo(3, 8, MED_HEXA8, ITI_Hexaedron8, Hexaedron8_indirection);
  _addTypeInfo(3, 10, MED_TETRA10, ITI_Tetraedron10);
  _addTypeInfo(3, 12, MED_OCTA12, ITI_Octaedron12);
  _addTypeInfo(3, 13, MED_PYRA13, ITI_NullType); // Non supporté
  _addTypeInfo(3, 15, MED_PENTA15, ITI_NullType); // Non supporté
  _addTypeInfo(3, 18, MED_PENTA18, ITI_NullType); // Non supporté
  _addTypeInfo(3, 20, MED_HEXA20, ITI_Hexaedron20, Hexaedron20_indirection);
  _addTypeInfo(3, 27, MED_HEXA27, ITI_NullType); // Non supporté

  // Mailles dont la géométrie à une connectivité variable.
  // Pour l'instant on ne supporte aucun de ces types dans Arcane.
  // On traite quand même ces éléments pour afficher une erreur s'ils sont
  // présents dans le maillage. En mettant la valeur (0) pour le nombre
  // de noeuds on signale à _readItems() qu'on ne sait pas traiter ces éléments.
  ///
  _addTypeInfo(2, 0, MED_POLYGON, ITI_NullType);
  _addTypeInfo(2, 0, MED_POLYGON2, ITI_NullType);
  _addTypeInfo(3, 0, MED_POLYHEDRON, ITI_NullType);

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
    _readAvailableTypes(fid, meshname);
    _readFamilies(fid, meshname);
    _readAndCreateCells(mesh, mesh_dimension, fid, meshname);
  }
  // La méthode IPrimaryMesh::endAllocate() est collective donc tout
  // le monde doit l'appeler même si les rangs autre que le rang 0
  // qui n'ont pas de mailles.
  mesh->endAllocate();

  // Liste des noms des groupes de mailles créées
  // Elle servira à transférer la liste des groupes à tous les rangs.
  UniqueArray<String> cell_group_names;
  IItemFamily* cell_family = mesh->cellFamily();
  if (is_read_items) {
    // Maintenant qu'on a créé toutes les mailles, on créé les groupes correspondants
    // Pour cela on parcours tous les instances de 'm_med_groups' et si une a des entités
    // alors ce sont des mailles à ajouter à un groupe.
    // ATTENTION ATTENTION:
    // NOTE: Les groupes doivent être communs à tout les rangs. Il faut les broadcaster
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
  // S'assure que tous les rangs connaissent les groupes
  if (is_read_items) {
    Int32 nb_group = cell_group_names.size();
    pm->broadcast(ArrayView<Int32>(1, &nb_group), 0);
    for (String name : cell_group_names)
      pm->broadcastString(name, 0);
  }
  else {
    Int32 nb_group = 0;
    pm->broadcast(ArrayView<Int32>(1, &nb_group), 0);
    String current_group_name;
    for (Int32 i = 0; i < nb_group; ++i) {
      pm->broadcastString(current_group_name, 0);
      CellGroup cell_group = cell_family->findGroup(current_group_name, true);
    }
  }

  if (is_read_items) {
    // Lit les coordonnées
    return _readNodesCoordinates(mesh, nb_node, spacedim, fid, meshname);
  }
  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère la liste des types géométriques présents dans le maillage.
 */
void MEDMeshReader::
_readAvailableTypes(med_idt fid, const char* meshname)
{
  // Récupère le nombre de type géométriques
  med_bool coordinatechangement;
  med_bool geotransformation;
  med_int nb_geo = MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, MED_GEO_ALL,
                                  MED_CONNECTIVITY, MED_NODAL, &coordinatechangement,
                                  &geotransformation);
  if (nb_geo < 0)
    ARCANE_FATAL("Can not read number of geometric entities nb_geo={0}", nb_geo);
  info() << "MED: nb_geotype = " << nb_geo;

  // Boucle sur les types présents
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
  // A priori il n'y a pas de uniqueId() pour les entités dans MED (TODO: à vérifier)
  // Donc on numérote les mailles en commencant par zéro et on incrémente à chaque
  // maille créée.
  Int64 cell_unique_id = 0;

  // Alloue les mailles types par type.
  // Parcours les types disponibles et traite ceux qui correspondent à la dimension
  // du maillage.
  for (med_int geotype : m_med_geotypes_in_mesh) {
    Int32 index_in_list = m_med_geotype_to_arcane_type_index[geotype];
    const MEDToArcaneItemInfo& iinfo = m_med_to_arcane_types[index_in_list];

    Int32 item_dimension = iinfo.dimension();
    // On ne traite que les entités de la dimension du maillage.
    if (item_dimension != mesh_dimension)
      continue;
    UniqueArray<med_int> med_connectivity;
    UniqueArray<med_int> med_family_values;
    Int32 nb_item = _readItems(fid, meshname, iinfo, med_connectivity, med_family_values);
    if (nb_item == 0)
      continue;
    Int16 arcane_type = iinfo.arcaneType();
    Int32 nb_item_node = iinfo.nbNode();
    Int32 m_nb_family_values = med_family_values.size();
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
      Int64 current_cell_unique_id = cell_unique_id;
      ++cell_unique_id;
      cells_infos[cells_infos_index] = arcane_type;
      ++cells_infos_index;
      cells_infos[cells_infos_index] = current_cell_unique_id;
      ++cells_infos_index;
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
      if (i < m_nb_family_values) {
        // Il y a une famille associée à l'entité
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
/*!
 * \brief Lit les informations des entités d'un type donné.
 *
 * Lit les informations des entités dont le type est donné par \a iinfo. Les entités sont
 * des mailles au sens MED, donc des Edge, Face ou Cell.
 * En retour, indique le nombre d'entités lues.
 * \a connectivity contiendra la connectivités pour les entités lues et
 * \a family_values le tableau pour chaque entité de la famille a laquelle elle
 * appartient. A noter qu'il est possible que \a family_values soit vide s'il n'y
 * a pas de famille associée aux entités.
 */
Int32 MEDMeshReader::
_readItems(med_idt fid, const char* meshname, const MEDToArcaneItemInfo& iinfo,
           Array<med_int>& connectivity, Array<med_int>& family_values)
{
  const bool is_verbose = false;

  connectivity.clear();
  family_values.clear();

  int med_item_type = iinfo.medType();
  med_bool coordinatechangement;
  med_bool geotransformation;
  med_int nb_med_item = ::MEDmeshnEntity(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL, med_item_type,
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
    // Indique un élément qu'on ne sais pas traiter.
    ARCANE_THROW(NotImplementedException, "Reading items with MED type '{0}'", med_item_type);

  connectivity.resize(nb_node * nb_med_item);
  int err = MEDmeshElementConnectivityRd(fid, meshname, MED_NO_DT, MED_NO_IT, MED_CELL,
                                         med_item_type, MED_NODAL, MED_FULL_INTERLACE,
                                         connectivity.data());
  if (err < 0)
    ARCANE_FATAL("Can not read connectivity MED med_item_type '{0}' error={1}",
                 med_item_type, err);

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

  // Récupère le nombre de familles
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

    // Lit les groupes de la famille
    if (nb_group == 0)
      continue;

    // Dans MED, les groupes ont une taille fixe maximale MED_LNAME_SIZE
    UniqueArray<char> all_group_names(MED_LNAME_SIZE * nb_group + 1);
    med_int family_number = 0;
    if (MEDfamilyInfo(fid, meshname, i + 1, familyname.data(), &family_number, all_group_names.data()) < 0)
      ARCANE_FATAL("Can not read group names from family index={0}", i);

    MEDFamilyInfo med_family(family_number);
    Int32 group_index = m_med_groups.size();
    med_family.m_index = group_index;
    MEDGroupInfo med_group(group_index);

    // Récupère les noms des groupes de la famille
    for (Int32 z = 0; z < nb_group; ++z) {
      //info() << " groupname=" << group_names << " number=" << familynumber;
      SmallSpan<char> med_group_name = all_group_names.smallSpan().subSpan(MED_LNAME_SIZE * z, MED_LNAME_SIZE);
      // Les groupes dans MED peuvent contenir des caractères non supportés par Arcane.
      // On les enlève.
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
