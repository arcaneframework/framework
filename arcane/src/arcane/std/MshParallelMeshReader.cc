// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MshParallelMeshReader.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Lecture parallèle d'un fichier au format MSH.				                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/AbstractService.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IVariableAccessor.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/BasicService.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/ItemPrinter.h"

// Element types in .msh file format, found in gmsh-2.0.4/Common/GmshDefines.h
#include "arcane/std/internal/IosFile.h"
#include "arcane/std/internal/IosGmsh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTES:
 * - La bibliothèque `gmsh` fournit un script 'open.py' dans le répertoire
 *   'demos/api' qui permet de générer un fichier '.msh' à partir d'un '.geo'.
 * - Il est aussi possible d'utiliser directement l'exécutable 'gmsh' avec
 *   l'option '-save-all' pour sauver un fichier '.msh' à partir d'un '.geo'
 *
 * TODO:
 * - lire les tags des noeuds(uniqueId())
 * - supporter les partitions
 * - pouvoir utiliser la bibliothèque 'gmsh' directement.
 * - améliorer la lecture parallèle en évitant que tous les sous-domaines
 *   lisent le fichier (même si seul le sous-domaine 0 alloue les entités)
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur de fichiers de maillage au format `msh`.
 *
 * Le format `msh` est celui utilisé par la bibliothèque 
 * [gmsh](https://gmsh.info/).
 *
 * Le lecteur supporte les versions `2.0` et `4.1` de ce format.
 *
 * Seules une partie des fonctionnalités du format sont supportées:
 *
 * - seuls les éléments d'ordre 1 sont supportés.
 * - les coordonnées paramétriques ne sont pas supportées
 * - seules les sections `$Nodes` et `$Entities` sont lues
 */
class MshParallelMeshReader
: public TraceAccessor
, public IMshMeshReader
{
 public:

  using eReturnType = typename IMeshReader::eReturnType;

  /*!
   * \brief Infos d'un bloc pour $Elements pour la version 4.
   *
   * Dans cette version, les éléments d'un bloc sont tous
   * de même type (par exemple que des IT_Quad4 ou IT_Triangle3.
   *
   * Si on a \a nb_entity, alors uids.size()==nb_entity
   * et connectivity.size()==nb_entity*item_nb_node
   */
  struct MeshV4ElementsBlock
  {
    Int32 index = 0; //!< Index du bloc dans la liste
    Int32 nb_entity = 0; //!< Nombre d'entités du bloc
    Integer item_type = -1; //!< Type Arcane de l'entité
    Int32 dimension = -1; //!< Dimension de l'entité
    Int32 item_nb_node = 0; //!< Nombre de noeuds de l'entité.
    Int32 entity_tag = -1;
    UniqueArray<Int64> uids;
    UniqueArray<Int64> connectivity;
  };

  /*!
   * \brief Infos sur un nom physique.
   *
   */
  struct MeshPhysicalName
  {
    MeshPhysicalName(Int32 _dimension, Int32 _tag, const String& _name)
    : dimension(_dimension)
    , tag(_tag)
    , name(_name)
    {}
    MeshPhysicalName() = default;
    bool isNull() const { return dimension == (-1); }
    Int32 dimension = -1;
    Int32 tag = -1;
    String name;
  };

  /*!
   * \brief Infos du bloc '$PhysicalNames'.
   */
  struct MeshPhysicalNameList
  {
    MeshPhysicalNameList()
    : m_physical_names(4)
    {}
    void add(Int32 dimension, Int32 tag, String name)
    {
      m_physical_names[dimension].add(MeshPhysicalName{ dimension, tag, name });
    }
    MeshPhysicalName find(Int32 dimension, Int32 tag) const
    {
      for (auto& x : m_physical_names[dimension])
        if (x.tag == tag)
          return x;
      return {};
    }

   private:

    UniqueArray<UniqueArray<MeshPhysicalName>> m_physical_names;
  };

  //! Infos pour les entités 0D
  struct MeshV4EntitiesNodes
  {
    MeshV4EntitiesNodes(Int32 _tag, Int32 _physical_tag)
    : tag(_tag)
    , physical_tag(_physical_tag)
    {}
    Int32 tag;
    Int32 physical_tag;
  };

  //! Infos pour les entités 1D, 2D et 3D
  struct MeshV4EntitiesWithNodes
  {
    MeshV4EntitiesWithNodes(Int32 _dim, Int32 _tag, Int32 _physical_tag)
    : dimension(_dim)
    , tag(_tag)
    , physical_tag(_physical_tag)
    {}
    Int32 dimension;
    Int32 tag;
    Int32 physical_tag;
  };

  struct MeshInfo
  {
   public:

    MeshInfo()
    : node_coords_map(5000, true)
    {}

   public:

    MeshV4EntitiesWithNodes* findEntities(Int32 dimension, Int32 tag)
    {
      for (auto& x : entities_with_nodes_list[dimension - 1])
        if (x.tag == tag)
          return &x;
      return nullptr;
    }

    MeshV4EntitiesNodes* findNodeEntities(Int32 tag)
    {
      for (auto& x : entities_nodes_list)
        if (x.tag == tag)
          return &x;
      return nullptr;
    }

   public:

    Integer nb_elements = 0;
    Integer nb_cell_node = 0;
    UniqueArray<Int32> cells_nb_node;
    UniqueArray<Int32> cells_type;
    UniqueArray<Int64> cells_uid;
    UniqueArray<Int64> cells_connectivity;
    UniqueArray<Real3> node_coords;
    HashTableMapT<Int64, Real3> node_coords_map;
    MeshPhysicalNameList physical_name_list;
    UniqueArray<MeshV4EntitiesNodes> entities_nodes_list;
    UniqueArray<MeshV4EntitiesWithNodes> entities_with_nodes_list[3];
    UniqueArray<MeshV4ElementsBlock> blocks;
  };

 public:

  explicit MshParallelMeshReader(ITraceMng* tm)
  : TraceAccessor(tm)
  {}

  eReturnType readMeshFromMshFile(IMesh* mesh, const String& file_name) override;

 private:

  IMesh* m_mesh = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  Int32 m_master_io_rank = A_NULL_RANK;
  bool m_is_parallel = false;
  Ref<IosFile> m_ios_file; // nullptr sauf pour le rang maitre.

 private:

  eReturnType _readNodesFromAsciiMshV4File(IosFile& ios_file, MeshInfo& mesh_info);
  Integer _readElementsFromAsciiMshV4File(IosFile* ios_file, MeshInfo& mesh_info);
  eReturnType _readMeshFromNewMshFile(IosFile* iso_file);
  void _allocateCells(MeshInfo& mesh_info);
  void _allocateGroups(MeshInfo& mesh_info);
  void _addFaceGroup(MeshV4ElementsBlock& block, const String& group_name);
  void _addCellGroup(MeshV4ElementsBlock& block, const String& group_name);
  void _addNodeGroup(MeshV4ElementsBlock& block, const String& group_name);
  Integer _switchMshType(Integer, Integer&);
  void _readPhysicalNames(IosFile* ios_file, MeshInfo& mesh_info);
  void _readEntitiesV4(IosFile* ios_file, MeshInfo& mesh_info);
  String _getNextLineAndBroadcast();
  Int32 _getIntegerAndBroadcast();
  void _getIntegersAndBroadcast(ArrayView<Int32> values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  /*!
   * \brief Calcule le début et le nombre d'éléments pour un interval.
   *
   * Calcul le début et le nombre d'éléments du index-ème interval
   * si le tableau contient \a size élément et qu'on veut \a nb_interval.
   */
  inline std::pair<Int32, Int32>
  _interval(Int32 index, Int32 nb_interval, Int32 size)
  {
    Int32 isize = size / nb_interval;
    Int32 ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index + 1) == nb_interval)
      isize = size - ibegin;
    return { ibegin, isize };
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MshParallelMeshReader::
_getNextLineAndBroadcast()
{
  IosFile* f = m_ios_file.get();
  String s;
  if (f)
    s = f->getNextLine();
  if (m_is_parallel) {
    if (f)
      info() << "BroadcastNextLine: " << s;
    m_parallel_mng->broadcastString(s, m_master_io_rank);
  }
  info() << "GetNextLine: " << s;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MshParallelMeshReader::
_getIntegerAndBroadcast()
{
  IosFile* f = m_ios_file.get();
  FixedArray<Int32, 1> v;
  if (f)
    v[0] = f->getInteger();
  if (m_is_parallel) {
    m_parallel_mng->broadcast(v.view(), m_master_io_rank);
  }
  return v[0];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_getIntegersAndBroadcast(ArrayView<Int32> values)
{
  IosFile* f = m_ios_file.get();
  if (f)
    for (Int32& v : values)
      v = f->getInteger();
  if (m_is_parallel)
    m_parallel_mng->broadcast(values, m_master_io_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MshParallelMeshReader::
_switchMshType(Integer mshElemType, Integer& nNodes)
{
  switch (mshElemType) {
  case (IT_NullType): // used to decode IT_NullType: IT_HemiHexa7|IT_Line9
    switch (nNodes) {
    case (7):
      return IT_HemiHexa7;
    default:
      info() << "Could not decode IT_NullType with nNodes=" << nNodes;
      throw IOException("_switchMshType", "Could not decode IT_NullType with nNodes");
    }
    break;
  case (MSH_PNT):
    nNodes = 1;
    return IT_Vertex; //printf("1-node point");
  case (MSH_LIN_2):
    nNodes = 2;
    return IT_Line2; //printf("2-node line");
  case (MSH_TRI_3):
    nNodes = 3;
    return IT_Triangle3; //printf("3-node triangle");
  case (MSH_QUA_4):
    nNodes = 4;
    return IT_Quad4; //printf("4-node quadrangle");
  case (MSH_TET_4):
    nNodes = 4;
    return IT_Tetraedron4; //printf("4-node tetrahedron");
  case (MSH_HEX_8):
    nNodes = 8;
    return IT_Hexaedron8; //printf("8-node hexahedron");
  case (MSH_PRI_6):
    nNodes = 6;
    return IT_Pentaedron6; //printf("6-node prism");
  case (MSH_PYR_5):
    nNodes = 5;
    return IT_Pyramid5; //printf("5-node pyramid");
  case (MSH_TRI_10):
    nNodes = 10;
    return IT_Heptaedron10; //printf("10-node second order tetrahedron")
  case (MSH_TRI_12):
    nNodes = 12;
    return IT_Octaedron12; //printf("Unknown MSH_TRI_12");

  case (MSH_TRI_6):
  case (MSH_QUA_9):
  case (MSH_HEX_27):
  case (MSH_PRI_18):
  case (MSH_PYR_14):
  case (MSH_QUA_8):
  case (MSH_HEX_20):
  case (MSH_PRI_15):
  case (MSH_PYR_13):
  case (MSH_TRI_9):
  case (MSH_TRI_15):
  case (MSH_TRI_15I):
  case (MSH_TRI_21):
  default:
    ARCANE_THROW(NotSupportedException, "Unknown GMSH element type '{0}'", mshElemType);
  }
  return IT_NullType;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *
 * \code
 $Nodes
 * numEntityBlocks(size_t) numNodes(size_t)
 *   minNodeTag(size_t) maxNodeTag(size_t)
 * entityDim(int) entityTag(int) parametric(int; 0 or 1)
 *   numNodesInBlock(size_t)
 *   nodeTag(size_t)
 *   ...
 *   x(double) y(double) z(double)
 *      < u(double; if parametric and entityDim >= 1) >
 *      < v(double; if parametric and entityDim >= 2) >
 *      < w(double; if parametric and entityDim == 3) >
 *   ...
 * ...
 *$EndNodes
 * \endcode
 */
IMeshReader::eReturnType MshParallelMeshReader::
_readNodesFromAsciiMshV4File(IosFile& ios_file, MeshInfo& mesh_info)
{
  // Première ligne du fichier
  Integer nb_entity = ios_file.getInteger();
  Integer total_nb_node = ios_file.getInteger();
  Integer min_node_tag = ios_file.getInteger();
  Integer max_node_tag = ios_file.getInteger();
  ios_file.getNextLine(); // Skip current \n\r
  if (total_nb_node < 0)
    ARCANE_THROW(IOException, "Invalid number of nodes : '{0}'", total_nb_node);
  info() << "[Nodes] nb_entity=" << nb_entity
         << " total_nb_node=" << total_nb_node
         << " min_tag=" << min_node_tag
         << " max_tag=" << max_node_tag;

  UniqueArray<Int32> nodes_uids;
  for (Integer i_entity = 0; i_entity < nb_entity; ++i_entity) {
    // Dimension de l'entité (pas utile)
    [[maybe_unused]] Integer entity_dim = ios_file.getInteger();
    // Tag de l'entité (pas utile)
    [[maybe_unused]] Integer entity_tag = ios_file.getInteger();
    Integer parametric_coordinates = ios_file.getInteger();
    Integer nb_node2 = ios_file.getInteger();
    ios_file.getNextLine();

    info(4) << "[Nodes] index=" << i_entity << " entity_dim=" << entity_dim << " entity_tag=" << entity_tag
            << " parametric=" << parametric_coordinates
            << " nb_node2=" << nb_node2;
    if (parametric_coordinates != 0)
      ARCANE_THROW(NotSupportedException, "Only 'parametric coordinates' value of '0' is supported (current={0})", parametric_coordinates);
    // Il est possible que le nombre de noeuds soit 0.
    // Dans ce cas, il faut directement passer à la ligne suivante
    if (nb_node2 == 0)
      continue;
    nodes_uids.resize(nb_node2);
    for (Integer i = 0; i < nb_node2; ++i) {
      // Conserve le uniqueId() du noeuds.
      nodes_uids[i] = ios_file.getInteger();
      //info() << "I=" << i << " ID=" << nodes_uids[i];
    }
    for (Integer i = 0; i < nb_node2; ++i) {
      Real nx = ios_file.getReal();
      Real ny = ios_file.getReal();
      Real nz = ios_file.getReal();
      Real3 xyz(nx, ny, nz);
      mesh_info.node_coords_map.add(nodes_uids[i], xyz);
      //info() << "I=" << i << " ID=" << nodes_uids[i] << " COORD=" << xyz;
    }
    ios_file.getNextLine(); // Skip current \n\r
  }
  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des éléments (mailles,faces,...)
 *
 * Dans la version 4, les éléments sont rangés par genre (eItemKind)
 *
 * \code
 *$Elements
 *  numEntityBlocks(size_t) numElements(size_t)
 *    minElementTag(size_t) maxElementTag(size_t)
 *  entityDim(int) entityTag(int) elementType(int; see below)
 *    numElementsInBlock(size_t)
 *    elementTag(size_t) nodeTag(size_t) ...
 *    ...
 *  ...
 *$EndElements
 * \endcode
 */
Integer MshParallelMeshReader::
_readElementsFromAsciiMshV4File(IosFile* ios_file, MeshInfo& mesh_info)
{
  IParallelMng* pm = m_parallel_mng;

  FixedArray<Int32, 4> elements_info;
  _getIntegersAndBroadcast(elements_info.view());

  Integer nb_block = elements_info[0];
  Integer number_of_elements = elements_info[1];
  Integer min_element_tag = elements_info[2];
  Integer max_element_tag = elements_info[3];

  if (ios_file)
    ios_file->getNextLine(); // Skip current \n\r

  info() << "[Elements] nb_block=" << nb_block
         << " nb_elements=" << number_of_elements
         << " min_element_tag=" << min_element_tag
         << " max_element_tag=" << max_element_tag;

  if (number_of_elements < 0)
    ARCANE_THROW(IOException, "Invalid number of elements: {0}", number_of_elements);

  UniqueArray<MeshV4ElementsBlock>& blocks = mesh_info.blocks;
  blocks.resize(nb_block);

  {
    // Numérote les blocs (pour le débug)
    Integer index = 0;
    for (MeshV4ElementsBlock& block : blocks) {
      block.index = index;
      ++index;
    }
  }
  for (MeshV4ElementsBlock& block : blocks) {
    FixedArray<Int32, 4> block_info;
    _getIntegersAndBroadcast(block_info.view());

    Integer entity_dim = block_info[0];
    Integer entity_tag = block_info[1];
    Integer entity_type = block_info[2];
    Integer nb_entity_in_block = block_info[3];

    Integer item_nb_node = 0;
    Integer item_type = _switchMshType(entity_type, item_nb_node);

    info(4) << "[Elements] index=" << block.index << " entity_dim=" << entity_dim
            << " entity_tag=" << entity_tag
            << " entity_type=" << entity_type << " nb_in_block=" << nb_entity_in_block
            << " item_type=" << item_type << " item_nb_node=" << item_nb_node;

    block.nb_entity = nb_entity_in_block;
    block.item_type = item_type;
    block.item_nb_node = item_nb_node;
    block.dimension = entity_dim;
    block.entity_tag = entity_tag;

    Int64 nb_uid = 0;
    Int64 nb_connectivity = 0;
    if (entity_type == MSH_PNT) {
      // Si le type est un point, le traitement semble
      // un peu particulier. Il y a dans ce cas
      // deux entiers dans la ligne suivante:
      // - un entier qui ne semble pas être utilisé
      // - le numéro unique du noeud qui nous intéresse
      if (ios_file) {
        [[maybe_unused]] Int64 unused_id = ios_file->getInt64();
        Int64 item_unique_id = ios_file->getInt64();
        info(4) << "Adding unique node uid=" << item_unique_id;
        block.uids.add(item_unique_id);
      }
      nb_uid = 1;
      nb_connectivity = 0;
    }
    else {
      if (ios_file) {
        info(4) << "Reading block nb_entity=" << nb_entity_in_block << " item_nb_node=" << item_nb_node;
        for (Integer i = 0; i < nb_entity_in_block; ++i) {
          Int64 item_unique_id = ios_file->getInt64();
          block.uids.add(item_unique_id);
          for (Integer j = 0; j < item_nb_node; ++j)
            block.connectivity.add(ios_file->getInt64());
        }
      }
      nb_uid = nb_entity_in_block;
      nb_connectivity = nb_uid * item_nb_node;
    }
    // Envoie le tableau aux autres rangs
    if (m_is_parallel) {
      if (!pm->isMasterIO()) {
        block.uids.resize(nb_uid);
        block.connectivity.resize(nb_connectivity);
      }
      pm->broadcast(block.uids, m_master_io_rank);
      pm->broadcast(block.connectivity, m_master_io_rank);
    }
    if (ios_file)
      ios_file->getNextLine(); // Skip current \n\r
  }

  // Maintenant qu'on a tout les blocs, la dimension du maillage est
  // la plus grande dimension des blocs
  Integer mesh_dimension = -1;
  for (MeshV4ElementsBlock& block : blocks)
    mesh_dimension = math::max(mesh_dimension, block.dimension);
  if (mesh_dimension < 0)
    ARCANE_FATAL("Invalid computed mesh dimension '{0}'", mesh_dimension);
  if (mesh_dimension != 2 && mesh_dimension != 3)
    ARCANE_THROW(NotSupportedException, "mesh dimension '{0}'. Only 2D or 3D meshes are supported", mesh_dimension);
  info() << "Computed mesh dimension = " << mesh_dimension;

  const Int32 my_rank = m_parallel_mng->commRank();
  const Int32 nb_rank = m_parallel_mng->commSize();

  // On ne conserve que les blocs de notre dimension
  // pour créér les mailles. On divise chaque bloc
  // en N partie (avec N le nombre de rangs MPI) et
  // chaque rang ne conserve qu'une partie. Ainsi
  // chaque sous-domaine aura une partie du maillage
  // afin de garantir un équilibrage sur le nombre
  // de mailles.
  for (MeshV4ElementsBlock& block : blocks) {
    if (block.dimension != mesh_dimension)
      continue;

    Integer item_type = block.item_type;
    Integer item_nb_node = block.item_nb_node;

    auto [first_index, nb_item] = _interval(my_rank, nb_rank, block.nb_entity);
    if (m_is_parallel) {
      UniqueArray<Int64> new_uids(block.uids.span().subSpan(first_index, nb_item));
      UniqueArray<Int64> new_connectivity(block.connectivity.span().subSpan(first_index * item_nb_node, nb_item * item_nb_node));
      block.uids = new_uids;
      block.connectivity = new_connectivity;
      block.nb_entity = nb_item;
    }

    info(4) << "Keeping block index=" << block.index << " nb_entity=" << block.nb_entity
            << " first_index=" << first_index << " nb_item=" << nb_item;

    for (Integer i = 0; i < block.nb_entity; ++i) {
      mesh_info.cells_type.add(item_type);
      mesh_info.cells_nb_node.add(item_nb_node);
      mesh_info.cells_uid.add(block.uids[i]);
      auto v = block.connectivity.subView(i * item_nb_node, item_nb_node);
      mesh_info.cells_connectivity.addRange(v);
    }
  }

  return mesh_dimension;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_allocateCells(MeshInfo& mesh_info)
{
  IMesh* mesh = m_mesh;
  Integer nb_elements = mesh_info.cells_type.size();
  info() << "nb_of_elements=cells_type.size()=" << nb_elements;
  Integer nb_cell_node = mesh_info.cells_connectivity.size();
  info() << "nb_cell_node=cells_connectivity.size()=" << nb_cell_node;

  // Création des mailles
  info() << "Building cells, nb_cell=" << nb_elements << " nb_cell_node=" << nb_cell_node;
  // Infos pour la création des mailles
  // par maille: 1 pour son unique id,
  //             1 pour son type,
  //             1 pour chaque noeud
  UniqueArray<Int64> cells_infos;
  Integer connectivity_index = 0;
  UniqueArray<Real3> local_coords;
  for (Integer i = 0; i < nb_elements; ++i) {
    Integer current_cell_nb_node = mesh_info.cells_nb_node[i];
    Integer cell_type = mesh_info.cells_type[i];
    Int64 cell_uid = mesh_info.cells_uid[i];
    cells_infos.add(cell_type);
    cells_infos.add(cell_uid); //cell_unique_id

    ArrayView<Int64> local_info(current_cell_nb_node, &mesh_info.cells_connectivity[connectivity_index]);
    cells_infos.addRange(local_info);
    connectivity_index += current_cell_nb_node;
  }

  IPrimaryMesh* pmesh = mesh->toPrimaryMesh();
  info() << "## Allocating ##";
  pmesh->allocateCells(nb_elements, cells_infos, false);
  info() << "## Ending ##";
  pmesh->endAllocate();
  info() << "## Done ##";

  // Positionne les coordonnées
  {
    VariableNodeReal3& nodes_coord_var(pmesh->nodesCoordinates());
    bool has_map = mesh_info.node_coords.empty();
    if (has_map) {
      ENUMERATE_NODE (i, mesh->ownNodes()) {
        Node node = *i;
        auto* data = mesh_info.node_coords_map.lookup(node.uniqueId().asInt64());
        if (data)
          nodes_coord_var[node] = data->value();
      }
      nodes_coord_var.synchronize();
    }
    else {
      ENUMERATE_NODE (node, mesh->allNodes()) {
        nodes_coord_var[node] = mesh_info.node_coords.item(node->uniqueId().asInt32());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_allocateGroups(MeshInfo& mesh_info)
{
  IMesh* mesh = m_mesh;
  Int32 mesh_dim = mesh->dimension();
  Int32 face_dim = mesh_dim - 1;
  for (MeshV4ElementsBlock& block : mesh_info.blocks) {
    Int32 block_index = block.index;
    Int32 block_dim = block.dimension;
    // On alloue un groupe s'il a un nom physique associé.
    // Pour cela, il faut déjà qu'il soit associé à une entité.
    Int32 block_entity_tag = block.entity_tag;
    if (block_entity_tag < 0) {
      info(5) << "[Groups] Skipping block index=" << block_index << " because it has no entity";
      continue;
    }
    MeshPhysicalName physical_name;
    if (block_dim == 0) {
      MeshV4EntitiesNodes* entity = mesh_info.findNodeEntities(block_entity_tag);
      if (!entity) {
        info(5) << "[Groups] Skipping block index=" << block_index
                << " because entity tag is invalid";
        continue;
      }
      Int32 entity_physical_tag = entity->physical_tag;
      physical_name = mesh_info.physical_name_list.find(block_dim, entity_physical_tag);
    }
    else {
      MeshV4EntitiesWithNodes* entity = mesh_info.findEntities(block_dim, block_entity_tag);
      if (!entity) {
        info(5) << "[Groups] Skipping block index=" << block_index
                << " because entity tag is invalid";
        continue;
      }
      Int32 entity_physical_tag = entity->physical_tag;
      physical_name = mesh_info.physical_name_list.find(block_dim, entity_physical_tag);
    }
    if (physical_name.isNull()) {
      info(5) << "[Groups] Skipping block index=" << block_index
              << " because entity physical tag is invalid";
      continue;
    }
    info(4) << "[Groups] Block index=" << block_index << " dim=" << block_dim
            << " name='" << physical_name.name << "'";
    if (block_dim == mesh_dim) {
      _addCellGroup(block, physical_name.name);
    }
    else if (block_dim == face_dim) {
      _addFaceGroup(block, physical_name.name);
    }
    else {
      _addNodeGroup(block, physical_name.name);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_addFaceGroup(MeshV4ElementsBlock& block, const String& group_name)
{
  IMesh* mesh = m_mesh;
  const Int32 nb_entity = block.nb_entity;

  // Il peut y avoir plusieurs blocs pour le même groupe.
  // On récupère le groupe s'il existe déjà.
  FaceGroup face_group = mesh->faceFamily()->findGroup(group_name, true);

  UniqueArray<Int32> faces_id; // Numéro de la face dans le maillage \a mesh
  faces_id.reserve(nb_entity);

  const Int32 item_nb_node = block.item_nb_node;
  const Int32 face_nb_node = nb_entity * item_nb_node;

  UniqueArray<Int64> faces_first_node_unique_id(nb_entity);
  UniqueArray<Int32> faces_first_node_local_id(nb_entity);
  UniqueArray<Int64> faces_nodes_unique_id(face_nb_node);
  Integer faces_nodes_unique_id_index = 0;

  UniqueArray<Int64> orig_nodes_id(item_nb_node);
  UniqueArray<Integer> face_nodes_index(item_nb_node);

  IItemFamily* node_family = mesh->nodeFamily();
  NodeInfoListView mesh_nodes(node_family);

  // Réordonne les identifiants des faces retrouver la face dans le maillage
  for (Integer i_face = 0; i_face < nb_entity; ++i_face) {
    for (Integer z = 0; z < item_nb_node; ++z)
      orig_nodes_id[z] = block.connectivity[faces_nodes_unique_id_index + z];

    mesh_utils::reorderNodesOfFace2(orig_nodes_id, face_nodes_index);
    for (Integer z = 0; z < item_nb_node; ++z)
      faces_nodes_unique_id[faces_nodes_unique_id_index + z] = orig_nodes_id[face_nodes_index[z]];
    faces_first_node_unique_id[i_face] = orig_nodes_id[face_nodes_index[0]];
    faces_nodes_unique_id_index += item_nb_node;
  }

  node_family->itemsUniqueIdToLocalId(faces_first_node_local_id, faces_first_node_unique_id, false);

  faces_nodes_unique_id_index = 0;
  for (Integer i_face = 0; i_face < nb_entity; ++i_face) {
    const Integer n = item_nb_node;
    Int32 face_first_node_lid = faces_first_node_local_id[i_face];
    if (face_first_node_lid != NULL_ITEM_LOCAL_ID) {
      Int64ConstArrayView face_nodes_id(item_nb_node, &faces_nodes_unique_id[faces_nodes_unique_id_index]);
      Node current_node(mesh_nodes[faces_first_node_local_id[i_face]]);
      Face face = mesh_utils::getFaceFromNodesUnique(current_node, face_nodes_id);

      // En parallèle, il est possible que la face ne soit pas dans notre sous-domaine
      // même si un de ses noeud l'est
      if (face.null()) {
        if (!m_is_parallel) {
          OStringStream ostr;
          ostr() << "(Nodes:";
          for (Integer z = 0; z < n; ++z)
            ostr() << ' ' << face_nodes_id[z];
          ostr() << " - " << current_node.localId() << ")";
          ARCANE_FATAL("INTERNAL: MeshMeshReader face index={0} with nodes '{1}' is not in node/face connectivity",
                       i_face, ostr.str());
        }
      }
      else
        faces_id.add(face.localId());
    }

    faces_nodes_unique_id_index += n;
  }
  info(4) << "Adding " << faces_id.size() << " faces from block index=" << block.index
          << " to group '" << face_group.name() << "'";
  face_group.addItems(faces_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_addCellGroup(MeshV4ElementsBlock& block, const String& group_name)
{
  IMesh* mesh = m_mesh;
  const Int32 nb_entity = block.nb_entity;

  // Il peut y avoir plusieurs blocs pour le même groupe.
  // On récupère le groupe s'il existe déjà.
  IItemFamily* cell_family = mesh->cellFamily();
  CellGroup cell_group = cell_family->findGroup(group_name, true);

  UniqueArray<Int32> cells_id(nb_entity); // Numéro de la face dans le maillage \a mesh

  cell_family->itemsUniqueIdToLocalId(cells_id, block.uids);

  info(4) << "Adding " << cells_id.size() << " cells from block index=" << block.index
          << " to group '" << cell_group.name() << "'";
  cell_group.addItems(cells_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_addNodeGroup(MeshV4ElementsBlock& block, const String& group_name)
{
  IMesh* mesh = m_mesh;
  const Int32 nb_entity = block.nb_entity;

  // Il peut y avoir plusieurs blocs pour le même groupe.
  // On récupère le groupe s'il existe déjà.
  IItemFamily* node_family = mesh->nodeFamily();
  NodeGroup node_group = node_family->findGroup(group_name, true);

  UniqueArray<Int32> nodes_id(nb_entity);

  node_family->itemsUniqueIdToLocalId(nodes_id, block.uids, false);

  // En parallèle, il est possible que certains noeuds du groupe ne soient
  // pas dans notre sous-domaine. Il faut les filtrer.
  if (m_is_parallel) {
    auto nodes_begin = nodes_id.begin();
    Int64 new_size = std::remove(nodes_begin, nodes_id.end(), NULL_ITEM_LOCAL_ID) - nodes_begin;
    nodes_id.resize(new_size);
  }

  info(4) << "Adding " << nodes_id.size() << " nodes from block index=" << block.index
          << " to group '" << node_group.name() << "'" << " nb_entity=" << nb_entity;

  if (nb_entity < 10) {
    info(4) << "Nodes UIDS=" << block.uids;
    info(4) << "Nodes LIDS=" << nodes_id;
  }
  node_group.addItems(nodes_id);

  if (nb_entity < 10) {
    VariableNodeReal3& coords(mesh->nodesCoordinates());
    ENUMERATE_ (Node, inode, node_group) {
      info(4) << "Node id=" << ItemPrinter(*inode) << " coord=" << coords[inode];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * $PhysicalNames // same as MSH version 2
 *    numPhysicalNames(ASCII int)
 *    dimension(ASCII int) physicalTag(ASCII int) "name"(127 characters max)
 *    ...
 * $EndPhysicalNames
 */
void MshParallelMeshReader::
_readPhysicalNames(IosFile* ios_file, MeshInfo& mesh_info)
{
  String quote_mark = "\"";
  Int32 nb_name = _getIntegerAndBroadcast();
  info() << "nb_physical_name=" << nb_name;
  if (ios_file)
    ios_file->getNextLine();
  for (Int32 i = 0; i < nb_name; ++i) {
    Int32 dim = _getIntegerAndBroadcast();
    Int32 tag = _getIntegerAndBroadcast();
    String s = _getNextLineAndBroadcast();
    if (dim < 0 || dim > 3)
      ARCANE_FATAL("Invalid value for physical name dimension dim={0}", dim);
    // Les noms des groupes peuvent commencer par des espaces et contiennent
    // des guillemets qu'il faut supprimer.
    s = String::collapseWhiteSpace(s);
    if (s.startsWith(quote_mark))
      s = s.substring(1);
    if (s.endsWith(quote_mark))
      s = s.substring(0, s.length() - 1);
    mesh_info.physical_name_list.add(dim, tag, s);
    info(4) << "[PhysicalName] index=" << i << " dim=" << dim << " tag=" << tag << " name='" << s << "'";
  }
  String s = _getNextLineAndBroadcast();
  if (s != "$EndPhysicalNames")
    ARCANE_FATAL("found '{0}' and expected '$EndPhysicalNames'", s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Lecture des entités.
 *
 * Le format est:
 *
 * \verbatim
 *    $Entities
 *      numPoints(size_t) numCurves(size_t)
 *        numSurfaces(size_t) numVolumes(size_t)
 *      pointTag(int) X(double) Y(double) Z(double)
 *        numPhysicalTags(size_t) physicalTag(int) ...
 *      ...
 *      curveTag(int) minX(double) minY(double) minZ(double)
 *        maxX(double) maxY(double) maxZ(double)
 *        numPhysicalTags(size_t) physicalTag(int) ...
 *        numBoundingPoints(size_t) pointTag(int) ...
 *      ...
 *      surfaceTag(int) minX(double) minY(double) minZ(double)
 *        maxX(double) maxY(double) maxZ(double)
 *        numPhysicalTags(size_t) physicalTag(int) ...
 *        numBoundingCurves(size_t) curveTag(int) ...
 *      ...
 *      volumeTag(int) minX(double) minY(double) minZ(double)
 *        maxX(double) maxY(double) maxZ(double)
 *        numPhysicalTags(size_t) physicalTag(int) ...
 *        numBoundngSurfaces(size_t) surfaceTag(int) ...
 *      ...
 *    $EndEntities
 * \endverbatim
 */
void MshParallelMeshReader::
_readEntitiesV4(IosFile* ios_file, MeshInfo& mesh_info)
{
  FixedArray<Int32, 4> nb_dim_item;
  if (ios_file) {
    nb_dim_item[0] = ios_file->getInteger();
    nb_dim_item[1] = ios_file->getInteger();
    nb_dim_item[2] = ios_file->getInteger();
    nb_dim_item[3] = ios_file->getInteger();
  }
  m_parallel_mng->broadcast(nb_dim_item.view(), m_master_io_rank);

  info(4) << "[Entities] nb_0d=" << nb_dim_item[0] << " nb_1d=" << nb_dim_item[1]
          << " nb_2d=" << nb_dim_item[2] << " nb_3d=" << nb_dim_item[3];
  // Après le format, on peut avoir les entités mais cela est optionnel
  // Si elles sont présentes, on lit le fichier jusqu'à la fin de cette section.
  if (ios_file)
    ios_file->getNextLine();
  for (Int32 i = 0; i < nb_dim_item[0]; ++i) {
    FixedArray<Int32, 2> tag_info;
    if (ios_file) {
      Int32 tag = ios_file->getInteger();
      Real x = ios_file->getReal();
      Real y = ios_file->getReal();
      Real z = ios_file->getReal();
      Int32 num_physical_tag = ios_file->getInteger();
      if (num_physical_tag > 1)
        ARCANE_FATAL("NotImplemented numPhysicalTag>1 (n={0})", num_physical_tag);
      Int32 physical_tag = -1;
      if (num_physical_tag == 1)
        physical_tag = ios_file->getInteger();
      info(4) << "[Entities] point tag=" << tag << " x=" << x << " y=" << y << " z=" << z << " phys_tag=" << physical_tag;

      tag_info[0] = tag;
      tag_info[1] = physical_tag;
    }
    m_parallel_mng->broadcast(tag_info.view(), m_master_io_rank);
    mesh_info.entities_nodes_list.add(MeshV4EntitiesNodes(tag_info[0], tag_info[1]));
    if (ios_file)
      ios_file->getNextLine();
  }

  for (Int32 dim = 1; dim <= 3; ++dim) {
    for (Int32 i = 0; i < nb_dim_item[dim]; ++i) {
      FixedArray<Int32, 3> dim_and_tag_info;
      if (ios_file) {
        Int32 tag = ios_file->getInteger();
        Real min_x = ios_file->getReal();
        Real min_y = ios_file->getReal();
        Real min_z = ios_file->getReal();
        Real max_x = ios_file->getReal();
        Real max_y = ios_file->getReal();
        Real max_z = ios_file->getReal();
        Int32 num_physical_tag = ios_file->getInteger();
        if (num_physical_tag > 1)
          ARCANE_FATAL("NotImplemented numPhysicalTag>1 (n={0})", num_physical_tag);
        Int32 physical_tag = -1;
        if (num_physical_tag == 1)
          physical_tag = ios_file->getInteger();
        Int32 num_bounding_group = ios_file->getInteger();
        for (Int32 k = 0; k < num_bounding_group; ++k) {
          [[maybe_unused]] Int32 group_tag = ios_file->getInteger();
        }
        info(4) << "[Entities] dim=" << dim << " tag=" << tag
                << " min_x=" << min_x << " min_y=" << min_y << " min_z=" << min_z
                << " max_x=" << max_x << " max_y=" << max_y << " max_z=" << max_z
                << " phys_tag=" << physical_tag;
        dim_and_tag_info[0] = dim;
        dim_and_tag_info[1] = tag;
        dim_and_tag_info[2] = physical_tag;
      }
      m_parallel_mng->broadcast(dim_and_tag_info.view(), m_master_io_rank);
      {
        Int32 dim = dim_and_tag_info[0];
        Int32 tag = dim_and_tag_info[1];
        Int32 physical_tag = dim_and_tag_info[2];
        mesh_info.entities_with_nodes_list[dim - 1].add(MeshV4EntitiesWithNodes(dim, tag, physical_tag));
      }
      if (ios_file)
        ios_file->getNextLine();
    }
  }
  String s = _getNextLineAndBroadcast();
  if (s != "$EndEntities")
    ARCANE_FATAL("found '{0}' and expected '$EndEntities'", s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *
 The version 2.0 of the '.msh' file format is Gmsh's new native mesh file format. It is very
 similar to the old one (see Section 9.1.1 [Version 1.0], page 139), but is more general: it
 contains information about itself and allows to associate an arbitrary number of integer tags
 with each element. Ialso exists in both ASCII and binary form.
 The '.msh' file format, version 2.0, is divided in three main sections, defining the file
 format ($MeshFormat-$EndMeshFormat), the nodes ($Nodes-$EndNodes) and the elements
 ($Elements-$EndElements) in the mesh:
 /code
 * $MeshFormat
 * 2.0 file-type data-size
 *               (one-binary) is an integer of value 1 written in binary form.
 *               This integer is used for detecting if the computer
 *               on which the binary file was written and the computer
 *               on which the file is read are of the same type
 *               (little or big endian).
 *
 * $EndMeshFormat
 * $Nodes
 * number-of-nodes
 * node-number x-coord y-coord z-coord
 * ...
 * $EndNodes
 * $Elements
 * number-of-elements
 * elm-number elm-type number-of-tags < tag > ... node-number-list
 * ...	
 * $EndElements
 \endcode
*/
IMeshReader::eReturnType MshParallelMeshReader::
_readMeshFromNewMshFile(IosFile* ios_file)
{
  IMesh* mesh = m_mesh;
  const char* func_name = "MshParallelMeshReader::_readMeshFromNewMshFile()";
  info() << "[_readMeshFromNewMshFile] Reading 'msh' file in parallel";
  MeshInfo mesh_info;
  const int MSH_BINARY_TYPE = 1;

  if (ios_file) {
    Real version = ios_file->getReal();
    if (version != 4.1)
      ARCANE_THROW(IOException, "Wrong msh file version '{0}'. Only version '4.1' is supported in parallel", version);
    Integer file_type = ios_file->getInteger(); // is an integer equal to 0 in the ASCII file format, equal to 1 for the binary format
    if (file_type == MSH_BINARY_TYPE)
      ARCANE_THROW(IOException, "Binary mode is not supported!");

    Integer data_size = ios_file->getInteger(); // is an integer equal to the size of the floating point numbers used in the file
    ARCANE_UNUSED(data_size);

    ios_file->getNextLine(); // Skip current \n\r

    // $EndMeshFormat
    if (!ios_file->lookForString("$EndMeshFormat"))
      ARCANE_THROW(IOException, "$EndMeshFormat not found");
  }

  // TODO: Les différentes sections ($Nodes, $Entitites, ...) peuvent
  // être dans n'importe quel ordre (à part $Nodes qui doit être avant $Elements)
  // Faire un méthode qui gère cela.

  String next_line = _getNextLineAndBroadcast();
  // Noms des groupes
  if (next_line == "$PhysicalNames") {
    _readPhysicalNames(ios_file, mesh_info);
    next_line = _getNextLineAndBroadcast();
  }

  // Après le format, on peut avoir les entités mais cela est optionnel
  // Si elles sont présentes, on lit le fichier jusqu'à la fin de cette section.
  if (next_line == "$Entities") {
    _readEntitiesV4(ios_file, mesh_info);
    next_line = _getNextLineAndBroadcast();
  }
  // $Nodes
  if (next_line != "$Nodes")
    ARCANE_THROW(IOException, "Unexpected string '{0}'. Valid values are '$Nodes'", next_line);

  if (ios_file) {
    // Fetch nodes number and the coordinates
    if (_readNodesFromAsciiMshV4File(*ios_file, mesh_info) != IMeshReader::RTOk)
      ARCANE_THROW(IOException, "Ascii nodes coords error");

    // $EndNodes
    if (!ios_file->lookForString("$EndNodes"))
      ARCANE_THROW(IOException, "$EndNodes not found");
  }

  // $Elements
  Integer mesh_dimension = -1;
  {
    if (ios_file)
      if (!ios_file->lookForString("$Elements"))
        ARCANE_THROW(IOException, "$Elements not found");

    mesh_dimension = _readElementsFromAsciiMshV4File(ios_file, mesh_info);

    // $EndElements
    if (ios_file)
      if (!ios_file->lookForString("$EndElements"))
        throw IOException(func_name, "$EndElements not found");
  }

  info() << "Computed mesh dimension = " << mesh_dimension;

  IPrimaryMesh* pmesh = mesh->toPrimaryMesh();
  pmesh->setDimension(mesh_dimension);

  _allocateCells(mesh_info);
  _allocateGroups(mesh_info);
  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lit le maillage contenu dans le fichier \a filename et le construit dans \a mesh
 */
IMeshReader::eReturnType MshParallelMeshReader::
readMeshFromMshFile(IMesh* mesh, const String& filename)
{
  info() << "Trying to read in parallel 'msh' file '" << filename;
  m_mesh = mesh;
  IParallelMng* pm = mesh->parallelMng();
  m_parallel_mng = pm;
  bool is_master_io = pm->isMasterIO();
  Int32 master_io_rank = pm->masterIORank();
  m_is_parallel = pm->isParallel();
  m_master_io_rank = master_io_rank;
  FixedArray<Int32, 1> file_readable;
  if (is_master_io) {
    bool is_readable = platform::isFileReadable(filename);
    info() << "Is file readable ?=" << is_readable;
    file_readable[0] = is_readable ? 1 : 0;
    if (!is_readable)
      error() << "Unable to read file '" << filename << "'";
  }
  pm->broadcast(file_readable.view(), master_io_rank);
  if (file_readable[0] == 0) {
    return IMeshReader::RTError;
  }

  std::ifstream ifile;
  Ref<IosFile> ios_file;
  if (is_master_io) {
    ifile.open(filename.localstr());
    ios_file = makeRef<IosFile>(new IosFile(&ifile));
  }
  m_ios_file = ios_file;
  String mesh_format_str = _getNextLineAndBroadcast();
  if (IosFile::isEqualString(mesh_format_str, "$MeshFormat"))
    return _readMeshFromNewMshFile(ios_file.get());
  info() << "The file does not begin with '$MeshFormat' returning RTError";
  return IMeshReader::RTError;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IMshMeshReader>
createMshParallelMeshReader(ITraceMng* tm)
{
  return makeRef<IMshMeshReader>(new MshParallelMeshReader(tm));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
