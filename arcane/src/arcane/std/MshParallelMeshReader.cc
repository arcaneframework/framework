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

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IMeshReader.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshUtils.h"

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
    Int64 nb_entity = 0; //!< Nombre d'entités du bloc
    Integer item_type = -1; //!< Type Arcane de l'entité
    Int32 dimension = -1; //!< Dimension de l'entité
    Int32 item_nb_node = 0; //!< Nombre de noeuds de l'entité.
    Int64 entity_tag = -1;
    UniqueArray<Int64> uids; //! < Liste des uniqueId() du bloc
    UniqueArray<Int64> connectivities; //!< Liste des connectivités du bloc.
  };

  /*!
   * \brief Infos sur un nom physique.
   *
   */
  struct MeshPhysicalName
  {
    MeshPhysicalName(Int32 _dimension, Int64 _tag, const String& _name)
    : dimension(_dimension)
    , tag(_tag)
    , name(_name)
    {}
    MeshPhysicalName() = default;
    bool isNull() const { return dimension == (-1); }
    Int32 dimension = -1;
    Int64 tag = -1;
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
    void add(Int32 dimension, Int64 tag, String name)
    {
      m_physical_names[dimension].add(MeshPhysicalName{ dimension, tag, name });
    }
    MeshPhysicalName find(Int32 dimension, Int64 tag) const
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
    MeshV4EntitiesNodes(Int64 _tag, Int64 _physical_tag)
    : tag(_tag)
    , physical_tag(_physical_tag)
    {}
    Int64 tag;
    Int64 physical_tag;
  };

  //! Infos pour les entités 1D, 2D et 3D
  struct MeshV4EntitiesWithNodes
  {
    MeshV4EntitiesWithNodes(Int32 _dim, Int64 _tag, Int64 _physical_tag)
    : dimension(_dim)
    , tag(_tag)
    , physical_tag(_physical_tag)
    {}
    Int32 dimension;
    Int64 tag;
    Int64 physical_tag;
  };

  struct MeshInfo
  {
   public:

    MeshInfo()
    {}

   public:

    MeshV4EntitiesWithNodes* findEntities(Int32 dimension, Int64 tag)
    {
      for (auto& x : entities_with_nodes_list[dimension - 1])
        if (x.tag == tag)
          return &x;
      return nullptr;
    }

    MeshV4EntitiesNodes* findNodeEntities(Int64 tag)
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
    //! Coordonnées des noeuds de ma partie
    UniqueArray<Real3> nodes_coordinates;
    //! UniqueId() des noeuds de ma partie.
    UniqueArray<Int64> nodes_unique_id;
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

  //! Nombre de partitions pour la lecture des noeuds et blocs
  Int32 m_nb_part = 4;
  //! Liste des rangs qui participent à la conservation des données
  UniqueArray<Int32> m_parts_rank;

 private:

  eReturnType _readNodesFromFileAscii(IosFile* ios_file, MeshInfo& mesh_info);
  Integer _readElementsFromFileAscii(IosFile* ios_file, MeshInfo& mesh_info);
  eReturnType _readMeshFromNewMshFile(IosFile* iso_file);
  void _setNodesCoordinates(MeshInfo& mesh_info);
  void _allocateCells(MeshInfo& mesh_info);
  void _allocateGroups(MeshInfo& mesh_info);
  void _addFaceGroup(MeshV4ElementsBlock& block, const String& group_name);
  void _addFaceGroupOnePart(ConstArrayView<Int64> connectivities, Int32 item_nb_node,
                            const String& group_name, Int32 block_index);
  void _addCellOrNodeGroup(MeshV4ElementsBlock& block, const String& group_name, IItemFamily* family);
  void _addCellOrNodeGroupOnePart(ConstArrayView<Int64> uids, const String& group_name,
                                  Int32 block_index, IItemFamily* family);
  void _addNodeGroup(MeshV4ElementsBlock& block, const String& group_name);
  Int32 _switchMshType(Int64, Int32&);
  void _readPhysicalNames(IosFile* ios_file, MeshInfo& mesh_info);
  void _readEntitiesV4(IosFile* ios_file, MeshInfo& mesh_info);
  String _getNextLineAndBroadcast();
  Int32 _getIntegerAndBroadcast();
  void _getInt64ArrayAndBroadcast(ArrayView<Int64> values);
  void _readOneElementBlock(MeshV4ElementsBlock& block);
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
  inline std::pair<Int64, Int64>
  _interval(Int32 index, Int32 nb_interval, Int64 size)
  {
    Int64 isize = size / nb_interval;
    Int64 ibegin = index * isize;
    // Pour le dernier interval, prend les elements restants
    if ((index + 1) == nb_interval)
      isize = size - ibegin;
    return { ibegin, isize };
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lis la valeur de la prochaine ligne et la broadcast aux autres rangs.
 */
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
_getInt64ArrayAndBroadcast(ArrayView<Int64> values)
{
  IosFile* f = m_ios_file.get();
  if (f)
    for (Int64& v : values)
      v = f->getInt64();
  if (m_is_parallel)
    m_parallel_mng->broadcast(values, m_master_io_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MshParallelMeshReader::
_switchMshType(Int64 mshElemType, Int32& nNodes)
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
 * \brief Lecture des noeuds du maillage.
 *
 * Lit les uniqueId() des noeuds et les coordonnées associées.
 *
 * Le format est le suivant:
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
_readNodesFromFileAscii(IosFile* ios_file, MeshInfo& mesh_info)
{
  IParallelMng* pm = m_parallel_mng;
  const Int32 my_rank = pm->commRank();

  FixedArray<Int64, 4> nodes_info;
  _getInt64ArrayAndBroadcast(nodes_info.view());

  Int64 nb_entity = nodes_info[0];
  Int64 total_nb_node = nodes_info[1];
  Int64 min_node_tag = nodes_info[2];
  Int64 max_node_tag = nodes_info[3];

  if (ios_file)
    ios_file->getNextLine(); // Skip current \n\r

  if (total_nb_node < 0)
    ARCANE_THROW(IOException, "Invalid number of nodes : '{0}'", total_nb_node);

  info() << "[Nodes] nb_entity=" << nb_entity
         << " total_nb_node=" << total_nb_node
         << " min_tag=" << min_node_tag
         << " max_tag=" << max_node_tag
         << " read_nb_part=" << m_nb_part;

  UniqueArray<Int64> nodes_uids;
  UniqueArray<Real3> nodes_coordinates;
  for (Integer i_entity = 0; i_entity < nb_entity; ++i_entity) {

    FixedArray<Int64, 4> entity_infos;
    _getInt64ArrayAndBroadcast(entity_infos.view());
    if (ios_file)
      ios_file->getNextLine();

    // Dimension de l'entité (pas utile)
    [[maybe_unused]] Int64 entity_dim = entity_infos[0];
    // Tag de l'entité (pas utile)
    [[maybe_unused]] Int64 entity_tag = entity_infos[1];
    Int64 parametric_coordinates = entity_infos[2];
    Int64 nb_node2 = entity_infos[3];

    info() << "[Nodes] index=" << i_entity << " entity_dim=" << entity_dim << " entity_tag=" << entity_tag
           << " parametric=" << parametric_coordinates
           << " nb_node2=" << nb_node2;

    if (parametric_coordinates != 0)
      ARCANE_THROW(NotSupportedException, "Only 'parametric coordinates' value of '0' is supported (current={0})", parametric_coordinates);

    // Il est possible que le nombre de noeuds soit 0.
    // Dans ce cas, il faut directement passer à la ligne suivante
    if (nb_node2 == 0)
      continue;

    // Partitionne la lecture en \a m_nb_part
    // Pour chaque i_entity , on a d'abord la liste des identifiants puis la liste des coordonnées

    for (Int32 i_part = 0; i_part < m_nb_part; ++i_part) {
      auto part_info = _interval(i_part, m_nb_part, nb_node2);
      Int64 nb_to_read = part_info.second;
      Int32 dest_rank = m_parts_rank[i_part];
      info() << "Reading UIDS part i=" << i_part << " dest_rank=" << dest_rank << " nb_to_read=" << nb_to_read;
      if (my_rank == dest_rank || my_rank == m_master_io_rank) {
        nodes_uids.resize(nb_to_read);
      }

      // Le rang maitre lit les informations des noeuds pour la partie concernée
      // et les transfère au rang destination
      if (ios_file) {
        for (Integer i = 0; i < nb_to_read; ++i) {
          // Conserve le uniqueId() du noeuds.
          nodes_uids[i] = ios_file->getInt64();
          //info() << "I=" << i << " ID=" << nodes_uids[i];
        }
        if (dest_rank != m_master_io_rank) {
          pm->send(nodes_uids, dest_rank);
        }
      }
      else if (my_rank == dest_rank) {
        pm->recv(nodes_uids, m_master_io_rank);
      }

      // Conserve les informations de ma partie
      if (my_rank == dest_rank) {
        mesh_info.nodes_unique_id.addRange(nodes_uids);
      }
    }

    // Lecture par partie des coordonnées
    for (Int32 i_part = 0; i_part < m_nb_part; ++i_part) {
      auto part_info = _interval(i_part, m_nb_part, nb_node2);
      Int64 nb_to_read = part_info.second;
      Int32 dest_rank = m_parts_rank[i_part];
      info() << "Reading COORDS part i=" << i_part << " dest_rank=" << dest_rank << " nb_to_read=" << nb_to_read;
      if (my_rank == dest_rank || my_rank == m_master_io_rank) {
        nodes_coordinates.resize(nb_to_read);
      }

      // Le rang maitre lit les informations des noeuds pour la partie concernée
      // et les transfère au rang destination
      if (ios_file) {
        for (Integer i = 0; i < nb_to_read; ++i) {
          Real nx = ios_file->getReal();
          Real ny = ios_file->getReal();
          Real nz = ios_file->getReal();
          nodes_coordinates[i] = Real3(nx, ny, nz);
          //info() << "I=" << i << " ID=" << nodes_uids[i] << " COORD=" << Real3(nx, ny, nz);
        }
        if (dest_rank != m_master_io_rank) {
          pm->send(nodes_coordinates, dest_rank);
        }
      }
      else if (my_rank == dest_rank) {
        pm->recv(nodes_coordinates, m_master_io_rank);
      }

      // Conserve les informations de ma partie
      if (my_rank == dest_rank) {
        mesh_info.nodes_coordinates.addRange(nodes_coordinates);
      }
    }

    if (ios_file)
      ios_file->getNextLine(); // Skip current \n\r
  }
  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lit un bloc d'entité de type 'Element'.
 */
void MshParallelMeshReader::
_readOneElementBlock(MeshV4ElementsBlock& block)
{
  IosFile* ios_file = m_ios_file.get();
  IParallelMng* pm = m_parallel_mng;
  const Int32 my_rank = pm->commRank();
  const Int64 nb_entity_in_block = block.nb_entity;
  const Int32 item_nb_node = block.item_nb_node;

  info() << "Reading block nb_entity=" << nb_entity_in_block << " item_nb_node=" << item_nb_node;

  UniqueArray<Int64> uids;
  UniqueArray<Int64> connectivities;

  for (Int32 i_part = 0; i_part < m_nb_part; ++i_part) {
    auto part_info = _interval(i_part, m_nb_part, nb_entity_in_block);
    const Int32 dest_rank = m_parts_rank[i_part];
    const Int64 nb_to_read = part_info.second;
    info() << "Reading block part i_part=" << i_part
           << " nb_to_read=" << nb_to_read << " dest_rank=" << dest_rank;

    const Int64 nb_uid = nb_to_read;
    const Int64 nb_connectivity = nb_uid * item_nb_node;
    if (my_rank == dest_rank || my_rank == m_master_io_rank) {
      uids.resize(nb_uid);
      connectivities.resize(nb_connectivity);
    }
    if (ios_file) {
      // Utilise des Int64 pour garantir qu'on ne déborde pas.
      for (Int64 i = 0; i < nb_uid; ++i) {
        Int64 item_unique_id = ios_file->getInt64();
        uids[i] = item_unique_id;
        for (Int32 j = 0; j < item_nb_node; ++j)
          connectivities[(i * item_nb_node) + j] = ios_file->getInt64();
      }
      if (dest_rank != m_master_io_rank) {
        pm->send(uids, dest_rank);
        pm->send(connectivities, dest_rank);
      }
    }
    else if (my_rank == dest_rank) {
      pm->recv(uids, m_master_io_rank);
      pm->recv(connectivities, m_master_io_rank);
    }
    if (my_rank == dest_rank) {
      block.uids.addRange(uids);
      block.connectivities.addRange(connectivities);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des éléments (mailles,faces,...)
 *
 * Voici la description du format:
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
 *
 * Dans la version 4, les éléments sont rangés par genre (eItemKind).
 * Chaque bloc d'entité peut-être de dimension différente Il n'y a pas
 * de dimension associé au maillage. On considère donc que la dimension
 * du maillage est la dimension la plus élevée des blocs qu'on lit.
 * Cela signifie aussi qu'on est obligé de lire tous les blocs avant de pouvoir
 * connaitre la dimension du maillage.
 *
 * En parallèle, chaque bloc est distribué sur les rangs de \a m_parts_rank.
 *
 * \return la dimension du maillage.
 */
Integer MshParallelMeshReader::
_readElementsFromFileAscii(IosFile* ios_file, MeshInfo& mesh_info)
{
  IParallelMng* pm = m_parallel_mng;

  FixedArray<Int64, 4> elements_info;
  _getInt64ArrayAndBroadcast(elements_info.view());

  Int64 nb_block = elements_info[0];
  Int64 number_of_elements = elements_info[1];
  Int64 min_element_tag = elements_info[2];
  Int64 max_element_tag = elements_info[3];

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

    FixedArray<Int64, 4> block_info;
    _getInt64ArrayAndBroadcast(block_info.view());

    Int32 entity_dim = CheckedConvert::toInt32(block_info[0]);
    Int64 entity_tag = block_info[1];
    Int64 entity_type = block_info[2];
    Int64 nb_entity_in_block = block_info[3];

    Integer item_nb_node = 0;
    Integer item_type = _switchMshType(entity_type, item_nb_node);

    info() << "[Elements] index=" << block.index << " entity_dim=" << entity_dim
           << " entity_tag=" << entity_tag
           << " entity_type=" << entity_type << " nb_in_block=" << nb_entity_in_block
           << " item_type=" << item_type << " item_nb_node=" << item_nb_node;

    block.nb_entity = nb_entity_in_block;
    block.item_type = item_type;
    block.item_nb_node = item_nb_node;
    block.dimension = entity_dim;
    block.entity_tag = entity_tag;

    if (entity_type == MSH_PNT) {
      // Si le type est un point, le traitement semble un peu particulier.
      // Il y a dans ce cas deux entiers dans la ligne suivante:
      // - un entier qui ne semble pas être utilisé
      // - le numéro unique du noeud qui nous intéresse
      Int64 item_unique_id = NULL_ITEM_UNIQUE_ID;
      if (ios_file) {
        [[maybe_unused]] Int64 unused_id = ios_file->getInt64();
        item_unique_id = ios_file->getInt64();
        info() << "Adding unique node uid=" << item_unique_id;
      }
      if (m_is_parallel)
        pm->broadcast(ArrayView<Int64>(1, &item_unique_id), m_master_io_rank);
      block.uids.add(item_unique_id);
    }
    else {
      _readOneElementBlock(block);
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

  // On ne conserve que les blocs de notre dimension pour créér les mailles.
  // On divise chaque bloc en N partie (avec N le nombre de rangs MPI) et
  // chaque rang ne conserve qu'une partie. Ainsi chaque sous-domaine aura une
  // partie du maillage afin de garantir un équilibrage sur le nombre de mailles.
  for (MeshV4ElementsBlock& block : blocks) {
    if (block.dimension != mesh_dimension)
      continue;

    Integer item_type = block.item_type;
    Integer item_nb_node = block.item_nb_node;

    const Int64 block_nb_item = block.uids.size();
    for (Int64 i = 0; i < block_nb_item; ++i) {
      mesh_info.cells_type.add(item_type);
      mesh_info.cells_nb_node.add(item_nb_node);
      mesh_info.cells_uid.add(block.uids[i]);
      auto v = block.connectivities.subView(i * item_nb_node, item_nb_node);
      mesh_info.cells_connectivity.addRange(v);
    }
  }

  return mesh_dimension;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne les coordonnées des noeuds.
 *
 * La liste est répartie sur les processeurs des rangs de \a m_parts_rank.
 * On boucle sur les rangs de cette liste et chaque rang envoie aux autres
 * la liste des coordonnées des noeuds qu'il contient. Chaque rang regarde
 * si des noeuds de cette liste lui appartiennent et si c'est le cas positionnent
 * les coordonnées.
 *
 * \note Cet algorithme envoie par morceau tous les noeuds et n'est donc pas
 * vraiement parallèle sur le temps d'exécution. On pourrait améliorer cela
 * si on connaissait l'intervalle des uniqueId() de chaque partie et ainsi
 * demander directement au rang concerné les coordonnées qu'on souhaite.
 */
void MshParallelMeshReader::
_setNodesCoordinates(MeshInfo& mesh_info)
{
  UniqueArray<Int64> uids_storage;
  UniqueArray<Real3> coords_storage;
  UniqueArray<Int32> local_ids;

  IParallelMng* pm = m_parallel_mng;
  const Int32 my_rank = pm->commRank();

  IItemFamily* node_family = m_mesh->nodeFamily();
  VariableNodeReal3& nodes_coord_var(m_mesh->nodesCoordinates());

  for (Int32 dest_rank : m_parts_rank) {
    Int32 nb_item = 0;
    if (my_rank == dest_rank) {
      nb_item = mesh_info.nodes_unique_id.size();
    }
    // Envoie le nombre de noeuds aux autres
    pm->broadcast(ArrayView<Int32>(1, &nb_item), dest_rank);
    ConstArrayView<Int64> uids;
    ConstArrayView<Real3> coords;
    if (my_rank == dest_rank) {
      pm->broadcast(mesh_info.nodes_unique_id, dest_rank);
      pm->broadcast(mesh_info.nodes_coordinates, dest_rank);
      uids = mesh_info.nodes_unique_id.view();
      coords = mesh_info.nodes_coordinates.view();
    }
    else {
      uids_storage.resize(nb_item);
      coords_storage.resize(nb_item);
      pm->broadcast(uids_storage, dest_rank);
      pm->broadcast(coords_storage, dest_rank);
      uids = uids_storage.view();
      coords = coords_storage.view();
    }
    local_ids.resize(nb_item);

    // Converti les uniqueId() en localId(). S'ils sont non nuls
    // c'est que l'entité est dans mon sous-domaine et donc on peut
    // positionner sa coordonnée
    node_family->itemsUniqueIdToLocalId(local_ids, uids, false);
    for (Int32 i = 0; i < nb_item; ++i) {
      NodeLocalId nid(local_ids[i]);
      if (!nid.isNull())
        nodes_coord_var[nid] = coords[i];
    }
  }
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

  // Positionne les coordonnées des noeuds
  _setNodesCoordinates(mesh_info);
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
    Int64 block_entity_tag = block.entity_tag;
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
      Int64 entity_physical_tag = entity->physical_tag;
      physical_name = mesh_info.physical_name_list.find(block_dim, entity_physical_tag);
    }
    else {
      MeshV4EntitiesWithNodes* entity = mesh_info.findEntities(block_dim, block_entity_tag);
      if (!entity) {
        info(5) << "[Groups] Skipping block index=" << block_index
                << " because entity tag is invalid";
        continue;
      }
      Int64 entity_physical_tag = entity->physical_tag;
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
      _addCellOrNodeGroup(block, physical_name.name, mesh->cellFamily());
    }
    else if (block_dim == face_dim) {
      _addFaceGroup(block, physical_name.name);
    }
    else {
      _addCellOrNodeGroup(block, physical_name.name, mesh->nodeFamily());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des faces au groupe \a group_name.
 */
void MshParallelMeshReader::
_addFaceGroup(MeshV4ElementsBlock& block, const String& group_name)
{
  IParallelMng* pm = m_parallel_mng;
  const Int32 my_rank = pm->commRank();
  const Int32 item_nb_node = block.item_nb_node;

  UniqueArray<Int64> connectivities;
  for (Int32 dest_rank : m_parts_rank) {
    Int64 nb_connectivity = 0;
    ArrayView<Int64> connectivities_view;
    if (my_rank == dest_rank) {
      nb_connectivity = block.connectivities.size();
      connectivities_view = block.connectivities;
    }
    pm->broadcast(ArrayView<Int64>(1, &nb_connectivity), dest_rank);
    if (my_rank != dest_rank) {
      connectivities.resize(nb_connectivity);
      connectivities_view = connectivities;
    }
    pm->broadcast(connectivities_view, dest_rank);
    _addFaceGroupOnePart(connectivities_view, item_nb_node, group_name, block.index);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_addFaceGroupOnePart(ConstArrayView<Int64> connectivities, Int32 item_nb_node,
                     const String& group_name, Int32 block_index)
{
  IMesh* mesh = m_mesh;
  const Int32 nb_entity = connectivities.size() / item_nb_node;

  // Il peut y avoir plusieurs blocs pour le même groupe.
  // On récupère le groupe s'il existe déjà.
  FaceGroup face_group = mesh->faceFamily()->findGroup(group_name, true);

  UniqueArray<Int32> faces_id; // Numéro de la face dans le maillage \a mesh
  faces_id.reserve(nb_entity);

  //const Int32 item_nb_node = block.item_nb_node;
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
      orig_nodes_id[z] = connectivities[faces_nodes_unique_id_index + z];

    MeshUtils::reorderNodesOfFace2(orig_nodes_id, face_nodes_index);
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
      Face face = MeshUtils::getFaceFromNodesUnique(current_node, face_nodes_id);

      // En parallèle, il est possible que la face ne soit pas dans notre sous-domaine
      // même si un de ses noeuds l'est
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
  info(4) << "Adding " << faces_id.size() << " faces from block index=" << block_index
          << " to group '" << face_group.name() << "'";
  face_group.addItems(faces_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des faces au groupe \a group_name.
 */
void MshParallelMeshReader::
_addCellOrNodeGroup(MeshV4ElementsBlock& block, const String& group_name, IItemFamily* family)
{
  IParallelMng* pm = m_parallel_mng;
  const Int32 my_rank = pm->commRank();

  UniqueArray<Int64> uids;
  for (Int32 dest_rank : m_parts_rank) {
    Int64 nb_uid = 0;
    ArrayView<Int64> uids_view;
    if (my_rank == dest_rank) {
      nb_uid = block.uids.size();
      uids_view = block.uids;
    }
    pm->broadcast(ArrayView<Int64>(1, &nb_uid), dest_rank);
    if (my_rank != dest_rank) {
      uids.resize(nb_uid);
      uids_view = uids;
    }
    pm->broadcast(uids_view, dest_rank);
    _addCellOrNodeGroupOnePart(uids_view, group_name, block.index, family);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_addCellOrNodeGroupOnePart(ConstArrayView<Int64> uids, const String& group_name,
                           Int32 block_index, IItemFamily* family)
{
  const Int32 nb_entity = uids.size();

  // Il peut y avoir plusieurs blocs pour le même groupe.
  // On récupère le groupe s'il existe déjà.
  ItemGroup group = family->findGroup(group_name, true);

  UniqueArray<Int32> items_lid(nb_entity);

  family->itemsUniqueIdToLocalId(items_lid, uids, false);

  // En parallèle, il est possible que certaines entités du groupe ne soient
  // pas dans notre sous-domaine. Il faut les filtrer.
  if (m_is_parallel) {
    auto items_begin = items_lid.begin();
    Int64 new_size = std::remove(items_begin, items_lid.end(), NULL_ITEM_LOCAL_ID) - items_begin;
    items_lid.resize(new_size);
  }

  info() << "Adding " << items_lid.size() << " items from block index=" << block_index
         << " to group '" << group_name << "' for family=" << family->name();

  group.addItems(items_lid);
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
    FixedArray<Int64, 2> tag_info;
    if (ios_file) {
      Int64 tag = ios_file->getInt64();
      Real x = ios_file->getReal();
      Real y = ios_file->getReal();
      Real z = ios_file->getReal();
      Int64 num_physical_tag = ios_file->getInt64();
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
      FixedArray<Int64, 3> dim_and_tag_info;
      if (ios_file) {
        Int64 tag = ios_file->getInt64();
        Real min_x = ios_file->getReal();
        Real min_y = ios_file->getReal();
        Real min_z = ios_file->getReal();
        Real max_x = ios_file->getReal();
        Real max_y = ios_file->getReal();
        Real max_z = ios_file->getReal();
        Int64 num_physical_tag = ios_file->getInt64();
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
        Int32 dim = CheckedConvert::toInt32(dim_and_tag_info[0]);
        Int64 tag = dim_and_tag_info[1];
        Int64 physical_tag = dim_and_tag_info[2];
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

  // Fetch nodes number and the coordinates
  if (_readNodesFromFileAscii(ios_file, mesh_info) != IMeshReader::RTOk)
    ARCANE_THROW(IOException, "Ascii nodes coords error");

  if (ios_file) {
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

    mesh_dimension = _readElementsFromFileAscii(ios_file, mesh_info);

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
  const Int32 nb_rank = pm->commSize();

  // Détermine les rangs qui vont conserver les données
  m_nb_part = math::min(m_nb_part, nb_rank);
  m_parts_rank.resize(m_nb_part);
  for (Int32 i = 0; i < m_nb_part; ++i) {
    m_parts_rank[i] = i % nb_rank;
  }

  bool is_master_io = pm->isMasterIO();
  Int32 master_io_rank = pm->masterIORank();
  m_is_parallel = pm->isParallel();
  m_master_io_rank = master_io_rank;
  FixedArray<Int32, 1> file_readable;
  // Seul le rang maître va lire le fichier.
  // On vérifie d'abord qu'il est lisible
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
