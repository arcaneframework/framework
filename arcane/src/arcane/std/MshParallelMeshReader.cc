// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MshParallelMeshReader.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Lecture parallèle d'un fichier au format MSH.				                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IMeshReader.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/internal/MshMeshGenerationInfo.h"

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
 * - supporter les partitions
 * - pouvoir utiliser la bibliothèque 'gmsh' directement.
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
 * Le lecteur supporte la version `4.1` de ce format.
 *
 * Actuellement, les balises suivantes sont gérées:
 *
 * - `$PhysicalNames`
 * - `$Entities`
 * - `$Nodes`
 * - `$Elements`
 * - `$Periodic` (en cours)
 */
class MshParallelMeshReader
: public TraceAccessor
, public IMshMeshReader
{
 public:

  using MshPhysicalName = impl::MshMeshGenerationInfo::MshPhysicalName;
  using MshEntitiesNodes = impl::MshMeshGenerationInfo::MshEntitiesNodes;
  using MshEntitiesWithNodes = impl::MshMeshGenerationInfo::MshEntitiesWithNodes;
  using MshPeriodicOneInfo = impl::MshMeshGenerationInfo::MshPeriodicOneInfo;
  using MshPeriodicInfo = impl::MshMeshGenerationInfo::MshPeriodicInfo;

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
  struct MshElementBlock
  {
    Int32 index = -1; //!< Index du bloc dans la liste
    Int64 nb_entity = 0; //!< Nombre d'entités du bloc
    ItemTypeId item_type; //!< Type Arcane de l'entité
    Int32 dimension = -1; //!< Dimension de l'entité
    Int32 item_nb_node = 0; //!< Nombre de noeuds de l'entité.
    Int64 entity_tag = -1;
    bool is_built_as_cells = false; //!< Indique si les entités du bloc sont des mailles
    UniqueArray<Int64> uids; //! < Liste des uniqueId() du bloc
    UniqueArray<Int64> connectivities; //!< Liste des connectivités du bloc.
  };

  /*!
   * \brief Infos d'un bloc pour $Nodes.
   */
  struct MshNodeBlock
  {
    Int32 index = -1; //!< Index du bloc dans la liste
    Int32 nb_node = 0; //!< Nombre d'entités du bloc
    Int32 entity_dim = -1; //!< Dimension de l'entité associée
    Int64 entity_tag = -1; //!< Tag de l'entité associée
    //! Index dans MshMeshAllocateInfo des noeuds de ce bloc.
    Int64 index_in_allocation_info = -1;
  };

  /*!
   * \brief Informations pour créer les entités d'un genre.
   */
  class MshItemKindInfo
  {
    // TODO: Allouer la connectivité par bloc pour éviter de trop grosses allocations

   public:

    void addItem(Int16 type_id, Int64 unique_id, SmallSpan<const Int64> nodes_uid)
    {
      ++nb_item;
      items_infos.add(type_id);
      items_infos.add(unique_id);
      Int64 index = items_infos.size();
      items_infos.addRange(nodes_uid);
      if (type_id == IT_Tetraedron10)
        std::swap(items_infos[index + 9], items_infos[index + 8]);
    }

   public:

    Int32 nb_item = 0;
    UniqueArray<Int64> items_infos;
  };

  //! Informations pour créer les entités Arcane.
  class MshMeshAllocateInfo
  {
   public:

    MshItemKindInfo cells_infos;

    //! Coordonnées des noeuds de ma partie
    UniqueArray<Real3> nodes_coordinates;
    //! UniqueId() des noeuds de ma partie.
    UniqueArray<Int64> nodes_unique_id;
    //! Tableau associatif (uniqueId(),rang) auquel le noeud appartiendra.
    std::unordered_map<Int64, Int32> nodes_rank_map;
    UniqueArray<MshElementBlock> element_blocks;
    UniqueArray<MshNodeBlock> node_blocks;
  };

 public:

  explicit MshParallelMeshReader(ITraceMng* tm)
  : TraceAccessor(tm)
  {}

  eReturnType readMeshFromMshFile(IMesh* mesh, const String& filename, bool use_internal_partition) override;

 private:

  IMesh* m_mesh = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  Int32 m_master_io_rank = A_NULL_RANK;
  bool m_is_parallel = false;
  Ref<IosFile> m_ios_file; // nullptr sauf pour le rang maitre.
  impl::MshMeshGenerationInfo* m_mesh_info = nullptr;
  MshMeshAllocateInfo m_mesh_allocate_info;
  //! Nombre de partitions pour la lecture des noeuds et blocs
  Int32 m_nb_part = 4;
  //! Liste des rangs qui participent à la conservation des données
  UniqueArray<Int32> m_parts_rank;
  //! Vrai si le format est binaire
  bool m_is_binary = false;

 private:

  void _readNodesFromFile();
  void _readNodesOneEntity(MshNodeBlock& node_block);
  Integer _readElementsFromFile();
  void _readMeshFromFile();
  void _setNodesCoordinates();
  void _allocateCells();
  void _allocateGroups();
  void _addFaceGroup(MshElementBlock& block, const String& group_name);
  void _addFaceGroupOnePart(ConstArrayView<Int64> connectivities, Int32 item_nb_node,
                            const String& group_name, Int32 block_index);
  void _addCellOrNodeGroup(ArrayView<Int64> block_uids, Int32 block_index,
                           const String& group_name, IItemFamily* family, bool filter_invalid);
  void _addCellOrNodeGroupOnePart(ConstArrayView<Int64> uids, const String& group_name,
                                  Int32 block_index, IItemFamily* family, bool filter_invalid);
  Int16 _switchMshType(Int64, Int32&) const;
  void _readPhysicalNames();
  void _readEntities();
  void _readPeriodic();
  void _readOneEntity(Int32 entity_dim, Int32 entity_index_in_dim);
  bool _getIsEndOfFileAndBroadcast();
  String _getNextLineAndBroadcast();
  Int32 _getIntegerAndBroadcast();
  Int64 _getInt64AndBroadcast();
  void _getInt64ArrayAndBroadcast(ArrayView<Int64> values);
  void _getInt32ArrayAndBroadcast(ArrayView<Int32> values);
  void _getDoubleArrayAndBroadcast(ArrayView<double> values);
  void _readOneElementBlock(MshElementBlock& block);
  void _computeNodesPartition();
  void _computeOwnItems(MshElementBlock& block, MshItemKindInfo& item_kind_info, bool is_generate_uid);
  Real3 _getReal3();
  Int32 _getInt32();
  Int64 _getInt64();
  void _goToNextLine();
  void _readAndCheck(const String& expected_value);
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
  if (f) {
    s = f->getNextLine();
  }
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
/*!
 * \brief Retourne \a true si on est à la fin du fichier.
 */
bool MshParallelMeshReader::
_getIsEndOfFileAndBroadcast()
{
  IosFile* f = m_ios_file.get();
  Int32 is_end_int = 0;
  if (f) {
    is_end_int = f->isEnd() ? 1 : 0;
    info() << "IsEndOfFile_Master: " << is_end_int;
  }
  if (m_is_parallel) {
    if (f)
      info() << "IsEndOfFile: " << is_end_int;
    m_parallel_mng->broadcast(ArrayView<Int32>(1, &is_end_int), m_master_io_rank);
  }
  bool is_end = (is_end_int != 0);
  info() << "IsEnd: " << is_end;
  return is_end;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MshParallelMeshReader::
//_getASCIIIntegerAndBroadcast()
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

Int64 MshParallelMeshReader::
_getInt64AndBroadcast()
{
  IosFile* f = m_ios_file.get();
  FixedArray<Int64, 1> v;
  if (f)
    v[0] = _getInt64();
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
  if (f) {
    if (m_is_binary) {
      f->binaryRead(values);
    }
    else {
      for (Int64& v : values)
        v = f->getInt64();
    }
  }
  if (m_is_parallel)
    m_parallel_mng->broadcast(values, m_master_io_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_getInt32ArrayAndBroadcast(ArrayView<Int32> values)
{
  IosFile* f = m_ios_file.get();
  if (f) {
    if (m_is_binary) {
      f->binaryRead(values);
    }
    else {
      for (Int32& v : values)
        v = f->getInteger();
    }
  }
  if (m_is_parallel)
    m_parallel_mng->broadcast(values, m_master_io_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_getDoubleArrayAndBroadcast(ArrayView<double> values)
{
  IosFile* f = m_ios_file.get();
  if (f) {
    if (m_is_binary) {
      f->binaryRead(values);
    }
    else {
      for (double& v : values)
        v = f->getReal();
    }
  }
  if (m_is_parallel)
    m_parallel_mng->broadcast(values, m_master_io_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 MshParallelMeshReader::
_getReal3()
{
  IosFile* f = m_ios_file.get();
  ARCANE_CHECK_POINTER(f);
  Real3 v;
  if (m_is_binary) {
    f->binaryRead(SmallSpan<Real3>(&v, 1));
  }
  else {
    Real x = f->getReal();
    Real y = f->getReal();
    Real z = f->getReal();
    v = Real3(x, y, z);
  }
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MshParallelMeshReader::
_getInt32()
{
  IosFile* f = m_ios_file.get();
  ARCANE_CHECK_POINTER(f);
  Int32 v = 0;
  if (m_is_binary)
    f->binaryRead(SmallSpan<Int32>(&v, 1));
  else
    v = f->getInteger();
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MshParallelMeshReader::
_getInt64()
{
  IosFile* f = m_ios_file.get();
  ARCANE_CHECK_POINTER(f);
  Int64 v = 0;
  if (m_is_binary)
    f->binaryRead(SmallSpan<Int64>(&v, 1));
  else
    v = f->getInt64();
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_goToNextLine()
{
  if (m_ios_file.get())
    m_ios_file->getNextLine();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int16 MshParallelMeshReader::
_switchMshType(Int64 mshElemType, Int32& nNodes) const
{
  switch (mshElemType) {
  case IT_NullType: // used to decode IT_NullType: IT_HemiHexa7|IT_Line9
    switch (nNodes) {
    case 7:
      return IT_HemiHexa7;
    default:
      info() << "Could not decode IT_NullType with nNodes=" << nNodes;
      throw IOException("_convertToMshType", "Could not decode IT_NullType with nNodes");
    }
    break;
  case MSH_PNT:
    nNodes = 1;
    return IT_Vertex;
  case MSH_LIN_2:
    nNodes = 2;
    return IT_Line2;
  case MSH_TRI_3:
    nNodes = 3;
    return IT_Triangle3;
  case MSH_QUA_4:
    nNodes = 4;
    return IT_Quad4;
  case MSH_TET_4:
    nNodes = 4;
    return IT_Tetraedron4;
  case MSH_HEX_8:
    nNodes = 8;
    return IT_Hexaedron8;
  case MSH_PRI_6:
    nNodes = 6;
    return IT_Pentaedron6;
  case MSH_PYR_5:
    nNodes = 5;
    return IT_Pyramid5;
  case MSH_TRI_10:
    nNodes = 10;
    return IT_Heptaedron10;
  case MSH_TRI_12:
    nNodes = 12;
    return IT_Octaedron12;
  case MSH_TRI_6:
    nNodes = 6;
    return IT_Triangle6;
  case MSH_TET_10:
    nNodes = 10;
    return IT_Tetraedron10;
  case MSH_QUA_9:
  case MSH_HEX_27:
  case MSH_PRI_18:
  case MSH_PYR_14:
  case MSH_QUA_8:
  case MSH_HEX_20:
  case MSH_PRI_15:
  case MSH_PYR_13:
  case MSH_TRI_9:
  case MSH_TRI_15:
  case MSH_TRI_15I:
  case MSH_TRI_21:
  default:
    ARCANE_THROW(NotSupportedException, "Unknown GMSH element type '{0}'", mshElemType);
  }
  return IT_NullType;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_computeNodesPartition()
{
  // Détermine la bounding box de la maille
  Real max_value = FloatInfo<Real>::maxValue();
  Real min_value = -max_value;
  Real3 min_box(max_value, max_value, max_value);
  Real3 max_box(min_value, min_value, min_value);
  const Int64 nb_node = m_mesh_allocate_info.nodes_coordinates.largeSize();
  for (Real3 pos : m_mesh_allocate_info.nodes_coordinates) {
    min_box = math::min(min_box, pos);
    max_box = math::max(max_box, pos);
  }

  //! Rank auquel appartient les noeuds de ma partie.
  UniqueArray<Int32> nodes_part(nb_node, 0);

  // Pour la partition, on ne prend en compte que la coordonnée X.
  // On est sur qu'elle est valable pour toutes les dimensions du maillage.
  // On partitionne avec des intervalles de même longueur.
  // NOTE: Cela fonctionne bien si l'ensemble des noeuds est bien réparti.
  // Si ce n'est pas le cas on pourrait utiliser une bisection en coupant
  // à chaque fois sur la moyenne.
  Real min_x = min_box.x;
  Real max_x = max_box.x;
  IParallelMng* pm = m_parallel_mng;
  Real global_min_x = pm->reduce(Parallel::ReduceMin, min_x);
  Real global_max_x = pm->reduce(Parallel::ReduceMax, max_x);
  info() << "MIN_MAX_X=" << global_min_x << " " << global_max_x;

  Real diff_v = (global_max_x - global_min_x) / static_cast<Real>(m_nb_part);
  // Ne devrait pas arriver mais c'est nécessaire pour éviter d'éventuelles
  // divisions par zéro.
  if (!math::isNearlyEqual(global_min_x, global_max_x)) {
    for (Int64 i = 0; i < nb_node; ++i) {
      Int32 part = static_cast<Int32>((m_mesh_allocate_info.nodes_coordinates[i].x - global_min_x) / diff_v);
      part = std::clamp(part, 0, m_nb_part - 1);
      nodes_part[i] = part;
    }
  }
  UniqueArray<Int32> nb_node_per_rank(m_nb_part, 0);
  // Construit la table de hashage des rangs
  for (Int64 i = 0; i < nb_node; ++i) {
    Int32 rank = m_parts_rank[nodes_part[i]];
    Int64 uid = m_mesh_allocate_info.nodes_unique_id[i];
    ++nb_node_per_rank[rank];
    m_mesh_allocate_info.nodes_rank_map.insert(std::make_pair(uid, rank));
  }
  pm->reduce(Parallel::ReduceSum, nb_node_per_rank);
  info() << "NB_NODE_PER_RANK=" << nb_node_per_rank;
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
void MshParallelMeshReader::
_readNodesFromFile()
{
  FixedArray<Int64, 4> nodes_info;
  _getInt64ArrayAndBroadcast(nodes_info.view());

  Int64 nb_node_block = static_cast<Int32>(nodes_info[0]);
  Int64 total_nb_node = nodes_info[1];
  Int64 min_node_tag = nodes_info[2];
  Int64 max_node_tag = nodes_info[3];

  if (!m_is_binary)
    _goToNextLine();

  if (total_nb_node < 0)
    ARCANE_THROW(IOException, "Invalid number of nodes : '{0}'", total_nb_node);

  info() << "[Nodes] nb_node_block=" << nb_node_block
         << " total_nb_node=" << total_nb_node
         << " min_tag=" << min_node_tag
         << " max_tag=" << max_node_tag
         << " read_nb_part=" << m_nb_part
         << " nb_rank=" << m_parallel_mng->commSize();

  UniqueArray<MshNodeBlock>& node_blocks = m_mesh_allocate_info.node_blocks;
  node_blocks.resize(nb_node_block);

  for (Int32 i = 0; i < nb_node_block; ++i) {
    MshNodeBlock& node_block = node_blocks[i];
    node_block.index = i;
    _readNodesOneEntity(node_block);
  }

  _computeNodesPartition();

  if (m_is_binary)
    _goToNextLine();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_readNodesOneEntity(MshNodeBlock& node_block)
{
  IosFile* ios_file = m_ios_file.get();
  IParallelMng* pm = m_parallel_mng;
  const Int32 my_rank = pm->commRank();

  UniqueArray<Int64> nodes_uids;
  UniqueArray<Real3> nodes_coordinates;

  FixedArray<Int32, 3> entity_infos;
  _getInt32ArrayAndBroadcast(entity_infos.view());
  Int32 nb_node2 = CheckedConvert::toInt32(_getInt64AndBroadcast());

  if (!m_is_binary)
    _goToNextLine();

  // Dimension de l'entité (pas utile)
  Int32 entity_dim = entity_infos[0];
  node_block.entity_dim = entity_dim;
  // Tag de l'entité associée
  Int32 entity_tag = entity_infos[1];
  node_block.entity_tag = entity_tag;

  node_block.index_in_allocation_info = m_mesh_allocate_info.nodes_coordinates.size();
  node_block.nb_node = nb_node2;

  Int32 parametric_coordinates = entity_infos[2];

  info() << "[Nodes] index=" << node_block.index << " entity_dim=" << entity_dim << " entity_tag=" << entity_tag
         << " parametric=" << parametric_coordinates
         << " nb_node2=" << nb_node2;

  if (parametric_coordinates != 0)
    ARCANE_THROW(NotSupportedException, "Only 'parametric coordinates' value of '0' is supported (current={0})", parametric_coordinates);

  // Il est possible que le nombre de noeuds soit 0.
  // Dans ce cas, il faut directement passer à la ligne suivante
  if (nb_node2 == 0)
    return;

  // Partitionne la lecture en \a m_nb_part
  // Pour chaque i_entity, on a d'abord la liste des identifiants puis la liste des coordonnées

  for (Int32 i_part = 0; i_part < m_nb_part; ++i_part) {
    Int64 nb_to_read = _interval(i_part, m_nb_part, nb_node2).second;
    Int32 dest_rank = m_parts_rank[i_part];
    info() << "Reading UIDS part i=" << i_part << " dest_rank=" << dest_rank << " nb_to_read=" << nb_to_read;
    if (my_rank == dest_rank || my_rank == m_master_io_rank) {
      nodes_uids.resize(nb_to_read);
    }

    // Le rang maitre lit les informations des noeuds pour la partie concernée
    // et les transfère au rang destination
    if (ios_file) {
      if (m_is_binary) {
        ios_file->binaryRead(nodes_uids.view());
      }
      else {
        for (Integer i = 0; i < nb_to_read; ++i) {
          // Conserve le uniqueId() du noeuds.
          nodes_uids[i] = ios_file->getInt64();
          //info() << "I=" << i << " ID=" << nodes_uids[i];
        }
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
      m_mesh_allocate_info.nodes_unique_id.addRange(nodes_uids);
    }
  }

  // Lecture par partie des coordonnées
  for (Int32 i_part = 0; i_part < m_nb_part; ++i_part) {
    Int64 nb_to_read = _interval(i_part, m_nb_part, nb_node2).second;
    Int32 dest_rank = m_parts_rank[i_part];
    info() << "Reading COORDS part i=" << i_part << " dest_rank=" << dest_rank << " nb_to_read=" << nb_to_read;
    if (my_rank == dest_rank || my_rank == m_master_io_rank) {
      nodes_coordinates.resize(nb_to_read);
    }

    // Le rang maitre lit les informations des noeuds pour la partie concernée
    // et les transfère au rang destination
    if (ios_file) {
      if (m_is_binary) {
        ios_file->binaryRead(nodes_coordinates.view());
      }
      else {
        for (Integer i = 0; i < nb_to_read; ++i) {
          nodes_coordinates[i] = _getReal3();
          //info() << "I=" << i << " ID=" << nodes_uids[i] << " COORD=" << Real3(nx, ny, nz);
        }
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
      m_mesh_allocate_info.nodes_coordinates.addRange(nodes_coordinates);
    }
  }

  if (!m_is_binary)
    _goToNextLine();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lit un bloc d'entité de type 'Element'.
 */
void MshParallelMeshReader::
_readOneElementBlock(MshElementBlock& block)
{
  IosFile* ios_file = m_ios_file.get();
  IParallelMng* pm = m_parallel_mng;
  const Int32 my_rank = pm->commRank();
  const Int64 nb_entity_in_block = block.nb_entity;
  const Int32 item_nb_node = block.item_nb_node;

  info() << "Reading block nb_entity=" << nb_entity_in_block << " item_nb_node=" << item_nb_node;

  UniqueArray<Int64> uids;
  UniqueArray<Int64> connectivities;

  UniqueArray<Int64> tmp_uid_and_connectivities;

  for (Int32 i_part = 0; i_part < m_nb_part; ++i_part) {
    const Int64 nb_to_read = _interval(i_part, m_nb_part, nb_entity_in_block).second;
    const Int32 dest_rank = m_parts_rank[i_part];

    info() << "Reading block part i_part=" << i_part
           << " nb_to_read=" << nb_to_read << " dest_rank=" << dest_rank;

    const Int64 nb_uid = nb_to_read;
    const Int64 nb_connectivity = nb_uid * item_nb_node;
    if (my_rank == dest_rank || my_rank == m_master_io_rank) {
      uids.resize(nb_uid);
      connectivities.resize(nb_connectivity);
    }
    if (ios_file) {
      if (m_is_binary) {
        Int64 nb_to_read = nb_uid * (item_nb_node + 1);
        tmp_uid_and_connectivities.resize(nb_to_read);
        ios_file->binaryRead(tmp_uid_and_connectivities.view());
        Int64 index = 0;
        for (Int64 i = 0; i < nb_uid; ++i) {
          Int64 item_unique_id = tmp_uid_and_connectivities[index];
          ++index;
          uids[i] = item_unique_id;
          for (Int32 j = 0; j < item_nb_node; ++j) {
            connectivities[(i * item_nb_node) + j] = tmp_uid_and_connectivities[index];
            ++index;
          }
        }
      }
      else {
        // Utilise des Int64 pour garantir qu'on ne déborde pas.
        for (Int64 i = 0; i < nb_uid; ++i) {
          Int64 item_unique_id = ios_file->getInt64();
          uids[i] = item_unique_id;
          for (Int32 j = 0; j < item_nb_node; ++j)
            connectivities[(i * item_nb_node) + j] = ios_file->getInt64();
        }
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
 * \verbatim
 * $Elements
 *   numEntityBlocks(size_t) numElements(size_t)
 *     minElementTag(size_t) maxElementTag(size_t)
 *   entityDim(int) entityTag(int) elementType(int; see below)
 *     numElementsInBlock(size_t)
 *     elementTag(size_t) nodeTag(size_t) ...
 *     ...
 *   ...
 * $EndElements
 * \endverbatim
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
_readElementsFromFile()
{
  IosFile* ios_file = m_ios_file.get();
  IParallelMng* pm = m_parallel_mng;

  FixedArray<Int64, 4> elements_info;
  _getInt64ArrayAndBroadcast(elements_info.view());

  Int64 nb_block = elements_info[0];
  Int64 number_of_elements = elements_info[1];
  Int64 min_element_tag = elements_info[2];
  Int64 max_element_tag = elements_info[3];

  if (!m_is_binary)
    _goToNextLine();

  info() << "[Elements] nb_block=" << nb_block
         << " nb_elements=" << number_of_elements
         << " min_element_tag=" << min_element_tag
         << " max_element_tag=" << max_element_tag;

  if (number_of_elements < 0)
    ARCANE_THROW(IOException, "Invalid number of elements: {0}", number_of_elements);

  UniqueArray<MshElementBlock>& blocks = m_mesh_allocate_info.element_blocks;
  blocks.resize(nb_block);

  {
    // Numérote les blocs (pour le débug)
    Integer index = 0;
    for (MshElementBlock& block : blocks) {
      block.index = index;
      ++index;
    }
  }

  for (MshElementBlock& block : blocks) {

    FixedArray<Int32, 3> block_info;
    _getInt32ArrayAndBroadcast(block_info.view());

    Int32 entity_dim = block_info[0];
    Int32 entity_tag = block_info[1];
    Int32 entity_type = block_info[2];
    Int64 nb_entity_in_block = _getInt64AndBroadcast();

    Integer item_nb_node = 0;
    ItemTypeId item_type(_switchMshType(entity_type, item_nb_node));

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
        [[maybe_unused]] Int64 unused_id = _getInt64();
        item_unique_id = _getInt64();
        info() << "Adding unique node uid=" << item_unique_id;
      }
      if (m_is_parallel)
        pm->broadcast(ArrayView<Int64>(1, &item_unique_id), m_master_io_rank);
      block.uids.add(item_unique_id);
    }
    else {
      _readOneElementBlock(block);
    }
    if (!m_is_binary)
      _goToNextLine();
  }

  if (m_is_binary)
    _goToNextLine();

  // Maintenant qu'on a tous les blocs, la dimension du maillage est
  // la plus grande dimension des blocs
  Integer mesh_dimension = -1;
  for (const MshElementBlock& block : blocks)
    mesh_dimension = math::max(mesh_dimension, block.dimension);
  if (mesh_dimension < 0)
    ARCANE_FATAL("Invalid computed mesh dimension '{0}'", mesh_dimension);
  if (mesh_dimension != 2 && mesh_dimension != 3)
    ARCANE_THROW(NotSupportedException, "mesh dimension '{0}'. Only 2D or 3D meshes are supported", mesh_dimension);
  info() << "Computed mesh dimension = " << mesh_dimension;

  bool allow_multi_dim_cell = m_mesh->meshKind().isNonManifold();
  bool use_experimental_type_for_cell = false;
  if (allow_multi_dim_cell) {
    // Par défaut utilise les nouveaux types.
    use_experimental_type_for_cell = true;
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_EXPERIMENTAL_CELL_TYPE", true))
      use_experimental_type_for_cell = (v.value() != 0);
  }
  info() << "Use experimental cell type?=" << use_experimental_type_for_cell;
  ItemTypeMng* item_type_mng = m_mesh->itemTypeMng();
  for (MshElementBlock& block : blocks) {
    const Int32 block_dim = block.dimension;
    String item_type_name = item_type_mng->typeFromId(block.item_type)->typeName();
    info() << "Reading block dim=" << block_dim << " type_name=" << item_type_name;
    if (block_dim == mesh_dimension)
      _computeOwnItems(block, m_mesh_allocate_info.cells_infos, false);
    else if (allow_multi_dim_cell) {
      // Regarde si on peut créé des mailles de dimension inférieures à celles
      // du maillage.
      bool use_sub_dim_cell = false;
      if (mesh_dimension == 3 && (block_dim == 2 || block_dim == 1))
        // Maille 1D ou 2D dans un maillage 3D
        use_sub_dim_cell = true;
      else if (mesh_dimension == 2 && block_dim == 1)
        // Maille 1D dans un maillage 2D
        use_sub_dim_cell = true;
      if (!use_experimental_type_for_cell)
        use_sub_dim_cell = false;
      if (use_sub_dim_cell) {
        // Ici on va créer des mailles 2D dans un maillage 3D.
        // On converti le type de base en un type équivalent pour les mailles.
        if (mesh_dimension == 3) {
          if (block.item_type == IT_Triangle3)
            block.item_type = ItemTypeId(IT_Cell3D_Triangle3);
          else if (block.item_type == IT_Quad4)
            block.item_type = ItemTypeId(IT_Cell3D_Quad4);
          else if (block.item_type == IT_Line2)
            block.item_type = ItemTypeId(IT_Cell3D_Line2);
          else
            ARCANE_FATAL("Not supported sub dimension cell type={0} for 3D mesh", item_type_name);
        }
        else if (mesh_dimension == 2) {
          if (block.item_type == IT_Line2)
            block.item_type = ItemTypeId(IT_CellLine2);
          else
            ARCANE_FATAL("Not supported sub dimension cell type={0} for 2D mesh", item_type_name);
        }
        block.is_built_as_cells = true;
        _computeOwnItems(block, m_mesh_allocate_info.cells_infos, false);
      }
    }
  }

  return mesh_dimension;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  template <typename DataType> inline ArrayView<DataType>
  _broadcastArrayWithSize(IParallelMng* pm, ArrayView<DataType> values,
                          UniqueArray<DataType>& work_values, Int32 dest_rank, Int64 size)
  {
    const Int32 my_rank = pm->commRank();
    ArrayView<DataType> view = values;
    if (my_rank != dest_rank) {
      work_values.resize(size);
      view = work_values.view();
    }
    pm->broadcast(view, dest_rank);
    return view;
  }
  /*!
   * \brief Broadcast un tableau et retourne une vue dessus.
   *
   * Si on est le rang \a dest_rank, alors on broadcast \a values.
   * Les autres rangs récupèrent la valeur dans \a work_values et
   * retournent une vue dessus.
   */
  template <typename DataType> inline ArrayView<DataType>
  _broadcastArray(IParallelMng* pm, ArrayView<DataType> values,
                  UniqueArray<DataType>& work_values, Int32 dest_rank)
  {
    const Int32 my_rank = pm->commRank();
    Int64 size = 0;
    // Envoie la taille
    if (dest_rank == my_rank)
      size = values.size();
    pm->broadcast(ArrayView<Int64>(1, &size), dest_rank);
    return _broadcastArrayWithSize(pm, values, work_values, dest_rank, size);
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_computeOwnItems(MshElementBlock& block, MshItemKindInfo& item_kind_info, bool is_generate_uid)
{
  // On ne conserve que les entités dont le premier noeud appartient à notre rang.

  IParallelMng* pm = m_parallel_mng;
  const Int32 my_rank = pm->commRank();

  const ItemTypeId item_type = block.item_type;
  const Int32 item_nb_node = block.item_nb_node;

  UniqueArray<Int64> connectivities;
  UniqueArray<Int64> uids;
  UniqueArray<Int32> nodes_rank;

  const Int32 nb_part = m_parts_rank.size();
  info() << "Compute own items block_index=" << block.index << " nb_part=" << nb_part;
  for (Int32 i_part = 0; i_part < nb_part; ++i_part) {
    const Int32 dest_rank = m_parts_rank[i_part];
    // Broadcast la i_part-ème partie des uids et connectivités des mailles
    ArrayView<Int64> connectivities_view = _broadcastArray(pm, block.connectivities.view(), connectivities, dest_rank);
    ArrayView<Int64> uids_view = _broadcastArray(pm, block.uids.view(), uids, dest_rank);

    Int32 nb_item = uids_view.size();
    nodes_rank.resize(nb_item);
    nodes_rank.fill(-1);

    // Parcours les entités. Chaque entité appartiendra au rang
    // de son premier noeud. Si cette partie correspond à mon rang, alors
    // on conserve la maille.
    for (Int32 i = 0; i < nb_item; ++i) {
      Int64 first_node_uid = connectivities_view[i * item_nb_node];
      auto x = m_mesh_allocate_info.nodes_rank_map.find(first_node_uid);
      if (x == m_mesh_allocate_info.nodes_rank_map.end())
        // Le noeud n'est pas dans ma liste
        continue;
      Int32 rank = x->second;
      nodes_rank[i] = rank;
    }
    pm->reduce(Parallel::ReduceMax, nodes_rank);
    for (Int32 i = 0; i < nb_item; ++i) {
      const Int32 rank = nodes_rank[i];
      if (rank != my_rank)
        // Le noeud n'est pas dans ma partie
        continue;
      // Le noeud est chez moi, j'ajoute l'entité à la liste des
      // entités que je vais créer.
      ConstArrayView<Int64> v = connectivities_view.subView(i * item_nb_node, item_nb_node);
      Int64 uid = uids_view[i];
      if (is_generate_uid)
        // Le uniqueId() sera généré automatiquement
        uid = NULL_ITEM_UNIQUE_ID;
      item_kind_info.addItem(item_type, uid, v);
    }
  }
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
 * vraiment parallèle sur le temps d'exécution. On pourrait améliorer cela
 * si on connaissait l'intervalle des uniqueId() de chaque partie et ainsi
 * demander directement au rang concerné les coordonnées qu'on souhaite.
 */
void MshParallelMeshReader::
_setNodesCoordinates()
{
  UniqueArray<Int64> uids_storage;
  UniqueArray<Real3> coords_storage;
  UniqueArray<Int32> local_ids;

  IParallelMng* pm = m_parallel_mng;

  const IItemFamily* node_family = m_mesh->nodeFamily();
  VariableNodeReal3& nodes_coord_var(m_mesh->nodesCoordinates());

  for (Int32 dest_rank : m_parts_rank) {
    ConstArrayView<Int64> uids = _broadcastArray(pm, m_mesh_allocate_info.nodes_unique_id.view(), uids_storage, dest_rank);
    ConstArrayView<Real3> coords = _broadcastArray(pm, m_mesh_allocate_info.nodes_coordinates.view(), coords_storage, dest_rank);

    Int32 nb_item = uids.size();
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
_allocateCells()
{
  IMesh* mesh = m_mesh;
  Integer nb_elements = m_mesh_allocate_info.cells_infos.nb_item;
  info() << "nb_of_elements=cells_type.size()=" << nb_elements;
  Integer nb_cell_node = m_mesh_allocate_info.cells_infos.items_infos.size();
  info() << "nb_cell_node=cells_connectivity.size()=" << nb_cell_node;

  // Création des mailles
  info() << "Building cells, nb_cell=" << nb_elements << " nb_cell_node=" << nb_cell_node;
  IPrimaryMesh* pmesh = mesh->toPrimaryMesh();
  info() << "## Allocating ##";
  pmesh->allocateCells(nb_elements, m_mesh_allocate_info.cells_infos.items_infos, false);
  info() << "## Ending ##";
  pmesh->endAllocate();
  info() << "## Done ##";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_allocateGroups()
{
  IMesh* mesh = m_mesh;
  Int32 mesh_dim = mesh->dimension();
  Int32 face_dim = mesh_dim - 1;
  UniqueArray<MshEntitiesWithNodes> entity_list;
  UniqueArray<MshPhysicalName> physical_name_list;
  IItemFamily* node_family = mesh->nodeFamily();

  for (MshElementBlock& block : m_mesh_allocate_info.element_blocks) {
    entity_list.clear();
    physical_name_list.clear();
    Int32 block_index = block.index;
    Int32 block_dim = block.dimension;
    // On alloue un groupe s'il a un nom physique associé.
    // Pour cela, il faut déjà qu'il soit associé à une entité.
    Int64 block_entity_tag = block.entity_tag;
    if (block_entity_tag < 0) {
      info(5) << "[Groups] Skipping block index=" << block_index << " because it has no entity";
      continue;
    }
    if (block_dim == 0) {
      const MshEntitiesNodes* entity = m_mesh_info->findNodeEntities(block_entity_tag);
      if (!entity) {
        info(5) << "[Groups] Skipping block index=" << block_index
                << " because entity tag is invalid";
        continue;
      }
      Int64 entity_physical_tag = entity->physicalTag();
      MshPhysicalName physical_name = m_mesh_info->findPhysicalName(block_dim, entity_physical_tag);
      physical_name_list.add(physical_name);
    }
    else {
      m_mesh_info->findEntities(block_dim, block_entity_tag, entity_list);
      for (const MshEntitiesWithNodes& x : entity_list) {
        Int64 entity_physical_tag = x.physicalTag();
        MshPhysicalName physical_name = m_mesh_info->findPhysicalName(block_dim, entity_physical_tag);
        physical_name_list.add(physical_name);
      }
    }
    for (const MshPhysicalName& physical_name : physical_name_list) {
      if (physical_name.isNull()) {
        info(5) << "[Groups] Skipping block index=" << block_index
                << " because entity physical tag is invalid";
        continue;
      }
      String group_name = physical_name.name();
      info(4) << "[Groups] Block index=" << block_index << " dim=" << block_dim
              << " name='" << group_name << "' built_as_cells=" << block.is_built_as_cells;
      if (block_dim == mesh_dim || block.is_built_as_cells) {
        _addCellOrNodeGroup(block.uids, block.index, group_name, mesh->cellFamily(), false);
      }
      else if (block_dim == face_dim) {
        _addFaceGroup(block, group_name);
      }
      else {
        // Il s'agit de noeuds
        _addCellOrNodeGroup(block.uids, block.index, group_name, node_family, false);
      }
    }
  }

  // Crée les groupes de noeuds associés aux blocs de $Nodes
  {
    //bool has_periodic = m_mesh_info->m_periodic_info.hasValues();
    UniqueArray<Int64>& nodes_uids = m_mesh_allocate_info.nodes_unique_id;
    // Créé les groupes de noeuds issus des blocks dans $Nodes
    for (const MshNodeBlock& block : m_mesh_allocate_info.node_blocks) {
      ArrayView<Int64> block_uids = nodes_uids.subView(block.index_in_allocation_info, block.nb_node);
      MshPhysicalName physical_name = m_mesh_info->findPhysicalName(block.entity_dim, block.entity_tag);
      String group_name = physical_name.name();
      // Si on a des infos de périodicité, on crée toujours les groupes de noeuds correspondants
      // aux blocs, car ils peuvent être référencés par les infos de périodicité.
      // Dans ce cas, on génère un nom de groupe.
      // NOTE: désactive cela car cela peut générer un très grand nombre de nombre de groupes
      // lorsqu'il y a beaucoup d'informations de périodicité et en plus on n'utilise pas encore
      // cela lors de l'écriture des informations périodiques.
      //if (physical_name.isNull() && has_periodic)
      //group_name = String::format("ArcaneMshInternalNodesDim{0}Entity{1}", block.entity_dim, block.entity_tag);
      if (!group_name.null())
        _addCellOrNodeGroup(block_uids, block.index, group_name, node_family, true);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute des faces au groupe \a group_name.
 */
void MshParallelMeshReader::
_addFaceGroup(MshElementBlock& block, const String& group_name)
{
  IParallelMng* pm = m_parallel_mng;
  const Int32 item_nb_node = block.item_nb_node;

  UniqueArray<Int64> connectivities;
  for (Int32 dest_rank : m_parts_rank) {
    ArrayView<Int64> connectivities_view = _broadcastArray(pm, block.connectivities.view(), connectivities, dest_rank);
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

  const Int32 face_nb_node = nb_entity * item_nb_node;

  UniqueArray<Int64> faces_first_node_unique_id(nb_entity);
  UniqueArray<Int32> faces_first_node_local_id(nb_entity);
  UniqueArray<Int64> faces_nodes_unique_id(face_nb_node);
  Integer faces_nodes_unique_id_index = 0;

  UniqueArray<Int64> orig_nodes_id(item_nb_node);
  UniqueArray<Integer> face_nodes_index(item_nb_node);

  IItemFamily* node_family = mesh->nodeFamily();
  NodeInfoListView mesh_nodes(node_family);

  // Réordonne les identifiants des faces retrouver la face dans le maillage.
  // Pour cela, on récupère le premier noeud de la face et on regarde s'il
  // se trouve dans notre sous-domaine. Si oui, la face sera ajoutée à notre
  // partie du maillage
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

  const bool is_non_manifold = m_mesh->meshKind().isNonManifold();
  faces_nodes_unique_id_index = 0;
  for (Integer i_face = 0; i_face < nb_entity; ++i_face) {
    const Integer n = item_nb_node;
    Int32 face_first_node_lid = faces_first_node_local_id[i_face];
    if (face_first_node_lid != NULL_ITEM_LOCAL_ID) {
      Int64ConstArrayView face_nodes_id(item_nb_node, &faces_nodes_unique_id[faces_nodes_unique_id_index]);
      Node current_node(mesh_nodes[faces_first_node_local_id[i_face]]);
      Face face = MeshUtils::getFaceFromNodesUniqueId(current_node, face_nodes_id);

      // En parallèle, il est possible que la face ne soit pas dans notre sous-domaine
      // même si un de ses noeuds l'est
      if (face.null()) {
        if (!m_is_parallel) {
          OStringStream ostr;
          ostr() << "(Nodes:";
          for (Integer z = 0; z < n; ++z)
            ostr() << ' ' << face_nodes_id[z];
          ostr() << " - " << current_node.localId() << ")";
          String error_string = "INTERNAL: MeshMeshReader face index={0} with nodes '{1}' is not in node/face connectivity.";
          if (!is_non_manifold)
            error_string = error_string + "\n This errors may occur if the mesh is non-manifold."
                                          "\n See Arcane documentation to specify the mesh is a non manifold one.\n";
          ARCANE_FATAL(error_string, i_face, ostr.str());
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
 * \brief Ajoute des mailles ou noeuds au groupe \a group_name.
 */
void MshParallelMeshReader::
_addCellOrNodeGroup(ArrayView<Int64> block_uids, Int32 block_index,
                    const String& group_name, IItemFamily* family, bool filter_invalid)
{
  IParallelMng* pm = m_parallel_mng;

  UniqueArray<Int64> uids;
  for (Int32 dest_rank : m_parts_rank) {
    ArrayView<Int64> uids_view = _broadcastArray(pm, block_uids, uids, dest_rank);
    _addCellOrNodeGroupOnePart(uids_view, group_name, block_index, family, filter_invalid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_addCellOrNodeGroupOnePart(ConstArrayView<Int64> uids, const String& group_name,
                           Int32 block_index, IItemFamily* family, bool filter_invalid)
{
  const Int32 nb_entity = uids.size();

  // Il peut y avoir plusieurs blocs pour le même groupe.
  // On récupère le groupe s'il existe déjà.
  ItemGroup group = family->findGroup(group_name, true);

  UniqueArray<Int32> items_lid(nb_entity);

  family->itemsUniqueIdToLocalId(items_lid, uids, false);

  // En parallèle, il est possible que certaines entités du groupe ne soient
  // pas dans notre sous-domaine. Il faut les filtrer.
  // De même, s'il s'agit d'un groupe issu d'une $Entity node, il est possible
  // que le noeud n'existe pas s'il n'est pas attaché à une maille
  // (TODO: à vérifier avec sod3d-misc.msh)
  if (m_is_parallel || filter_invalid) {
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
_readPhysicalNames()
{
  // NOTE: même en binaire, la partie $PhysicalNames est écrite en ASCII.
  String quote_mark = "\"";
  Int32 nb_name = _getIntegerAndBroadcast();
  info() << "nb_physical_name=" << nb_name;

  _goToNextLine();

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
    m_mesh_info->physical_name_list.add(dim, tag, s);
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
_readEntities()
{
  IosFile* ios_file = m_ios_file.get();

  FixedArray<Int64, 4> nb_dim_item;
  _getInt64ArrayAndBroadcast(nb_dim_item.view());

  info() << "[Entities] nb_0d=" << nb_dim_item[0] << " nb_1d=" << nb_dim_item[1]
         << " nb_2d=" << nb_dim_item[2] << " nb_3d=" << nb_dim_item[3];
  // Après le format, on peut avoir les entités mais cela est optionnel
  // Si elles sont présentes, on lit le fichier jusqu'à la fin de cette section.
  if (!m_is_binary)
    _goToNextLine();

  // Lecture des entités associées à des points
  for (Int64 i = 0; i < nb_dim_item[0]; ++i) {
    FixedArray<Int64, 2> tag_info;
    if (ios_file) {
      Int32 tag = _getInt32();
      Real3 xyz = _getReal3();
      Int64 num_physical_tag = _getInt64();
      if (num_physical_tag > 1)
        ARCANE_FATAL("NotImplemented numPhysicalTag>1 (n={0}, index={1} xyz={2})",
                     num_physical_tag, i, xyz);

      Int32 physical_tag = -1;
      if (num_physical_tag == 1)
        physical_tag = _getInt32();
      info() << "[Entities] point tag=" << tag << " pos=" << xyz << " phys_tag=" << physical_tag;

      tag_info[0] = tag;
      tag_info[1] = physical_tag;
    }
    m_parallel_mng->broadcast(tag_info.view(), m_master_io_rank);
    m_mesh_info->entities_nodes_list.add(MshEntitiesNodes(tag_info[0], tag_info[1]));
    if (!m_is_binary)
      _goToNextLine();
  }

  // Lecture des entités de dimensions 1, 2 et 3.
  for (Int32 i_dim = 1; i_dim <= 3; ++i_dim)
    for (Int32 i = 0; i < nb_dim_item[i_dim]; ++i)
      _readOneEntity(i_dim, i);

  // En binaire, il faut aller au début de la ligne suivante.
  if (m_is_binary)
    _goToNextLine();
  _readAndCheck("$EndEntities");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MshParallelMeshReader::
_readOneEntity(Int32 entity_dim, Int32 entity_index_in_dim)
{
  IosFile* ios_file = m_ios_file.get();

  // Infos pour les tags
  // [0] entity_dim
  // [1] tag
  // [2] nb_physical_tag
  // [2+1] physical_tag1
  // [2+N] physical_tagN
  // ...
  // [3+nb_physical_tag] nb_boundary
  // [3+nb_physical_tag+1] boundary_tag1
  // [3+nb_physical_tag+n] boundary_tagN
  // ...
  info() << "[Entities] Reading entity dim=" << entity_dim << " index_in_dim=" << entity_index_in_dim;
  SmallArray<Int64, 128> dim_and_tag_info;
  dim_and_tag_info.add(entity_dim);
  if (ios_file) {
    Int32 tag = _getInt32();
    dim_and_tag_info.add(tag);
    Real3 min_pos = _getReal3();
    Real3 max_pos = _getReal3();
    Int64 nb_physical_tag = _getInt64();
    dim_and_tag_info.add(nb_physical_tag);
    for (Int32 z = 0; z < nb_physical_tag; ++z) {
      Int32 physical_tag = _getInt32();
      dim_and_tag_info.add(physical_tag);
      info(4) << "[Entities] z=" << z << " physical_tag=" << physical_tag;
    }
    // TODO: Lire les informations numBounding...
    Int64 nb_bounding_group = _getInt64();
    dim_and_tag_info.add(nb_bounding_group);
    for (Int64 z = 0; z < nb_bounding_group; ++z) {
      Int32 boundary_tag = _getInt32();
      info(4) << "[Entities] z=" << z << " boundary_tag=" << boundary_tag;
    }
    info(4) << "[Entities] dim=" << entity_dim << " tag=" << tag
            << " min_pos=" << min_pos << " max_pos=" << max_pos
            << " nb_phys_tag=" << nb_physical_tag
            << " nb_bounding=" << nb_bounding_group;
  }
  Int32 info_size = dim_and_tag_info.size();
  m_parallel_mng->broadcast(ArrayView<Int32>(1, &info_size), m_master_io_rank);
  dim_and_tag_info.resize(info_size);
  m_parallel_mng->broadcast(dim_and_tag_info.view(), m_master_io_rank);

  {
    Int32 dim = CheckedConvert::toInt32(dim_and_tag_info[0]);
    Int64 tag = dim_and_tag_info[1];
    Int64 nb_physical_tag = dim_and_tag_info[2];
    for (Int32 z = 0; z < nb_physical_tag; ++z) {
      Int64 physical_tag = dim_and_tag_info[3 + z];
      info(4) << "[Entities] adding info dim=" << entity_dim << " tag=" << tag
              << " physical_tag=" << physical_tag;
      m_mesh_info->entities_with_nodes_list[dim - 1].add(MshEntitiesWithNodes(dim, tag, physical_tag));
    }
  }

  if (!m_is_binary)
    _goToNextLine();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des informations périodiques.
 *
 * Le format est:
 *
 * \verbatim
 *   $Periodic
 *     numPeriodicLinks(size_t)
 *     entityDim(int) entityTag(int) entityTagMaster(int)
 *     numAffine(size_t) value(double) ...
 *     numCorrespondingNodes(size_t)
 *       nodeTag(size_t) nodeTagMaster(size_t)
 *       ...
 *     ...
 *   $EndPeriodic
 * \endverbatim
 */
void MshParallelMeshReader::
_readPeriodic()
{
  Int64 nb_link = _getInt64AndBroadcast();
  info() << "[Periodic] nb_link=" << nb_link;

  // TODO: pour l'instant, tous les rangs conservent les
  // données car on suppose qu'il n'y en a pas beaucoup.
  // A terme, il faudra aussi distribuer ces informations.
  MshPeriodicInfo& periodic_info = m_mesh_info->m_periodic_info;
  periodic_info.m_periodic_list.resize(nb_link);
  for (Int64 ilink = 0; ilink < nb_link; ++ilink) {
    MshPeriodicOneInfo& one_info = periodic_info.m_periodic_list[ilink];
    FixedArray<Int32, 3> entity_info;
    _getInt32ArrayAndBroadcast(entity_info.view());

    info() << "[Periodic] link_index=" << ilink << " dim=" << entity_info[0] << " entity_tag=" << entity_info[1]
           << " entity_tag_master=" << entity_info[2];
    one_info.m_entity_dim = entity_info[0];
    one_info.m_entity_tag = entity_info[1];
    one_info.m_entity_tag_master = entity_info[2];

    Int64 num_affine = _getInt64AndBroadcast();
    info() << "[Periodic] num_affine=" << num_affine;
    one_info.m_affine_values.resize(num_affine);
    _getDoubleArrayAndBroadcast(one_info.m_affine_values);
    one_info.m_nb_corresponding_node = CheckedConvert::toInt32(_getInt64AndBroadcast());
    info() << "[Periodic] nb_corresponding_node=" << one_info.m_nb_corresponding_node;
    one_info.m_corresponding_nodes.resize(one_info.m_nb_corresponding_node * 2);
    _getInt64ArrayAndBroadcast(one_info.m_corresponding_nodes);
    info() << "[Periodic] corresponding_nodes=" << one_info.m_corresponding_nodes;
  }

  _goToNextLine();

  String s = _getNextLineAndBroadcast();
  if (s != "$EndPeriodic")
    ARCANE_FATAL("found '{0}' and expected '$EndPeriodic'", s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tente de lire la valeur \a value.
 */
void MshParallelMeshReader::
_readAndCheck(const String& expected_value)
{
  String s;
  if (m_is_binary) {
    constexpr Int32 MAX_SIZE = 128;
    FixedArray<Byte, MAX_SIZE> buf_bytes;
    Span<const Byte> expected_bytes = expected_value.bytes();
    Int32 read_size = CheckedConvert::toInt32(expected_bytes.size());
    SmallSpan<Byte> bytes_to_read = buf_bytes.span().subSpan(0, read_size);
    if (read_size >= MAX_SIZE)
      ARCANE_FATAL("local buffer is too small (size={0} max={1})", read_size, MAX_SIZE);
    IosFile* f = m_ios_file.get();
    if (f) {
      f->binaryRead(bytes_to_read);
      s = String(bytes_to_read);
      info() << "S=" << s;
      if (m_is_parallel) {
        m_parallel_mng->broadcastString(s, m_master_io_rank);
      }
    }
    _goToNextLine();
  }
  else {
    s = _getNextLineAndBroadcast();
  }
  if (s != expected_value)
    ARCANE_FATAL("found '{0}' and expected '{1}'", s, expected_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 */
void MshParallelMeshReader::
_readMeshFromFile()
{
  IosFile* ios_file = m_ios_file.get();
  IMesh* mesh = m_mesh;
  info() << "Reading 'msh' file in parallel";

  const int MSH_BINARY_TYPE = 1;

  if (ios_file) {
    Real version = ios_file->getReal();
    if (version != 4.1)
      ARCANE_THROW(IOException, "Wrong msh file version '{0}'. Only version '4.1' is supported in parallel", version);
    Integer file_type = ios_file->getInteger(); // is an integer equal to 0 in the ASCII file format, equal to 1 for the binary format
    if (file_type == MSH_BINARY_TYPE)
      m_is_binary = true;
    info() << "IsBinary?=" << m_is_binary;
    if (m_is_binary)
      pwarning() << "MSH reader for binary format is experimental";
    Int32 data_size = ios_file->getInteger(); // is an integer equal to the size of the floating point numbers used in the file
    ARCANE_UNUSED(data_size);
    if (data_size != 8)
      ARCANE_FATAL("Only 'size_t' of size '8' is allowed (current size is '{0}')", data_size);
    // En binaire, il a un entier à lire qui vaut 1 et qui permet de savoir si on n'est en big/little endian.
    if (m_is_binary) {
      // Il faut lire jusqu'à la fin de la ligne pour le binaire.
      _goToNextLine();
      Int32 int_value_one = 0;
      ios_file->binaryRead(SmallSpan<Int32>(&int_value_one, 1));
      if (int_value_one != 1)
        ARCANE_FATAL("Bad endianess for file. Read int as value '{0}' (expected=1)", int_value_one);
    }

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
    _readPhysicalNames();
    next_line = _getNextLineAndBroadcast();
  }

  // Après le format, on peut avoir les entités mais cela est optionnel
  // Si elles sont présentes, on lit le fichier jusqu'à la fin de cette section.
  if (next_line == "$Entities") {
    _readEntities();
    next_line = _getNextLineAndBroadcast();
  }
  // $Nodes
  if (next_line != "$Nodes")
    ARCANE_THROW(IOException, "Unexpected string '{0}'. Valid values are '$Nodes'", next_line);

  // Fetch nodes number and the coordinates
  _readNodesFromFile();

  // $EndNodes
  if (ios_file && !ios_file->lookForString("$EndNodes"))
    ARCANE_THROW(IOException, "$EndNodes not found");

  // $Elements
  if (ios_file && !ios_file->lookForString("$Elements"))
    ARCANE_THROW(IOException, "$Elements not found");

  Int32 mesh_dimension = _readElementsFromFile();

  // $EndElements
  if (ios_file && !ios_file->lookForString("$EndElements"))
    ARCANE_THROW(IOException, "$EndElements not found");

  info() << "Computed mesh dimension = " << mesh_dimension;

  IPrimaryMesh* pmesh = mesh->toPrimaryMesh();
  pmesh->setDimension(mesh_dimension);

  info() << "NextLine=" << next_line;

  bool is_end = _getIsEndOfFileAndBroadcast();
  if (!is_end) {
    next_line = _getNextLineAndBroadcast();
    // $Periodic
    if (next_line == "$Periodic") {
      _readPeriodic();
    }
  }

  _allocateCells();
  _setNodesCoordinates();
  _allocateGroups();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lit le maillage contenu dans le fichier \a filename et le construit dans \a mesh
 */
IMeshReader::eReturnType MshParallelMeshReader::
readMeshFromMshFile(IMesh* mesh, const String& filename, bool use_internal_partition)
{
  m_mesh_info = impl::MshMeshGenerationInfo::getReference(mesh, true);
  info() << "Trying to read in parallel 'msh' file '" << filename << "'"
         << " use_internal_partition=" << use_internal_partition;
  m_mesh = mesh;
  IParallelMng* pm = mesh->parallelMng();
  // Lit en séquentiel si les fichiers sont déjà découpés
  if (!use_internal_partition)
    pm = pm->sequentialParallelMng();
  m_parallel_mng = pm;
  const Int32 nb_rank = pm->commSize();

  // Détermine les rangs qui vont conserver les données.
  // Il n'est pas obligatoire que tous les rangs participent
  // à la conservation des données. L'idéal avec un
  // grand nombre de rangs serait qu'un rang sur 2 ou 4 participent.
  // Cependant, cela génère alors des partitions vides (pour
  // les rangs qui ne participent pas) et cela peut
  // faire planter certains partitionneurs (comme ParMetis)
  // lorsqu'il y a des partitions vides. Il faudrait d'abord
  // corriger ce problème.
  m_nb_part = nb_rank;
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
  if (IosFile::isEqualString(mesh_format_str, "$MeshFormat")) {
    _readMeshFromFile();
    if (!use_internal_partition) {
      info() << "Synchronize groups and variables";
      mesh->synchronizeGroupsAndVariables();
    }
    return IMeshReader::RTOk;
  }

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
