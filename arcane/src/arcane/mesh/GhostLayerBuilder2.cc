// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerBuilder2.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Construction des couches fantômes.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/parallel/BitonicSortT.H"

#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/IItemFamilySerializer.h"
#include "arcane/core/ParallelMngUtils.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

#include <algorithm>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction des couches fantômes.
 */
class GhostLayerBuilder2
: public TraceAccessor
{
  class BoundaryNodeInfo;
  class BoundaryNodeBitonicSortTraits;
  class BoundaryNodeToSendInfo;

 public:

  using ItemInternalMap = DynamicMeshKindInfos::ItemInternalMap;
  using SubDomainItemMap = HashTableMapT<Int32, SharedArray<Int32>>;

 public:

  //! Construit une instance pour le maillage \a mesh
  GhostLayerBuilder2(DynamicMeshIncrementalBuilder* mesh_builder, bool is_allocate, Int32 version);

 public:

  void addGhostLayers();

 private:

  DynamicMesh* m_mesh = nullptr;
  DynamicMeshIncrementalBuilder* m_mesh_builder = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  bool m_is_verbose = false;
  bool m_is_allocate = false;
  Int32 m_version = -1;
  bool m_use_optimized_node_layer = true;
  bool m_use_only_minimal_cell_uid = true;

 private:

  void _printItem(ItemInternal* ii, std::ostream& o);
  void _markBoundaryItems(ArrayView<Int32> node_layer);
  void _sendAndReceiveCells(SubDomainItemMap& cells_to_send);
  void _sortBoundaryNodeList(Array<BoundaryNodeInfo>& boundary_node_list);
  void _addGhostLayer(Integer current_layer, Int32ConstArrayView node_layer);
  void _markBoundaryNodes(ArrayView<Int32> node_layer);
  void _markBoundaryNodesFromEdges(ArrayView<Int32> node_layer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostLayerBuilder2::
GhostLayerBuilder2(DynamicMeshIncrementalBuilder* mesh_builder,bool is_allocate,Int32 version)
: TraceAccessor(mesh_builder->mesh()->traceMng())
, m_mesh(mesh_builder->mesh())
, m_mesh_builder(mesh_builder)
, m_parallel_mng(m_mesh->parallelMng())
, m_is_allocate(is_allocate)
, m_version(version)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_GHOSTLAYER_USE_OPTIMIZED_LAYER", true)) {
    Int32 vv = v.value();
    m_use_optimized_node_layer = (vv == 1 || vv == 3);
    m_use_only_minimal_cell_uid = (v == 2 || vv == 3);
  }
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_GHOSTLAYER_VERBOSE", true)) {
    m_is_verbose = (v.value()!=0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_printItem(ItemInternal* ii,std::ostream& o)
{
  o << ItemPrinter(ii);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Structure contenant les informations des noeuds frontières.
 *
 * Cette structure est utilisée pour communiquer avec les autres rangs.
 * Il faut donc qu'elle soit de type POD. Pour la communication avec les autres
 * rang on la converti en un type de base qui est un Int64. Il faut donc aussi
 * que sa taille soit un multiple de celle d'un Int64..
 */
class GhostLayerBuilder2::BoundaryNodeInfo
{
 public:

  using BasicType = Int64;
  static constexpr Int64 nbBasicTypeSize() { return 3; }

 public:

  static ConstArrayView<BasicType> asBasicBuffer(ConstArrayView<BoundaryNodeInfo> values)
  {
    Int32 message_size = messageSize(values);
    const BoundaryNodeInfo* fsi_base = values.data();
    auto* ptr = reinterpret_cast<const Int64*>(fsi_base);
    return ConstArrayView<BasicType>(message_size, ptr);
  }

  static ArrayView<BasicType> asBasicBuffer(ArrayView<BoundaryNodeInfo> values)
  {
    Int32 message_size = messageSize(values);
    BoundaryNodeInfo* fsi_base = values.data();
    auto* ptr = reinterpret_cast<Int64*>(fsi_base);
    return ArrayView<BasicType>(message_size, ptr);
  }

  static Int32 messageSize(ConstArrayView<BoundaryNodeInfo> values)
  {
    static_assert((sizeof(Int64) * nbBasicTypeSize()) == sizeof(BoundaryNodeInfo));
    Int64 message_size_i64 = values.size() * nbBasicTypeSize();
    Int32 message_size = CheckedConvert::toInteger(message_size_i64);
    return message_size;
  }

  static Int32 nbElement(Int32 message_size)
  {
    if ((message_size % nbBasicTypeSize()) != 0)
      ARCANE_FATAL("Message size '{0}' is not a multiple of basic size '{1}'", message_size, nbBasicTypeSize());
    Int32 nb_element = message_size / nbBasicTypeSize();
    return nb_element;
  }

 public:

  struct HashFunction
  {
    size_t operator()(const BoundaryNodeInfo& a) const
    {
      size_t h1 = std::hash<Int64>{}(a.node_uid);
      size_t h2 = std::hash<Int64>{}(a.cell_uid);
      size_t h3 = std::hash<Int32>{}(a.cell_owner);
      return h1 ^ h2 ^ h3;
    }
  };
  friend bool operator==(const BoundaryNodeInfo& a, const BoundaryNodeInfo& b)
  {
    return (a.node_uid == b.node_uid && a.cell_uid == b.cell_uid && a.cell_owner == b.cell_owner);
  }

 public:

  Int64 node_uid = NULL_ITEM_UNIQUE_ID;
  Int64 cell_uid = NULL_ITEM_UNIQUE_ID;
  Int32 cell_owner = -1;
  Int32 padding = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor pour trier les BoundaryNodeInfo via le tri bitonic.
 */
class GhostLayerBuilder2::BoundaryNodeBitonicSortTraits
{
 public:
  static bool compareLess(const BoundaryNodeInfo& k1,const BoundaryNodeInfo& k2)
  {
    Int64 k1_node_uid = k1.node_uid;
    Int64 k2_node_uid = k2.node_uid;
    if (k1_node_uid<k2_node_uid)
      return true;
    if (k1_node_uid>k2_node_uid)
      return false;

    Int64 k1_cell_uid = k1.cell_uid;
    Int64 k2_cell_uid = k2.cell_uid;
    if (k1_cell_uid<k2_cell_uid)
      return true;
    if (k1_cell_uid>k2_cell_uid)
      return false;

    return (k1.cell_owner<k2.cell_owner);
  }

  static Parallel::Request send(IParallelMng* pm,Int32 rank,ConstArrayView<BoundaryNodeInfo> values)
  {
    auto buf_view = BoundaryNodeInfo::asBasicBuffer(values);
    return pm->send(buf_view, rank, false);
  }

  static Parallel::Request recv(IParallelMng* pm,Int32 rank,ArrayView<BoundaryNodeInfo> values)
  {
    auto buf_view = BoundaryNodeInfo::asBasicBuffer(values);
    return pm->recv(buf_view, rank, false);
  }

  static Integer messageSize(ConstArrayView<BoundaryNodeInfo> values)
  {
    return BoundaryNodeInfo::messageSize(values);
  }

  static BoundaryNodeInfo maxValue()
  {
    BoundaryNodeInfo bni;
    bni.node_uid = INT64_MAX;
    bni.cell_uid = INT64_MAX;
    bni.cell_owner = -1;
    return bni;
  }

  static bool isValid(const BoundaryNodeInfo& bni)
  {
    return bni.node_uid!=INT64_MAX;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GhostLayerBuilder2::BoundaryNodeToSendInfo
{
 public:
  Integer m_index;
  Integer m_nb_cell;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute les couches de mailles fantomes.
 *
 * Cette version utilise un tri pour déterminer les infos 
 *
 * Avant appel à cette fonction, il ne faut plus qu'il y ait de mailles
 * fantômes: toutes les mailles du maillages doivent appartenir à ce sous-domaine.
 * (TODO: ajouter test pour cela).
 *
 * Si on demande plusieurs couches de mailles fantômes, on procède en plusieurs
 * étapes pour le même algo. D'abord on envoie la première couche, puis la première
 * et la seconde, puis trois couches et ainsi de suite. Cela n'est probablement
 * pas optimum en terme de communication mais permet de traiter tous les cas,
 * notamment le cas ou il faut traverser plusieurs sous-domaines pour
 * ajouter des couches de mailles fantômes.
 * 
 * \todo: faire les optimisations spécifiées dans les commentaires
 * dans cette fonction.
 * \todo: faire en sorte qu'on ne travaille que avec la connectivité
 * maille/noeud.
 * 
 */
void GhostLayerBuilder2::
addGhostLayers()
{
  IParallelMng* pm = m_parallel_mng;
  if (!pm->isParallel())
    return;
  Integer nb_ghost_layer = m_mesh->ghostLayerMng()->nbGhostLayer();
  info() << "** GHOST LAYER BUILDER V" << m_version << " with sort (nb_ghost_layer=" << nb_ghost_layer << ")";

  // Couche fantôme à laquelle appartient le noeud.
  UniqueArray<Integer> node_layer(m_mesh->nodeFamily()->maxLocalId(), -1);

  // Marque les noeuds frontières
  // On le fait toujours même si on ne veut pas de couche de mailles fantômes
  _markBoundaryItems(node_layer);

  if (nb_ghost_layer == 0)
    return;
  const Int32 my_rank = pm->commRank();

  const bool is_non_manifold = m_mesh->meshKind().isNonManifold();
  if (is_non_manifold && (m_version != 3))
    ARCANE_FATAL("Only version 3 of ghostlayer builder is supported for non manifold meshes");

  ItemInternalMap& cells_map = m_mesh->cellsMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();

  Integer boundary_nodes_uid_count = 0;

  // Vérifie qu'il n'y a pas de mailles fantômes avec la version 3.
  // Si c'est le cas, affiche un avertissement et indique de passer à la version 4.
  if (m_version == 3) {
    Integer nb_ghost = 0;
    cells_map.eachItem([&](Item cell) {
      if (!cell.isOwn())
        ++nb_ghost;
    });
    if (nb_ghost != 0)
      warning() << "Invalid call to addGhostLayers() with version 3 because mesh "
                << " already has '" << nb_ghost << "' ghost cells. The computed ghost cells"
                << " may be wrong. Use version 4 of ghost layer builder if you want to handle this case";
  }

  // Couche fantôme à laquelle appartient la maille.
  UniqueArray<Integer> cell_layer(m_mesh->cellFamily()->maxLocalId(), -1);

  if (m_version >= 4) {
    _markBoundaryNodes(node_layer);
    nodes_map.eachItem([&](Item node) {
      if (node_layer[node.localId()] == 1)
        ++boundary_nodes_uid_count;
    });
  }
  else {
    // Parcours les nœuds et calcule le nombre de nœuds frontières
    // et marque la première couche
    nodes_map.eachItem([&](Item node) {
      Int32 f = node.itemBase().flags();
      if (f & ItemFlags::II_Shared) {
        node_layer[node.localId()] = 1;
        ++boundary_nodes_uid_count;
      }
    });
  }

  info() << "NB BOUNDARY NODE=" << boundary_nodes_uid_count;

  for (Integer current_layer = 1; current_layer <= nb_ghost_layer; ++current_layer) {
    //Integer current_layer = 1;
    info() << "Processing layer " << current_layer;
    cells_map.eachItem([&](Cell cell) {
      // Ne traite pas les mailles qui ne m'appartiennent pas
      if (m_version >= 4 && cell.owner() != my_rank)
        return;
      //Int64 cell_uid = cell->uniqueId();
      Int32 cell_lid = cell.localId();
      if (cell_layer[cell_lid] != (-1))
        return;
      bool is_current_layer = false;
      for (Int32 inode_local_id : cell.nodeIds()) {
        Integer layer = node_layer[inode_local_id];
        //info() << "NODE_LAYER lid=" << i_node->localId() << " layer=" << layer;
        if (layer == current_layer) {
          is_current_layer = true;
          break;
        }
      }
      if (is_current_layer) {
        cell_layer[cell_lid] = current_layer;
        //info() << "Current layer celluid=" << cell_uid;
        // Si non marqué, initialise à la couche courante + 1.
        for (Int32 inode_local_id : cell.nodeIds()) {
          Integer layer = node_layer[inode_local_id];
          if (layer == (-1)) {
            //info() << "Marque node uid=" << i_node->uniqueId();
            node_layer[inode_local_id] = current_layer + 1;
          }
        }
      }
    });
  }

  // Marque les nœuds pour lesquels on n'a pas encore assigné la couche fantôme.
  // Pour eux, on indique qu'on est sur la couche 'nb_ghost_layer+1'.
  // Le but est de ne jamais transférer ces noeuds.
  // NOTE: Ce mécanisme a été ajouté en juillet 2024 pour la version 3.14.
  //       S'il fonctionne bien on pourra ne conserver que cette méthode.
  if (m_use_optimized_node_layer) {
    Integer nb_no_layer = 0;
    nodes_map.eachItem([&](Node node) {
      Int32 lid = node.localId();
      Int32 layer = node_layer[lid];
      if (layer <= 0) {
        node_layer[lid] = nb_ghost_layer + 1;
        ++nb_no_layer;
      }
    });
    info() << "Mark remaining nodes nb=" << nb_no_layer;
  }

  for (Integer i = 1; i <= nb_ghost_layer; ++i)
    _addGhostLayer(i, node_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détermine les noeuds frontières.
 *
 * Cet algorithme fonctionne même s'il y a déjà des mailles fantômes.
 * Pour déterminer les noeuds frontières il faut déjà déterminer les
 * faces frontières. Une face est frontière si elle est dans l'un des deux cas:
 * - elle n'a qu'une maille connectée qui appartient à ce sous-domaine.
 * - elle est connectée à deux mailles dont une exactement appartient à ce
 *   domaine.
 * Une fois les faces frontières trouvées, on considère que les noeuds frontières
 * sont ceux qui appartiennent à une face frontière.
 */
void GhostLayerBuilder2::
_markBoundaryNodes(ArrayView<Int32> node_layer)
{
  IParallelMng* pm = m_mesh->parallelMng();
  const Int32 my_rank = pm->commRank();
  ItemInternalMap& faces_map = m_mesh->facesMap();
  // TODO: regarder s'il est correcte de modifier ItemFlags::II_SubDomainBoundary
  const int shared_and_boundary_flags = ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary;
  // Parcours les faces et marque les nœuds, arêtes et faces frontières
  faces_map.eachItem([&](Face face) {
    Int32 nb_own = 0;
    for (Integer i = 0, n = face.nbCell(); i < n; ++i)
      if (face.cell(i).owner() == my_rank)
        ++nb_own;
    if (nb_own == 1) {
      face.mutableItemBase().addFlags(shared_and_boundary_flags);
      //++nb_sub_domain_boundary_face;
      for (Item inode : face.nodes()) {
        inode.mutableItemBase().addFlags(shared_and_boundary_flags);
        node_layer[inode.localId()] = 1;
      }
      for (Item iedge : face.edges())
        iedge.mutableItemBase().addFlags(shared_and_boundary_flags);
    }
  });
  _markBoundaryNodesFromEdges(node_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_addGhostLayer(Integer current_layer, Int32ConstArrayView node_layer)
{
  info() << "Processing ghost layer " << current_layer;

  SharedArray<BoundaryNodeInfo> boundary_node_list;
  //boundary_node_list.reserve(boundary_nodes_uid_count);

  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  bool is_verbose = m_is_verbose;

  ItemInternalMap& cells_map = m_mesh->cellsMap();
  ItemInternalMap& nodes_map = m_mesh->nodesMap();

  Int64 nb_added_for_different_rank = 0;
  Int64 nb_added_for_in_layer = 0;

  const Int32 max_local_id = m_mesh->nodeFamily()->maxLocalId();

  // Tableaux contenant pour chaque nœud le uid de la plus petite maille connectée
  // et le rang associé. Si le uid est A_NULL_UNIQUE_ID il ne faut pas ajouter ce nœud.
  UniqueArray<Int64> node_cell_uids(max_local_id, NULL_ITEM_UNIQUE_ID);

  const bool do_only_minimal_uid = m_use_only_minimal_cell_uid;
  // On doit envoyer tous les nœuds dont le numéro de couche est différent de (-1).
  // NOTE: pour la couche au dessus de 1, il ne faut envoyer qu'une seule valeur.
  cells_map.eachItem([&](Cell cell) {
    // Ne traite pas les mailles qui ne m'appartiennent pas
    if (m_version >= 4 && cell.owner() != my_rank)
      return;
    Int64 cell_uid = cell.uniqueId();
    for (Node node : cell.nodes()) {
      Int32 node_lid = node.localId();
      bool do_it = false;
      if (cell.owner() != my_rank) {
        do_it = true;
        ++nb_added_for_different_rank;
      }
      else {
        Integer layer = node_layer[node_lid];
        do_it = layer <= current_layer;
        if (do_it)
          ++nb_added_for_in_layer;
      }
      if (do_it) {
        Int32 node_lid = node.localId();
        if (do_only_minimal_uid) {
          Int64 current_uid = node_cell_uids[node_lid];
          if ((current_uid == NULL_ITEM_UNIQUE_ID) || cell_uid < current_uid) {
            node_cell_uids[node_lid] = cell_uid;
            if (is_verbose)
              info() << "AddNode node_uid=" << node.uniqueId() << " cell=" << cell_uid;
          }
          else if (is_verbose)
            info() << "AddNode node_uid=" << node.uniqueId() << " cell=" << cell_uid << " not done current=" << current_uid;
        }
        else {
          Int64 node_uid = node.uniqueId();
          BoundaryNodeInfo nci;
          nci.node_uid = node_uid;
          nci.cell_uid = cell_uid;
          nci.cell_owner = my_rank;
          boundary_node_list.add(nci);
          if (is_verbose)
            info() << "AddNode node_uid=" << node.uniqueId() << " cell=" << cell_uid;
        }
      }
    }
  });

  if (do_only_minimal_uid) {
    nodes_map.eachItem([&](Node node) {
      Int32 lid = node.localId();
      Int64 cell_uid = node_cell_uids[lid];
      if (cell_uid != NULL_ITEM_UNIQUE_ID) {
        Int64 node_uid = node.uniqueId();
        BoundaryNodeInfo nci;
        nci.node_uid = node_uid;
        nci.cell_uid = cell_uid;
        nci.cell_owner = my_rank;
        boundary_node_list.add(nci);
      }
    });
  }

  info() << "NB BOUNDARY NODE LIST=" << boundary_node_list.size()
         << " nb_added_for_different_rank=" << nb_added_for_different_rank
         << " nb_added_for_in_layer=" << nb_added_for_in_layer
         << " do_only_minimal=" << do_only_minimal_uid;

  _sortBoundaryNodeList(boundary_node_list);
  SharedArray<BoundaryNodeInfo> all_boundary_node_info = boundary_node_list;

  UniqueArray<BoundaryNodeToSendInfo> node_list_to_send;
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer bi_n = all_bni.size();
    for (Integer i = 0; i < bi_n; ++i) {
      const BoundaryNodeInfo& bni = all_bni[i];
      // Recherche tous les éléments de all_bni qui ont le même noeud.
      // Cela représente toutes les mailles connectées à ce noeud.
      Int64 node_uid = bni.node_uid;
      Integer last_i = i;
      for (; last_i < bi_n; ++last_i)
        if (all_bni[last_i].node_uid != node_uid)
          break;
      Integer nb_same_node = (last_i - i);
      if (is_verbose)
        info() << "NB_SAME_NODE uid=" << node_uid << " n=" << nb_same_node << " last_i=" << last_i;
      // Maintenant, regarde si les mailles connectées à ce noeud ont le même propriétaire.
      // Si c'est le cas, il s'agit d'un vrai noeud frontière et il n'y a donc rien à faire.
      // Sinon, il faudra envoyer la liste des mailles à tous les PE dont les rangs apparaissent dans cette liste
      Int32 owner = bni.cell_owner;
      bool has_ghost = false;
      for (Integer z = 0; z < nb_same_node; ++z)
        if (all_bni[i + z].cell_owner != owner) {
          has_ghost = true;
          break;
        }
      if (has_ghost) {
        BoundaryNodeToSendInfo si;
        si.m_index = i;
        si.m_nb_cell = nb_same_node;
        node_list_to_send.add(si);
        if (is_verbose)
          info() << "Add ghost uid=" << node_uid << " index=" << i << " nb_same_node=" << nb_same_node;
      }
      i = last_i - 1;
    }
  }

  IntegerUniqueArray nb_info_to_send(nb_rank, 0);
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer nb_node_to_send = node_list_to_send.size();
    std::set<Int32> ranks_done;
    for (Integer i = 0; i < nb_node_to_send; ++i) {
      Integer index = node_list_to_send[i].m_index;
      Integer nb_cell = node_list_to_send[i].m_nb_cell;

      ranks_done.clear();

      for (Integer kz = 0; kz < nb_cell; ++kz) {
        Int32 krank = all_bni[index + kz].cell_owner;
        if (ranks_done.find(krank) == ranks_done.end()) {
          ranks_done.insert(krank);
          // Pour chacun, il faudra envoyer
          // - le nombre de mailles (1*Int64)
          // - le uid du noeud (1*Int64)
          // - le uid et le rank de chaque maille (2*Int64*nb_cell)
          //TODO: il est possible de stocker les rangs sur Int32
          nb_info_to_send[krank] += (nb_cell * 2) + 2;
        }
      }
    }
  }

  if (is_verbose) {
    for (Integer i = 0; i < nb_rank; ++i) {
      Integer nb_to_send = nb_info_to_send[i];
      if (nb_to_send != 0)
        info() << "NB_TO_SEND rank=" << i << " n=" << nb_to_send;
    }
  }

  Integer total_nb_to_send = 0;
  IntegerUniqueArray nb_info_to_send_indexes(nb_rank, 0);
  for (Integer i = 0; i < nb_rank; ++i) {
    nb_info_to_send_indexes[i] = total_nb_to_send;
    total_nb_to_send += nb_info_to_send[i];
  }
  info() << "TOTAL_NB_TO_SEND=" << total_nb_to_send;

  UniqueArray<Int64> resend_infos(total_nb_to_send);
  {
    ConstArrayView<BoundaryNodeInfo> all_bni = all_boundary_node_info;
    Integer nb_node_to_send = node_list_to_send.size();
    std::set<Int32> ranks_done;
    for (Integer i = 0; i < nb_node_to_send; ++i) {
      Integer node_index = node_list_to_send[i].m_index;
      Integer nb_cell = node_list_to_send[i].m_nb_cell;
      Int64 node_uid = all_bni[node_index].node_uid;

      ranks_done.clear();

      for (Integer kz = 0; kz < nb_cell; ++kz) {
        Int32 krank = all_bni[node_index + kz].cell_owner;
        if (ranks_done.find(krank) == ranks_done.end()) {
          ranks_done.insert(krank);
          Integer send_index = nb_info_to_send_indexes[krank];
          resend_infos[send_index] = node_uid;
          ++send_index;
          resend_infos[send_index] = nb_cell;
          ++send_index;
          for (Integer zz = 0; zz < nb_cell; ++zz) {
            resend_infos[send_index] = all_bni[node_index + zz].cell_uid;
            ++send_index;
            resend_infos[send_index] = all_bni[node_index + zz].cell_owner;
            ++send_index;
          }
          nb_info_to_send_indexes[krank] = send_index;
        }
      }
    }
  }

  IntegerUniqueArray nb_info_to_recv(nb_rank, 0);
  {
    Timer::SimplePrinter sp(traceMng(), "Sending size with AllToAll");
    pm->allToAll(nb_info_to_send, nb_info_to_recv, 1);
  }

  if (is_verbose)
    for (Integer i = 0; i < nb_rank; ++i)
      info() << "NB_TO_RECV: I=" << i << " n=" << nb_info_to_recv[i];

  Integer total_nb_to_recv = 0;
  for (Integer i = 0; i < nb_rank; ++i)
    total_nb_to_recv += nb_info_to_recv[i];

  // Il y a de fortes chances que cela ne marche pas si le tableau est trop grand,
  // il faut proceder avec des tableaux qui ne depassent pas 2Go a cause des
  // Int32 de MPI.
  // TODO: Faire le AllToAll en plusieurs fois si besoin.
  // TOOD: Fusionner ce code avec celui de FaceUniqueIdBuilder2.
  UniqueArray<Int64> recv_infos;
  {
    Int32 vsize = sizeof(Int64) / sizeof(Int64);
    Int32UniqueArray send_counts(nb_rank);
    Int32UniqueArray send_indexes(nb_rank);
    Int32UniqueArray recv_counts(nb_rank);
    Int32UniqueArray recv_indexes(nb_rank);
    Int32 total_send = 0;
    Int32 total_recv = 0;
    for (Integer i = 0; i < nb_rank; ++i) {
      send_counts[i] = (Int32)(nb_info_to_send[i] * vsize);
      recv_counts[i] = (Int32)(nb_info_to_recv[i] * vsize);
      send_indexes[i] = total_send;
      recv_indexes[i] = total_recv;
      total_send += send_counts[i];
      total_recv += recv_counts[i];
    }
    recv_infos.resize(total_nb_to_recv);

    Int64ConstArrayView send_buf(total_nb_to_send * vsize, (Int64*)resend_infos.data());
    Int64ArrayView recv_buf(total_nb_to_recv * vsize, (Int64*)recv_infos.data());

    info() << "BUF_SIZES: send=" << send_buf.size() << " recv=" << recv_buf.size();
    {
      Timer::SimplePrinter sp(traceMng(), "Send values with AllToAll");
      pm->allToAllVariable(send_buf, send_counts, send_indexes, recv_buf, recv_counts, recv_indexes);
    }
  }

  SubDomainItemMap cells_to_send(50, true);

  // TODO: il n'y a a priori pas besoin d'avoir les mailles ici mais
  // seulement la liste des procs a qui il faut envoyer. Ensuite,
  // si le proc connait a qui il doit envoyer, il peut envoyer les mailles
  // à ce moment la. Cela permet d'envoyer moins d'infos dans le AllToAll précédent

  {
    Integer index = 0;
    UniqueArray<Int32> my_cells;
    SharedArray<Int32> ranks_to_send;
    std::set<Int32> ranks_done;
    while (index < total_nb_to_recv) {
      Int64 node_uid = recv_infos[index];
      ++index;
      Int64 nb_cell = recv_infos[index];
      ++index;
      Node current_node(nodes_map.findItem(node_uid));
      if (is_verbose)
        info() << "NODE uid=" << node_uid << " nb_cell=" << nb_cell << " idx=" << (index - 2);
      my_cells.clear();
      ranks_to_send.clear();
      ranks_done.clear();
      for (Integer kk = 0; kk < nb_cell; ++kk) {
        Int64 cell_uid = recv_infos[index];
        ++index;
        Int32 cell_owner = CheckedConvert::toInt32(recv_infos[index]);
        ++index;
        if (kk == 0 && current_layer == 1 && m_is_allocate)
          // Je suis la maille de plus petit uid et donc je
          // positionne le propriétaire du noeud.
          // TODO: ne pas faire cela ici, mais le faire dans une routine à part.
          nodes_map.findItem(node_uid).toMutable().setOwner(cell_owner, my_rank);
        if (is_verbose)
          info() << " CELL=" << cell_uid << " owner=" << cell_owner;
        if (cell_owner == my_rank) {
          impl::ItemBase dcell = cells_map.tryFind(cell_uid);
          if (dcell.null())
            ARCANE_FATAL("Internal error: cell uid={0} is not in our mesh", cell_uid);
          if (do_only_minimal_uid) {
            // Ajoute toutes les mailles autour de mon noeud
            for (CellLocalId c : current_node.cellIds())
              my_cells.add(c);
          }
          else
            my_cells.add(dcell.localId());
        }
        else {
          if (ranks_done.find(cell_owner) == ranks_done.end()) {
            ranks_to_send.add(cell_owner);
            ranks_done.insert(cell_owner);
          }
        }
      }

      if (is_verbose) {
        info() << "CELLS TO SEND: node_uid=" << node_uid
               << " nb_rank=" << ranks_to_send.size()
               << " nb_cell=" << my_cells.size();
        info(4) << "CELLS TO SEND: node_uid=" << node_uid
                << " rank=" << ranks_to_send
                << " cell=" << my_cells;
      }

      for (Integer zrank = 0, zn = ranks_to_send.size(); zrank < zn; ++zrank) {
        Int32 send_rank = ranks_to_send[zrank];
        SubDomainItemMap::Data* d = cells_to_send.lookupAdd(send_rank);
        Int32Array& c = d->value();
        for (Integer zid = 0, zid_size = my_cells.size(); zid < zid_size; ++zid) {
          // TODO: regarder si maille pas déjà présente et ne pas l'ajouter si ce n'est pas nécessaire.
          c.add(my_cells[zid]);
        }
      }
    }
  }

  info() << "GHOST V3 SERIALIZE CELLS";
  _sendAndReceiveCells(cells_to_send);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Trie parallèle de la liste des infos sur les noeuds frontières.
 *
 * Récupère en entrée une liste de noeuds frontières et la trie en parallèle
 * en s'assurant qu'après le tri les infos d'un même noeud sont sur le
 * même proc.
 */
void GhostLayerBuilder2::
_sortBoundaryNodeList(Array<BoundaryNodeInfo>& boundary_node_list)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  bool is_verbose = m_is_verbose;

  Parallel::BitonicSort<BoundaryNodeInfo, BoundaryNodeBitonicSortTraits> boundary_node_sorter(pm);
  boundary_node_sorter.setNeedIndexAndRank(false);

  {
    Timer::SimplePrinter sp(traceMng(), "Sorting boundary nodes");
    boundary_node_sorter.sort(boundary_node_list);
  }

  if (is_verbose) {
    ConstArrayView<BoundaryNodeInfo> all_bni = boundary_node_sorter.keys();
    Integer n = all_bni.size();
    for (Integer i = 0; i < n; ++i) {
      const BoundaryNodeInfo& bni = all_bni[i];
      info() << "NODES_KEY i=" << i
             << " node=" << bni.node_uid
             << " cell=" << bni.cell_uid
             << " rank=" << bni.cell_owner;
    }
  }

  // TODO: il n'y a pas besoin d'envoyer toutes les mailles.
  // pour déterminer le propriétaire d'un noeud, il suffit
  // que chaque PE envoie sa maille de plus petit UID.
  // Ensuite, chaque noeud a besoin de savoir la liste
  // des sous-domaines connectés pour renvoyer l'info. Chaque
  // sous-domaine en sachant cela saura a qui il doit envoyer
  // les mailles fantomes.

  {
    ConstArrayView<BoundaryNodeInfo> all_bni = boundary_node_sorter.keys();
    Integer n = all_bni.size();
    // Comme un même noeud peut être présent dans la liste du proc précédent, chaque PE
    // (sauf le 0) envoie au proc précédent le début sa liste qui contient les même noeuds.

    UniqueArray<BoundaryNodeInfo> end_node_list;
    Integer begin_own_list_index = 0;
    if (n != 0 && my_rank != 0) {
      if (BoundaryNodeBitonicSortTraits::isValid(all_bni[0])) {
        Int64 node_uid = all_bni[0].node_uid;
        for (Integer i = 0; i < n; ++i) {
          if (all_bni[i].node_uid != node_uid) {
            begin_own_list_index = i;
            break;
          }
          else
            end_node_list.add(all_bni[i]);
        }
      }
    }
    info() << "BEGIN_OWN_LIST_INDEX=" << begin_own_list_index << " end_node_list_size=" << end_node_list.size();
    if (is_verbose) {
      for (Integer k = 0, kn = end_node_list.size(); k < kn; ++k)
        info() << " SEND node_uid=" << end_node_list[k].node_uid
               << " cell_uid=" << end_node_list[k].cell_uid;
    }

    UniqueArray<BoundaryNodeInfo> end_node_list_recv;

    UniqueArray<Parallel::Request> requests;
    Integer recv_message_size = 0;
    Integer send_message_size = BoundaryNodeBitonicSortTraits::messageSize(end_node_list);

    // Envoie et réceptionne d'abord les tailles.
    if (my_rank != (nb_rank - 1)) {
      requests.add(pm->recv(IntegerArrayView(1, &recv_message_size), my_rank + 1, false));
    }
    if (my_rank != 0) {
      requests.add(pm->send(IntegerConstArrayView(1, &send_message_size), my_rank - 1, false));
    }
    info() << "Send size=" << send_message_size << " Recv size=" << recv_message_size;
    pm->waitAllRequests(requests);
    requests.clear();

    if (recv_message_size != 0) {
      Int32 nb_element = BoundaryNodeInfo::nbElement(recv_message_size);
      end_node_list_recv.resize(nb_element);
      requests.add(BoundaryNodeBitonicSortTraits::recv(pm, my_rank + 1, end_node_list_recv));
    }
    if (send_message_size != 0)
      requests.add(BoundaryNodeBitonicSortTraits::send(pm, my_rank - 1, end_node_list));

    pm->waitAllRequests(requests);

    boundary_node_list.clear();
    boundary_node_list.addRange(all_bni.subConstView(begin_own_list_index, n - begin_own_list_index));
    boundary_node_list.addRange(end_node_list_recv);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_sendAndReceiveCells(SubDomainItemMap& cells_to_send)
{
  auto exchanger{ ParallelMngUtils::createExchangerRef(m_parallel_mng) };

  const bool is_verbose = m_is_verbose;

  // Envoie et réceptionne les mailles fantômes
  for (SubDomainItemMap::Enumerator i_map(cells_to_send); ++i_map;) {
    Int32 sd = i_map.data()->key();
    Int32Array& items = i_map.data()->value();

    // Comme la liste par sous-domaine peut contenir plusieurs
    // fois la même maille, on trie la liste et on supprime les
    // doublons
    std::sort(std::begin(items), std::end(items));
    auto new_end = std::unique(std::begin(items), std::end(items));
    items.resize(CheckedConvert::toInteger(new_end - std::begin(items)));
    if (is_verbose)
      info(4) << "CELLS TO SEND SD=" << sd << " Items=" << items;
    else
      info(4) << "CELLS TO SEND SD=" << sd << " nb=" << items.size();
    exchanger->addSender(sd);
  }
  exchanger->initializeCommunicationsMessages();
  for (Integer i = 0, ns = exchanger->nbSender(); i < ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToSend(i);
    Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    Int32ConstArrayView items_to_send = cells_to_send[rank];
    m_mesh->serializeCells(s, items_to_send);
  }
  exchanger->processExchange();
  info(4) << "END EXCHANGE CELLS";
  for (Integer i = 0, ns = exchanger->nbReceiver(); i < ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* s = sm->serializer();
    m_mesh->addCells(s);
  }
  m_mesh_builder->printStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Marque les entitées au bord du sous-domaine.
 *
 * Cela suppose que les faces aient déja été marquées avec le flag II_Boundary
 * et que leur propriétaire soit correctement positionné (i.e: le même pour
 * tous les sous-domaines).
 */
void GhostLayerBuilder2::
_markBoundaryItems(ArrayView<Int32> node_layer)
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  ItemInternalMap& faces_map = m_mesh->facesMap();

  const int shared_and_boundary_flags = ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary;

  // Parcours les faces et marque les nœuds, arêtes et faces frontières
  faces_map.eachItem([&](Face face) {
    bool is_sub_domain_boundary_face = false;
    if (face.itemBase().flags() & ItemFlags::II_Boundary) {
      is_sub_domain_boundary_face = true;
    }
    else {
      if (face.nbCell() == 2 && (face.cell(0).owner() != my_rank || face.cell(1).owner() != my_rank))
        is_sub_domain_boundary_face = true;
    }
    if (is_sub_domain_boundary_face) {
      face.mutableItemBase().addFlags(shared_and_boundary_flags);
      for (Item inode : face.nodes())
        inode.mutableItemBase().addFlags(shared_and_boundary_flags);
      for (Item iedge : face.edges())
        iedge.mutableItemBase().addFlags(shared_and_boundary_flags);
    }
  });
  _markBoundaryNodesFromEdges(node_layer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerBuilder2::
_markBoundaryNodesFromEdges(ArrayView<Int32> node_layer)
{
  const bool is_non_manifold = m_mesh->meshKind().isNonManifold();
  if (!is_non_manifold)
    return;

  const int shared_and_boundary_flags = ItemFlags::II_Shared | ItemFlags::II_SubDomainBoundary;

  info() << "Mark boundary nodes from edges for non-manifold mesh";
  // Parcours l'ensemble des arêtes.
  // Si une arête est connectée à une seule maille de dimension 2
  // dont on est le propriétaire, alors il s'agit d'une arête de bord
  // et on marque les noeuds correspondants.
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  ItemInternalMap& edges_map = m_mesh->edgesMap();
  edges_map.eachItem([&](Edge edge) {
    Int32 nb_cell = edge.nbCell();
    Int32 nb_dim2_cell = 0;
    Int32 nb_own_dim2_cell = 0;
    for (Cell cell : edge.cells()) {
      Int32 dim = cell.typeInfo()->dimension();
      if (dim == 2) {
        ++nb_dim2_cell;
        if (cell.owner() == my_rank)
          ++nb_own_dim2_cell;
      }
    }
    if (nb_dim2_cell == nb_cell && nb_own_dim2_cell == 1) {
      edge.mutableItemBase().addFlags(shared_and_boundary_flags);
      for (Item inode : edge.nodes()) {
        inode.mutableItemBase().addFlags(shared_and_boundary_flags);
        node_layer[inode.localId()] = 1;
      }
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Cette fonction gère les versions 3 et 4 de calcul des entités fantômes.
extern "C++" void
_buildGhostLayerNewVersion(DynamicMesh* mesh, bool is_allocate, Int32 version)
{
  GhostLayerBuilder2 glb(mesh->m_mesh_builder, is_allocate, version);
  glb.addGhostLayers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
