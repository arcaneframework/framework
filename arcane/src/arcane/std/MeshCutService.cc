// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCutService.cc                                           (C) 2000-2026 */
/*                                                                           */
/* Service allowing the creation of a mesh with a cut of another mesh.       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"

#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshSection.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/MeshBuildInfo.h"

#include "arcane/std/MeshCut_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
/*!
 * \brief Noeud sur arête.
 * Représente le noeud à l'intersection entre une arête du maillage 3D et
 * un plan.
 * Si les deux noeuds de la struct sont identiques, c'est qu'il y a un
 * noeud du maillage 3D qui est sur le plan.
 */
struct NodeOnEdge
{
  NodeOnEdge(const Node first_node, const Node second_node)
  {
    const Int64 a = first_node.uniqueId();
    const Int64 b = second_node.uniqueId();
    if (a < b) {
      m_node0 = first_node;
      m_node1 = second_node;
      m_uid_node0 = a;
      m_uid_node1 = b;
    }
    else {
      m_node0 = second_node;
      m_node1 = first_node;
      m_uid_node0 = b;
      m_uid_node1 = a;
    }
  }

  NodeOnEdge(Int64 first_node, Int64 second_node)
  {
    if (first_node < second_node) {
      m_uid_node0 = first_node;
      m_uid_node1 = second_node;
    }
    else {
      m_uid_node0 = second_node;
      m_uid_node1 = first_node;
    }
  }

  NodeOnEdge() = default;

  bool operator<(const NodeOnEdge& other) const
  {
    if (m_uid_node0 != other.m_uid_node0) {
      return m_uid_node0 < other.m_uid_node0;
    }
    return m_uid_node1 < other.m_uid_node1;
  }

  bool operator==(const NodeOnEdge& other) const
  {
    return m_uid_node0 == other.m_uid_node0 && m_uid_node1 == other.m_uid_node1;
  }

  Node m_node0;
  Node m_node1;

  Int64 m_uid_node0 = -1;
  Int64 m_uid_node1 = -1;

  Int64 m_uid_new_node = -1;
  Int32 m_owner_new_node = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Face du maillage 2D issue de deux noeuds d'arêtes.
 */
struct FaceLite
{
  FaceLite(const Ref<NodeOnEdge>& node0, const Ref<NodeOnEdge>& node1)
  : m_node0(node0->m_uid_node0 < node1->m_uid_node0 ? node0 : node1)
  , m_node1(node0->m_uid_node0 < node1->m_uid_node0 ? node1 : node0)
  {
    const Int64 node00 = node0->m_uid_node0;
    const Int64 node01 = node0->m_uid_node1;
    const Int64 node10 = node1->m_uid_node0;
    const Int64 node11 = node1->m_uid_node1;

    if (node00 < node10) {
      m_node0 = node0;
      m_node1 = node1;
    }
    else if (node00 > node10) {
      m_node0 = node1;
      m_node1 = node0;
    }
    else {
      if (node01 == node11) {
        ARCANE_FATAL("Impossible (normalement...)");
      }
      if (node01 < node11) {
        m_node0 = node0;
        m_node1 = node1;
      }
      else {
        m_node0 = node1;
        m_node1 = node0;
      }
    }
  }

  bool operator<(const FaceLite& other) const
  {
    return m_node0->operator<(*(other.m_node0.get())) || (m_node0->operator==(*(other.m_node0.get())) && m_node1->operator<(*(other.m_node1.get())));
  }

  bool operator==(const FaceLite& other) const
  {
    return m_node0->operator==(*(other.m_node0.get())) && m_node1->operator==(*(other.m_node1.get()));
  }

  Ref<NodeOnEdge> m_node0;
  Ref<NodeOnEdge> m_node1;

  Int64 m_uid_new_face = -1;
  Int32 m_owner_new_face = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Position d'un noeud d'arête.
 */
struct NodeIntersection
{
  NodeIntersection(const Node first_node, const Node second_node, const Real3& intersection_pos)
  : m_new_node(makeRef(new NodeOnEdge(first_node, second_node)))
  , m_intersection_pos(intersection_pos)
  {}

  NodeIntersection(Int64 first_node, Int64 second_node, const Real3& intersection_pos)
  : m_new_node(makeRef(new NodeOnEdge(first_node, second_node)))
  , m_intersection_pos(intersection_pos)
  {}

  NodeIntersection() = default;

  bool operator<(const NodeIntersection& other) const
  {
    return m_new_node->operator<(*(other.m_new_node.get()));
  }

  bool operator==(const NodeIntersection& other) const
  {
    return m_new_node->operator==(*(other.m_new_node.get()));
  }

  Ref<NodeOnEdge> m_new_node;
  Real3 m_intersection_pos{ -1 };
};

struct UnknownNode
{
  UnknownNode(Int64 node0_uid, Int64 node1_uid, Int32 who) : m_node0_uid(node0_uid), m_node1_uid(node1_uid), m_who(who){}
  bool null() const {return m_who == -1;}
  bool operator==(const UnknownNode& other) const{return m_node0_uid == other.m_node0_uid && m_node1_uid == other.m_node1_uid;}
  Int64 m_node0_uid;
  Int64 m_node1_uid;
  Int32 m_who;
};
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service allowing the creation of a mesh with a cut of another mesh.
 *
 * For each plan, a cut will be realized. All cuts will be stored in the mesh
 * created by this service. To get this mesh, you can call \a meshSection()
 * method.
 */
class MeshCutService
: public ArcaneMeshCutObject
{
 public:

  explicit MeshCutService(const ServiceBuildInfo& sbi)
  : ArcaneMeshCutObject(sbi)
  , m_creation_type(sbi.creationType())
  {}

 public:

  void addPlane(const Real3& p0, const Real3& normal) override;

  void setVariables(VariableCollection variables) override;

  void setServiceMeshUniqueId(Int32 unique_id) override;

  VariableCollection variables() override { return {}; }

  void updateSection() override;

  MeshHandle meshSection() override
  {
    return m_cloned_mesh->handle();
  }

 private:

  void _createMesh();
  void _createNodesAndCells(Int32 plan_pos, Int32& sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 ajust_node_pos, Int32& sd_nb_cell, UniqueArray<Int64>& new_cells, Int32& sd_nb_face, UniqueArray<FaceLite>& new_faces);
  void _makeUniqueCellUID(Int32 sd_nb_cell, UniqueArray<Int64>& new_cells, UniqueArray<NodeIntersection>& new_nodes);

  void _fillNodeUID(Int32& sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 current_plan_pos);
  Int32 _makeUniqueNodeUID(Int32 sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 current_plan);
  void _fillFaceUID(Int32& sd_nb_face, UniqueArray<FaceLite>& new_faces, Int32 current_plan_pos);
  Int32 _makeUniqueFaceUID(Int32 sd_nb_face, UniqueArray<FaceLite>& new_faces, Int32 current_plan);
  void _compute();

  std::optional<Int64> _find(Span<FaceLite> new_faces, Int64 node_uid0, Int64 node_uid1);

  void _addFaces(UniqueArray<FaceLite>& new_faces) const;
  void _addCells(UniqueArray<Int64>& new_cells);
  void _setCoordNodesAndOwner(UniqueArray<NodeIntersection>& new_nodes);
  void _setFacesOwner(UniqueArray<FaceLite>& new_faces);

 private:

  eServiceType m_creation_type;
  // VariableCollection m_variables_ori;
  IPrimaryMesh* m_cloned_mesh = nullptr;
  UniqueArray<std::pair<Real3, Real3>> m_plans;
  Int32 m_mesh_uid = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHCUT(MeshCutService, MeshCutService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
addPlane(const Real3& p0, const Real3& normal)
{
  m_plans.add({ p0, math::normalizeReal3(normal) });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
setVariables(VariableCollection variables)
{
  ARCANE_UNUSED(variables);
  ARCANE_NOT_YET_IMPLEMENTED("Not supported yet");
  // m_variables_ori = variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
setServiceMeshUniqueId(Int32 unique_id)
{
  m_mesh_uid = unique_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
updateSection()
{
  if (mesh()->ghostLayerMng()->nbGhostLayer() < 1) {
    ARCANE_FATAL("A ghost layer is required for this service");
  }
  _createMesh();
  _compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_createMesh()
{
  if (m_cloned_mesh != nullptr) {
    m_cloned_mesh->modifier()->clearItems();
  }

  if (mesh()->dimension() != 3) {
    ARCANE_FATAL("Only 3D meshes are supported");
  }

  IMeshMng* mm = subDomain()->meshMng();

  if (m_mesh_uid == -1) {
    if (m_creation_type == ST_CaseOption) {
      m_mesh_uid = options()->getUniqueIdServiceMesh();
    }
    else {
      m_mesh_uid = 0;
    }
  }

  String service_mesh_name = mesh()->name() + "_MeshCut" + m_mesh_uid;

  MeshHandle* mesh_handle = mm->findMeshHandle(service_mesh_name, false);

  if (mesh_handle == nullptr) {
    IParallelMng* pm = subDomain()->parallelMng();
    MeshBuildInfo mbi(service_mesh_name);
    // auto mesh_kind = mbi.meshKind();
    // mesh_kind.setMeshDimensionKind(eMeshCellDimensionKind::NonManifold);
    // mbi.addMeshKind(mesh_kind);
    mbi.addParallelMng(makeRef(pm));
    m_cloned_mesh = mm->meshFactoryMng()->createMesh(mbi);
    m_cloned_mesh->modifier()->setDynamic(true);
    m_cloned_mesh->setDimension(2);
    m_cloned_mesh->endAllocate();
  }
  else {
    m_cloned_mesh = mesh_handle->mesh()->toPrimaryMesh();
    m_cloned_mesh->modifier()->clearItems();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_createNodesAndCells(Int32 plan_pos, Int32& sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 ajust_node_pos, Int32& sd_nb_cell, UniqueArray<Int64>& new_cells, Int32& sd_nb_face, UniqueArray<FaceLite>& new_faces)
{
  auto [p0, normal] = m_plans[plan_pos];

  const Int32 mesh_dim = mesh()->dimension();

  VariableNodeReal3& node_coord = mesh()->nodesCoordinates();
  VariableNodeReal node_dist(VariableBuildInfo(mesh(), "NodeDist"));

  UniqueArray<Int32> face_already_computed;

  // Calcul de la distance signée de chaque noeud par rapport au plan de coupe.
  ENUMERATE_ (Node, inode, allNodes()) {
    Real d = math::dot({ node_coord[inode] - p0 }, normal);
    node_dist[inode] = math::isNearlyZeroWithEpsilon(d, 1e-10) ? 0 : d;
  }

  // Tableau qui contiendra tous les noeuds d'une future maille.
  // Une maille 3D, si elle est coupée par le plan, donnera forcément une
  // maille 2D. Les noeuds de cette maille seront stockés dans ce tableau,
  // ainsi que, pour chaque noeud, sa position et l'arête dont il est issu
  // (pour éviter les doublons).
  UniqueArray<NodeIntersection> point_coords_tmp;

  ENUMERATE_ (Cell, icell, ownCells()) {
    Cell cell = *icell;

    bool has_face_on_plane = false;
    {
      bool cell_useful = true;
      Int32 nb_node_on_plane = 0;

      // On regarde si la maille traverse le plan ou si des noeuds sont sur le
      // plan.
      {
        bool has_neg = false;
        bool has_pos = false;
        for (Node node : cell.nodes()) {
          Real d = node_dist[node];
          if (d < 0)
            has_neg = true;
          else if (d > 0)
            has_pos = true;
          else {
            nb_node_on_plane++;
          }
        }

        // Tous les noeuds sont du même coté du plan.
        if (!(has_neg && has_pos))
          cell_useful = false;
      }

      // Si le nombre de noeuds sur le plan correspond au nombre de noeuds
      // minimum d'une face, il s'agit peut-être d'une face confondue au plan.
      // Dans ce cas, il est nécessaire de faire un traitement spécial pour
      // éviter un doublon de mailles dans le maillage final.
      if (nb_node_on_plane >= mesh_dim) {
        for (Face face : cell.faces()) {
          // Pour éviter que deux processus créent la même maille.
          if (!face.isOwn())
            continue;
          if (face.nbNode() != nb_node_on_plane)
            continue;

          bool has_egal = true;
          for (Node node : face.nodes()) {
            if (node_dist[node] != 0) {
              has_egal = false;
              break;
            }
          }

          // On a trouvé la face confondue.
          if (has_egal) {

            // Si elle a déjà été traitée, on passe pour éviter un doublon.
            if (face_already_computed.contains(face.localId())) {
              cell_useful = false;
              break;
            }

            // On sait que la face est confondue au plan et que la future
            // maille créée grâce à cette face n'existe pas encore.
            // On peut donc directement ajouter les noeuds de la face dans la
            // liste des noeuds de la future maille.
            face_already_computed.add(face.localId());
            for (Node node : face.nodes()) {
              auto ni = NodeIntersection{ node, node, node_coord[node] };
              point_coords_tmp.add(ni);
            }
            has_face_on_plane = true;
            cell_useful = true;
            break;
          }
        }
      }

      if (!cell_useful)
        continue;
    }

    if (!has_face_on_plane) {
      // Tableaux définissant les arêtes pour chaque type d'élément.
      // Chaque arête = [n0, n1] indices noeuds locaux de maille.
      //
      static constexpr Integer edges_tetraedron4[6][2] = {
        { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 }
      };
      static constexpr Integer edges_pyramid5[8][2] = {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 }, { 0, 4 }, { 1, 4 }, { 2, 4 }, { 3, 4 }
      };
      static constexpr Integer edges_pentaedron6[9][2] = {
        { 0, 1 }, { 1, 2 }, { 2, 0 }, { 0, 3 }, { 1, 4 }, { 2, 5 }, { 3, 4 }, { 4, 5 }, { 5, 3 }
      };
      static constexpr Integer edges_hexaedron8[12][2] = {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }, { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 }
      };

      Integer nb_edges = 0;
      const Integer(*edge_def)[2] = nullptr;

      auto type = cell.itemTypeId();

      if (type == ITI_Tetraedron4) {
        nb_edges = 6;
        edge_def = edges_tetraedron4;
      }
      else if (type == ITI_Pyramid5) {
        nb_edges = 8;
        edge_def = edges_pyramid5;
      }
      else if (type == ITI_Pentaedron6) {
        nb_edges = 9;
        edge_def = edges_pentaedron6;
      }
      else if (type == ITI_Hexaedron8) {
        nb_edges = 12;
        edge_def = edges_hexaedron8;
      }

      else {
        ARCANE_FATAL("Cell type not supported -- type: {0}", type);
      }

      // Itère sur toutes les arêtes de la maille.
      for (Integer i = 0; i < nb_edges; ++i) {
        Node node0 = cell.node(edge_def[i][0]);
        Node node1 = cell.node(edge_def[i][1]);

        bool need_compute_intersection = true;

        // Si le noeud 0 est sur le plan.
        if (node_dist[node0] == 0) {
          const Real3 p = node_coord[node0];

          auto ni = NodeIntersection{ node0, node0, p };

          if (!point_coords_tmp.contains(ni)) {
            point_coords_tmp.add(ni);
          }
          need_compute_intersection = false;
        }

        // Si le noeud 1 est sur le plan.
        if (node_dist[node1] == 0) {
          const Real3 p = node_coord[node1];

          auto ni = NodeIntersection{ node1, node1, p };

          if (!point_coords_tmp.contains(ni)) {
            point_coords_tmp.add(ni);
          }
          need_compute_intersection = false;
        }

        // Si l'arrête passe à travers le plan.
        if (need_compute_intersection && node_dist[node0] * node_dist[node1] < 0) {

          // Paramètre d'interpolation t dans [0,1] pour le point d'intersection
          // le long de l'arête de node0 à node1.
          Real t = std::abs(node_dist[node0]) / (std::abs(node_dist[node0]) + std::abs(node_dist[node1]));

          // Calcul du point d'intersection par interpolation linéaire.
          Real3 p;
          p.x = node_coord[node0].x + t * (node_coord[node1].x - node_coord[node0].x);
          p.y = node_coord[node0].y + t * (node_coord[node1].y - node_coord[node0].y);
          p.z = node_coord[node0].z + t * (node_coord[node1].z - node_coord[node0].z);

          auto ni = NodeIntersection{ node0, node1, p };

          if (!point_coords_tmp.contains(ni)) {
            point_coords_tmp.add(ni);
          }
        }
      }
    }

    // Si le nombre de futurs noeuds est suffisant pour en faire une maille.
    if (point_coords_tmp.size() >= mesh_dim) {

      // Type cell.
      if (point_coords_tmp.size() > 6)
        ARCANE_FATAL("Pas implem : {0}", point_coords_tmp.size());

      new_cells.add(point_coords_tmp.size());
      new_cells.add(sd_nb_cell);

      {
        // Calcul du barycentre de tous les points d'intersection.
        Real3 bary{ 0 };
        for (const auto& node : point_coords_tmp) {
          bary += node.m_intersection_pos;
        }
        bary /= point_coords_tmp.size();

        // On choisit un vecteur de référence arbitraire non parallèle au normal.
        // Si normal.x est grand (proche de l'axe X), utiliser l'axe Y ; sinon utiliser l'axe X.
        const Real3 arbitrary = (std::abs(normal.x) > 0.9) ? Real3{ 0.0, 1.0, 0.0 } : Real3{ 1.0, 0.0, 0.0 };
        // u et v forment une base orthonormale du plan de coupe.
        // Ils sont perpendiculaires au normal et l'un à l'autre.
        const Real3 u = math::normalizedCrossProduct3(arbitrary, normal);
        const Real3 v = math::normalizedCrossProduct3(normal, u);

        // On trie les points d'intersection par angle polaire autour du barycentre.
        // Cela garantit que le polygone résultant est correctement ordonné (sens inverse des aiguilles d'une montre).
        UniqueArray<Int64> indices;
        indices.reserve(point_coords_tmp.size());
        for (Int64 i = 0; i < point_coords_tmp.size(); ++i) {
          indices.add(i);
        }

        std::sort(indices.begin(), indices.end(),
                  [&](const Int64 ia, const Int64 ib) {
                    const Real3& pa = point_coords_tmp[ia].m_intersection_pos;
                    const Real3& pb = point_coords_tmp[ib].m_intersection_pos;

                    // Vecteurs allant du barycentre vers chaque point.
                    const Real3 va{ pa - bary };
                    const Real3 vb{ pb - bary };

                    // Projeter sur la base 2D du plan (u, v).
                    const Real a_x = math::dot(va, u);
                    const Real a_y = math::dot(va, v);

                    const Real b_x = math::dot(vb, u);
                    const Real b_y = math::dot(vb, v);

                    // Comparer les angles en utilisant atan2.
                    const Real angle_a = std::atan2(a_y, a_x);
                    const Real angle_b = std::atan2(b_y, b_x);

                    return angle_a < angle_b;
                  });

        // On crée la nouvelle maille.
        // On en profite pour modifier le tableau "indices" :
        // - avant la boucle, ce tableau contient les indices de positions de
        //   noeuds dans le tableau point_coords_tmp.
        // - après la boucle, ce tableau contiendra les indices de positions
        //   de ces mêmes noeuds mais dans le tableau "global" new_nodes.
        for (Int64& idx : indices) {
          // On regarde si le noeud est déjà présent dans le tableau "global".
          std::optional<Int64> pos = new_nodes.span().findFirst(point_coords_tmp[idx]);
          if (pos) {
            // Attention : on ajoute la position du noeud dans le tableau
            // "global", pas le UID de celui-ci ! En effet, on ne le connait
            // pas encore, ce sera ajusté dans la méthode
            // "_makeUniqueCellUID()".
            new_cells.add(static_cast<Int32>(pos.value()) + ajust_node_pos);
            idx = pos.value();
          }
          // S'il n'est pas présent, on le crée.
          else {
            NodeIntersection& elem = point_coords_tmp[idx];

            // Si les deux noeuds ont le même proprio, on sait que le nouveau noeud aura le même proprio.
            if (elem.m_new_node->m_node0.owner() == elem.m_new_node->m_node1.owner()) {
              elem.m_new_node->m_owner_new_node = elem.m_new_node->m_node0.owner();
              // Si les deux noeuds sont à nous, on est sûr d'être le
              // propriétaire du nouveau noeud. C'est donc à nous de donner le
              // UniqueID (unique pour ce processus, la correction sera faite
              // plus tard, quand on saura le nombre de noeuds de chaque
              // processus).
              if (elem.m_new_node->m_node0.isOwn()) {
                elem.m_new_node->m_uid_new_node = sd_nb_node++;
              }
              else {
                // -2 = nécessite d'être ajouté plus tard.
                elem.m_new_node->m_uid_new_node = -2;
              }
            }
            new_cells.add(new_nodes.size() + ajust_node_pos);
            idx = new_nodes.size();
            new_nodes.add(elem);
          }
        }

        // Maintenant, on ajoute les faces.
        Int64 idxm1 = indices[indices.size() - 1];
        for (Int64 idx : indices) {
          FaceLite fl(new_nodes[idx].m_new_node, new_nodes[idxm1].m_new_node);
          if (!new_faces.contains(fl)) {
            // info() << "Add face"
            //        << " -- N00 " << fl.m_node0->m_uid_node0
            //        << " -- N01 " << fl.m_node0->m_uid_node1
            //        << " -- N10 " << fl.m_node1->m_uid_node0
            //        << " -- N11 " << fl.m_node1->m_uid_node1;

            // Même principe qu'avec les noeuds juste au-dessus.
            if (new_nodes[idx].m_new_node->m_owner_new_node == new_nodes[idxm1].m_new_node->m_owner_new_node) {
              fl.m_owner_new_face = new_nodes[idx].m_new_node->m_owner_new_node;
              if (fl.m_owner_new_face == subDomain()->subDomainId()) {
                fl.m_uid_new_face = sd_nb_face++;
              }
              else {
                fl.m_uid_new_face = -2;
              }
            }
            new_faces.add(fl);
          }
          idxm1 = idx;
        }
      }
      sd_nb_cell++;
    }
    point_coords_tmp.clear();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_makeUniqueCellUID(Int32 sd_nb_cell, UniqueArray<Int64>& new_cells, UniqueArray<NodeIntersection>& new_nodes)
{
  IParallelMng* pm = subDomain()->parallelMng();

  Int32 decal = sd_nb_cell;
  pm->scan(MessagePassing::ReduceSum, ArrayView{ 1, &decal });

  // Int32 nb_cells_global = decal;
  // pm->broadcast(ArrayView{ 1, &nb_cells_global }, pm->commSize() - 1);

  decal -= sd_nb_cell;

  // info() << "[" << pm->commRank() << "] Scan result (rectified) : " << decal;

  // En plus de décaler les uids des mailles pour les rendre unique sur tout le
  // domaine, on ajoute les vrais UID des noeuds à la place des positions des
  // noeuds dans le tableau "new_nodes".
  Int32 pos0 = 0;
  while (pos0 < new_cells.size()) {
    Int64 type = new_cells[pos0++];
    new_cells[pos0++] += decal;

    for (Int32 i = 0; i < type; ++i) {
      Int64& pos_to_uid = new_cells[pos0++];
      pos_to_uid = new_nodes[pos_to_uid].m_new_node->m_uid_new_node;
    }
  }
  // return nb_cells_global;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_fillNodeUID(Int32& sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 current_plan_pos)
{

  // À partir d'ici, nous avons un tableau de noeuds. Les noeuds dont on est
  // sûr qu'ils nous appartiennent ont déjà un UID/Owner.
  //
  // Il y a deux autres types de noeuds :
  // - le noeud dont on pense connaitre le proprio à qui demander le UID,
  // - le noeud dont on ne connait pas encore le proprio ni le UID.

  IParallelMng* pm = subDomain()->parallelMng();
  Int32 my_proc = pm->commRank();
  UniqueArray<UniqueArray<Ref<NodeOnEdge>>> requested_nodes(subDomain()->nbSubDomain());

  //
  //
  // Première étape : détermination des proprios potentiels et création des
  // requêtes.
  //
  // Détermination des proprios :
  // Si un noeud n'a pas encore de proprio, on choisit le proprio parmi les
  // mailles des noeuds (N0 et N1) de l'arête dont est issu notre noeud (NN).
  //
  // N0      NN      N1
  // *-------*-------*
  //
  // NN = elem.m_new_node
  // N0 = elem.m_new_node->m_node0
  // N1 = elem.m_new_node->m_node1
  //
  // Il peut y avoir des cas où le proprio choisi ne possède pas le noeud.
  // Ce cas sera traité plus tard dans la méthode.
  //
  // À partir de là, on a un proprio pour tous les noeuds.
  //
  // Création des requêtes :
  // On a un message pour chaque processus.
  // Ce message est composé de pairs de UID : les UID des noeuds de l'arête
  // dont est issu notre noeud (seul moyen d'identifier notre noeud sur tous
  // les processus (si l'on ne souhaite pas utiliser sa position)).
  //
  // Exemple :
  // request_uid[][] :
  // P0 : --Nous--
  // P1 : [NN09_N0_UID, NN09_N1_UID, NN01_N0_UID, NN01_N1_UID, NN00_N0_UID, NN00_N1_UID]
  // P2 : [NN14_N0_UID, NN14_N1_UID, NN15_N0_UID, NN15_N1_UID]
  // P3 : []
  //
  // request_uid[][] :
  // P0 : []
  // P1 : [NN09_N0_UID, NN09_N1_UID]
  // P2 : --Nous--
  // P3 : []
  //
  // request_uid[][] :
  // P0 : []
  // P1 : [NN00_N0_UID, NN00_N1_UID]
  // P2 : []
  // P3 : --Nous--
  //
  // On stocke aussi, dans un autre tableau, les NodeOnEdge correspondant
  // aux paires de UID, afin de les retrouver facilement lorsque l'on aura
  // reçu les réponses des processus.
  //
  // Exemple :
  // requested_nodes[][] :
  // P0 : --Nous--
  // P1 : [NN09, NN01, NN00]
  // P2 : [NN14, NN15]
  // P3 : []
  //
  //
  {
    UniqueArray<UniqueArray<Int64>> request_uid(subDomain()->nbSubDomain());

    info() << "[Node][" << my_proc << "] Step 1";

    for (auto& elem : new_nodes) {
      // Si le uid est déjà mis, pas besoin de le rechercher...
      if (elem.m_new_node->m_uid_new_node >= 0) {
        continue;
      }

      // Si le proprio n'est pas définit, on doit faire une recherche.
      if (elem.m_new_node->m_owner_new_node < 0) {

        Node node0 = elem.m_new_node->m_node0;
        Node node1 = elem.m_new_node->m_node1;

        // Le propriétaire du noeud est le propriétaire de la maille ayant le plus
        // petit UID, parmi les mailles en commun entre les deux noeuds d'origine.

        // TODO Ajouter traitement particulier pour le cas où node0 == node1
        Int64 min_uid = INT64_MAX;
        Int32 owner_min = -1;
        for (Cell cell0 : node0.cells()) {
          for (Cell cell1 : node1.cells()) {
            if (cell0 == cell1) {
              if (cell0.uniqueId() < min_uid) {
                min_uid = cell0.uniqueId();
                owner_min = cell0.owner();
              }
            }
          }
        }

        // S'il l'on est le proprio, on peut définir le uid du noeud.
        if (owner_min == subDomain()->subDomainId()) {
          elem.m_new_node->m_owner_new_node = subDomain()->subDomainId();
          elem.m_new_node->m_uid_new_node = sd_nb_node++;

          info() << "[Node][" << my_proc << " -> " << elem.m_new_node->m_owner_new_node << "] Set UID"
                 << " -- UID0 : " << elem.m_new_node->m_uid_node0
                 << " -- UID1 : " << elem.m_new_node->m_uid_node1
                 << " -- New UID : " << elem.m_new_node->m_uid_new_node;
        }

        // Sinon, on doit aller demander le uid au proprio.
        else {
          elem.m_new_node->m_owner_new_node = owner_min;
          elem.m_new_node->m_uid_new_node = -2;

          // Une requête est composée uniquement des deux UID des noeuds de
          // l'arête dont est issue le nouveau noeud. C'est le seul moyen
          // d'identifier ce noeud pour l'instant (si on exclut l'identification
          // par sa position).
          request_uid[elem.m_new_node->m_owner_new_node].add(elem.m_new_node->m_uid_node0);
          request_uid[elem.m_new_node->m_owner_new_node].add(elem.m_new_node->m_uid_node1);

          requested_nodes[elem.m_new_node->m_owner_new_node].add(elem.m_new_node);

          info() << "[Node][" << my_proc << " -> " << elem.m_new_node->m_owner_new_node << "] Ask1"
                 << " -- UID0 : " << elem.m_new_node->m_uid_node0
                 << " -- UID1 : " << elem.m_new_node->m_uid_node1;
        }
      }

      // Si le proprio est déjà défini, on doit lui demander le uid du noeud.
      else {
        request_uid[elem.m_new_node->m_owner_new_node].add(elem.m_new_node->m_uid_node0);
        request_uid[elem.m_new_node->m_owner_new_node].add(elem.m_new_node->m_uid_node1);

        requested_nodes[elem.m_new_node->m_owner_new_node].add(elem.m_new_node);

        info() << "[Node][" << my_proc << " -> " << elem.m_new_node->m_owner_new_node << "] Ask2"
               << " -- UID0 : " << elem.m_new_node->m_uid_node0
               << " -- UID1 : " << elem.m_new_node->m_uid_node1;
      }
    }

    // On envoie les requêtes.
    {
      UniqueArray<Parallel::Request> requests(subDomain()->nbSubDomain() * 2);

      info() << "[Node][" << my_proc << "] Step 1.2";

      for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
        if (sr == subDomain()->subDomainId()) {
          continue;
        }
        Int32 size = request_uid[sr].size();
        requests[sr * 2] = pm->send(ArrayView{ 1, &size }, sr, false);
        requests[sr * 2 + 1] = pm->send(request_uid[sr], sr, false);
        info() << "[Node][" << my_proc << " -> " << sr << "] Requests : " << request_uid[sr];
      }

      pm->waitAllRequests(requests);
    }
  }

  //
  //
  // Deuxième étape : traitement des requêtes des autres processus et gestion
  // des inconnus.
  //
  // Traitement des requêtes :
  //
  // On effectue le traitement à la réception (donc processus par processus).
  // Pour chaque paire de UID (N0-N1), on la recherche dans notre tableau des
  // noeuds.
  // On l'a trouve, parfait, on place l'UID du noeud (NN) dans le tableau de
  // réponse.
  //
  // Gestion des inconnus :
  //
  // Si on ne trouve pas la paire dans notre tableau, on est dans le cas où le
  // noeud NN est placé sur un noeud du maillage 3D (donc N0 == N1 et
  // position(N0) == position(NN)).
  // Lorsque l'on itère sur les mailles de N0, on peut tomber sur des mailles
  // qui ne sont pas coupées par le plan, donc qui ne possède pas le noeud.
  //
  // On n'a pas le noeud dans notre tableau. Par contre, on sait qui demande
  // ce noeud ! Grâce aux mailles fantômes, on est sûr que tous les processus
  // qui souhaite le UID du noeud NN vont le demander au même processus. Nous
  // pouvons donc construire la liste de ces processus demandeurs. Processus
  // demandeurs dont on est sûr qu'ils possèdent le noeud NN !
  // Nous allons donc leur envoyer cette liste et il se débrouilleront pour
  // trouver le bon proprio parmi les processus de cette liste !
  //
  // Réponses :
  //
  // Le tableau des réponses doit donc contenir deux parties :
  // - les UIDs, s'ils sont trouvés,
  // - les processus possédant les noeuds inconnus.
  //
  // Pour avoir la limite entre ces deux parties, on place la taille de la
  // première partie en première position du tableau de réponse.
  //
  // Si un UID est trouvé, on l'ajoute. S'il n'est pas trouvé, on place "-1".
  //
  // La seconde partie est structurée ainsi :
  // - Pour chaque inconnu (pour chaque "-1") :
  //   - UID du noeud N0
  //   - UID du noeud N1
  //   - Nombre de processus demandeurs (donc possédant forcément le noeud NN),
  //   - Rangs des processus demandeurs.
  //
  // TODO : A changer :
  // L'ordre de la seconde partie n'a pas d'importance.
  // En effet, l'ordre de cette partie est le même pour tous les processus (il
  // n'y a pas de tri par processus selon l'ordre des requêtes (qui
  // correspondrait à l'ordre des "-1")).
  // C'est pour ça qu'il y a la présence des "UID du noeud N0" et "UID du
  // noeud N1".
  // (trier avant envoi économiserait de la mémoire et réduirait la taille de
  // la réponse, donc TODO).
  //
  //
  // (Pour l'exemple, on suit le précédent exemple)
  // answers_uid[][] :
  // P0 : [3,    -1, NN01_UID, -1,    NN09_N0_UID, NN09_N1_UID, 2, 0, 2, NN00_N0_UID, NN00_N1_UID, 2, 0, 3]
  // P1 : --Nous--
  // P2 : [1,    -1,                  NN09_N0_UID, NN09_N1_UID, 2, 0, 2                                   ]
  // P3 : [1,    -1,                  NN00_N0_UID, NN00_N1_UID, 2, 0, 3                                   ]
  //
  //
  {
    UniqueArray<UniqueArray<Int64>> answers_uid(subDomain()->nbSubDomain());
    {
      UniqueArray<UnknownNode> unknown_node;
      info() << "[Node][" << my_proc << "] Step 2";

      for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
        if (sr == subDomain()->subDomainId()) {
          continue;
        }
        Int32 size = 0;
        pm->recv(ArrayView{ 1, &size }, sr);
        UniqueArray<Int64> requested_uid(size);
        pm->recv(requested_uid, sr);

        // Taille de la première partie.
        answers_uid[sr].add(0);

        // On traite chaque paire de noeud de la requête.
        for (Int32 ipair_uid = 0; ipair_uid < requested_uid.size(); ipair_uid += 2) {

          std::optional<Int64> pos = new_nodes.span().findFirst(NodeIntersection{ requested_uid[ipair_uid], requested_uid[ipair_uid + 1], Real3{ 0 } });
          if (pos) {
            answers_uid[sr].add(new_nodes[pos.value()].m_new_node->m_uid_new_node);
            info() << "[Node][" << my_proc << " <- " << sr << "] Found"
                   << " -- UID0 : " << requested_uid[ipair_uid]
                   << " -- UID1 : " << requested_uid[ipair_uid + 1]
                   << " -- New UID : " << new_nodes[pos.value()].m_new_node->m_uid_new_node;
          }

          // Il peut arriver que l'on nous demande un noeud que nous n'avons pas.
          // Par exemple, si un plan arrive pile sur une des arêtes de nos mailles.
          // Par contre, on sait qui en a besoin, donc qui possède une maille avec
          // le noeud en question.
          // Pour donner cette information, on va ajouter "-1" puis, à la fin de
          // la réponse, on va placer les processus en question.
          else {
            answers_uid[sr].add(-1);
            unknown_node.add({ requested_uid[ipair_uid], requested_uid[ipair_uid + 1], sr });
            info() << "[Node][" << my_proc << " <- " << sr << "] NOT Found"
                   << " -- UID0 : " << requested_uid[ipair_uid]
                   << " -- UID1 : " << requested_uid[ipair_uid + 1];
          }
        }
        answers_uid[sr][0] = answers_uid[sr].size() - 1;
      }

      UniqueArray<Int64> who;
      info() << "[Node][" << my_proc << "] Step 2.2";

      // Dès qu'il y a eu une paire de noeuds inconnus dans une requête, il y
      // a eu un enregistrement de fait dans le tableau unknown_node.
      // Il faut maintenant lister, pour chaque paire, qui l'a aussi demandé
      // (rechercher les doublons) et envoyer cette liste à chacun d'entre eux.
      for (Int32 i = 0; i < unknown_node.size(); ++i) {
        if (unknown_node[i].null())
          continue;
        who.clear();
        who.add(unknown_node[i].m_who);
        unknown_node[i].m_who = -1;
        for (Int32 j = i + 1; j < unknown_node.size(); ++j) {
          if (unknown_node[j].null())
            continue;
          if (unknown_node[j] == unknown_node[i]) {
            who.add(unknown_node[j].m_who);
            unknown_node[j].m_who = -1;
          }
        }
        info() << "[Node] Additionnal infos -- Node0UID : " << unknown_node[i].m_node0_uid << " -- Node1UID : " << unknown_node[i].m_node1_uid << " -- Who : " << who;
        for (auto proc : who) {
          answers_uid[proc].add(unknown_node[i].m_node0_uid);
          answers_uid[proc].add(unknown_node[i].m_node1_uid);
          answers_uid[proc].add(who.size());
          answers_uid[proc].addRange(who);
        }
      }
    }

    {
      UniqueArray<Parallel::Request> requests(subDomain()->nbSubDomain() * 2);

      // On envoie les réponses.
      info() << "[Node][" << my_proc << "] Step 2.3";
      for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
        if (sr == subDomain()->subDomainId()) {
          continue;
        }

        Int32 size = answers_uid[sr].size();
        requests[sr * 2] = pm->send(ArrayView{ 1, &size }, sr, false);
        requests[sr * 2 + 1] = pm->send(answers_uid[sr], sr, false);

        info() << "[Node][" << my_proc << " -> " << sr << "] Send " << answers_uid[sr];
      }

      pm->waitAllRequests(requests);
    }
  }


  //
  //
  // Troisième étape : Mise à jour des UIDs des noeuds et recherche
  // complémentaire de proprios.
  //
  // On commence par découper la réponse en deux, pour retrouver les deux
  // parties tèl que défini plus haut.
  // On itère sur la première partie. L'ordre des UIDs reçus est le même que
  // l'ordre des noeuds du tableau "requested_nodes" complété dans la première
  // partie. On peut ainsi aisément récupérer l'objet correspondant
  // (`node_on_edge` dans le code juste en dessous).
  //
  // Si nous avons reçu un UID valide, on met à jour le noeud correspondant
  // et on passe au suivant.
  //
  // Si nous avons "-1", c'est que le propriétaire que nous avons défini dans
  // la première partie n'est pas le bon. Nous devons donc organiser un nouvel
  // envoi.
  //
  // Recherche complémentaire de proprios :
  //
  // Comme l'ordre des infos complémentaires de la seconde partie du message
  // n'est pas forcément le même que l'ordre de la première partie (voir TODO
  // au dessus), on doit d'abord rechercher, dans la seconde partie, la paire
  // de UID correspondant au noeud que nous traitons actuellement.
  //
  // On trouve forcément une correspondance.
  //
  // On est sûr que l'ordre dans lequel est organisée la seconde partie est le
  // même pour tous. On va donc se caler sur cette ordre pour transmettre le
  // prochain message.
  //
  // Lorsqu'une correspondance est trouvée, on modifie le message. On remplace
  // UID0 par la position de l'objet `Ref<NodeOnEdge>` dans le tableau
  // `requested_nodes` (pour éviter une seconde recherche).
  //
  // Exemple :
  // Rappel tableau requested_nodes[][] (Première étape de la méthode) :
  // requested_nodes[][] :
  // P0 : --Nous--
  // P1 : [NN09, NN01, NN00]
  //
  // Avant modif :
  // answered_uid[] :
  // P0 : --Nous--
  // P1 : [3,    -1, NN01_UID, -1,    NN09_N0_UID, NN09_N1_UID, 2, 0, 2,   NN00_N0_UID, NN00_N1_UID, 2, 0, 3]
  //
  // Après modif :
  // answered_uid[] :
  // P0 : --Nous--
  // P1 : [3,    -1, NN01_UID, -1,    0,           NN09_N1_UID, 2, 0, 2,   2,           NN00_N1_UID, 2, 0, 3]
  //
  // Explication de l'exemple :
  // On remplace "NN09_N0_UID" par "0" car le noeud "NN09" se trouve à la
  // position 0 du tableau `requested_nodes[1][]` (le premier indice est le
  // rang du processus cible (donc 1 ici)).
  // On remplace "NN00_N0_UID" par "2" car le noeud "NN00" se trouve à la
  // position 2 du tableau `requested_nodes[1][]`.
  //
  // Cette algorithme est en deux étapes pour préserver simplement l'ordre de
  // la seconde partie de la réponse.
  //
  // Une fois `answered_uid[1]` modifié (donc une fois que l'on a fini
  // d'itérer sur la première partie de la réponse), on passe à la recherche
  // du propriétaire des noeuds restants.
  //
  // On itère sur la seconde partie de la réponse. Tous les processus
  // itèreront dans le même ordre !
  //
  // On récupère l'objet `Ref<NodeOnEdge>` à l'aide de sa position dans le
  // tableau `requested_nodes`. On reproduit l'algo de recherche de proprio de
  // la première étape de la méthode en s'assurant de tomber sur un proprio de
  // la liste des proprios potentiels.
  //
  // Par rapport à la première étape de la méthode, on sait de qui recevoir
  // des informations et on sait dans quel ordre les interpréter. Donc inutile
  // d'envoyer des messages vides si aucune infos n'est nécessaire.
  //
  // On va utiliser le tableau "request_uid" pour stocker les messages à
  // envoyer mais aussi la taille des messages que l'on devra recevoir.
  //
  // Pour les messages à envoyer, on stocke uniquement les UIDs qui devront
  // être appliqués.
  // Pour chaque UID stocké du coté du processus envoyeur, l'objet
  // `Ref<NodeOnEdge>` correspondant est stocké dans le tableau
  // `requested_nodes2` du coté du processus receveur.
  //
  // Exemple :
  // L'algo a déterminé que
  // - "NN09" appartient à "2" (et non à "0") et
  // - "NN00" appartient à "0" (et non à "3").
  // 
  // request_uid[][] :
  // P0 : --Nous-- [0, 0, 1, 0]
  // P1 :          []
  // P2 :          []
  // P3 :          [NN00_UID]
  //
  // (à lire : P0 attend un message de taille 1 de la part de P2)
  // (         P0 envoie NN00_UID à P3)
  //
  // request_uid[][] :
  // P0 :          [NN09_UID]
  // P1 :          []
  // P2 : --Nous-- [0, 0, 0, 0]
  // P3 :          []
  //
  // request_uid[][] :
  // P0 :          []
  // P1 :          []
  // P2 :          []
  // P3 : --Nous-- [1, 0, 0, 0]
  //
  //
  UniqueArray<UniqueArray<Int64>> request_uid(subDomain()->nbSubDomain());
  UniqueArray<UniqueArray<Ref<NodeOnEdge>>> requested_nodes2(subDomain()->nbSubDomain());

  // On reçoit et traite les réponses.
  info() << "[Node][" << my_proc << "] Step 3";

  // Permet de savoir si les prochaines étapes sont utiles.
  bool need_more_comm = false;

  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }

    Int32 total_size = 0;
    pm->recv(ArrayView{ 1, &total_size }, sr);

    UniqueArray<Int64> answered_uid(total_size);
    pm->recv(answered_uid, sr);

    Int32 answer_to_request_size = static_cast<Int32>(answered_uid[0]);

    Span<Int64> answer_to_request = answered_uid.subView(1, answer_to_request_size);
    Span<Int64> additionnal_answer = answered_uid.subView(answer_to_request_size+1, answered_uid.size() - (answer_to_request_size+1));

    info() << "[Node][" << my_proc << " <- " << sr << "] Decoupe"
           << " -- answer_to_request : " << answer_to_request
           << " -- additionnal_answer : " << additionnal_answer;


    for (Int32 answer = 0; answer < answer_to_request.size(); ++answer) {
      Ref<NodeOnEdge> node_on_edge = requested_nodes[sr][answer];

      if (answer_to_request[answer] != -1) {
        node_on_edge->m_uid_new_node = answer_to_request[answer];
        info() << "[Node][" << my_proc << "] Apply"
               << " -- UID0 : " << node_on_edge->m_uid_node0
               << " -- UID1 : " << node_on_edge->m_uid_node1
               << " -- New UID : " << node_on_edge->m_uid_new_node;
      }
      else {

        Int64 pos = 0;
        while (pos < additionnal_answer.size()) {
          Int64& uid0 = additionnal_answer[pos++];
          Int64 uid1 = additionnal_answer[pos++];
          Int64 decal = additionnal_answer[pos++];
          if (uid0 == node_on_edge->m_uid_node0 && uid1 == node_on_edge->m_uid_node1) {
            uid0 = answer;
            break;
          }
          pos += decal;
        }
        if (pos >= additionnal_answer.size()) {
          ARCANE_FATAL("Unknown node -- UID0 : {0} -- UID1 : {1}", node_on_edge->m_uid_node0, node_on_edge->m_uid_node1);
        }
      }
    }

    Int32 pos = 0;
    Span<Int64> sub_additionnal_answer;

    while (pos < additionnal_answer.size()) {
      Int64 pos_node_in_array = additionnal_answer[pos++];
      pos++;
      Int64 decal = additionnal_answer[pos++];

      Ref<NodeOnEdge> node_on_edge = requested_nodes[sr][pos_node_in_array];
      sub_additionnal_answer = additionnal_answer.subSpan(pos, decal);

      pos += decal;

      info() << "[Node][" << my_proc << " <- " << sr << "] Sub"
             << " -- sub_additionnal_answer : " << sub_additionnal_answer;

      Node node0 = node_on_edge->m_node0;
      Node node1 = node_on_edge->m_node1;

      // Le propriétaire du noeud est le propriétaire de la maille ayant le plus
      // petit UID, parmi les mailles en commun entre les deux noeuds d'origine.

      // TODO Ajouter traitement particulier pour le cas où node0 == node1
      Int64 min_uid = INT64_MAX;
      Int32 owner_min = -1;
      for (Cell cell0 : node0.cells()) {
        for (Cell cell1 : node1.cells()) {
          if (cell0 == cell1) {
            if (cell0.uniqueId() < min_uid && sub_additionnal_answer.contains(cell0.owner())) {
              min_uid = cell0.uniqueId();
              owner_min = cell0.owner();
            }
          }
        }
      }

      // S'il l'on est le proprio, on peut définir le uid du noeud.
      if (owner_min == subDomain()->subDomainId()) {
        node_on_edge->m_owner_new_node = subDomain()->subDomainId();
        node_on_edge->m_uid_new_node = sd_nb_node++;

        // On enregistre le UID à envoyer.
        if (sub_additionnal_answer.size() > 1) {
          need_more_comm = true;

          info() << "[Node][" << my_proc << "] Send"
                 << " -- UID : " << node_on_edge->m_uid_new_node
                 << " -- for Node UID0 : " << node_on_edge->m_uid_node0
                 << " -- UID1 : " << node_on_edge->m_uid_node1
                 << " -- to : " << sub_additionnal_answer;

          for (auto proc : sub_additionnal_answer) {
            if (proc == my_proc)
              continue;
            request_uid[proc].add(node_on_edge->m_uid_new_node);
          }
        }
      }

      // Sinon, on doit aller demander le uid au proprio.
      else {
        need_more_comm = true;
        node_on_edge->m_owner_new_node = owner_min;
        node_on_edge->m_uid_new_node = -2;

        if (request_uid[my_proc].empty()) {
          request_uid[my_proc].resize(subDomain()->nbSubDomain(), 0);
        }

        request_uid[my_proc][owner_min]++;

        requested_nodes2[owner_min].add(node_on_edge);

        info() << "[Node][" << my_proc << " -> " << node_on_edge->m_owner_new_node << "] Recv UID for Node"
               << " -- UID0 : " << node_on_edge->m_uid_node0
               << " -- UID1 : " << node_on_edge->m_uid_node1;
      }
    }
  }

  // Si nous n'avons pas de UID manquants et que personne n'a besoin de l'un
  // de nos UIDs, on passe cette étape.
  // Sinon :
  if (need_more_comm) {
    {
      UniqueArray<Parallel::Request> requests(subDomain()->nbSubDomain());

      // On envoie les UID complémentaires.
      info() << "[Node][" << my_proc << "] Step 3.2";

      for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
        if (sr == subDomain()->subDomainId()) {
          info() << "[Node][" << my_proc << " -> " << sr << "] My RequestsC : " << request_uid[sr];
          continue;
        }
        if (request_uid[sr].empty()) continue;
        requests[sr] = pm->send(request_uid[sr], sr, false);
        info() << "[Node][" << my_proc << " -> " << sr << "] RequestsC : " << request_uid[sr];
      }

      pm->waitAllRequests(requests);
    }

    //
    //
    // Quatrième étape : Mise à jour des UIDs manquants.
    //
    // On est sûr d'avoir une réponse complète, donc c'est plus simple.
    // On récupère le nombre de messages à recevoir de la part de l'autre
    // processus et on traite son message dans l'ordre. Le tableau
    // `requested_nodes2[][]` est dans le même ordre ce qui facilite les
    // choses.
    //
    //

    // On reçoit et traite les UID complémentaires.
    info() << "[Node][" << my_proc << "] Step 4";

    for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
      if (sr == subDomain()->subDomainId()) {
        continue;
      }

      Int64 size = request_uid[my_proc][sr];
      if (size == 0) continue;

      info() << "[Node][" << my_proc << " -> " << sr << "] Size recv : " << size;


      UniqueArray<Int64> requested_uid(size);
      pm->recv(requested_uid, sr);

      for (Int32 i = 0; i < size; i += 1) {
        Ref<NodeOnEdge> node_on_edge = requested_nodes2[sr][i];
        node_on_edge->m_uid_new_node = requested_uid[i];
        info() << "[Node][" << my_proc << "] Apply2"
               << " -- UID0 : " << node_on_edge->m_uid_node0
               << " -- UID1 : " << node_on_edge->m_uid_node1
               << " -- New UID : " << node_on_edge->m_uid_new_node;
      }
    }
  }

  info() << "[Node][" << subDomain()->parallelMng()->commRank() << "]";
  subDomain()->parallelMng()->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MeshCutService::
_makeUniqueNodeUID(Int32 sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 current_plan)
{
  IParallelMng* pm = subDomain()->parallelMng();

  UniqueArray<Int32> all_nb_node(pm->commSize());
  pm->allGather(ArrayView{ 1, &sd_nb_node }, all_nb_node);

  Int32 sum = 0;
  for (auto& elem : all_nb_node) {
    const Int32 old = elem;
    elem = sum;
    sum += old;
  }

  // info() << "[" << pm->commRank() << "] Gather result (rectified) : " << all_nb_node;

  // On ne doit traiter que les nouveaux noeuds.
  Span<NodeIntersection> current_plan_new_nodes = new_nodes.subView(current_plan, new_nodes.size() - current_plan);

  for (auto& elem : current_plan_new_nodes) {
    elem.m_new_node->m_uid_new_node += all_nb_node[elem.m_new_node->m_owner_new_node];
  }
  return sum;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_fillFaceUID(Int32& sd_nb_face, UniqueArray<FaceLite>& new_faces, Int32 current_plan_pos)
{
  // Cet algo est repris de celui traitant les noeuds.
  IParallelMng* pm = subDomain()->parallelMng();
  Int32 my_proc = pm->commRank();
  UniqueArray<UniqueArray<Int32>> requested_faces(subDomain()->nbSubDomain());

  Span<FaceLite> current_plan_new_faces = new_faces.subView(current_plan_pos, new_faces.size() - current_plan_pos);

  {
    Int32 iter = -1;
    UniqueArray<UniqueArray<Int64>> request_uid(subDomain()->nbSubDomain());

    info() << "[Face][" << my_proc << "] Step 1";

    for (auto& elem : current_plan_new_faces) {
      iter++;
      if (elem.m_uid_new_face >= 0) {
        continue;
      }
      if (elem.m_owner_new_face < 0) {

        Node node00 = elem.m_node0->m_node0;
        Node node01 = elem.m_node0->m_node1;
        Node node10 = elem.m_node1->m_node0;
        Node node11 = elem.m_node1->m_node1;

        Int64 min_uid = INT64_MAX;
        Int32 owner_min = -1;

        // TODO AH : C'est quand même TURBO moche
        for (Cell cell00 : node00.cells()) {
          for (Cell cell01 : node01.cells()) {
            for (Cell cell10 : node10.cells()) {
              for (Cell cell11 : node11.cells()) {
                if (cell00 == cell01 && cell00 == cell10 && cell00 == cell11) {
                  if (cell00.uniqueId() < min_uid) {
                    min_uid = cell00.uniqueId();
                    owner_min = cell00.owner();
                  }
                }
              }
            }
          }
        }

        if (owner_min == subDomain()->subDomainId()) {
          elem.m_owner_new_face = subDomain()->subDomainId();
          elem.m_uid_new_face = sd_nb_face++;

          info() << "[Face][" << my_proc << " -> " << elem.m_owner_new_face << "] Set UID"
                 << " -- UID0 : " << elem.m_node0->m_uid_new_node
                 << " -- UID1 : " << elem.m_node1->m_uid_new_node
                 << " -- New UID : " << elem.m_uid_new_face;
        }
        else {
          elem.m_owner_new_face = owner_min;
          elem.m_uid_new_face = -2;
          request_uid[elem.m_owner_new_face].add(elem.m_node0->m_uid_new_node);
          request_uid[elem.m_owner_new_face].add(elem.m_node1->m_uid_new_node);

          requested_faces[elem.m_owner_new_face].add(iter);

          info() << "[Face][" << my_proc << " -> " << elem.m_owner_new_face << "] Ask1"
                 << " -- UID0 : " << elem.m_node0->m_uid_new_node
                 << " -- UID1 : " << elem.m_node1->m_uid_new_node;
        }
      }
      else {
        request_uid[elem.m_owner_new_face].add(elem.m_node0->m_uid_new_node);
        request_uid[elem.m_owner_new_face].add(elem.m_node1->m_uid_new_node);

        requested_faces[elem.m_owner_new_face].add(iter);

        info() << "[Face][" << my_proc << " -> " << elem.m_owner_new_face << "] Ask2"
               << " -- UID0 : " << elem.m_node0->m_uid_new_node
               << " -- UID1 : " << elem.m_node1->m_uid_new_node;
      }
    }

    {
      UniqueArray<Parallel::Request> requests(subDomain()->nbSubDomain() * 2);

      // On envoie les requêtes.
      info() << "[Face][" << my_proc << "] Step 2";

      for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
        if (sr == subDomain()->subDomainId()) {
          continue;
        }
        Int32 size = request_uid[sr].size();
        requests[sr * 2] = pm->send(ArrayView{ 1, &size }, sr, false);
        requests[sr * 2 + 1] = pm->send(request_uid[sr], sr, false);
        info() << "[Face][" << my_proc << " -> " << sr << "] Requests : " << request_uid[sr];
      }

      pm->waitAllRequests(requests);
    }
  }

  UniqueArray<UniqueArray<Int64>> answers_uid(subDomain()->nbSubDomain());

  UniqueArray<UnknownNode> unknown_face;

  // On reçoit et traite les demandes.
  info() << "[Face][" << my_proc << "] Step 3";

  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }
    Int32 size = 0;
    pm->recv(ArrayView{ 1, &size }, sr);
    UniqueArray<Int64> requested_uid(size);
    pm->recv(requested_uid, sr);

    answers_uid[sr].add(0);

    for (Int32 ipair_uid = 0; ipair_uid < requested_uid.size(); ipair_uid += 2) {
      std::optional<Int64> pos = _find(new_faces, requested_uid[ipair_uid], requested_uid[ipair_uid + 1]);
      if (pos) {
        answers_uid[sr].add(new_faces[pos.value()].m_uid_new_face);
        info() << "[Face][" << my_proc << " <- " << sr << "] Found"
               << " -- UID0 : " << requested_uid[ipair_uid]
               << " -- UID1 : " << requested_uid[ipair_uid + 1]
               << " -- New UID : " << new_faces[pos.value()].m_uid_new_face;
      }

      // Il peut arriver que l'on nous demande une face que nous n'avons pas.
      // Par exemple, si un plan arrive pile sur une des arêtes de nos mailles.
      // Par contre, on sait qui en a besoin, donc qui possède une maille avec
      // la face en question.
      // Pour donner cette information, on va ajouter "-1" puis, à la fin de
      // la réponse, on va placer les processus en question.
      else {
        answers_uid[sr].add(-1);
        unknown_face.add({ requested_uid[ipair_uid], requested_uid[ipair_uid + 1], sr });
        info() << "[Face][" << my_proc << " <- " << sr << "] NOT Found"
               << " -- UID0 : " << requested_uid[ipair_uid]
               << " -- UID1 : " << requested_uid[ipair_uid + 1];
      }
    }
    answers_uid[sr][0] = answers_uid[sr].size() - 1;
  }

  UniqueArray<Int64> who;
  for (Int32 i = 0; i < unknown_face.size(); ++i) {
    if (unknown_face[i].null())
      continue;
    who.clear();
    who.add(unknown_face[i].m_who);
    unknown_face[i].m_who = -1;
    for (Int32 j = i + 1; j < unknown_face.size(); ++j) {
      if (unknown_face[j].null())
        continue;
      if (unknown_face[j] == unknown_face[i]) {
        who.add(unknown_face[j].m_who);
        unknown_face[j].m_who = -1;
      }
    }
    info() << "[Face] Additionnal infos -- Node0UID : " << unknown_face[i].m_node0_uid << " -- Node1UID : " << unknown_face[i].m_node1_uid << " -- Who : " << who;
    for (auto proc : who) {
      answers_uid[proc].add(unknown_face[i].m_node0_uid);
      answers_uid[proc].add(unknown_face[i].m_node1_uid);
      answers_uid[proc].add(who.size());
      answers_uid[proc].addRange(who);
    }
  }

  {
    UniqueArray<Parallel::Request> requests(subDomain()->nbSubDomain() * 2);

    // On envoie les réponses.
    info() << "[Face][" << my_proc << "] Step 4";
    for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
      if (sr == subDomain()->subDomainId()) {
        continue;
      }

      Int32 size = answers_uid[sr].size();
      requests[sr * 2] = pm->send(ArrayView{ 1, &size }, sr, false);
      requests[sr * 2 + 1] = pm->send(answers_uid[sr], sr, false);
    }

    pm->waitAllRequests(requests);
  }

  UniqueArray<UniqueArray<Int64>> request_uid(subDomain()->nbSubDomain());
  UniqueArray<UniqueArray<Int32>> requested_faces2(subDomain()->nbSubDomain());

  // On reçoit et traite les réponses.
  info() << "[Face][" << my_proc << "] Step 5";

  Int32 iter_requested_faces = 0;
  bool need_more_comm = false;
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }

    Int32 total_size = 0;
    pm->recv(ArrayView{ 1, &total_size }, sr);

    UniqueArray<Int64> answered_uid(total_size);
    pm->recv(answered_uid, sr);

    Int32 answer_to_request_size = answered_uid[0];

    Span<Int64> answer_to_request = answered_uid.subView(1, answer_to_request_size);
    Span<Int64> additionnal_answer = answered_uid.subView(answer_to_request_size + 1, answered_uid.size() - (answer_to_request_size + 1));

    info() << "[Face][" << my_proc << " <- " << sr << "] Decoupe"
           << " -- answer_to_request : " << answer_to_request
           << " -- additionnal_answer : " << additionnal_answer;

    for (Int32 answer = 0; answer < answer_to_request.size(); ++answer) {
      FaceLite& face_lite = new_faces[requested_faces[sr][answer]];

      if (answer_to_request[answer] != -1) {
        face_lite.m_uid_new_face = answer_to_request[answer];
        info() << "[Face][" << my_proc << "] Apply"
               << " -- UID0 : " << face_lite.m_node0->m_uid_new_node
               << " -- UID1 : " << face_lite.m_node1->m_uid_new_node
               << " -- New UID : " << face_lite.m_uid_new_face;
      }
      else {
        need_more_comm = true;
        Span<Int64> sub_additionnal_answer;

        Int32 pos_sort = 0;
        Int64 pos = 0;
        while (pos < additionnal_answer.size()) {
          Int64 uid0 = additionnal_answer[pos++];
          Int64 uid1 = additionnal_answer[pos++];
          Int64 decal = additionnal_answer[pos++];
          if (uid0 == face_lite.m_node0->m_uid_new_node && uid1 == face_lite.m_node1->m_uid_new_node) {
            sub_additionnal_answer = additionnal_answer.subSpan(pos, decal);
            break;
          }
          pos += decal;
          pos_sort++;
        }
        if (sub_additionnal_answer.data() == nullptr) {
          ARCANE_FATAL("Unknown face -- UID0 : {0} -- UID1 : {1}", face_lite.m_node0->m_uid_new_node, face_lite.m_node1->m_uid_new_node);
        }

        info() << "[Face][" << my_proc << " <- " << sr << "] Sub"
               << " -- sub_additionnal_answer : " << sub_additionnal_answer;

        Node node00 = face_lite.m_node0->m_node0;
        Node node01 = face_lite.m_node0->m_node1;
        Node node10 = face_lite.m_node1->m_node0;
        Node node11 = face_lite.m_node1->m_node1;

        Int64 min_uid = INT64_MAX;
        Int32 owner_min = -1;

        // TODO AH : C'est quand même TURBO moche
        for (Cell cell00 : node00.cells()) {
          for (Cell cell01 : node01.cells()) {
            for (Cell cell10 : node10.cells()) {
              for (Cell cell11 : node11.cells()) {
                if (cell00 == cell01 && cell00 == cell10 && cell00 == cell11) {
                  if (cell00.uniqueId() < min_uid && sub_additionnal_answer.contains(cell00.owner())) {
                    min_uid = cell00.uniqueId();
                    owner_min = cell00.owner();
                  }
                }
              }
            }
          }
        }

        if (owner_min == subDomain()->subDomainId()) {
          face_lite.m_owner_new_face = subDomain()->subDomainId();
          face_lite.m_uid_new_face = sd_nb_face++;

          info() << "[Face][" << my_proc << "] Send"
                 << " -- UID : " << face_lite.m_uid_new_face
                 << " -- for Node UID0 : " << face_lite.m_node0->m_uid_new_node
                 << " -- UID1 : " << face_lite.m_node1->m_uid_new_node
                 << " -- to : " << sub_additionnal_answer;

          for (auto proc : sub_additionnal_answer) {
            if (proc == my_proc)
              continue;
            request_uid[proc].add(face_lite.m_uid_new_face);
          }
        }

        // Sinon, on doit aller demander le uid au proprio.
        else {
          face_lite.m_owner_new_face = owner_min;
          face_lite.m_uid_new_face = -2;

          request_uid[my_proc].add(owner_min);
          // request_uid[my_proc].add(iter_requested_faces);
          request_uid[my_proc].add(pos_sort);

          requested_faces2[owner_min].add(requested_faces[sr][answer]);

          info() << "[Face][" << my_proc << " -> " << face_lite.m_owner_new_face << "] Recv UID for Node"
                 << " -- UID0 : " << face_lite.m_node0->m_uid_new_node
                 << " -- UID1 : " << face_lite.m_node1->m_uid_new_node;
        }
      }
      iter_requested_faces++;
    }
  }

  if (need_more_comm) {
    {
      UniqueArray<Parallel::Request> requests(subDomain()->nbSubDomain());
      // On envoie les UID complémentaires.
      info() << "[Face][" << my_proc << "] Step 6";

      for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
        if (sr == subDomain()->subDomainId()) {
          info() << "[Face][" << my_proc << " -> " << sr << "] My RequestsC : " << request_uid[sr];
          continue;
        }
        if (request_uid[sr].empty()) continue;
        requests[sr] = pm->send(request_uid[sr], sr, false);
        info() << "[Face][" << my_proc << " -> " << sr << "] RequestsC : " << request_uid[sr];
      }

      pm->waitAllRequests(requests);
    }

    // On reçoit et traite les UID complémentaires.
    info() << "[Face][" << my_proc << "] Step 7";

    for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
      if (sr == subDomain()->subDomainId()) {
        continue;
      }

      Int32 size = 0;

      for (Int32 i = 0; i < request_uid[my_proc].size(); i += 2) {
        if (request_uid[my_proc][i] == sr) {
          size++;
        }
      }
      
      if (size == 0) continue;

      info() << "[Face][" << my_proc << " -> " << sr << "] Size recv : " << size;

      UniqueArray<Int64> requested_uid(size);
      pm->recv(requested_uid, sr);

      for (Int32 i = 0, ii = 0; i < request_uid[my_proc].size(); i += 2, ++ii) {
        if (request_uid[my_proc][i] == sr) {
          FaceLite& face_lite = new_faces[requested_faces2[sr][ii]];

          face_lite.m_uid_new_face = requested_uid[request_uid[my_proc][i+1]];
          info() << "[Face][" << my_proc << "] Apply2"
                 << " -- UID0 : " << face_lite.m_node0->m_uid_new_node
                 << " -- UID1 : " << face_lite.m_node1->m_uid_new_node
                 << " -- New UID : " << face_lite.m_uid_new_face;
        }
      }
    }
  }

  info() << "[Face][" << subDomain()->parallelMng()->commRank() << "]";
  subDomain()->parallelMng()->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MeshCutService::
_makeUniqueFaceUID(Int32 sd_nb_face, UniqueArray<FaceLite>& new_faces, Int32 current_plan)
{
  IParallelMng* pm = subDomain()->parallelMng();

  UniqueArray<Int32> all_nb_node(pm->commSize());
  pm->allGather(ArrayView{ 1, &sd_nb_face }, all_nb_node);

  Int32 sum = 0;
  for (auto& elem : all_nb_node) {
    const Int32 old = elem;
    elem = sum;
    sum += old;
  }

  // info() << "[" << pm->commRank() << "] Gather result (rectified) : " << all_nb_node;

  Span<FaceLite> current_plan_new_faces = new_faces.subView(current_plan, new_faces.size() - current_plan);

  for (auto& elem : current_plan_new_faces) {
    elem.m_uid_new_face += all_nb_node[elem.m_owner_new_face];
  }
  return sum;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_compute()
{
  Int32 nb_cell = 0;
  Int32 g_nb_node = 0;
  Int32 g_nb_face = 0;

  UniqueArray<Int64> new_cells;

  UniqueArray<NodeIntersection> new_nodes;
  UniqueArray<FaceLite> new_faces;

  for (Int32 i = 0; i < m_plans.size(); ++i) {

    info() << "Plan : " << m_plans[i].first << ", " << m_plans[i].second;

    UniqueArray<NodeIntersection> plan_new_nodes;
    UniqueArray<FaceLite> plan_new_faces;

    Int32 ajust = new_nodes.size();

    Int32 previous_g_nb_node = g_nb_node;
    Int32 previous_g_nb_face = g_nb_face;

    Int32 nb_node_for_this_plan = g_nb_node;
    Int32 nb_face_for_this_plan = g_nb_face;

    info() << "[" << subDomain()->parallelMng()->commRank() << "] _createNodesAndCells";
    _createNodesAndCells(i, g_nb_node, plan_new_nodes, ajust, nb_cell, new_cells, g_nb_face, plan_new_faces);

    for (auto& elem : plan_new_nodes) {
      info() << "New node"
             << " -- UID : " << elem.m_new_node->m_uid_new_node
             << " -- Owner : " << elem.m_new_node->m_owner_new_node
             << " -- Pos : " << elem.m_intersection_pos
             << " -- Edge node0 : " << elem.m_new_node->m_uid_node0
             << " -- Edge node1 : " << elem.m_new_node->m_uid_node1;
    }

    info() << "[" << subDomain()->parallelMng()->commRank() << "] _fillNodeUID";
    _fillNodeUID(g_nb_node, plan_new_nodes, 0);
    nb_node_for_this_plan = g_nb_node - nb_node_for_this_plan;

    nb_node_for_this_plan = _makeUniqueNodeUID(nb_node_for_this_plan, plan_new_nodes, 0);
    g_nb_node = previous_g_nb_node + nb_node_for_this_plan;

    for (auto& elem : plan_new_nodes) {
      info() << "Fix node"
             << " -- UID : " << elem.m_new_node->m_uid_new_node
             << " -- Owner : " << elem.m_new_node->m_owner_new_node
             << " -- Pos : " << elem.m_intersection_pos
             << " -- Edge node0 : " << elem.m_new_node->m_uid_node0
             << " -- Edge node1 : " << elem.m_new_node->m_uid_node1;
    }

    for (auto& elem : plan_new_faces) {
      info() << "New face"
             << " -- UID : " << elem.m_uid_new_face
             << " -- Owner : " << elem.m_owner_new_face
             << " -- Node0 : " << elem.m_node0->m_uid_new_node
             << " -- Node00 : " << elem.m_node0->m_uid_node0
             << " -- Node01 : " << elem.m_node0->m_uid_node1
             << " -- Node1 : " << elem.m_node1->m_uid_new_node
             << " -- Node10 : " << elem.m_node1->m_uid_node0
             << " -- Node11 : " << elem.m_node1->m_uid_node1;
    }
    _fillFaceUID(g_nb_face, plan_new_faces, 0);
    nb_face_for_this_plan = g_nb_face - nb_face_for_this_plan;

    nb_face_for_this_plan = _makeUniqueFaceUID(nb_face_for_this_plan, plan_new_faces, 0);
    g_nb_face = previous_g_nb_face + nb_face_for_this_plan;

    for (auto& elem : plan_new_faces) {
      info() << "Fix face"
             << " -- UID : " << elem.m_uid_new_face
             << " -- Owner : " << elem.m_owner_new_face
             << " -- Node0 : " << elem.m_node0->m_uid_new_node
             << " -- Node00 : " << elem.m_node0->m_uid_node0
             << " -- Node01 : " << elem.m_node0->m_uid_node1
             << " -- Node1 : " << elem.m_node1->m_uid_new_node
             << " -- Node10 : " << elem.m_node1->m_uid_node0
             << " -- Node11 : " << elem.m_node1->m_uid_node1;
    }

    new_nodes.addRange(plan_new_nodes);
    new_faces.addRange(plan_new_faces);
  }

  _makeUniqueCellUID(nb_cell, new_cells, new_nodes);

  {
    // Int32 pos0 = 0;
    // while (pos0 < new_cells.size()) {
    //   StringBuilder logs;
    //   logs += "New cell -- Type : ";
    //   Int64 type = new_cells[pos0++];
    //   logs += type;
    //   logs += " -- UID : ";
    //   logs += new_cells[pos0++];
    //
    //   for (Int32 i = 0; i < type; ++i) {
    //     logs += " -- Node";
    //     logs += i;
    //     logs += " : ";
    //     logs += new_cells[pos0++];
    //   }
    //   info() << logs;
    // }
  }

  _addFaces(new_faces);
  _addCells(new_cells);
  m_cloned_mesh->modifier()->endUpdate();
  _setCoordNodesAndOwner(new_nodes);
  _setFacesOwner(new_faces);

  m_cloned_mesh->nodeFamily()->notifyItemsOwnerChanged();
  m_cloned_mesh->faceFamily()->notifyItemsOwnerChanged();

  info() << "New mesh -- NbNode : " << m_cloned_mesh->nbNode() << " -- NbCells : " << m_cloned_mesh->nbCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::optional<Int64> MeshCutService::
_find(Span<FaceLite> new_faces, Int64 node_uid0, Int64 node_uid1)
{
  for (Int32 i = 0; i < new_faces.size(); ++i) {
    if (new_faces[i].m_node0->m_uid_new_node == node_uid0 && new_faces[i].m_node1->m_uid_new_node == node_uid1) {
      return i;
    }
  }
  return std::nullopt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_addFaces(UniqueArray<FaceLite>& new_faces) const
{
  UniqueArray<Int64> faces_infos;
  faces_infos.reserve(new_faces.size() * 4);

  Int32 nb_faces = 0;

  for (auto& elem : new_faces) {
    faces_infos.add(ITI_Line2);
    faces_infos.add(elem.m_uid_new_face);
    faces_infos.add(elem.m_node0->m_uid_new_node);
    faces_infos.add(elem.m_node1->m_uid_new_node);
    nb_faces++;
  }

  m_cloned_mesh->modifier()->addFaces(nb_faces, faces_infos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_addCells(UniqueArray<Int64>& new_cells)
{
  Int32 nb_cells = 0;

  Int32 pos0 = 0;
  while (pos0 < new_cells.size()) {
    const Int64 type = new_cells[pos0];

    if (type == 3)
      new_cells[pos0++] = ITI_Triangle3;
    else if (type == 4)
      new_cells[pos0++] = ITI_Quad4;
    else if (type == 5)
      new_cells[pos0++] = ITI_Pentagon5;
    else if (type == 6)
      new_cells[pos0++] = ITI_Hexagon6;
    else
      ARCANE_FATAL("Pas implem : {0}", type);

    pos0 += 1 + static_cast<Int32>(type); // (1)=UID + (type)=UIDNodes

    nb_cells++;
  }

  m_cloned_mesh->modifier()->addCells(nb_cells, new_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_setCoordNodesAndOwner(UniqueArray<NodeIntersection>& new_nodes)
{
  VariableNodeReal3& node_coords(m_cloned_mesh->nodesCoordinates());
  ENUMERATE_ (Node, inode, m_cloned_mesh->allNodes()) {
    const Int64 uid = inode->uniqueId();
    for (const auto& elem : new_nodes) {
      if (elem.m_new_node->m_uid_new_node == uid) {
        node_coords[inode] = elem.m_intersection_pos;
        inode->mutableItemBase().setOwner(elem.m_new_node->m_owner_new_node, subDomain()->subDomainId());
        // info() << "NodeUID : " << uid << " -- Coord : " << node_coords[inode];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_setFacesOwner(UniqueArray<FaceLite>& new_faces)
{
  ENUMERATE_ (Face, iface, m_cloned_mesh->allFaces()) {
    const Int64 uid = iface->uniqueId();
    for (const auto& elem : new_faces) {
      if (elem.m_uid_new_face == uid) {
        iface->mutableItemBase().setOwner(elem.m_owner_new_face, subDomain()->subDomainId());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
