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

  Node m_node0{};
  Node m_node1{};
  Int64 m_uid_node0 = -1;
  Int64 m_uid_node1 = -1;
  Int64 m_uid_new_node = -1;
  Int32 m_owner_new_node = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct FaceLite
{
  FaceLite(const Ref<NodeOnEdge>& node0, const Ref<NodeOnEdge>& node1)
  : m_node0(node0->m_uid_node0 < node1->m_uid_node0 ? node0 : node1)
  , m_node1(node0->m_uid_node0 < node1->m_uid_node0 ? node1 : node0)
  {}

  bool operator<(const FaceLite& other) const
  {
    return m_node0->operator<(*(other.m_node0.get())) && m_node1->operator<(*(other.m_node1.get()));
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
  {}

 public:

  void addPlane(const Real3& p0, const Real3& normal) override;

  void setVariables(VariableCollection variables) override;

  void updateSection() override;

  MeshHandle meshSection() override
  {
    return m_cloned_mesh->handle();
  }

  VariableCollection variables() override { return {}; }

 private:

  void _createMesh();
  void _createNodesAndCells(Int32 plan_pos, Int32& sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32& sd_nb_cell, UniqueArray<Int64>& new_cells, Int32& sd_nb_face, UniqueArray<FaceLite>& new_faces);
  Int32 _makeUniqueCellUID(Int32 sd_nb_cell, UniqueArray<Int64>& new_cells, UniqueArray<NodeIntersection>& new_nodes);

  void _fillNodeUID(Int32& sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 current_plan);
  Int32 _makeUniqueNodeUID(Int32 sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 current_plan);
  void _fillFaceUID(Int32& sd_nb_face, UniqueArray<FaceLite>& new_faces, Int32 current_plan);
  Int32 _makeUniqueFaceUID(Int32 sd_nb_face, UniqueArray<FaceLite>& new_faces, Int32 current_plan);
  void _compute();

  std::optional<Int64> _find(Span<FaceLite> new_faces, Int64 node_uid0, Int64 node_uid1);

  void _addFaces(UniqueArray<FaceLite>& new_faces) const;
  void _addCells(UniqueArray<Int64>& new_cells);
  void _setCoordNodesAndOwner(UniqueArray<NodeIntersection>& new_nodes);
  void _setFacesOwner(UniqueArray<FaceLite>& new_faces);

 private:

  // VariableCollection m_variables_ori;
  IPrimaryMesh* m_cloned_mesh = nullptr;
  UniqueArray<std::pair<Real3, Real3>> m_plans;
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
updateSection()
{
  if (mesh()->ghostLayerMng()->nbGhostLayer() < 1) {
    ARCANE_FATAL("A ghost layer is required for this service");
  }
  if (m_cloned_mesh == nullptr) {
    _createMesh();
  }
  else {
    m_cloned_mesh->modifier()->clearItems();
  }
  _compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_createMesh()
{
  if (mesh()->dimension() != 3) {
    ARCANE_FATAL("Only 3D meshes are supported");
  }

  IMeshMng* mm = subDomain()->meshMng();
  IParallelMng* pm = subDomain()->parallelMng();
  // TODO gérer cas où il y a plusieurs services pour même maillage.
  MeshBuildInfo mbi(mesh()->name() + "_MeshCut");
  mbi.addParallelMng(makeRef(pm));
  m_cloned_mesh = mm->meshFactoryMng()->createMesh(mbi);
  m_cloned_mesh->modifier()->setDynamic(true);
  m_cloned_mesh->setDimension(2);
  m_cloned_mesh->endAllocate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_createNodesAndCells(Int32 plan_pos, Int32& sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32& sd_nb_cell, UniqueArray<Int64>& new_cells, Int32& sd_nb_face, UniqueArray<FaceLite>& new_faces)
{
  auto [p0, normal] = m_plans[plan_pos];

  const Int32 mesh_dim = mesh()->dimension();

  VariableNodeReal3& node_coord = mesh()->nodesCoordinates();
  VariableNodeReal node_dist(VariableBuildInfo(mesh(), "NodeDist"));

  UniqueArray<Int32> face_already_computed;

  // Calcul de la distance signée de chaque noeud par rapport au plan de coupe.
  ENUMERATE_ (Node, inode, allNodes()) {
    Real d = math::dot({ node_coord[inode] - p0 }, normal);
    node_dist[inode] = math::isNearlyZero(d) ? 0 : d;
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
          // Pour éviter que deux processus créés la même maille.
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
      // hexa_edge: 12 arêtes pour un hexaédre (8 noeuds), chaque arête = [n0, n1] indices locaux.
      static constexpr Integer hexa_edge[12][2] = {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }, { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 }
      };
      // tetra_edge: 6 arêtes pour un tétraèdre (4 noeuds).
      static constexpr Integer tetra_edge[6][2] = {
        { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 }
      };
      // quad_edge: 4 arêtes pour un quadrangle (4 noeuds).
      static constexpr Integer quad_edge[4][2] = {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 }
      };
      // tri_edge: 3 arêtes pour un triangle (3 noeuds).
      static constexpr Integer tri_edge[3][2] = {
        { 0, 1 }, { 1, 2 }, { 2, 0 }
      };

      Integer nb_edges = 0;
      const Integer(*edge_def)[2] = nullptr;

      Integer nb_node = cell.nbNode();

      if (nb_node == 8) {
        nb_edges = 12;
        edge_def = hexa_edge;
      }
      else if (nb_node == 4) {
        nb_edges = 6;
        edge_def = tetra_edge;
      }
      else if (nb_node == 4) {
        nb_edges = 4;
        edge_def = quad_edge;
      }
      else if (nb_node == 3) {
        nb_edges = 3;
        edge_def = tri_edge;
      }
      else {
        ARCANE_FATAL("Cell type not supported -- nbNode: {0}", nb_node);
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

        for (Int64& idx : indices) {
          std::optional<Int64> pos = new_nodes.span().findFirst(point_coords_tmp[idx]);
          if (pos) {
            new_cells.add(static_cast<Int32>(pos.value()));
            idx = pos.value();
          }
          else {
            auto& elem = point_coords_tmp[idx];

            // Si les deux noeuds sont à nous, on est sûr d'être le propriétaire du nouveau noeud.
            if (elem.m_new_node->m_node0.owner() == elem.m_new_node->m_node1.owner()) {
              elem.m_new_node->m_owner_new_node = elem.m_new_node->m_node0.owner(); //bof
              if (elem.m_new_node->m_node0.isOwn()) {
                elem.m_new_node->m_uid_new_node = sd_nb_node++;
              }
              else {
                elem.m_new_node->m_uid_new_node = -2;
              }
            }
            new_cells.add(new_nodes.size());
            idx = new_nodes.size();
            new_nodes.add(elem);
          }
        }

        Int64 idxm1 = indices[indices.size() - 1];
        for (Int64 idx : indices) {
          FaceLite fl(new_nodes[idx].m_new_node, new_nodes[idxm1].m_new_node);
          if (!new_faces.contains(fl)) {
            info() << "Add face"
                   << " -- N00 " << fl.m_node0->m_uid_node0
                   << " -- N01 " << fl.m_node0->m_uid_node1
                   << " -- N10 " << fl.m_node1->m_uid_node0
                   << " -- N11 " << fl.m_node1->m_uid_node1;
            if (new_nodes[idx].m_new_node->m_owner_new_node == new_nodes[idxm1].m_new_node->m_owner_new_node) {
              fl.m_owner_new_face = new_nodes[idx].m_new_node->m_owner_new_node;
              if (fl.m_owner_new_face == subDomain()->subDomainId()) {
                fl.m_uid_new_face = sd_nb_face++;
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

Int32 MeshCutService::
_makeUniqueCellUID(Int32 sd_nb_cell, UniqueArray<Int64>& new_cells, UniqueArray<NodeIntersection>& new_nodes)
{
  IParallelMng* pm = subDomain()->parallelMng();

  Int32 decal = sd_nb_cell;
  pm->scan(MessagePassing::ReduceSum, ArrayView{ 1, &decal });

  Int32 nb_cells_global = decal;
  pm->broadcast(ArrayView{ 1, &nb_cells_global }, pm->commSize() - 1);

  decal -= sd_nb_cell;

  info() << "[" << pm->commRank() << "] Scan result (rectified) : " << decal;

  Int32 pos0 = 0;
  while (pos0 < new_cells.size()) {
    Int64 type = new_cells[pos0++];
    new_cells[pos0++] += decal;

    for (Int32 i = 0; i < type; ++i) {
      Int64& pos_to_uid = new_cells[pos0++];
      pos_to_uid = new_nodes[pos_to_uid].m_new_node->m_uid_new_node;
    }
  }
  return nb_cells_global;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_fillNodeUID(Int32& sd_nb_node, UniqueArray<NodeIntersection>& new_nodes, Int32 current_plan)
{
  IParallelMng* pm = subDomain()->parallelMng();
  Int32 my_proc = pm->commRank();
  UniqueArray<UniqueArray<Int64>> request_uid(subDomain()->nbSubDomain());

  Span<NodeIntersection> current_plan_new_nodes = new_nodes.subView(current_plan, new_nodes.size() - current_plan);

  // On détermine le futur proprio de chaque noeud.
  for (auto& elem : current_plan_new_nodes) {
    if (elem.m_new_node->m_uid_new_node >= 0) {
      continue;
    }
    if (elem.m_new_node->m_owner_new_node < 0) {

      Node node0 = elem.m_new_node->m_node0;
      Node node1 = elem.m_new_node->m_node1;

      // Le propriétaire du noeud est le propriétaire de la maille ayant le plus
      // petit UID, parmi les mailles en commun entre les deux noeuds d'origine.
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

      if (owner_min == subDomain()->subDomainId()) {
        elem.m_new_node->m_owner_new_node = subDomain()->subDomainId();
        elem.m_new_node->m_uid_new_node = sd_nb_node++;
      }
      else {
        elem.m_new_node->m_owner_new_node = owner_min;
        elem.m_new_node->m_uid_new_node = -2;
        request_uid[elem.m_new_node->m_owner_new_node].add(elem.m_new_node->m_uid_node0);
        request_uid[elem.m_new_node->m_owner_new_node].add(elem.m_new_node->m_uid_node1);

        info() << "[" << my_proc << "] Ask1"
               << " -- UID0 : " << elem.m_new_node->m_uid_node0
               << " -- UID1 : " << elem.m_new_node->m_uid_node1;
      }
    }
    else {
      request_uid[elem.m_new_node->m_owner_new_node].add(elem.m_new_node->m_uid_node0);
      request_uid[elem.m_new_node->m_owner_new_node].add(elem.m_new_node->m_uid_node1);

      info() << "[" << my_proc << "] Ask2"
             << " -- UID0 : " << elem.m_new_node->m_uid_node0
             << " -- UID1 : " << elem.m_new_node->m_uid_node1;
    }
  }

  UniqueArray<Parallel::Request> requests(subDomain()->nbSubDomain() * 2);

  // Si le noeud n'est pas à nous, il faut demander son UID au processus proprio.
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }
    Int32 size = request_uid[sr].size();
    requests[sr * 2] = pm->send(ArrayView{ 1, &size }, sr, false);
    requests[sr * 2 + 1] = pm->send(request_uid[sr], sr, false);
    info() << "[" << my_proc << " -> " << sr << "] Requests : " << request_uid[sr];
  }

  pm->waitAllRequests(requests);
  // pm->freeRequests(requests);

  UniqueArray<UniqueArray<Int64>> answers_uid(subDomain()->nbSubDomain());

  // On reçoit et traite les demandes.
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }
    Int32 size = 0;
    pm->recv(ArrayView{ 1, &size }, sr);
    UniqueArray<Int64> requested_uid(size);
    pm->recv(requested_uid, sr);

    for (Int32 ipair_uid = 0; ipair_uid < requested_uid.size(); ipair_uid += 2) {
      std::optional<Int64> pos = new_nodes.span().findFirst(NodeIntersection{ requested_uid[ipair_uid], requested_uid[ipair_uid + 1], Real3{ 0 } });
      if (pos) {
        answers_uid[sr].add(new_nodes[pos.value()].m_new_node->m_uid_new_node);
        info() << "[" << my_proc << "] Found"
               << " -- UID0 : " << requested_uid[ipair_uid]
               << " -- UID1 : " << requested_uid[ipair_uid + 1]
               << " -- New UID : " << new_nodes[pos.value()].m_new_node->m_uid_new_node;
      }
      else {
        warning() << "Not found";
        answers_uid[sr].add(-1);
      }
    }
  }

  // On envoie les réponses.
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }

    Int32 size = answers_uid[sr].size();
    requests[sr * 2] = pm->send(ArrayView{ 1, &size }, sr, false);
    requests[sr * 2 + 1] = pm->send(answers_uid[sr], sr, false);
  }

  pm->waitAllRequests(requests);
  // pm->freeRequests(requests);

  // On reçoit et traite les réponses.
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }

    Int32 size = 0;
    pm->recv(ArrayView{ 1, &size }, sr);
    UniqueArray<Int64> answered_uid(size);
    pm->recv(answered_uid, sr);

    for (Int32 ipair_uid = 0; ipair_uid < request_uid[sr].size(); ipair_uid += 2) {
      std::optional<Int64> pos = current_plan_new_nodes.findFirst(NodeIntersection{ request_uid[sr][ipair_uid], request_uid[sr][ipair_uid + 1], Real3{ 0 } });
      if (pos) {
        current_plan_new_nodes[pos.value()].m_new_node->m_uid_new_node = answered_uid[ipair_uid / 2];
        info() << "[" << my_proc << "] Apply"
               << " -- UID0 : " << current_plan_new_nodes[pos.value()].m_new_node->m_uid_node0
               << " -- UID1 : " << current_plan_new_nodes[pos.value()].m_new_node->m_uid_node1
               << " -- New UID : " << current_plan_new_nodes[pos.value()].m_new_node->m_uid_new_node;
      }
      else {
        ARCANE_FATAL("GL");
      }
    }
  }
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

  info() << "[" << pm->commRank() << "] Gather result (rectified) : " << all_nb_node;

  info() << "current_plan : " << current_plan << " -- new_nodes.size() : " << new_nodes.size();

  ArrayView<NodeIntersection> current_plan_new_nodes = new_nodes.subView(current_plan, new_nodes.size() - current_plan);

  for (auto& elem : current_plan_new_nodes) {
    // info() << "[" << pm->commRank() << "] Old UID : " << elem.m_new_node->m_uid_new_node;
    elem.m_new_node->m_uid_new_node += all_nb_node[elem.m_new_node->m_owner_new_node];
    // info() << "[" << pm->commRank() << "] New UID : " << elem.m_new_node->m_uid_new_node;
  }
  return sum;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_fillFaceUID(Int32& sd_nb_face, UniqueArray<FaceLite>& new_faces, Int32 current_plan)
{
  IParallelMng* pm = subDomain()->parallelMng();
  Int32 my_proc = pm->commRank();
  UniqueArray<UniqueArray<Int64>> request_uid(subDomain()->nbSubDomain());

  Span<FaceLite> current_plan_new_faces = new_faces.subView(current_plan, new_faces.size() - current_plan);

  for (auto& elem : current_plan_new_faces) {

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
      }
      else {
        elem.m_owner_new_face = owner_min;
        elem.m_uid_new_face = -2;
        request_uid[elem.m_owner_new_face].add(elem.m_node0->m_uid_new_node);
        request_uid[elem.m_owner_new_face].add(elem.m_node1->m_uid_new_node);

        info() << "[" << my_proc << "] Ask3"
               << " -- UID0 : " << elem.m_node0->m_uid_new_node
               << " -- UID1 : " << elem.m_node1->m_uid_new_node;
      }
    }
    else {
      request_uid[elem.m_owner_new_face].add(elem.m_node0->m_uid_new_node);
      request_uid[elem.m_owner_new_face].add(elem.m_node1->m_uid_new_node);

      info() << "[" << my_proc << "] Ask4"
             << " -- UID0 : " << elem.m_node0->m_uid_new_node
             << " -- UID1 : " << elem.m_node1->m_uid_new_node;
    }
  }

  UniqueArray<Parallel::Request> requests(subDomain()->nbSubDomain() * 2);

  // Si le noeud n'est pas à nous, il faut demander son UID au processus proprio.
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }
    Int32 size = request_uid[sr].size();
    requests[sr * 2] = pm->send(ArrayView{ 1, &size }, sr, false);
    requests[sr * 2 + 1] = pm->send(request_uid[sr], sr, false);
    info() << "[" << my_proc << " -> " << sr << "] Requests : " << request_uid[sr];
  }

  pm->waitAllRequests(requests);
  // pm->freeRequests(requests);

  UniqueArray<UniqueArray<Int64>> answers_uid(subDomain()->nbSubDomain());

  // On reçoit et traite les demandes.
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }
    Int32 size = 0;
    pm->recv(ArrayView{ 1, &size }, sr);
    UniqueArray<Int64> requested_uid(size);
    pm->recv(requested_uid, sr);

    for (Int32 ipair_uid = 0; ipair_uid < requested_uid.size(); ipair_uid += 2) {
      std::optional<Int64> pos = _find(new_faces, requested_uid[ipair_uid], requested_uid[ipair_uid + 1]);
      if (pos) {
        answers_uid[sr].add(new_faces[pos.value()].m_uid_new_face);
        info() << "[" << my_proc << "] Found"
               << " -- UID0 : " << requested_uid[ipair_uid]
               << " -- UID1 : " << requested_uid[ipair_uid + 1]
               << " -- New UID : " << new_faces[pos.value()].m_uid_new_face;
      }
      else {
        warning() << "Not found";
        answers_uid[sr].add(-1);
      }
    }
  }

  // On envoie les réponses.
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }

    Int32 size = answers_uid[sr].size();
    requests[sr * 2] = pm->send(ArrayView{ 1, &size }, sr, false);
    requests[sr * 2 + 1] = pm->send(answers_uid[sr], sr, false);
  }

  pm->waitAllRequests(requests);
  // pm->freeRequests(requests);

  // On reçoit et traite les réponses.
  for (Int32 sr = 0; sr < subDomain()->nbSubDomain(); ++sr) {
    if (sr == subDomain()->subDomainId()) {
      continue;
    }

    Int32 size = 0;
    pm->recv(ArrayView{ 1, &size }, sr);
    UniqueArray<Int64> answered_uid(size);
    pm->recv(answered_uid, sr);

    for (Int32 ipair_uid = 0; ipair_uid < request_uid[sr].size(); ipair_uid += 2) {
      std::optional<Int64> pos = _find(current_plan_new_faces, request_uid[sr][ipair_uid], request_uid[sr][ipair_uid + 1]);
      if (pos) {
        current_plan_new_faces[pos.value()].m_uid_new_face = answered_uid[ipair_uid / 2];
        info() << "[" << my_proc << "] Apply"
               << " -- UID0 : " << current_plan_new_faces[pos.value()].m_node0->m_uid_new_node
               << " -- UID1 : " << current_plan_new_faces[pos.value()].m_node1->m_uid_new_node
               << " -- New UID : " << current_plan_new_faces[pos.value()].m_uid_new_face;
      }
      else {
        ARCANE_FATAL("GL");
      }
    }
  }
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

  info() << "[" << pm->commRank() << "] Gather result (rectified) : " << all_nb_node;

  Span<FaceLite> current_plan_new_faces = new_faces.subView(current_plan, new_faces.size() - current_plan);

  for (auto& elem : current_plan_new_faces) {
    // info() << "[" << pm->commRank() << "] Old UID : " << elem.m_new_node->m_uid_new_node;
    elem.m_uid_new_face += all_nb_node[elem.m_owner_new_face];
    // info() << "[" << pm->commRank() << "] New UID : " << elem.m_new_node->m_uid_new_node;
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

    Int32 size_of_new_nodes = new_nodes.size();
    Int32 size_of_new_faces = new_faces.size();

    Int32 previous_g_nb_node = g_nb_node;
    Int32 previous_g_nb_face = g_nb_face;

    Int32 nb_node_for_this_plan = g_nb_node;
    Int32 nb_face_for_this_plan = g_nb_face;

    _createNodesAndCells(i, g_nb_node, new_nodes, nb_cell, new_cells, g_nb_face, new_faces);

    info() << "qaaaaa";

    _fillNodeUID(g_nb_node, new_nodes, size_of_new_nodes);
    nb_node_for_this_plan = g_nb_node - nb_node_for_this_plan;

    info() << "nb_node_for_this_plan : " << nb_node_for_this_plan;

    nb_node_for_this_plan = _makeUniqueNodeUID(nb_node_for_this_plan, new_nodes, size_of_new_nodes);
    g_nb_node = previous_g_nb_node + nb_node_for_this_plan;

    for (auto& elem : new_nodes) {
      info() << "New node"
             << " -- UID : " << elem.m_new_node->m_uid_new_node
             << " -- Owner : " << elem.m_new_node->m_owner_new_node
             << " -- Pos : " << elem.m_intersection_pos
             << " -- Edge node0 : " << elem.m_new_node->m_uid_node0
             << " -- Edge node1 : " << elem.m_new_node->m_uid_node1;
    }

    Int32 nb_face = 0;

    _fillFaceUID(g_nb_face, new_faces, size_of_new_faces);
    nb_face_for_this_plan = g_nb_face - nb_face_for_this_plan;

    info() << "nb_face_for_this_plan : " << nb_face_for_this_plan;

    nb_face_for_this_plan = _makeUniqueFaceUID(nb_face_for_this_plan, new_faces, size_of_new_faces);
    g_nb_face = previous_g_nb_face + nb_face_for_this_plan;

    for (auto& elem : new_faces) {
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
  }

  info() << "new_cells : " << new_cells;

  _makeUniqueCellUID(nb_cell, new_cells, new_nodes);

  {
    Int32 pos0 = 0;
    while (pos0 < new_cells.size()) {
      StringBuilder logs;
      logs += "New cell -- Type : ";
      Int64 type = new_cells[pos0++];
      logs += type;
      logs += " -- UID : ";
      logs += new_cells[pos0++];

      for (Int32 i = 0; i < type; ++i) {
        logs += " -- Node";
        logs += i;
        logs += " : ";
        logs += new_cells[pos0++];
      }
      info() << logs;
    }
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
  faces_infos.reserve(10000);

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
  UniqueArray<Int64> cells_infos;
  cells_infos.reserve(10000);

  Int32 nb_cells = 0;

  Int32 pos0 = 0;
  while (pos0 < new_cells.size()) {
    Int64 type = new_cells[pos0++];
    Int64 uid = new_cells[pos0++];

    if (type == 3)
      cells_infos.add(ITI_Triangle3);
    else if (type == 4)
      cells_infos.add(ITI_Quad4);
    else if (type == 5)
      cells_infos.add(ITI_Pentagon5);
    else if (type == 6)
      cells_infos.add(ITI_Hexagon6);
    else
      ARCANE_FATAL("Pas implem : {0}", type);

    cells_infos.add(uid);

    for (Int32 i = 0; i < type; ++i) {
      cells_infos.add(new_cells[pos0++]);
    }
    nb_cells++;
  }

  m_cloned_mesh->modifier()->addCells(nb_cells, cells_infos);
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
        info() << "NodeUID : " << uid << " -- Coord : " << node_coords[inode];
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
