// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCutService.cc                                           (C) 2000-2026 */
/*                                                                           */
/* TODO.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

class MeshCutService
: public ArcaneMeshCutObject
{
 public:

  explicit MeshCutService(const ServiceBuildInfo& sbi)
  : ArcaneMeshCutObject(sbi)
  {}

 public:

  void addPlan(const Real3& p0, const Real3& normal) override;

 public:

  void setVariables(VariableCollection variables) override;
  void updateSection() override;

  MeshHandle meshSection() override
  {
    return m_cloned_mesh->handle();
  }
  VariableCollection variables() override { return {}; }

 private:

  void _createMesh();
  void _createCells(Int32 plan_pos, Int32& nb_cell, Int32& nb_node, UniqueArray<Int64>& cells_infos, UniqueArray<Real3>& pos_node);
  void _compute();

 private:

  VariableCollection m_variables_ori;
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

struct NodeIntersection
{
  NodeIntersection(const Node aa, const Node bb, const Real3& inter)
  : m_intersection_pos(inter)
  {
    const Int32 a = aa.localId();
    const Int32 b = bb.localId();
    if (a < b) {
      m_lid_node0 = a;
      m_lid_node1 = b;
    }
    else {
      m_lid_node0 = b;
      m_lid_node1 = a;
    }
  }

  NodeIntersection() = default;

  bool operator<(const NodeIntersection& other) const
  {
    if (m_lid_node0 != other.m_lid_node0) {
      return m_lid_node0 < other.m_lid_node0;
    }
    return m_lid_node1 < other.m_lid_node1;
  }

  bool operator==(const NodeIntersection& other) const
  {
    return m_lid_node0 == other.m_lid_node0 && m_lid_node1 == other.m_lid_node1;
  }

  Int32 m_lid_node0 = -1;
  Int32 m_lid_node1 = -1;
  Real3 m_intersection_pos{ -1 };
  Int64 m_uid_new_node = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
addPlan(const Real3& p0, const Real3& normal)
{
  m_plans.add({ p0, math::normalizeReal3(normal) });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
setVariables(VariableCollection variables)
{
  m_variables_ori = variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
updateSection()
{
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
_createCells(Int32 plan_pos, Int32& sd_nb_cell, Int32& sd_nb_node, UniqueArray<Int64>& cells_infos, UniqueArray<Real3>& pos_node)
{
  auto [p0, normal] = m_plans[plan_pos];

  VariableNodeReal3& node_coord = mesh()->nodesCoordinates();
  VariableNodeReal node_dist(VariableBuildInfo(mesh(), "NodeDist"));

  UniqueArray<Int32> face_already_computed;

  // Calcul de la distance signée de chaque noeud par rapport au plan de coupe.
  ENUMERATE_ (Node, inode, allNodes()) {
    node_dist[inode] = math::dot({ node_coord[inode] - p0 }, normal);
  }

  UniqueArray<NodeIntersection> point_coords;
  UniqueArray<NodeIntersection> point_coords_tmp;

  ENUMERATE_ (Cell, icell, ownCells()) {
    ItemWithNodes cell = *icell;

    // On regarde si la maille traverse le plan ou si une de ces faces est collé dessus.
    bool colineaire = false;
    {
      bool cont = true;

      // Si au moins un noeud est sur le plan, peut-être que la face est colinéaire.
      bool has_egal = false;
      {
        // On vérifie que la maille est dans le plan.
        bool has_neg = false, has_pos = false;
        for (Node node : cell.nodes()) {
          Real d = node_dist[node];
          if (d < 0)
            has_neg = true;
          else if (d > 0)
            has_pos = true;
          else
            has_egal = true;
        }
        if (!(has_neg && has_pos))
          cont = false;
      }
      // On vérifie si une des faces est sur le plan.
      if (!cont && has_egal) {
        for (Face face : icell->faces()) {
          has_egal = true;
          for (Node node : face.nodes()) {
            if (!math::isNearlyZero(node_dist[node])) {
              has_egal = false;
              break;
            }
          }
          if (has_egal) {
            if (face_already_computed.contains(face.localId())) {
              cont = false;
              break;
            }
            face_already_computed.add(face.localId());
            for (Node node : face.nodes()) {
              auto ni = NodeIntersection{ node, node, node_coord[node] };
              point_coords_tmp.add(ni);
            }
            cont = true;
            cell = face;
            colineaire = true;
            break;
          }
        }
      }
      if (!cont) {
        continue;
      }
    }

    if (!colineaire) {
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
        ARCANE_FATAL("Type de maille non supporté: nbNode={0}", nb_node);
      }

      // Itère sur toutes les arêtes de la maille.
      for (Integer i = 0; i < nb_edges; ++i) {
        Node node0 = cell.node(edge_def[i][0]);
        Node node1 = cell.node(edge_def[i][1]);

        bool aaaa = true;

        // Si le noeud 0 est sur le plan.
        if (math::isNearlyZero(node_dist[node0])) {
          const Real3 p = node_coord[node0];

          auto ni = NodeIntersection{ node0, node0, p };

          if (!point_coords_tmp.contains(ni)) {
            point_coords_tmp.add(ni);
          }
          aaaa = false;
        }

        // Si le noeud 1 est sur le plan.
        if (math::isNearlyZero(node_dist[node1])) {
          const Real3 p = node_coord[node1];

          auto ni = NodeIntersection{ node1, node1, p };

          if (!point_coords_tmp.contains(ni)) {
            point_coords_tmp.add(ni);
          }
          aaaa = false;
        }

        // Si l'arrête passe à travers le plan.
        if (aaaa && node_dist[node0] * node_dist[node1] < 0) {

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

    if (point_coords_tmp.size() >= 3) {
      // On ajoute le uniqueId du nouveau noeud. Si ce noeud a déjà été créé, on récupère son uniqueId.
      {
        for (auto& new_node : point_coords_tmp) {
          auto pos = point_coords.span().findFirst(new_node);
          if (pos) {
            new_node.m_uid_new_node = point_coords[pos.value()].m_uid_new_node;
          }
          else {
            new_node.m_uid_new_node = sd_nb_node++;
            pos_node.add(new_node.m_intersection_pos);
            point_coords.add(new_node);
          }
        }
      }

      if (point_coords_tmp.size() == 3)
        cells_infos.add(ITI_Triangle3);
      else if (point_coords_tmp.size() == 4)
        cells_infos.add(ITI_Quad4);
      else if (point_coords_tmp.size() == 5)
        cells_infos.add(ITI_Pentagon5);
      else if (point_coords_tmp.size() == 6)
        cells_infos.add(ITI_Hexagon6);
      else
        ARCANE_FATAL("Pas implem : {0}", point_coords_tmp.size());

      cells_infos.add(sd_nb_cell);

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
                  [&](Int64 ia, Int64 ib) {
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

        for (Int64 idx : indices) {
          ARCANE_FATAL_IF(point_coords_tmp[idx].m_uid_new_node == -1, "aaa {0}", point_coords_tmp[idx].m_uid_new_node);
          cells_infos.add(point_coords_tmp[idx].m_uid_new_node);
        }
      }
      sd_nb_cell++;
    }
    point_coords_tmp.clear();
  }

  // for (auto elem : point_coords) {
  //   info() << "UID : " << elem.m_uid_new_node << "\t -- Node0 : " << elem.m_lid_node0 << "\t -- Node1 : " << elem.m_lid_node1 << "\t -- Pos : " << elem.m_intersection_pos;
  // }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutService::
_compute()
{
  UniqueArray<Int64> cells_infos;
  cells_infos.reserve(10000);

  UniqueArray<Real3> pos_node;
  pos_node.reserve(10000);

  Int32 nb_cell = 0;
  Int32 nb_node = 0;

  for (Int32 i = 0; i < m_plans.size(); ++i) {
    _createCells(i, nb_cell, nb_node, cells_infos, pos_node);
  }

  m_cloned_mesh->modifier()->addCells(nb_cell, cells_infos);
  m_cloned_mesh->modifier()->endUpdate();

  {
    VariableNodeReal3& node_coords(m_cloned_mesh->nodesCoordinates());
    ENUMERATE_ (Node, inode, m_cloned_mesh->allNodes()) {
      node_coords[inode] = pos_node[inode->uniqueId()];
    }
  }

  info() << "New mesh -- NbNode : " << m_cloned_mesh->nbNode() << " -- NbCells : " << m_cloned_mesh->nbCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
