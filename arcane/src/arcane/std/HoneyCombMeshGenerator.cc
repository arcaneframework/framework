// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HoneyCombMeshGenerator.cc                                   (C) 2000-2022 */
/*                                                                           */
/* Service de génération de maillage hexagonal.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real2.h"

#include "arcane/IMeshBuilder.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/IParallelMng.h"

#include "arcane/std/HoneyComb2DMeshGenerator_axl.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 *
 * \brief Classe d'aide pour générer un maillage séquentiel en se basant
 * sur la connectivité est les coordonnées des noeuds.
 */
class SimpleSequentialMeshBuilder
: public TraceAccessor
{
 private:

  class Real3Compare
  {
   public:

    bool operator()(const Real3& a, const Real3& b) const
    {
      if (!math::isNearlyEqualWithEpsilon(a.x, b.x, m_epsilon))
        return a.x < b.x;
      if (!math::isNearlyEqualWithEpsilon(a.y, b.y, m_epsilon))
        return a.y < b.y;
      if (!math::isNearlyEqualWithEpsilon(a.z, b.z, m_epsilon))
        return a.z < b.z;
      return false;
    }

   private:

    Real m_epsilon = 1e-14;
  };

 public:

  explicit SimpleSequentialMeshBuilder(IMesh* pm)
  : TraceAccessor(pm->traceMng())
  , m_mesh(pm)
  {
  }

 public:

  //! Ajoute ou récupère l'uniqueId() du noeud de coordonnées \a coord.
  Int32 addNode(Real3 coord)
  {
    auto p = m_nodes_coord_map.find(coord);
    if (p != m_nodes_coord_map.end())
      return p->second;
    Int32 v = m_nodes_coordinates.size();
    m_nodes_coordinates.add(coord);
    m_nodes_coord_map.insert(std::make_pair(coord, v));
    return v;
  }

  Int32 addCell(Int32 type, ConstArrayView<Real3> nodes_coords)
  {
    m_cells_infos.add(type);
    const Int32 v = m_nb_cell;
    m_cells_infos.add(v);
    ++m_nb_cell;
    for (Real3 coord : nodes_coords)
      m_cells_infos.add(addNode(coord));
    return v;
  }
  ConstArrayView<Int64> cellsInfos() const { return m_cells_infos; }
  Int32 nbCell() const { return m_nb_cell; }

  void setNodesCoordinates() const
  {
    Int32 nb_node = m_nodes_coordinates.size();
    UniqueArray<Int64> unique_ids(nb_node);
    for (Integer i = 0; i < nb_node; ++i)
      unique_ids[i] = i;
    UniqueArray<Int32> local_ids(nb_node);
    IItemFamily* node_family = m_mesh->nodeFamily();
    node_family->itemsUniqueIdToLocalId(local_ids, unique_ids);

    VariableNodeReal3& mesh_nodes_coordinates(m_mesh->nodesCoordinates());
    ItemInternalList nodes(node_family->itemsInternal());

    for (Integer i = 0; i < nb_node; ++i) {
      Node node{ nodes[local_ids[i]] };
      mesh_nodes_coordinates[node] = m_nodes_coordinates[i];
    }
  }

 private:

  using CoordMap = std::map<Real3, Int32, Real3Compare>;

  IMesh* m_mesh = nullptr;
  CoordMap m_nodes_coord_map;
  Int32 m_nb_cell = 0;
  UniqueArray<Int64> m_cells_infos;
  //! Correspondante uid->coord pour les noeuds
  UniqueArray<Real3> m_nodes_coordinates;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HoneyComb2DMeshGenerator
: public TraceAccessor
{
 public:

  //! Infos sur une ligne d'hexagone
  class CellLineInfo
  {
   public:

    CellLineInfo(Real2 first_center, Int32 nb_cell)
    : m_first_center(first_center)
    , m_nb_cell(nb_cell)
    {}

   public:

    Real2 m_first_center;
    Int32 m_nb_cell;
  };

 public:

  explicit HoneyComb2DMeshGenerator(IPrimaryMesh* mesh);

 public:

  void generateMesh(Real2 origin, Real pitch, Integer nb_ring)
  {
    m_pitch = pitch;
    m_nb_ring = nb_ring;
    m_origin = origin;
    _buildCells();
  }

 private:

  IPrimaryMesh* m_mesh = nullptr;
  Real m_pitch = 0.0;
  Integer m_nb_ring = 0;
  Real2 m_origin;

 private:

  void _buildCellNodes(Real x, Real y, ArrayView<Real3> coords);
  void _buildCells();
  void _buildCells2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HoneyComb2DMeshGenerator::
HoneyComb2DMeshGenerator(IPrimaryMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyComb2DMeshGenerator::
_buildCellNodes(Real x, Real y, ArrayView<Real3> coords)
{
  Real pitch = m_pitch;
  Real p = 0.5 * pitch;
  Real q = 0.5 * pitch / (math::sqrt(3.0));
  coords[0] = { x + q, y + p, 0.0 };
  coords[1] = { x - q, y + p, 0.0 };
  coords[2] = { x - 2 * q, y, 0.0 };
  coords[3] = { x - q, y - p, 0.0 };
  coords[4] = { x + q, y - p, 0.0 };
  coords[5] = { x + 2 * q, y, 0.0 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyComb2DMeshGenerator::
_buildCells()
{
  // En parallèle, seul le rang 0 construit le maillage
  IParallelMng* pm = m_mesh->parallelMng();
  if (pm->commRank() == 0)
    _buildCells2();
  else {
    m_mesh->setDimension(2);
    m_mesh->allocateCells(0, Int64ConstArrayView(), true);
  }
}

void HoneyComb2DMeshGenerator::
_buildCells2()
{
  SimpleSequentialMeshBuilder simple_mesh_builder(m_mesh);

  const Real pitch = m_pitch;
  const Integer nb_ring = m_nb_ring;

  info() << "Build Hexagonal 2D Cells nb_layer=" << nb_ring;

  UniqueArray<Real3> cell_nodes_coords(6);
  UniqueArray<CellLineInfo> cells_line_info;

  // u1, u2 and u3 are the directional vectors following the
  // directions of the hexagon
  Real2 u1(0.5 * pitch * math::sqrt(3.0), -0.5 * pitch);
  Real2 u2(0.5 * pitch * math::sqrt(3.0), 0.5 * pitch);
  Real2 u3(pitch * 0.0, pitch * 1.0);

  // Mailles au dessus du centre
  for (Integer i = (-nb_ring + 1); i < 1; ++i) {
    Real x0 = u3[0] * i - u1[0] * (nb_ring - 1);
    Real y0 = u3[1] * i - u1[1] * (nb_ring - 1);
    Real2 pos(x0, y0);
    pos += m_origin;
    Integer nb_cell_in_line = 2 * nb_ring - 1 + i;
    cells_line_info.add(CellLineInfo(pos, nb_cell_in_line));
  }

  // Mailles en dessous du centre
  for (Integer i = 1; i < nb_ring; ++i) {
    Real x0{ u2[0] * i - u1[0] * (nb_ring - 1) };
    Real y0{ u2[1] * i - u1[1] * (nb_ring - 1) };
    Real2 pos(x0, y0);
    pos += m_origin;
    Integer nb_cell_in_line = 2 * nb_ring - 1 - i;
    cells_line_info.add(CellLineInfo(pos, nb_cell_in_line));
  }

  for (const CellLineInfo& cli : cells_line_info) {
    Real2 xy = cli.m_first_center;
    for (Integer j = 0, n = cli.m_nb_cell; j < n; ++j) {
      _buildCellNodes(xy.x, xy.y, cell_nodes_coords);
      simple_mesh_builder.addCell(IT_Hexagon6, cell_nodes_coords);
      xy += u1;
    }
  }

  m_mesh->setDimension(2);
  m_mesh->allocateCells(simple_mesh_builder.nbCell(), simple_mesh_builder.cellsInfos(), true);

  simple_mesh_builder.setNodesCoordinates();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de génération de maillage cartésien en 2D.
 */
class HoneyComb2DMeshGeneratorService
: public ArcaneHoneyComb2DMeshGeneratorObject
{
 public:

  explicit HoneyComb2DMeshGeneratorService(const ServiceBuildInfo& sbi)
  : ArcaneHoneyComb2DMeshGeneratorObject(sbi)
  {}

 public:

  void fillMeshBuildInfo([[maybe_unused]] MeshBuildInfo& build_info) override
  {
  }
  void allocateMeshItems(IPrimaryMesh* pm) override
  {
    info() << "HoneyComb2DMeshGenerator: allocateMeshItems()";
    HoneyComb2DMeshGenerator g(pm);
    Real pitch = options()->pitchSize();
    Integer nb_layer = options()->nbLayer();
    if (pitch <= 0.0)
      ARCANE_FATAL("Invalid valid value '{0}' for pitch (should be > 0.0)", pitch);
    if (nb_layer <= 0)
      ARCANE_FATAL("Invalid valid value '{0}' for 'nb-layer' (should be > 0)", nb_layer);
    g.generateMesh(options()->origin(), pitch, nb_layer);
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HONEYCOMB2DMESHGENERATOR(HoneyComb2D, HoneyComb2DMeshGeneratorService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
