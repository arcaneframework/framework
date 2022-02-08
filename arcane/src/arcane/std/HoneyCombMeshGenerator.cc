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

#include "arcane/std/HoneyComb2DMeshGenerator_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HoneyComb2DMeshGenerator
: public TraceAccessor
{
 public:

  HoneyComb2DMeshGenerator(IPrimaryMesh* mesh);

 public:

  void generateMesh(Real pitch, Integer nb_ring)
  {
    m_pitch = pitch;
    m_nb_ring = nb_ring;
    _buildCells();
  }

 private:

  IPrimaryMesh* m_mesh = nullptr;
  Real m_pitch = 0.0;
  Integer m_nb_ring = 0;

 private:

  void _buildCellNodes(Real x, Real y, ArrayView<Real2> coords);
  void _buildCells();
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
_buildCellNodes(Real x, Real y, ArrayView<Real2> coords)
{
  Real pitch = m_pitch;
  Real p = 0.5 * pitch;
  Real q = 0.5 * pitch / (math::sqrt(3.0));
  coords[0] = { x + q, y + p };
  coords[1] = { x - q, y + p };
  coords[2] = { x - 2 * q, y };
  coords[3] = { x - q, y - p };
  coords[4] = { x + q, y - p };
  coords[5] = { x + 2 * q, y };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyComb2DMeshGenerator::
_buildCells()
{
  // Construit les mailles
  // Pour l'instant les noeuds sont dupliqués entre les mailles

  UniqueArray<Real2> cells_center;
  const Real pitch = m_pitch;
  const Integer nb_ring = m_nb_ring;

  info() << "Build Hexagonal 2D Cells nb_layer=" << nb_ring;

  // u1, u2 and u3 are the directional vectors following the
  // directions of the hexagon
  Real2 u1(0.5 * pitch * math::sqrt(3.0), -0.5 * pitch);
  Real2 u2(0.5 * pitch * math::sqrt(3.0), 0.5 * pitch);
  Real2 u3(pitch * 0.0, pitch * 1.0);

  for (Integer i = (-nb_ring + 1); i < 1; ++i) {
    Real x0 = u3[0] * i - u1[0] * (nb_ring - 1);
    Real y0 = u3[1] * i - u1[1] * (nb_ring - 1);
    Integer numberOfCellsPerLine = 2 * nb_ring - 1 + i;
    //for j in range(numberOfCellsPerLine):
    for (Integer j = 0; j < numberOfCellsPerLine; ++j) {
      cells_center.add({ x0, y0 });
      x0 += u1[0];
      y0 += u1[1];
      //info() << "ADD_CELL xy=" << cells_center.back();
    }
  }

  for (Integer i = 1; i < nb_ring; ++i) {
    Real x0{ u2[0] * i - u1[0] * (nb_ring - 1) };
    Real y0{ u2[1] * i - u1[1] * (nb_ring - 1) };
    Integer numberOfCellsPerLine = 2 * nb_ring - 1 - i;
    for (Integer j = 0; j < numberOfCellsPerLine; ++j) {
      cells_center.add({ x0, y0 });
      x0 += u1[0];
      y0 += u1[1];
      //info() << "ADD_CELL xy=" << cells_center.back();
    }
  }

  // Créé les mailles (les noeuds sont dupliqués)
  Integer nb_cell = cells_center.size();
  UniqueArray<Int64> cells_infos;
  cells_infos.reserve(nb_cell * 8);
  for (Integer i = 0; i < nb_cell; ++i) {
    // Type
    cells_infos.add(IT_Hexagon6);
    // UID
    cells_infos.add(i);
    for (Integer j = 0; j < 6; ++j)
      cells_infos.add((i * 6) + j);
  }
  m_mesh->setDimension(2);
  m_mesh->allocateCells(nb_cell, cells_infos, true);

  // Positionne les coordonnées des noeuds
  {
    VariableNodeReal3& nodes_coordinates(m_mesh->nodesCoordinates());
    ItemInternalList cells(m_mesh->cellFamily()->itemsInternal());
    UniqueArray<Real2> coords(6);
    // Suppose qu'on n'a pas changé l'ordre entre les localId() et uniqueId()
    for (Integer i = 0; i < nb_cell; ++i) {
      Real2 center = cells_center[i];
      Cell cell{ cells[i] };
      _buildCellNodes(center.x, center.y, coords);
      for (Integer j = 0; j < 6; ++j)
        nodes_coordinates[cell.node(j)] = Real3(coords[j].x, coords[j].y, 0.0);
    }
  }
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

  HoneyComb2DMeshGeneratorService(const ServiceBuildInfo& sbi)
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
    g.generateMesh(pitch, nb_layer);
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
