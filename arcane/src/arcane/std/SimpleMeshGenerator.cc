// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleMeshGenerator.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Service de génération de maillage 'Simple'.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleMeshGenerator.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/FixedArray.h"

#include "arcane/core/XmlNode.h"
#include "arcane/core/Item.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/internal/IMeshModifierInternal.h"

#include <arcane/utils/SmallArray.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleMeshGenerator::
SimpleMeshGenerator(IPrimaryMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_mesh_modifier_internal(mesh->modifier()->_modifierInternalApi())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute ou récupère le nœud ayant pour coordonnées \a pos.
 *
 * \return Le uniqueId() du noeud.
 */
Integer SimpleMeshGenerator::
_addNode(Real3 pos)
{
  auto i = m_coords_to_uid.find(pos);
  if (i == m_coords_to_uid.end()) {
    Integer current_id = m_nodes_unique_id.size();
    m_nodes_unique_id.add(current_id);
    m_nodes_coords.add(pos);
    m_coords_to_uid.insert(std::make_pair(pos, current_id));
    return current_id;
  }
  return i->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute ou récupère le nœud ayant pour coordonnées \a pos.
 *
 * \return Le uniqueId() du noeud.
 */
Integer SimpleMeshGenerator::
_addNode(Real x, Real y, Real z)
{
  return _addNode(Real3(x, y, z));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_addCell(Int16 type_id, ConstArrayView<Int64> nodes_id)
{
  Integer current_id = m_current_nb_cell;
  ++m_current_nb_cell;
  m_mesh_modifier_internal->addCell(ItemUniqueId(current_id), ItemTypeId(type_id), nodes_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute une maille avec les coordonnées des noeuds \a nodes_coords.
 */
void SimpleMeshGenerator::
_addCell(ItemTypeId type_id, ConstArrayView<Real3> nodes_coords)
{
  Integer current_id = m_current_nb_cell;
  ++m_current_nb_cell;
  const Int32 nb_node = nodes_coords.size();
  SmallArray<Int64> nodes_unique_id(nb_node);
  for (Int32 i = 0; i < nb_node; ++i)
    nodes_unique_id[i] = _addNode(nodes_coords[i]);
  m_mesh_modifier_internal->addCell(ItemUniqueId(current_id), type_id, nodes_unique_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleMeshGenerator::
readOptions(XmlNode node)
{
  XmlNode modeNode = node.child(String("mode"));
  m_mode = modeNode.valueAsInteger();
  if (m_mode < 1 || m_mode > 4) {
    info() << "Bad syntax for <meshgenerator>";
    info() << "Expected is <meshgenerator><simple><mode>mode</simple></meshgenerator>";
    info() << "with mode between 1 and 3";
    error() << "Bad syntax for <meshgenerator>";
    return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleHemiHexa7(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 7> ids;
  ids[0] = _addNode(x0, y0, z1);
  ids[1] = _addNode(x0 + 1., y0, z1);
  ids[2] = _addNode(x0 + 1., y0, z2);
  ids[3] = _addNode(x0 + 1., y0 + 1., z2);
  ids[4] = _addNode(x0, y0 + 1., z2);
  ids[5] = _addNode(x0, y0 + 1., z1);
  ids[6] = _addNode(x0 + 1., y0 + 1., z1);
  _addCell(IT_HemiHexa7, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleHemiHexa6(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 6> ids;
  ids[0] = _addNode(x0, y0, z1);
  ids[1] = _addNode(x0 + 1., y0, z1);
  ids[2] = _addNode(x0 + 1., y0, z2);
  ids[3] = _addNode(x0 + 1., y0 + 1., z1);
  ids[4] = _addNode(x0, y0 + 1., z2);
  ids[5] = _addNode(x0, y0 + 1., z1);
  _addCell(IT_HemiHexa6, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleHemiHexa5(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 5> ids;
  ids[0] = _addNode(x0, y0, z1);
  ids[1] = _addNode(x0 + 1., y0, z1);
  ids[2] = _addNode(x0 + 1., y0, z2);
  ids[3] = _addNode(x0 + 1., y0 + 1., z1);
  ids[4] = _addNode(x0, y0 + 1., z1);
  _addCell(IT_HemiHexa5, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleAntiWedgeLeft6(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 6> ids;
  ids[0] = _addNode(x0, y0, z1);
  ids[1] = _addNode(x0, y0 + 1., z1);
  ids[2] = _addNode(x0, y0 + .5, z2);
  ids[3] = _addNode(x0 + 1., y0, z1);
  ids[4] = _addNode(x0 + 1., y0 + 1., z1);
  ids[5] = _addNode(x0 + 1., y0 + .5, (z1 + z2) / 2);
  _addCell(IT_AntiWedgeLeft6, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleAntiWedgeRight6(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 6> ids;
  ids[0] = _addNode(x0, y0, z1);
  ids[1] = _addNode(x0, y0 + 1., z1);
  ids[2] = _addNode(x0, y0 + .5, (z1 + z2) / 2);
  ids[3] = _addNode(x0 + 1., y0, z1);
  ids[4] = _addNode(x0 + 1., y0 + 1., z1);
  ids[5] = _addNode(x0 + 1., y0 + .5, z2);
  _addCell(IT_AntiWedgeRight6, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleDiTetra5(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 5> ids;
  ids[0] = _addNode(x0, y0, z2);
  ids[1] = _addNode(x0 + 1., y0, z1);
  ids[2] = _addNode(x0 + 2., y0, z2);
  ids[3] = _addNode(x0 + 1., y0 + 1., (z2 + z1) / 2);
  ids[4] = _addNode(x0 + 1., y0 - 1., (z2 + z1) / 2);
  _addCell(IT_DiTetra5, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_fillHexaedronCoordinates(ItemTypeId type_id, ArrayView<Real3> coords, Real x0, Real y0, Real z1, Real z2)
{
  coords[0] = Real3(x0, y0, z1);
  coords[1] = Real3(x0 + 1., y0, z1);
  coords[2] = Real3(x0 + 1., y0 + 1., z1);
  coords[3] = Real3(x0, y0 + 1., z1);
  coords[4] = Real3(x0, y0, z2);
  coords[5] = Real3(x0 + 1., y0, z2);
  coords[6] = Real3(x0 + 1., y0 + 1., z2);
  coords[7] = Real3(x0, y0 + 1., z2);

  // Ajoute le centre des faces
  if (type_id == ITI_Hexaedron20 || type_id == ITI_Hexaedron27) {
    // Hexaedre d'ordre 2 ou plus
    coords[8] = _center(coords, 0, 1);
    coords[9] = _center(coords, 1, 2);
    coords[10] = _center(coords, 2, 3);
    coords[11] = _center(coords, 3, 0);
    coords[12] = _center(coords, 4, 5);
    coords[13] = _center(coords, 5, 6);
    coords[14] = _center(coords, 6, 7);
    coords[15] = _center(coords, 7, 4);
    coords[16] = _center(coords, 0, 4);
    coords[17] = _center(coords, 1, 5);
    coords[18] = _center(coords, 2, 6);
    coords[19] = _center(coords, 3, 7);
  }
  if (type_id == ITI_Hexaedron27) {
    // Ajoute le centre des faces
    // TODO: Utiliser le type LocalFace du type pour garantir
    // la cohérence de la numérotation
    coords[20] = _center(coords, 0, 4, 7, 3);
    coords[21] = _center(coords, 1, 2, 6, 5);
    coords[22] = _center(coords, 0, 1, 5, 4);
    coords[23] = _center(coords, 2, 3, 7, 6);
    coords[24] = _center(coords, 0, 3, 2, 1);
    coords[25] = _center(coords, 4, 5, 6, 7);

    // Ajoute le centre de la maille
    Real3 center;
    for (Int32 i = 0; i < 8; ++i)
      center += coords[i];
    center /= 8.0;
    coords[26] = center;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleHexaedron(ItemTypeId type_id, Real x0, Real y0, Real z1, Real z2)
{
  if (type_id != ITI_Hexaedron8 && type_id != ITI_Hexaedron27 && type_id != ITI_Hexaedron20)
    ARCANE_FATAL("Invalid hexaedron type '{0}'", type_id);
  FixedArray<Real3, 27> coords_array;
  Int32 nb_node = m_item_type_mng->typeFromId(type_id)->nbLocalNode();
  ArrayView<Real3> coords(coords_array.view().subView(0, nb_node));
  _fillHexaedronCoordinates(type_id, coords, x0, y0, z1, z2);
  _addCell(type_id, coords);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleHexaedron8(Real x0, Real y0, Real z1, Real z2)
{
  _createSimpleHexaedron(ITI_Hexaedron8, x0, y0, z1, z2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleOctaedron12(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 12> ids;
  ids[0] = _addNode(x0, y0 + 0.5, z1);
  ids[1] = _addNode(x0 + 0.25, y0, z1);
  ids[2] = _addNode(x0 + 0.75, y0, z1);
  ids[3] = _addNode(x0 + 1.0, y0 + 0.5, z1);
  ids[4] = _addNode(x0 + 0.75, y0 + 1., z1);
  ids[5] = _addNode(x0 + 0.25, y0 + 1., z1);

  ids[6] = _addNode(x0, y0 + 0.5, z2);
  ids[7] = _addNode(x0 + 0.25, y0, z2);
  ids[8] = _addNode(x0 + 0.75, y0, z2);
  ids[9] = _addNode(x0 + 1., y0 + 0.5, z2);
  ids[10] = _addNode(x0 + 0.75, y0 + 1., z2);
  ids[11] = _addNode(x0 + 0.25, y0 + 1., z2);
  _addCell(IT_Octaedron12, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleHeptaedron10(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 10> ids;
  ids[0] = _addNode(x0, y0 + 0.5, z1);
  ids[1] = _addNode(x0 + 1. / 2., y0, z1);
  ids[2] = _addNode(x0 + 2. / 2., y0 + 0.25, z1);
  ids[3] = _addNode(x0 + 2. / 2., y0 + 0.75, z1);
  ids[4] = _addNode(x0 + 1. / 2., y0 + 1, z1);

  ids[5] = _addNode(x0, y0 + 0.5, z2);
  ids[6] = _addNode(x0 + 1. / 2., y0, z2);
  ids[7] = _addNode(x0 + 2. / 2., y0 + 0.25, z2);
  ids[8] = _addNode(x0 + 2. / 2., y0 + 0.75, z2);
  ids[9] = _addNode(x0 + 1. / 2., y0 + 1., z2);
  _addCell(IT_Heptaedron10, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimplePentaedron6(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 6> ids;
  ids[0] = _addNode(x0, y0, z1);
  ids[1] = _addNode(x0 + 1., y0 + 1. / 2., z1);
  ids[2] = _addNode(x0, y0 + 1., z1);
  ids[3] = _addNode(x0, y0, z2);
  ids[4] = _addNode(x0 + 1., y0 + 1. / 2., z2);
  ids[5] = _addNode(x0, y0 + 1., z2);
  _addCell(IT_Pentaedron6, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimplePyramid5(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 5> ids;
  ids[0] = _addNode(x0, y0, z1);
  ids[1] = _addNode(x0 + 1., y0, z1);
  ids[2] = _addNode(x0 + 1., y0 + 1., z1);
  ids[3] = _addNode(x0, y0 + 1., z1);
  ids[4] = _addNode(x0 + 0.5, y0 + 0.5, z2);
  _addCell(IT_Pyramid5, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleMeshGenerator::
_createSimpleTetraedron4(Real x0, Real y0, Real z1, Real z2)
{
  FixedArray<Int64, 4> ids;
  ids[0] = _addNode(x0, y0, z1);
  ids[1] = _addNode(x0 + 1., y0, z1);
  ids[2] = _addNode(x0 + 0.5, y0 + 1., z1);
  ids[3] = _addNode(x0 + 0.5, y0 + 0.5, z2);
  _addCell(IT_Tetraedron4, ids.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleMeshGenerator::
generateMesh()
{
  IPrimaryMesh* mesh = m_mesh;

  info() << "Using simple mesh generator";
  mesh->setDimension(3);
  m_item_type_mng = mesh->itemTypeMng();

  if (m_mode == 1) {
    _createSimpleHexaedron8(2., 0., 0., 1.);
    _createSimpleOctaedron12(4., 0., 0., 1.);
    _createSimpleHeptaedron10(6., 0., 0., 1.);
    _createSimplePentaedron6(8., 0., 0., 1.);
    _createSimplePyramid5(10., 0., 0., 1.);
    _createSimpleTetraedron4(12., 0., 0., 1.);
    _createSimpleHemiHexa7(14., 0., 0., 1.);
    _createSimpleHemiHexa6(16., 0., 0., 1.);
    _createSimpleHemiHexa5(18., 0., 0., 1.);
    _createSimpleAntiWedgeLeft6(20., 0., 0., 1.);
    _createSimpleAntiWedgeRight6(22., 0., 0., 1.);
    _createSimpleDiTetra5(24., 0., 0., 1.);
  }

  if (m_mode == 2) {
    _createSimpleHexaedron8(0., 0., 0., 1.);
    _createSimpleHemiHexa7(0., 0., 1., 2.);
    _createSimpleHexaedron8(1., 0., 0., 1.);

    _createSimpleHexaedron8(2., 0., 0., 1.);
    _createSimpleHemiHexa6(2., 0., 1., 2.);
    _createSimpleHexaedron8(3., 0., 0., 1.);

    _createSimpleHexaedron8(4., 0., 0., 1.);
    _createSimpleHemiHexa5(4., 0., 1., 2.);
    _createSimpleHexaedron8(5., 0., 0., 1.);

    _createSimpleHexaedron8(6., 0., 0., 1.);
    _createSimpleAntiWedgeLeft6(6., 0., 1., 2.);
    _createSimpleHexaedron8(7., 0., 0., 1.);

    _createSimpleHexaedron8(8., 0., 0., 1.);
    _createSimpleAntiWedgeRight6(8., 0., 1., 2.);
    _createSimpleHexaedron8(9., 0., 0., 1.);

    _createSimpleHexaedron8(10., 0., 0., 1.);
    _createSimplePyramid5(10., 0., 1., 2.);
    _createSimpleHexaedron8(11., 0., 0., 1.);
  }

  if (m_mode == 3) {
    _createSimpleHexaedron8(2., 0., 0., 1.);
    _createSimpleOctaedron12(4., 0., 0., 1.);
    _createSimpleHeptaedron10(6., 0., 0., 1.);
    _createSimplePentaedron6(8., 0., 0., 1.);
    _createSimplePyramid5(10., 0., 0., 1.);
    _createSimpleTetraedron4(12., 0., 0., 1.);
  }

  if (m_mode == 4) {
    _createSimpleHexaedron(ITI_Hexaedron20, 0., 0., 0., 1.);
    _createSimpleHexaedron(ITI_Hexaedron20, 1., 0., 0., 1.);
    _createSimpleHexaedron(ITI_Hexaedron27, 2., 0., 0., 1.);
    _createSimpleHexaedron(ITI_Hexaedron27, 3., 0., 0., 1.);
  }

  UniqueArray<Int64> cells_infos(m_cells_infos.size());
  for (Integer i = 0; i < m_cells_infos.size(); ++i)
    cells_infos[i] = m_cells_infos[i];
  mesh->endAllocate();

  UniqueArray<Int64> nodes_unique_id(m_nodes_unique_id.size());
  for (Integer i = 0; i < m_nodes_unique_id.size(); ++i)
    nodes_unique_id[i] = m_nodes_unique_id[i];

  VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
  {
    // Remplit la variable contenant les coordonnées des noeuds
    UniqueArray<Int32> nodes_local_id(nodes_unique_id.size());
    IItemFamily* family = mesh->itemFamily(IK_Node);
    family->itemsUniqueIdToLocalId(nodes_local_id, nodes_unique_id);
    NodeInfoListView nodes_internal(family);
    Integer nb_node_local_id = nodes_local_id.size();
    for (Integer i = 0; i < nb_node_local_id; ++i) {
      Node node = nodes_internal[nodes_local_id[i]];
      //Int64 unique_id = nodes_unique_id[i];
      nodes_coord_var[node] = m_nodes_coords[i];
      //info() << "Set coord " << ItemPrinter(node) << " coord=" << nodes_coord_var[node];
    }
  }
  nodes_coord_var.synchronize();

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
