// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleMeshGenerator.h                                       (C) 2000-2025 */
/*                                                                           */
/* Service de génération de maillage 'Simple'.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_SIMPLEMESHGENERATOR_H
#define ARCANE_STD_SIMPLEMESHGENERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/IMeshModifierInternal.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/std/IMeshGenerator.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemTypeMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Générateur simple de chaque type d'entité de maillage.
 */
class SimpleMeshGenerator
: public TraceAccessor
, public IMeshGenerator
{
public:

  explicit SimpleMeshGenerator(IPrimaryMesh* mesh);

 public:

  ConstArrayView<Int32> communicatingSubDomains() const override { return {}; }
  bool readOptions(XmlNode node) override;
  bool generateMesh() override;

 protected:

  Integer _addNode(Real3 position);
  Integer _addNode(Real x,Real y,Real z);
  void _addCell(Int16 type_id, ConstArrayView<Int64> nodes_id);
  void _addCell(ItemTypeId type_id, ConstArrayView<Real3> nodes_coords);

 private:

  Int32 m_mode = 1;
  IPrimaryMesh* m_mesh = nullptr;
  ItemTypeMng* m_item_type_mng = nullptr;
  IMeshModifierInternal* m_mesh_modifier_internal = nullptr;
  UniqueArray<Int64> m_nodes_unique_id;
  UniqueArray<Real3> m_nodes_coords;
  UniqueArray<Int64> m_cells_infos;
  Integer m_current_nb_cell = 0;
  typedef std::map<Real3,Integer> Real3Map;
  /*!
   * \brief Mapping Coordonnées --> Indice unique.
   * Pour la fusion automatique des noeuds aux mêmes coordonnées,
   * utilise ce champ pour stocker les coordonnées déjà référencées.
   */
  Real3Map m_coords_to_uid;

 private:

  void _createSimpleDiTetra5(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleAntiWedgeRight6(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleAntiWedgeLeft6(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleHemiHexa5(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleHemiHexa6(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleHemiHexa7(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleOctaedron12(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleHeptaedron10(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleHexaedron8(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleHexaedron(ItemTypeId type_id, Real x0, Real y0, Real z1, Real z2);
  void _createSimplePentaedron6(Real x0,Real y0,Real z1,Real z2);
  void _createSimplePyramid5(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleTetraedron4(Real x0,Real y0,Real z1,Real z2);

  void _fillHexaedronCoordinates(ItemTypeId type_id, ArrayView<Real3> coords, Real x0, Real y0, Real z1, Real z2);

  static Real3 _center(Real3 n0, Real3 n1)
  {
    return (n0 + n1) / 2.0;
  }
  static Real3 _center(Real3 n0, Real3 n1, Real3 n2, Real3 n3)
  {
    return (n0 + n1 + n2 + n3) / 4.0;
  }
  static Real3 _center(ConstArrayView<Real3> coords, Int32 n0, Int32 n1)
  {
    return _center(coords[n0], coords[n1]);
  }
  static Real3 _center(ConstArrayView<Real3> coords, Int32 n0, Int32 n1, Int32 n2, Int32 n3)
  {
    return _center(coords[n0], coords[n1], coords[n2], coords[n3]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
