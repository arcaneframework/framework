// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleMeshGenerator.h                                       (C) 2000-2020 */
/*                                                                           */
/* Service de génération de maillage 'Simple'.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_SIMPLEMESHGENERATOR_H
#define ARCANE_STD_SIMPLEMESHGENERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/std/IMeshGenerator.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Génèrateur simple de chaque type d'entité de maillage.
 */
class SimpleMeshGenerator
: public TraceAccessor
, public IMeshGenerator
{
public:
  SimpleMeshGenerator(IPrimaryMesh* mesh);
 public:
  IntegerConstArrayView communicatingSubDomains() const override
  {
    return IntegerConstArrayView();
  }
  bool readOptions(XmlNode node) override;
  bool generateMesh() override;
 protected:
  Integer _addNode(const Real3& position);
  Integer _addNode(Real x,Real y,Real z);
  void _addCell(Integer type_id,IntegerConstArrayView nodes_id);
 private:
  Integer m_mode;
  IPrimaryMesh* m_mesh;
  Int64UniqueArray m_nodes_unique_id;
  Real3UniqueArray m_nodes_coords;
  Int64UniqueArray m_cells_infos;
  Integer m_current_nb_cell;
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
  void _createSimplePentaedron6(Real x0,Real y0,Real z1,Real z2);
  void _createSimplePyramid5(Real x0,Real y0,Real z1,Real z2);
  void _createSimpleTetraedron4(Real x0,Real y0,Real z1,Real z2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
