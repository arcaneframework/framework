// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HoneyCombMeshGenerator.h                                    (C) 2000-2022 */
/*                                                                           */
/* Service de génération de maillage hexagonal.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_HONEYCOMBMESHGENERATOR_H
#define ARCANE_STD_HONEYCOMBMESHGENERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/UniqueArray.h"

#include "arcane/ArcaneTypes.h"

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

  //! Infos sur une ligne d'hexagone
  class CellLineInfo;

 public:

  explicit HoneyComb2DMeshGenerator(IPrimaryMesh* mesh);

 public:

  void generateMesh(Real2 origin, Real pitch, Integer nb_ring);

 private:

  IPrimaryMesh* m_mesh = nullptr;
  Real m_pitch = 0.0;
  Integer m_nb_ring = 0;
  Real2 m_origin;

 private:

  void _buildCells();
  void _buildCells2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HoneyComb3DMeshGenerator
: public TraceAccessor
{
 public:

  //! Infos sur une ligne d'hexagone
  class CellLineInfo;

 public:

  explicit HoneyComb3DMeshGenerator(IPrimaryMesh* mesh);

 public:

  void generateMesh(Real2 origin, Real pitch, Integer nb_ring, ConstArrayView<Real> heights);

 private:

  IPrimaryMesh* m_mesh = nullptr;
  Real m_pitch = 0.0;
  Integer m_nb_ring = 0;
  Real2 m_origin;
  UniqueArray<Real> m_heights;

 private:

  void _buildCellNodes(Real x, Real y, Real z, ArrayView<Real3> coords);
  void _buildCells();
  void _buildCells2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
