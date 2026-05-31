// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SodMeshGenerator.h                                          (C) 2000-2024 */
/*                                                                           */
/* Service for generating a 'sod'-style mesh.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_SODMESHGENERATOR_H
#define ARCANE_STD_SODMESHGENERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/std/IMeshGenerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mesh generator for a shock tube.
 *
 * The mesh is a Cartesian mesh in unstructured format
 *
 * The tube has two zones, ZG and ZD, along the x-axis: the first
 * half of the cells is for ZG, the following for ZD.
 *
 * In parallel, layers are increased along the z-axis. Each subdomain
 * has \a nb_cell_z layers along z and shares a layer with the
 * previous subdomain and a layer with the next one. In this way,
 * each subdomain calculates the same thing and the number of iterations
 * of the case does not change regardless of the number of processors.
 * For boundary conditions, six surfaces are created: XMIN, XMAX for
 * the faces along X, YMIN and YMAX for the faces along Y and ZMIN and
 * ZMAX for those along Z.
 */
class SodMeshGenerator
: public TraceAccessor
, public IMeshGenerator
{
 public:

  class Impl;

 public:

  SodMeshGenerator(IPrimaryMesh* tm, bool use_zxy = false);
  ~SodMeshGenerator();

 public:

  IntegerConstArrayView communicatingSubDomains() const override;
  bool readOptions(XmlNode node) override;
  bool generateMesh() override;

 private:

  IPrimaryMesh* m_mesh;
  bool m_zyx_generate;
  std::unique_ptr<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
