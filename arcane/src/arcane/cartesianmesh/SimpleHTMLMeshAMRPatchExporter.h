// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleHTMLMeshAMRPatchExporter.h                            (C) 2000-2025 */
/*                                                                           */
/* Écrivain d'un maillage au format HTML, avec un SVG.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_SIMPLEHTMLMESHAMRPATCHEXPORTER_H
#define ARCANE_CARTESIANMESH_SIMPLEHTMLMESHAMRPATCHEXPORTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianPatch.h"

#include "arcane/core/ItemTypes.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exportation d'un maillage au format SVG.
 *
 * Cette classe ne fonctionne que pour les maillages 2D. Seules les composantes
 * `x` et `y` sont considérérées. Après avoir créée une instance, il est possible
 * d'appeler la méthode writeGroup() pour exporter les entités associées à groupe
 * de maille (noeuds, faces et mailles).
 */
class ARCANE_CARTESIANMESH_EXPORT SimpleHTMLMeshAMRPatchExporter
{
  class Impl;

 public:

  SimpleHTMLMeshAMRPatchExporter();
  SimpleHTMLMeshAMRPatchExporter(const SimpleHTMLMeshAMRPatchExporter& rhs) = delete;
  SimpleHTMLMeshAMRPatchExporter& operator=(const SimpleHTMLMeshAMRPatchExporter& rhs) = delete;
  ~SimpleHTMLMeshAMRPatchExporter();

  void addPatch(const CartesianPatch& patch);
  void write(std::ostream& ofile);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
