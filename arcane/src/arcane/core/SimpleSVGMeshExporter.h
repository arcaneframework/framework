// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleSVGMeshExporter.h                                     (C) 2000-2025 */
/*                                                                           */
/* Écrivain d'un maillage au format SVG.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SIMPLESVGMESHEXPORTER_H
#define ARCANE_CORE_SIMPLESVGMESHEXPORTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

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
class ARCANE_CORE_EXPORT SimpleSVGMeshExporter
{
  class Impl;
 public:
 /*!
  * \brief Créé une instance associée au flux \a ofile.
  *
  * Le flux \a ofile doit rester valide durant toute la durée de vie de l'instance.
  */
  SimpleSVGMeshExporter(std::ostream& ofile);
  SimpleSVGMeshExporter(const SimpleSVGMeshExporter& rhs) = delete;
  SimpleSVGMeshExporter& operator=(const SimpleSVGMeshExporter& rhs) = delete;
  ~SimpleSVGMeshExporter();
  /*!
   * \brief Exporte les entités du groupe \a cells.
   *
   * Actuellement, il n'est pas possible d'appeler plusieurs fois cette méthode.
   */
  void write(const CellGroup& cells);
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
