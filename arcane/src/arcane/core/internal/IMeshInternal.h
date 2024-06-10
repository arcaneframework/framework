// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshInternal.h                                             (C) 2000-2024 */
/*                                                                           */
/* Partie interne à Arcane de IMesh.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IMESHINTERNAL_H
#define ARCANE_CORE_INTERNAL_IMESHINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de IMesh.
 */
class ARCANE_CORE_EXPORT IMeshInternal
{
 public:

  virtual ~IMeshInternal() = default;

 public:

  /*!
   * \brief Positionne le type de maillage.
   *
   * Pour l'instant il ne faut utiliser cette méthode que pour spécifier
   * la structure du maillage (eMeshStructure).
   */
  virtual void setMeshKind(const MeshKind& v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
