// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshSubdivider.h                                           (C) 2000-2024 */
/*                                                                           */
/* Interface d'un service de subdivision d'un maillage.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHSUBDIVIDER_H
#define ARCANE_CORE_IMESHSUBDIVIDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief IMeshSubdivider
 * \warning Experimental. Do not use outside of Arcane
 */
class ARCANE_CORE_EXPORT IMeshSubdivider
{
 public:

  virtual ~IMeshSubdivider() = default; //<! Libère les ressources

 public:

  //! Subdivise le maillage \a mesh
  virtual void subdivideMesh(IPrimaryMesh* mesh) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
