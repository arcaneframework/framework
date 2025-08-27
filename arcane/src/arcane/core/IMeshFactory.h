// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshFactory.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service de de fabrique de maillage.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHFACTORY_H
#define ARCANE_CORE_IMESHFACTORY_H
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
 * \internal
 * \brief Interface du service gérant la lecture d'un maillage.
 */
class ARCANE_CORE_EXPORT IMeshFactory
{
 public:

  virtual ~IMeshFactory() {} //<! Libère les ressources

 public:

  //! Créé un maillage avec les informations de \a build_info
  virtual IPrimaryMesh* createMesh(IMeshMng* mm,const MeshBuildInfo& build_info) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

