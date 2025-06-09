// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshStats.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interface d'une classe donnant des informations sur le maillage.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHSTATS_H
#define ARCANE_CORE_IMESHSTATS_H
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
 * \brief Interface d'une classe donnant des informations sur le maillage.
 */
class ARCANE_CORE_EXPORT IMeshStats
{
 public:

  //! Libère les ressources
  virtual ~IMeshStats() = default;

 public:

  //! Création d'une instance par défaut
  static IMeshStats* create(ITraceMng* trace, IMesh* mesh, IParallelMng* pm);

 public:

  //! Imprime des infos sur le maillage
  virtual void dumpStats() = 0;

  //! Imprime des infos sur le graphe du maillage
  virtual void dumpGraphStats() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

