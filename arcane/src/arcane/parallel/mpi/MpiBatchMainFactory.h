// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiBatchMainFactory.h                                       (C) 2000-2016 */
/*                                                                           */
/* Fabrique principale pour 'MPI'.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIBATCHMAINFACTORY_H
#define ARCANE_PARALLEL_MPI_MPIBATCHMAINFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/impl/MainFactory.h"
#include "arcane/impl/ArcaneMain.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MPI_EXPORT MpiBatchMainFactory
: public MainFactory
{
 public:

  virtual IArcaneMain* createArcaneMain(const ApplicationInfo& app_info);
	/*!
	 * \brief Exécute l'application spécifiée par \a app_info en utilisant
	 * MPI comme gestionnaire de parallélisme.
	 */
	static int exec(const ApplicationInfo& app_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
