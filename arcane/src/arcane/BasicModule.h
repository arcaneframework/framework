// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicModule.h                                               (C) 2000-2006 */
/*                                                                           */
/* Module basique avec informations de maillage.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_BASICMODULE_H
#define ARCANE_BASICMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/AbstractModule.h"
#include "arcane/MeshAccessor.h"
#include "arcane/CommonVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module basique.
 *
 * \ingroup Module
 *
 * Classe de base d'un module permettant de récupérer aisément les informations
 * de maillages (IMesh) et les variables standards (CommonVariables).
 */
class ARCANE_CORE_EXPORT BasicModule
: public AbstractModule
, public MeshAccessor
, public CommonVariables
{
 protected:

  //! Constructeur à partir d'un \a ModuleBuildInfo
  BasicModule(const ModuleBuildInfo&);

 public:
	
  //! Destructeur
  virtual ~BasicModule();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

