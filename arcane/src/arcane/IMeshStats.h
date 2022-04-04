// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshStats.h                                                (C) 2000-2008 */
/*                                                                           */
/* Interface d'une classe donnant des informations sur le maillage.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHSTATS_H
#define ARCANE_IMESHSTATS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IParallelMng;

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
  virtual ~IMeshStats() {}

 public:
  
  //! Création d'une instance par défaut
  static IMeshStats* create(ITraceMng* trace,IMesh* mesh,IParallelMng* pm);

 public:

  //! Imprime des infos sur le maillage
  virtual void dumpStats() =0;
  
  //! Imprime des infos sur le graphe du maillage
  virtual void dumpGraphStats() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

