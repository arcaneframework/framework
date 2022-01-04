// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IProcessorAffinityService.h                                 (C) 2000-2018 */
/*                                                                           */
/* Interface d'un service de gestion de l'affinité des processeurs/coeurs.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IPROCESSORAFFINITYSERVICE_H
#define ARCANE_UTILS_IPROCESSORAFFINITYSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un service de de trace des appels de fonctions.
 */
class IProcessorAffinityService
{
 public:

  virtual ~IProcessorAffinityService() {} //<! Libère les ressources

 public:

  virtual void build() =0;

 public:

  virtual void printInfos() =0;

  virtual String cpuSetString() =0;

  //! Contraint le thread courant à rester sur le coeur d'indice \a cpu
  virtual void bindThread(Int32 cpu) =0;
  
  virtual Int32 numberOfCore() =0;
  virtual Int32 numberOfSocket() =0;
  virtual Int32 numberOfProcessingUnit() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

