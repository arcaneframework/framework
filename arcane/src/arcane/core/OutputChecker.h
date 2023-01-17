﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* OutputChecker.h                                             (C) 2000-2010 */
/*                                                                           */
/* Sorties basées sur un temps (physique ou CPU) ou un nombre d'itération.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_OUTPUTCHECKER_H
#define ARCANE_OUTPUTCHECKER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/VariableTypes.h"
#include "arcane/CaseOptions.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère les sorties basées sur un temps physique, temps CPU ou
 * un nombre d'itération.
 *
 Le temps CPU est exprimé en minutes.
 */
class ARCANE_CORE_EXPORT OutputChecker
{
 public:

  //! Type de sortie
  enum eOutType
  {
    OutTypeNone,       //!< Pas de sorties
    OutTypeGlobalTime, //!< Sortie basée sur le temps physique
    OutTypeCPUTime,    //!< Sortie basée sur le temps CPU consommé
    OutTypeIteration   //!< Sortie basée sur le nombre d'itérations
  };

 public:

  OutputChecker(ISubDomain* sd,const String& name);

 public:

  void initialize();
  void initialize(bool recompute_next_value);
  bool hasOutput() const { return m_out_type!=OutTypeNone; }
  bool check(Real old_time,Real current_time,Integer current_iteration,
             Integer current_cpu_time,const String& from_function=String());
  void assignGlobalTime(VariableScalarReal* variable,const CaseOptionReal* option);
  void assignCPUTime(VariableScalarInteger* variable,const CaseOptionInteger* option);
  void assignIteration(VariableScalarInteger* variable,const CaseOptionInteger* option);
  Real nextGlobalTime() const;
  Integer nextIteration() const;
  Integer nextCPUTime() const;

 private:

  ISubDomain* m_sub_domain;
  String m_name;
  eOutType m_out_type;
  VariableScalarInteger* m_next_iteration; //!< Itération de la prochaine sauvegarde
  VariableScalarReal* m_next_global_time;  //!< Temps physique de la prochaine sauvegarde
  VariableScalarInteger* m_next_cpu_time; //!< Temps CPU de la prochaine sauvegarde
  const CaseOptionInteger* m_step_iteration;
  const CaseOptionReal* m_step_global_time;
  const CaseOptionInteger* m_step_cpu_time;

 private:

  void _recomputeTypeGlobalTime();
  void _recomputeTypeCPUTime();
  void _recomputeTypeIteration();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

