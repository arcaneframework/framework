// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableDependInfo.h                                        (C) 2000-2025 */
/*                                                                           */
/* Informations sur une dépendance de variable.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEDEPENDINFO_H
#define ARCANE_CORE_VARIABLEDEPENDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceInfo.h"

#include "arcane/core/IVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une dépendance de variable.
 */
class ARCANE_CORE_EXPORT VariableDependInfo
{
 public:

  VariableDependInfo(IVariable* var, IVariable::eDependType depend_type,
                     const TraceInfo& trace_info);

 public:

  //! Variable
  IVariable* variable() const { return m_variable; }

  //! Type de dépendance.
  IVariable::eDependType dependType() const { return m_depend_type; }

  /*!
   * Infos (si disponible) sur l'endroit dans le code source où la dépendance
   * a été ajoutée.
   */
  const TraceInfo& traceInfo() const { return m_trace_info; }

 private:

  IVariable* m_variable = nullptr;
  IVariable::eDependType m_depend_type = IVariable::DPT_CurrentTime;
  TraceInfo m_trace_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
