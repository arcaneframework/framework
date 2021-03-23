// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableDependInfo.h                            (C) 2000-2014 */
/*                                                                           */
/* Informations sur une dépendance de variable.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_VARIABLEDEPENDINFO_H
#define ARCANE_MATERIALS_VARIABLEDEPENDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/utils/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur une dépendance de variable.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableDependInfo
{
 public:

  MeshMaterialVariableDependInfo(IMeshMaterialVariable* var,
                                 const TraceInfo& trace_info);

 public:

  //! Variable
  IMeshMaterialVariable* variable() const { return m_variable; }

  /*!
   * Infos (si disponible) sur l'endroit dans le code source où la dépendance
   * a été ajoutée.
   */
  const TraceInfo& traceInfo() const { return m_trace_info; }

 private:

  IMeshMaterialVariable* m_variable;
  TraceInfo m_trace_info;  
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
