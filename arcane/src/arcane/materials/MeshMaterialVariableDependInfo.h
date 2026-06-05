// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableDependInfo.h                            (C) 2000-2014 */
/*                                                                           */
/* Information about a variable dependency.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_VARIABLEDEPENDINFO_H
#define ARCANE_MATERIALS_VARIABLEDEPENDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/utils/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about a variable dependency.
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
   * Info (if available) about the location in the source code where the dependency
   * was added.
   */
  const TraceInfo& traceInfo() const { return m_trace_info; }

 private:

  IMeshMaterialVariable* m_variable;
  TraceInfo m_trace_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
