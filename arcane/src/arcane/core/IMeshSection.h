// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshSection.h                                      (C) 2000-2026 */
/*                                                                           */
/* TODO.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHSECTION_H
#define ARCANE_CORE_IMESHSECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableCollection;
class MeshHandle;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT IMeshSection
{
 public:

  //! Releases resources
  virtual ~IMeshSection() = default;

 public:

  virtual void addPlan(const Real3& p0, const Real3& normal) = 0;

  virtual void setVariables(VariableCollection variables) = 0;
  virtual VariableCollection variables() = 0;

  virtual void updateSection() = 0;
  virtual MeshHandle meshSection() = 0;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
