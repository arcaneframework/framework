// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicModule.h                                               (C) 2000-2025 */
/*                                                                           */
/* Basic module with mesh information.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_BASICMODULE_H
#define ARCANE_CORE_BASICMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/AbstractModule.h"
#include "arcane/core/MeshAccessor.h"
#include "arcane/core/CommonVariables.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Module
 *
 * \brief Basic module.
 *
 * Base class of a module allowing easy retrieval of
 * mesh information (IMesh) and standard variables (CommonVariables).
 */
class ARCANE_CORE_EXPORT BasicModule
: public AbstractModule
, public MeshAccessor
, public CommonVariables
{
 protected:

  //! Constructor from a \a ModuleBuildInfo
  explicit BasicModule(const ModuleBuildInfo&);

 public:

  //! Destructor
  ~BasicModule() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
