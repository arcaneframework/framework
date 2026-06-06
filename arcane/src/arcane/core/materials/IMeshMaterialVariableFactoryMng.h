// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableFactoryMng.h                           (C) 2000-2025 */
/*                                                                           */
/* Interface of the material variable factory manager.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLEFACTORYMNG_H
#define ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLEFACTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of the material variable factory manager.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariableFactoryMng
{
 public:

  virtual ~IMeshMaterialVariableFactoryMng() = default;

 public:

  //! Builds the instance
  virtual void build() = 0;

  //! Associated trace manager
  virtual ITraceMng* traceMng() const = 0;

  //! Registers the factory \a factory.
  virtual void registerFactory(Ref<IMeshMaterialVariableFactory> factory) = 0;

  //! Creates a material variable.
  virtual IMeshMaterialVariable*
  createVariable(const String& storage_type,
                 const MaterialVariableBuildInfo& build_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
