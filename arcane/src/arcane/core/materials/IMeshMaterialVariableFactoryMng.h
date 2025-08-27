// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableFactoryMng.h                           (C) 2000-2022 */
/*                                                                           */
/* Interface du gestionnaire de fabrique de variables matériaux.             */
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
 * \brief Interface du gestionnaire de fabrique de variables matériaux.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariableFactoryMng
{
 public:
  
  virtual ~IMeshMaterialVariableFactoryMng() = default;

 public:

  //! Construit l'instance
  virtual void build() =0;

  //! Gestionnaire de trace associé
  virtual ITraceMng* traceMng() const =0;

  //! Enregistre la fabrique \a factory.
  virtual void registerFactory(Ref<IMeshMaterialVariableFactory> factory) =0;

  //! Créé une variable matériau.
  virtual IMeshMaterialVariable*
  createVariable(const String& storage_type,
                 const MaterialVariableBuildInfo& build_info) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
