// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorRuntimeInitialisationInfo.h                      (C) 2000-2021 */
/*                                                                           */
/* Informations pour l'initialisation du runtime des accélérateurs.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_ACCELERATORRUNTIMEINITIALISATIONINFO_H
#define ARCANE_ACCELERATOR_CORE_ACCELERATORRUNTIMEINITIALISATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/PropertyDeclarations.h"

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour l'initialisation des accélérateurs.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT AcceleratorRuntimeInitialisationInfo
{
  ARCANE_DECLARE_PROPERTY_CLASS(AcceleratorRuntimeInitialisationInfo);
 private:
  class Impl;
 public:

  AcceleratorRuntimeInitialisationInfo();
  AcceleratorRuntimeInitialisationInfo(const AcceleratorRuntimeInitialisationInfo& rhs);
  ~AcceleratorRuntimeInitialisationInfo();
  AcceleratorRuntimeInitialisationInfo& operator=(const AcceleratorRuntimeInitialisationInfo& rhs);

 public:

  void setIsUsingAcceleratorRuntime(bool v);
  bool isUsingAcceleratorRuntime() const;

  //! Nom du runtime utilisé (pour l'instant uniquement 'cuda')
  void setAcceleratorRuntime(StringView name);
  String acceleratorRuntime() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void
arcaneInitializeRunner(Accelerator::Runner& runner,ITraceMng* tm,
                       const AcceleratorRuntimeInitialisationInfo& acc_info);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

