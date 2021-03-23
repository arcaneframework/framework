// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorRuntimeInitialisationInfo.h                      (C) 2000-2020 */
/*                                                                           */
/* Informations pour l'initialisation du runtime des accélérateurs.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ACCELERATORRUNTIMEINITIALISATIONINFO_H
#define ARCANE_UTILS_ACCELERATORRUNTIMEINITIALISATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/PropertyDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour l'initialisation des accélérateurs.
 */
class ARCANE_CORE_EXPORT AcceleratorRuntimeInitialisationInfo
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

