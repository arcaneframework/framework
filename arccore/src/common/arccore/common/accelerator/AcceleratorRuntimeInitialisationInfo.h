// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorRuntimeInitialisationInfo.h                      (C) 2000-2025 */
/*                                                                           */
/* Informations pour l'initialisation du runtime des accélérateurs.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_ACCELERATORRUNTIMEINITIALISATIONINFO_H
#define ARCCORE_COMMON_ACCELERATOR_ACCELERATORRUNTIMEINITIALISATIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour l'initialisation des accélérateurs.
 */
class ARCCORE_COMMON_EXPORT AcceleratorRuntimeInitialisationInfo
{
 private:

  class Impl;

 public:

  AcceleratorRuntimeInitialisationInfo();
  AcceleratorRuntimeInitialisationInfo(const AcceleratorRuntimeInitialisationInfo& rhs);
  ~AcceleratorRuntimeInitialisationInfo();
  AcceleratorRuntimeInitialisationInfo& operator=(const AcceleratorRuntimeInitialisationInfo& rhs);

 public:

  //! Indique si on utilise un runtime accélérateur
  void setIsUsingAcceleratorRuntime(bool v);
  bool isUsingAcceleratorRuntime() const;

  //! Nom du runtime utilisé (pour l'instant uniquement 'cuda', 'hip' ou 'sycl')
  void setAcceleratorRuntime(StringView name);
  String acceleratorRuntime() const;

  //! Positionne le device associé au Runner associé.
  void setDeviceId(DeviceId name);
  //! Device associé au Runner associé
  DeviceId deviceId() const;

  //! Politique d'exécution associée à acceleratorRuntime()
  eExecutionPolicy executionPolicy() const;

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

