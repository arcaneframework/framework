// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAccelerator.h                                              (C) 2000-2021 */
/*                                                                           */
/* Interface du gestionnaire des accélérateurs.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_IACCELERATORMNG_H
#define ARCANE_ACCELERATOR_CORE_IACCELERATORMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des accélérateurs.
 *
 * Il est nécessaire d'appeler initialize() avant de pouvoir accéder aux
 * méthodes telles que defaultRunner() ou defaultQueue().
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IAcceleratorMng
{
 public:

  virtual ~IAcceleratorMng() = default;

 public:

  /*!
   * \brief Initialise l'instance.
   *
   * \pre isInitialized()==false
   */
  virtual void initialize(const AcceleratorRuntimeInitialisationInfo& runtime_info) =0;

  //! Indique si l'instance a été initialisée via l'appel à initialize()
  virtual bool isInitialized() const =0;

  /*!
   * \brief Exécuteur par défaut.
   *
   * Le pointeur retourné reste la propriété de cette instance.
   *
   * \pre isInitialized()==true
   */
  virtual Runner* defaultRunner() =0;

  /*!
   * \brief File d'exécution par défaut.
   *
   * Le pointeur retourné reste la propriété de cette instance.
   *
   * \pre isInitialized()==true
   */
  virtual RunQueue* defaultQueue() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
