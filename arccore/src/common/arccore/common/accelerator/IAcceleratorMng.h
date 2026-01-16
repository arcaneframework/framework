// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAcceleratorMng.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des accélérateurs.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_IACCELERATORMNG_H
#define ARCCORE_COMMON_ACCELERATOR_IACCELERATORMNG_H
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
 * \brief Interface du gestionnaire des accélérateurs.
 *
 * Cette interface permet de récupérer une instance de Runner et RunQueue
 * associée à un contexte. Il faut appeler initialize() pour créer ces deux
 * instances qu'il est ensuite possible de récupérer via runner() ou queue().
 *
 * Il est nécessaire d'appeler initialize() avant de pouvoir accéder aux
 * méthodes telles que defaultRunner() ou defaultQueue().
 */
class ARCCORE_COMMON_EXPORT IAcceleratorMng
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
   * \note Cette méthode sera à terme obsolète.. Il est préférable d'utiliser
   * la méthode runner() à la place car elle est toujours valide.
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
   * \note Cette méthode sera à terme obsolète.. Il est préférable d'utiliser
   * la méthode queue() à la place car elle est toujours valide.
   *
   * * \pre isInitialized()==true
   */
  virtual RunQueue* defaultQueue() =0;

 public:

  /*!
   * \brief Exécuteur associé à l'instance.
   *
   * Si l'instance a été initialisée, retourne *defaultRunner().
   * Sinon, retourne une instance de Runner nulle.
   */
  virtual Runner runner() = 0;

  /*!
   * \brief File d'exécution associée à l'instance.
   *
   * Si l'instance a été initialisée, retourne *defaultQueue().
   * Sinon, retourne une file nulle.
   */
  virtual RunQueue queue() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
