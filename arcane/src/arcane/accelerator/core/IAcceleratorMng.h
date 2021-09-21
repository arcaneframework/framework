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
 * Avant l'appel à initialize(), seule la méthode defaultRunner() peut être
 * appelée.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IAcceleratorMng
{
 public:

  virtual ~IAcceleratorMng() = default;

 public:

  //! Initialize l'instance
  virtual void initialize() =0;

  /*!
   * \brief Exécuteur par défaut.
   *
   * Le pointeur retourné reste la propriété de cette instance.
   */
  virtual Runner* defaultRunner() =0;

  /*!
   * \brief File d'exécution par défaut.
   *
   * L'instance retournée est nulle si initialize() n'a pas encore été appelé.
   */
  virtual RunQueue* defaultQueue() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
