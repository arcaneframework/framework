// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParticleExchanger.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface d'un échangeur de particules asynchrone.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IASYNCPARTICLEEXCHANGER_H
#define ARCANE_CORE_IASYNCPARTICLEEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un échangeur de particules asynchrone.
 */
class ARCANE_CORE_EXPORT IAsyncParticleExchanger
{
 public:

  virtual ~IAsyncParticleExchanger() = default;

 public:

  virtual bool exchangeItemsAsync(Integer nb_particle_finish_exchange,
                                  Int32ConstArrayView local_ids,
                                  Int32ConstArrayView sub_domains_to_send,
                                  Int32Array* new_particle_local_ids,
                                  IFunctor* functor,
                                  bool has_local_flying_particles) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
