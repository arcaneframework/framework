// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParticleExchanger.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface of a particle exchanger.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARTICLEEXCHANGER_H
#define ARCANE_CORE_IPARTICLEEXCHANGER_H
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
 * \brief Interface of a particle exchanger.
 *
 * This class is used to exchange particles between sub-domains in
 * several steps and is generally used by trajectory codes. If one wishes to
 * perform a single exchange,
 * IItemFamily::exchangeItems() must be used.
 *
 * First of all, an instance must be initialized with a particle family
 * via initialize().
 *
 * To proceed with an exchange, you must first initialize
 * the exchange via beginNewExchange() using the number
 * of particles in the sub-domain involved in the exchange as an argument.
 * You must then call exchangeItems() for each phase
 * of particle exchange. For each call to exchangeItems(), you
 * must specify the number of particles that no longer participate
 * in the exchange. When there are no more involved particles,
 * exchangeItems() returns true and the exchange is finished.
 * It is possible to restart the entire process via
 * a new call to beginNewExchange().
 *
 * Between two phases of an exchange, it is possible to indicate
 * that new particles will participate via
 * addNewParticles()
 */
class ARCANE_CORE_EXPORT IParticleExchanger
{
 public:

  virtual ~IParticleExchanger() = default; //!< Releases resources

 public:

  virtual void build() = 0;

  //! Initializes the exchanger for the item_family \a item_family.
  virtual void initialize(IItemFamily* item_family) = 0;

 public:

  /*!
   * \brief Starts a new particle exchange.
   *
   * \a nb_particle is the number of particles in the sub-domain that will
   * take part in a potential exchange.
   *
   * This method is collective and must be called by all sub-domains.
   */
  virtual void beginNewExchange(Integer nb_particle) = 0;

  /*!
   * \brief Exchanges particles between sub-domains.
   *
   * This operation sends the particles from the item_family \a item_family whose
   * local indices are given by the list \a local_ids to the sub-domains
   * specified by \a sub_domains_to_send, and receives from these same sub-domains those
   * that this sub-domain owns. The sent particles are deleted
   * from the item_family \a item_family and the received ones are added.
   *
   * Variables associated with the item_family \a item_family are transferred
   * at the same time as the particles.
   *
   * This operation is collective and blocking.
   *
   * If \a item_group is not null, it will contain the list of
   * new entities in return.
   *
   * If \a wait_functor is not null, the functor is called during the sending
   * and receiving of messages. It is then possible to perform operations.
   * Operations must not use particles, nor variables on
   * the particles of the exchanged family.
   *
   * \retval \a true if all exchange phases are finished
   * \retval \a false otherwise
   *
   * \todo improve the documentation
   */
  virtual ARCANE_DEPRECATED bool exchangeItems(Integer nb_particle_finish_exchange,
                                               Int32ConstArrayView local_ids,
                                               Int32ConstArrayView sub_domains_to_send,
                                               ItemGroup item_group,
                                               IFunctor* wait_functor) = 0;

  /*!
   * \brief Exchanges particles between sub-domains.
   *
   * This operation sends the particles from the item_family \a item_family whose
   * local indices are given by the list \a local_ids to the sub-domains
   * specified by \a sub_domains_to_send, and receives from these same sub-domains those
   * that this sub-domain owns. The sent particles are deleted
   * from the item_family \a item_family and the received ones are added.
   *
   * Variables associated with the item_family \a item_family are transferred
   * at the same time as the particles.
   *
   * This operation is collective and blocking.
   *
   * If \a new_particle_local_ids is not null, it will contain in return
   * the array of local indices of the new entities.
   *
   * If \a wait_functor is not null, the functor is called during the sending
   * and receiving of messages. It is then possible to perform operations.
   * Operations must not use particles, nor variables on
   * the particles of the exchanged family.
   *
   * \retval \a true if all exchange phases are finished
   * \retval \a false otherwise
   *
   * \todo improve the documentation
   */
  virtual bool exchangeItems(Integer nb_particle_finish_exchange,
                             Int32ConstArrayView local_ids,
                             Int32ConstArrayView ranks_to_send,
                             Int32Array* new_particle_local_ids,
                             IFunctor* wait_functor) = 0;

  //! \internal
  virtual void sendItems(Integer nb_particle_finish_exchange,
                         Int32ConstArrayView local_ids,
                         Int32ConstArrayView sub_domains_to_send) = 0;

  //! \internal
  virtual bool waitMessages(Integer nb_pending_particle,
                            Int32Array* new_particle_local_ids,
                            IFunctor* functor) = 0;

  /*!
   * \brief Adds \a nb_particle to the current exchange.
   *
   * This method allows indicating that new particles
   * will participate in the exchange, for example following their creation.
   */
  virtual void addNewParticles(Integer nb_particle) = 0;

  //! Associated family.
  virtual IItemFamily* itemFamily() = 0;

  //! Sets the verbosity level (0 for no messages)
  virtual void setVerboseLevel(Integer level) = 0;

  //! Verbosity level
  virtual Integer verboseLevel() const = 0;

  //! Asynchronism management (returns nullptr if functionality is not available)
  virtual IAsyncParticleExchanger* asyncParticleExchanger() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
