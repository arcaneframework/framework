// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParticleExchanger.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface d'un échangeur de particules.                                   */
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
 * \brief Interface d'un échangeur de particules.
 *
 * Cette classe sert à échanger des particules entre sous-domaines en
 * plusieurs étapes et est en général utilisée par des codes
 * de trajectographie. Si l'on souhaite faire un échange en une fois,
 * il faut utiliser IItemFamily::exchangeItems().
 *
 * Avant tout, une instance doit être initialisée avec une famille
 * de particules via initialize().
 *
 * Pour procéder à un échange, il faut d'abord initialiser
 * l'échange via beginNewExchange() avec comme argument le nombre
 * de particules du sous-domaine concernées par l'échange.
 * Il faut ensuite appeler exchangeItems() pour chaque phase
 * d'échange de particules. A chaque appel à exchangeItems(), il
 * faut spécifier le nombre de particules ne participant plus
 * à l'échange. Lorsqu'il n'y a plus de particules concernée,
 * exchangeItems() return true et l'échange est terminé.
 * Il est possible de recommencer tout le processus via
 * un nouvel appel à beginNewExchange().
 *
 * Entre deux phases d'un échange, il est possible d'indiquer
 * que de nouvelles particules vont participer via
 * addNewParticles()
 * 
 * 
 */
class ARCANE_CORE_EXPORT IParticleExchanger
{
 public:

  virtual ~IParticleExchanger() = default; //!< Libère les ressources

 public:

  virtual void build() = 0;

  //! Initialize l'échangeur pour la famille \a item_family.
  virtual void initialize(IItemFamily* item_family) = 0;

 public:

  /*!
   * \brief Commence un nouvel échange de particules.
   *
   * \a nb_particule est le nombre de particules du sous-domaine qui vont
   * prendre part à un éventuel échange.
   *
   * Cette méthode est collective et doit être appelée par tout les sous-domaines.
   */
  virtual void beginNewExchange(Integer nb_particle) = 0;

  /*!
   * \brief Échange des particules entre sous-domaines.
   *
   * Cette opération envoie les particules de la famille \a item_family dont les
   * indices locaux sont donnés par la liste \a local_ids aux sous-domaines
   * specifiés par \a sub_domains_to_send, et réceptionne de ces mêmes sous-domaines celles
   * dont ce sous-domaine est propriétaire. Les particules envoyées sont supprimées
   * de la famille \a item_family et celles recues ajoutées.
   *
   * Les variables reposant sur la famille \a item_family sont transférées
   * en même temps que les particules.
   *
   * Cette opération est collective et bloquante.
   *
   * Si \a item_group n'est pas nul, il contiendra en retour la liste des
   * nouvelles entités.
   *
   * Si \a wait_functor n'est pas nul, le functor est appelé pendant l'envoie
   * et la réception des messages. Il est alors possible de faire des opérations.
   * Les opérations ne doivent pas utiliser de particules, ni des variables sur
   * les particules de la famille échangée.
   *
   * \retval \a true si toutes les phases d'échange sont terminés
   * \retval \a false sinon
   *
   * \todo améliorer la doc
   */
  virtual ARCANE_DEPRECATED bool exchangeItems(Integer nb_particle_finish_exchange,
                                               Int32ConstArrayView local_ids,
                                               Int32ConstArrayView sub_domains_to_send,
                                               ItemGroup item_group,
                                               IFunctor* wait_functor) = 0;

  /*!
   * \brief Échange des particules entre sous-domaines.
   *
   * Cette opération envoie les particules de la famille \a item_family dont les
   * indices locaux sont donnés par la liste \a local_ids aux sous-domaines
   * specifiés par \a sub_domains_to_send, et réceptionne de ces mêmes sous-domaines celles
   * dont ce sous-domaine est propriétaire. Les particules envoyées sont supprimées
   * de la famille \a item_family et celles recues ajoutées.
   *
   * Les variables reposant sur la famille \a item_family sont transférées
   * en même temps que les particules.
   *
   * Cette opération est collective et bloquante.
   *
   * Si \a new_particle_local_ids n'est pas nul, il contiendra en retour
   * le tableau des indices locaux des nouvelles entités.
   *
   * Si \a wait_functor n'est pas nul, le functor est appelé pendant l'envoie
   * et la réception des messages. Il est alors possible de faire des opérations.
   * Les opérations ne doivent pas utiliser de particules, ni des variables sur
   * les particules de la famille échangée.
   *
   * \retval \a true si toutes les phases d'échange sont terminés
   * \retval \a false sinon
   *
   * \todo améliorer la doc
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
   * \brief Ajoute \a nb_particle dans l'échange actuel.
   *
   * Cette méthode permet d'indiquer que de nouvelles particules
   * vont participer à l'échanger, par exemple suite à leur création.
   */
  virtual void addNewParticles(Integer nb_particle) = 0;

  //! Famille associée.
  virtual IItemFamily* itemFamily() = 0;

  //! Positionne le niveau de verbosité (0 pour aucune message)
  virtual void setVerboseLevel(Integer level) = 0;

  //! Niveau de verbosité
  virtual Integer verboseLevel() const = 0;

  //! Gestion de l'asynchronisme (retourne nullptr si fonctionnalité non disponible)
  virtual IAsyncParticleExchanger* asyncParticleExchanger() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
