// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AsyncParticleExchanger.h                                    (C) 2019-2020 */
/* Author : Hugo Taboada                                                     */
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/BasicParticleExchanger.h"
#include "arcane/mesh/BasicParticleExchanger.h"

#include "arcane/IAsyncParticleExchanger.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

class AsyncParticleExchanger
: public BasicService
, public IParticleExchanger
, public IAsyncParticleExchanger
{

  //Constructors and destructors
 public:
  explicit AsyncParticleExchanger(const ServiceBuildInfo& sbi);
  ~AsyncParticleExchanger() override;

  //IParticleExchanger.h interface
 public:
  void build() override;
  void initialize(IItemFamily* item_family) override;
  void beginNewExchange(Integer nb_particule) override;
  bool exchangeItems(Integer nb_particle_finish_exchange,
                     Int32ConstArrayView local_ids,
                     Int32ConstArrayView sub_domains_to_send,
                     ItemGroup item_group,
                     IFunctor* functor) override;
  bool exchangeItems(Integer nb_particle_finish_exchange,
                     Int32ConstArrayView local_ids,
                     Int32ConstArrayView sub_domains_to_send,
                     Int32Array* new_particle_local_ids,
                     IFunctor* functor) override;
  void sendItems(Integer nb_particle_finish_exchange,
                 Int32ConstArrayView local_ids,
                 Int32ConstArrayView sub_domains_to_send) override;
  bool waitMessages(Integer nb_pending_particles,
                    Int32Array* new_particle_local_ids,
                    IFunctor* functor) override;
  void addNewParticles(Integer nb_particle) override;
  IItemFamily* itemFamily() override;
  void setVerboseLevel(Integer level) override;
  Integer verboseLevel() const override;
  IAsyncParticleExchanger* asyncParticleExchanger() override;

  //IAsyncParticleExchanger.h interface
 public:
  bool exchangeItemsAsync(
  Integer nb_particle_finish_exchange,
  Int32ConstArrayView local_ids,
  Int32ConstArrayView sub_domains_to_send,
  Int32Array* new_particle_local_ids,
  IFunctor* functor,
  bool has_local_flying_particles) override;

  //Private member variables including bpe that is a composition with BasicParticleExchanger class.
 private:
  BasicParticleExchanger m_bpe;
  Integer m_nb_particle_send_before_reduction;
  Integer m_nb_particle_send_before_reduction_tmp;
  Integer m_sum_of_nb_particle_sent; //Somme des particules envoyé calculé dans la condition d'arrêt du mode asynchrone
  UniqueArray<Parallel::Request> m_reduce_requests; //Ce tableau doit avoir qu'une seule requête à la fois

  //Private functions used by internal implementation of AsyncParticleExchanger
 private:
  void _generateSendItemsAsync(Int32ConstArrayView local_ids, Int32ConstArrayView sub_domains_to_send);
  //bool _waitSomeMessages(Integer nb_pending_particles, ItemGroup item_group, Int32Array* new_particle_local_ids);
  bool _waitSomeMessages(ItemGroup item_group, Int32Array* new_particle_local_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh
