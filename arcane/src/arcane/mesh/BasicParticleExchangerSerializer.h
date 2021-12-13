// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicParticleExchangerSerializer.h                          (C) 2000-2021 */
/*                                                                           */
/* Serialisation de l'échangeur de particules basique.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_BASICPARTICLEEXCHANGERSERIALIZER_H
#define ARCANE_BASICPARTICLEEXCHANGERSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/ItemTypes.h"
#include "arcane/Parallel.h"
#include "arcane/VariableList.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Echangeur de particules basique (utilise une réduction bloquante).
 *
 * Avant de sérialiser/désérialiser des messages, il faut appeler la méthode
 * beginNewExchange() pour mettre à jour les informations nécessaires à la
 * sérialisation comme la liste des variables.
 */
class ARCANE_MESH_EXPORT BasicParticleExchangerSerializer
: public TraceAccessor
{
 public:
  
  BasicParticleExchangerSerializer(IItemFamily* family,Int32 my_rank);
  ~BasicParticleExchangerSerializer();

 public:

  void beginNewExchange();

  void setVerboseLevel(Integer level) { m_verbose_level = level; }
  Integer verboseLevel() const { return m_verbose_level; }

  void serializeMessage(ISerializeMessage* sm,
                        Int32ConstArrayView acc_ids,
                        Int64Array& items_to_send_uid,
                        Int64Array& items_to_send_cells_uid);
  void deserializeMessage(ISerializeMessage* message,
                          Int64Array& items_to_create_unique_id,
                          Int64Array& items_to_create_cells_unique_id,
                          Int32Array& items_to_create_local_id,
                          Int32Array& items_to_create_cells_local_id,
                          ItemGroup item_group,
                          Int32Array* new_particle_local_ids);

 private:

  IItemFamily* m_item_family = nullptr;

  Int32 m_my_rank = A_NULL_RANK;

  Int32 m_verbose_level = 1;
  Int32 m_debug_exchange_items_level = 0;

  //! Numéro du message. Utile pour le débug
  Int64 m_serialize_id = 1;

  //! Liste des variables à échanger
  VariableList m_variables_to_exchange;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

