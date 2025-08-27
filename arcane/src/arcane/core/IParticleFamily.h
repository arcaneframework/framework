// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParticleFamily.h                                           (C) 2000-2022 */
/*                                                                           */
/* Interface d'une famille de particules.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPARTICLEFAMILY_H
#define ARCANE_IPARTICLEFAMILY_H
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
 * \ingroup Mesh
 * \brief Interface d'une famille de particules.
 *
 * Une famille de particle est une famille d'entité (IItemFamily).
 * Cette interface ne contient que les méthodes spécifiques aux particules.
 * Pour les opérations génériques aux entités, il faut passer par
 * l'interface IItemFamily via itemFamily().
 *
 * Il peut y a voir plusieurs famille de particule par maillage.
 * Contrairement aux entités classiques du maillage (noeud, maille, ...),
 * les particules peuvent être créées directement.
 *
 */
class ARCANE_CORE_EXPORT IParticleFamily
{
 public:

  virtual ~IParticleFamily() {} //<! Libère les ressources

 public:

  virtual void build() = 0;

  //! set le flag pour gérer les particules ghost de la famille
  virtual void setEnableGhostItems(bool value) = 0;

  //! récupère le flag pour gérer les particules ghost de la famille
  virtual bool getEnableGhostItems() const = 0;

 public:

  //! Nom de la famille
  virtual String name() const = 0;

  //! Nom complet de la famille (avec celui du maillage)
  virtual String fullName() const = 0;

  //! Nombre d'entités
  virtual Integer nbItem() const = 0;

  //! Groupe de toutes les particules
  virtual ItemGroup allItems() const = 0;

 public:

  /*!
   * \brief Alloue des particules
   *
   * Alloue les particules dont les uniqueId() sont données par le
   * tablea \a unique_ids.
   *
   * Après appel à cette opération, il faut appeler endUpdate() pour notifier
   * à l'instance la fin des modifications. Il est possible d'enchaîner plusieurs
   * allocations avant d'appeler endUpdate(). Attention, la vue retournée 
   * peut être invalidée après l'appel à endUpdate() si la compression est active.
   * \a items_local_id doit avoir le même nombre d'éléments que unique_ids.
   */
  virtual ParticleVectorView addParticles(Int64ConstArrayView unique_ids,
                                          Int32ArrayView items_local_id) = 0;
  virtual ParticleVectorView addParticles2(Int64ConstArrayView unique_ids,
                                           Int32ConstArrayView owners,
                                           Int32ArrayView items) = 0;

  /*!
   * \brief Alloue des particules dans des mailles.
   *
   * Cette méthode est similaire à addParticles() mais permet de spécifier
   * directement les mailles dans lesquelles seront créées les particules.
   */
  virtual ParticleVectorView addParticles(Int64ConstArrayView unique_ids,
                                          Int32ConstArrayView cells_local_id,
                                          Int32ArrayView items_local_id) = 0;

  virtual void removeParticles(Int32ConstArrayView items_local_id) = 0;

  /*!
   * \sa IItemFamily::endUpdate().
   */
  virtual void endUpdate() = 0;

  /*!
   * \brief Déplace la particule \a particle dans la maille \a new_cell.
   */
  virtual void setParticleCell(Particle particle, Cell new_cell) = 0;

  /*!
   * \brief Déplace la list de particules \a particles dans les nouvelles mailles \a new_cells.
   */
  virtual void setParticlesCell(ParticleVectorView particles, CellVectorView new_cells) = 0;

 public:

  /*!
   * \brief Échange des entités.
   *
   * Cette méthode n'est supportée que pour les familles de particule.
   * Pour les éléments du maillage comme les noeuds, faces ou maille, il faut utiliser IMesh::exchangeItems().
   *
   * Les nouveaux propriétaires des entités sont données par la itemsNewOwner().
   *
   * Cette opération est bloquante et collective.
   */
  virtual void exchangeParticles() = 0;

 public:

  //! Interface sur la famille
  virtual IItemFamily* itemFamily() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
