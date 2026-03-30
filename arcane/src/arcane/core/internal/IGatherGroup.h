// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGatherGroup.h                                              (C) 2000-2026 */
/*                                                                           */
/* Interface pour gérer les regroupements sur un nombre restreint de         */
/* sous-domaines.                                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_IGATHERGROUP_H
#define ARCANE_CORE_INTERNAL_IGATHERGROUP_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface de classe permettant de regrouper les données de certains
 * sous-domaines sur d'autres sous-domaines.
 */
class ARCANE_CORE_EXPORT IGatherGroup
{

 public:

  virtual ~IGatherGroup() = default;

 public:

  /*!
   * \brief Méthode permettant de savoir si l'on doit effectuer le
   * regroupement ou si l'on peut directement écrire les données.
   *
   * Appel non collectif, mais la valeur retournée sera la même pour tous les
   * appelants.
   *
   * L'appel à gatherToMasterIO() peut tout de même être effectué, le tableau
   * \a in sera simplement copié dans le tableau \a out.
   */
  virtual bool isNeedGather() = 0;

  /*!
   * \brief Méthode permettant de regrouper les données de plusieurs
   * sous-domaines sur un ou plusieurs sous-domaines.
   *
   * Appel collectif.
   *
   * \param sizeof_elem La taille d'un élément.
   * \param in Notre tableau que l'on souhaite regrouper.
   * \param out Le tableau regroupé. Si l'on n'est pas écrivain, il n'y aura
   * aucune modification.
   */
  virtual void gatherToMasterIO(Int64 sizeof_elem, Span<const Byte> in, Span<Byte> out) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface de classe permettant de calculer et de conserver les
 * informations de regroupements.
 */
class ARCANE_CORE_EXPORT IGatherGroupInfo
{
 public:

  virtual ~IGatherGroupInfo() = default;

 public:

  /*!
   * \brief Méthode permettant de calculer les informations de regroupements.
   *
   * Appel collectif.
   *
   * Un second appel à cette méthode n'aura pas d'effet, sauf en cas d'appel à
   * la méthode \a needRecompute() avant.
   *
   * \param nb_elem_in Le nombre d'éléments que notre sous-domaine souhaite
   * envoyer au maitre.
   */
  virtual void computeSize(Int32 nb_elem_in) = 0;

  /*!
   * \brief Méthode permettant de demander un recalcul des informations de
   * regroupements. Pour cela, il faudra rappeler la méthode \a computeSize().
   */
  virtual void setNeedRecompute() = 0;

  /*!
   * \brief Méthode permettant de savoir si la méthode \a computeSize() a déjà
   * été appelée.
   */
  virtual bool isComputed() = 0;

  /*!
   * \brief Méthode permettant de connaitre le nombre d'éléments que notre
   * sous-domaine devra traiter après réception.
   */
  virtual Int32 nbElemOutput() = 0;

  /*!
   * \brief Méthode permettant de connaitre la taille, en octet, de l'ensemble
   * des éléments que notre sous-domaine devra traiter après réception.
   *
   * \param sizeof_type La taille d'un élément.
   */
  virtual Int32 sizeOfOutput(Int32 sizeof_type) = 0;

  /*!
   * \brief Méthode permettant de connaitre le nombre d'éléments que vont nous
   * envoyer chaque sous-domaine tier.
   */
  virtual SmallSpan<Int32> nbElemRecvGatherToMasterIO() = 0;

  /*!
   * \brief Méthode pemettant de connaitre le nombre de sous-domaines
   * écrivains.
   */
  virtual Int32 nbWriterGlobal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
