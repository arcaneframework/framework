// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAllocator.h                                                (C) 2000-2006 */
/*                                                                           */
/* Interface d'un allocateur typé.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IALLOCATOR_H
#define ARCANE_UTILS_IALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un allocateur typé
 */
template<typename T>
class IAllocatorT
{
 protected:

  IAllocatorT() {}
  /*! \brief Détruit l'allocateur.
   * 
   * Les objets alloués par l'allocateur doivent tous avoir été désalloués.
   *
   * Ce destructeur est protégé. Pour détruire l'instance, utiliser destroy().
   */
  virtual ~IAllocatorT() {}

 public:
  
  //! Détruit l'instance
  virtual void destroy() =0;

 public:
  
  /*! \brief Alloue de la mémoire pour \a new_capacity objets.
   *
   * En cas de succès, retourne le pointeur sur le premier élément alloué.
   * En cas d'échec, une exception est levée (std::bad_alloc).
   * La valeur retournée n'est jamais nul.
   * \a new_capacity doit être strictement positif.
   */
  virtual T* allocate(Integer new_capacity) = 0;

  /*! \brief Libère la mémoire.
   *
   * Libère la mémoire dont le premier élément est donnée par \a ptr.
   * \a capacity indique le nombre d'éléments de la zone mémoire. Il doit
   * être égal à la valeur donnée lors de l'appel à allocate().
   */
  virtual void deallocate(const T* ptr,Integer capacity) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

