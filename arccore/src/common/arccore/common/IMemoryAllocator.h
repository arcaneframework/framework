// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryAllocator.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface d'un allocateur mémoire.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_IMEMORYALLOCATOR_H
#define ARCCORE_COMMON_IMEMORYALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/MemoryAllocationArgs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un allocateur pour la mémoire.
 *
 * Cette classe définit une interface pour l'allocation mémoire utilisée
 * par les classes tableaux de Arccore (Array, UniqueArray).
 *
 * Une instance de cette classe doit rester valide tant qu'il existe
 * des tableaux l'utilisant. Comme l'allocateur est transféré lors des copies,
 * il est préférable que les allocateurs soient des objets statiques qui
 * dont la durée de vie est celle du programme.
 *
 * Les allocateurs n'ont pas d'état modifiables spécifiques et doivent fonctionner en
 * multi-threading.
 */
class ARCCORE_COMMON_EXPORT IMemoryAllocator
{
 public:

  /*!
   * \brief Détruit l'allocateur.
   *
   * Les objets alloués par l'allocateur doivent tous avoir été désalloués.
   */
  virtual ~IMemoryAllocator() = default;

 public:

  /*!
   * \brief Indique si l'allocateur supporte la sémantique de realloc.
   *
   * Les allocateurs par défaut du C (malloc/realloc/free) supportent
   * évidemment le realloc mais ce n'est pas forcément le cas
   * des allocateurs spécifiques avec alignement mémoire (comme
   * par exemple posix_memalign).
   */
  virtual bool hasRealloc(MemoryAllocationArgs) const { return false; }

  /*!
   * \brief Alloue de la mémoire pour \a new_size octets et retourne le pointeur.
   *
   * La sémantique est équivalent à malloc():
   * - \a new_size peut valoir zéro et dans ce cas le pointeur retourné
   * est soit nul, soit une valeur spécifique
   * - le pointeur retourné peut être nul si la mémoire n'a pas pu être allouée.
   */
  virtual AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) = 0;

  /*!
   * \brief Réalloue de la mémoire pour \a new_size octets et retourne le pointeur.
   *
   * Le pointeur \a current_ptr doit avoir été alloué via l'appel à
   * allocate() ou reallocate() de cette instance.
   *
   * La sémantique de cette méthode est équivalente à realloc():
   * - \a current_ptr peut-être nul auquel cas cet appel est équivalent
   * à allocate().
   * - le pointeur retourné peut être nul si la mémoire n'a pas pu être allouée.
   */
  virtual AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) = 0;

  /*!
   * \brief Libère la mémoire dont l'adresse de base est \a ptr.
   *
   * Le pointeur \a ptr doit avoir été alloué via l'appel à
   * allocate() ou reallocate() de cette instance.
   *
   * La sémantique de cette méthode équivalente à free() et donc \a ptr
   * peut être nul auquel cas aucune opération n'est effectuée.
   */
  virtual void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) = 0;

  /*!
   * \brief Ajuste la capacité suivant la taille d'élément.
   *
   * Cette méthode est utilisée pour éventuellement modifié le nombre
   * d'éléments alloués suivant leur taille. Cela permet par exemple
   * pour les allocateurs alignés de garantir que le nombre d'éléments
   * alloués est un multiple de cet alignement.
   * 
   */
  virtual Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const = 0;

  /*!
   * \brief Valeur de l'alignement garanti par l'allocateur.
   *
   * Cette méthode permet de s'assurer qu'un allocateur a un alignement suffisant
   * pour certaines opérations comme la vectorisation par exemple.
   *
   * S'il n'y a aucune garantie, retourne 0.
   */
  virtual size_t guaranteedAlignment(MemoryAllocationArgs args) const =0;

  /*!
   * \brief Valeur de l'alignement garanti par l'allocateur.
   *
   * \sa guaranteedAlignment()
   */
  ARCCORE_DEPRECATED_REASON("Y2024: Use guaranteedAlignment() instead")
  virtual size_t guarantedAlignment(MemoryAllocationArgs args) const;

  /*!
   * \brief Notifie du changement des arguments spécifiques à l'instance.
   *
   * \param ptr zone mémoire allouée
   * \param old_args ancienne valeur des arguments
   * \param new_args nouvelle valeur des arguments
   */
  virtual void notifyMemoryArgsChanged(MemoryAllocationArgs old_args, MemoryAllocationArgs new_args, AllocatedMemoryInfo ptr);

  /*!
   * \brief Copie la mémoire entre deux zones.
   *
   * L'implémentation par défaut utilise std::memcpy().
   *
   * \param args arguments de la zone mémoire
   * \param destination destination de la copie
   * \param destination source de la copie
   */
  virtual void copyMemory(MemoryAllocationArgs args, AllocatedMemoryInfo destination, AllocatedMemoryInfo source);

  //! Ressource mémoire fournie par l'allocateur
  virtual eMemoryResource memoryResource() const { return eMemoryResource::Unknown; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

