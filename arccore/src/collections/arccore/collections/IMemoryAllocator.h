// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryAllocator.h                                          (C) 2000-2020 */
/*                                                                           */
/* Interface d'un allocateur mémoire.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_IMEMORYALLOCATOR_H
#define ARCCORE_COLLECTIONS_IMEMORYALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
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
 * des tableaux l'utilisant.
 */
class ARCCORE_COLLECTIONS_EXPORT IMemoryAllocator
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
  virtual bool hasRealloc() const =0;

  /*!
   * \brief Alloue de la mémoire pour \a new_size octets et retourne le pointeur.
   *
   * La sémantique est équivalent à malloc():
   * - \a new_size peut valoir zéro et dans ce cas le pointeur retourné
   * est soit nul, soit une valeur spécifique
   * - le pointeur retourné peut être nul si la mémoire n'a pas pu être allouée.
   */
  virtual void* allocate(size_t new_size) =0;

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
  virtual void* reallocate(void* current_ptr,size_t new_size) =0;

  /*!
   * \brief Libère la mémoire dont l'adresse de base est \a ptr.
   *
   * Le pointeur \a ptr doit avoir été alloué via l'appel à
   * allocate() ou reallocate() de cette instance.
   *
   * La sémantique de cette méthode équivalente à free() et donc \a ptr
   * peut être nul auquel cas aucune opération n'est effectuée.
   */
  virtual void deallocate(void* ptr) =0;

  /*!
   * \brief Ajuste la capacité suivant la taille d'élément.
   *
   * Cette méthode est utilisée pour éventuellement modifié le nombre
   * d'éléments alloués suivant leur taille. Cela permet par exemple
   * pour les allocateurs alignés de garantir que le nombre d'éléments
   * alloués est un multiple de cet alignement.
   * 
   */
  virtual size_t adjustCapacity(size_t wanted_capacity,size_t element_size) =0;

 /*!
  * \brief Valeur de l'alignement garanti par l'allocateur.
  *
  * Cette méthode permet de s'assurer qu'un allocateur a un alignement suffisant
  * pour certaines opérations comme la vectorisation par exemple.
  *
  * S'il n'y a aucune garantie, retourne 0.
  */
  virtual size_t guarantedAlignment() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur mémoire via malloc/realloc/free.
 *
 * TODO: marquer les méthodes comme 'final'.
 */
class ARCCORE_COLLECTIONS_EXPORT DefaultMemoryAllocator
: public IMemoryAllocator
{
  friend class ArrayImplBase;
  friend class ArrayMetaData;

 private:

  static DefaultMemoryAllocator shared_null_instance;

 public:

  bool hasRealloc() const override;
  void* allocate(size_t new_size) override;
  void* reallocate(void* current_ptr,size_t new_size) override;
  void deallocate(void* ptr) override;
  size_t adjustCapacity(size_t wanted_capacity,size_t element_size) override;
  size_t guarantedAlignment() override { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur mémoire avec alignement mémoire spécifique.
 *
 * Cette classe s'utilise via les deux méthodes publiques Simd()
 * et CacheLine() qui retournent repectivement un allocateur avec
 * un alignement adéquat pour autoriser la vectorisation et un allocateur
 * aligné sur une ligne de cache.
 */
class ARCCORE_COLLECTIONS_EXPORT AlignedMemoryAllocator
: public IMemoryAllocator
{
 private:

  static AlignedMemoryAllocator SimdAllocator;
  static AlignedMemoryAllocator CacheLineAllocator;

 public:

  // TODO: essayer de trouver les bonnes valeurs en fonction de la cible.
  // 64 est OK pour toutes les architectures x64 à la fois pour le SIMD
  // et la ligne de cache.

  // IMPORTANT: Si on change la valeur ici, il faut changer la taille de
  // l'alignement de ArrayImplBase.

  // TODO Pour l'instant seul un alignement sur 64 est autorisé. Pour
  // autoriser d'autres valeurs, il faut modifier l'implémentation dans
  // ArrayImplBase.

  // TODO marquer les méthodes comme 'final'.

  //! Alignement pour les structures utilisant la vectorisation
  static constexpr Integer simdAlignment() { return 64; }
  //! Alignement pour une ligne de cache.
  static constexpr Integer cacheLineAlignment() { return 64; }

  /*!
   * \brief Allocateur garantissant l'alignement pour utiliser
   * la vectorisation sur la plateforme cible.
   *
   * Il s'agit de l'alignement pour le type plus restrictif et donc il
   * est possible d'utiliser cet allocateur pour toutes les structures vectorielles.
   */
  static AlignedMemoryAllocator* Simd()
  {
    return &SimdAllocator;
  }

  /*!
   * \brief Allocateur garantissant l'alignement sur une ligne de cache.
   */
  static AlignedMemoryAllocator* CacheLine()
  {
    return &CacheLineAllocator;
  }

 protected:

  AlignedMemoryAllocator(Integer alignment)
  : m_alignment((size_t)alignment){}

 public:

  bool hasRealloc() const override;
  void* allocate(size_t new_size) override;
  void* reallocate(void* current_ptr,size_t new_size) override;
  void deallocate(void* ptr) override;
  size_t adjustCapacity(size_t wanted_capacity,size_t element_size) override;
  size_t guarantedAlignment() override { return m_alignment; }

 private:

  size_t m_alignment;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur mémoire via malloc/realloc/free avec impression listing.
 *
 * Cet allocateur est principalement utilisé à des fins de debugging.
 * La sortie des informations se fait sur std::cout.
 */
class ARCCORE_COLLECTIONS_EXPORT PrintableMemoryAllocator
: public DefaultMemoryAllocator
{
  using Base = DefaultMemoryAllocator;

 public:

  void* allocate(size_t new_size) override;
  void* reallocate(void* current_ptr,size_t new_size) override;
  void deallocate(void* ptr) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

