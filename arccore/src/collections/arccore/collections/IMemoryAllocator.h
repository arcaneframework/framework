// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryAllocator.h                                          (C) 2000-2023 */
/*                                                                           */
/* Interface d'un allocateur mémoire.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_IMEMORYALLOCATOR_H
#define ARCCORE_COLLECTIONS_IMEMORYALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/MemoryAllocationArgs.h"

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
 * des tableaux l'utilisant. Comme l'allocateur est transféré lors des copies,
 * il est préférable que les allocateurs soient des objets statiques qui
 * dont la durée de vie est celle du programme.
 *
 * Les allocateurs n'ont pas d'état modifiables spécifiques et doivent fonctionner en
 * multi-threading.
 *
 * Depuis la version 2.3 de Arccore, les méthodes sans MemoryAllocationArgs sont
 * obsolètes.
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
  virtual bool hasRealloc(MemoryAllocationArgs args) const;

  /*!
   * \brief Alloue de la mémoire pour \a new_size octets et retourne le pointeur.
   *
   * La sémantique est équivalent à malloc():
   * - \a new_size peut valoir zéro et dans ce cas le pointeur retourné
   * est soit nul, soit une valeur non spécifiée.
   * - le pointeur retourné peut être nul si la mémoire n'a pas pu être allouée.
   */
  virtual AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size);

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
  virtual AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size);

  /*!
   * \brief Libère la mémoire dont l'adresse de base est \a ptr.
   *
   * Le pointeur \a ptr doit avoir été alloué via l'appel à
   * allocate() ou reallocate() de cette instance.
   *
   * La sémantique de cette méthode équivalente à free() et donc \a ptr
   * peut être nul auquel cas aucune opération n'est effectuée.
   */
  virtual void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr);

  /*!
   * \brief Ajuste la capacité suivant la taille d'élément.
   *
   * Cette méthode est utilisée pour éventuellement modifié le nombre
   * d'éléments alloués suivant leur taille. Cela permet par exemple
   * pour les allocateurs alignés de garantir que le nombre d'éléments
   * alloués est un multiple de cet alignement.
   * 
   */
  virtual Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const;

  /*!
   * \brief Valeur de l'alignement garanti par l'allocateur.
   *
   * Cette méthode permet de s'assurer qu'un allocateur a un alignement suffisant
   * pour certaines opérations comme la vectorisation par exemple.
   *
   * S'il n'y a aucune garantie, retourne 0.
   */
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
  virtual eMemoryRessource memoryRessource() const { return eMemoryRessource::Unknown; }

 public:

  // Méthodes historiques (avant 2023) sans arguments supplémentaires.
  // Elles sont laissées pour rester compatible avec l'existant mais seront
  // supprimées à terme.
  ARCCORE_DEPRECATED_REASON("Y2023: use hasRealloc(MemoryAllocationArgs) instead")
  virtual bool hasRealloc() const = 0;
  ARCCORE_DEPRECATED_REASON("Y2023: use allocate(MemoryAllocationArgs,Int64) instead")
  virtual void* allocate(size_t new_size) = 0;
  ARCCORE_DEPRECATED_REASON("Y2023: use reallocate(MemoryAllocationArgs,AllocatedMemoryInfo,Int64) instead")
  virtual void* reallocate(void* current_ptr, size_t new_size) = 0;
  ARCCORE_DEPRECATED_REASON("Y2023: use deallocate(MemoryAllocationArgs,AllocatedMemoryInfo) instead")
  virtual void deallocate(void* ptr) = 0;
  ARCCORE_DEPRECATED_REASON("Y2023: use adjustedCapacity(MemoryAllocationArgs,Int64,Int64) instead")
  virtual size_t adjustCapacity(size_t wanted_capacity, size_t element_size) = 0;
  ARCCORE_DEPRECATED_REASON("Y2023: use guarantedAlignment(MemoryAllocationArgs) instead")
  virtual size_t guarantedAlignment() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface de la version 3 de IMemoryAllocator.
 *
 * Elle contient les mêmes méthodes que IMemoryAllocator mais avec un argument
 * supplémentaire qui permet de spécialiser les allocations.
 */
class ARCCORE_COLLECTIONS_EXPORT IMemoryAllocator3
: public IMemoryAllocator
{
  using BaseClass = IMemoryAllocator;

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
  virtual size_t guarantedAlignment(MemoryAllocationArgs args) const = 0;

 private:

  bool hasRealloc() const final;
  void* allocate(size_t new_size) final;
  void* reallocate(void* current_ptr, size_t new_size) final;
  void deallocate(void* ptr) final;
  size_t adjustCapacity(size_t wanted_capacity, size_t element_size) final;
  size_t guarantedAlignment() final;
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
 public:

  using IMemoryAllocator::adjustCapacity;
  using IMemoryAllocator::allocate;
  using IMemoryAllocator::deallocate;
  using IMemoryAllocator::guarantedAlignment;
  using IMemoryAllocator::hasRealloc;
  using IMemoryAllocator::reallocate;

 public:

  bool hasRealloc() const override;
  void* allocate(size_t new_size) override;
  void* reallocate(void* current_ptr, size_t new_size) override;
  void deallocate(void* ptr) override;
  size_t adjustCapacity(size_t wanted_capacity, size_t element_size) override;
  size_t guarantedAlignment() override { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur mémoire via malloc/realloc/free.
 *
 * TODO: marquer les méthodes comme 'final'.
 */
class ARCCORE_COLLECTIONS_EXPORT DefaultMemoryAllocator3
: public IMemoryAllocator3
{
  friend class ArrayMetaData;

 public:

  using IMemoryAllocator::adjustCapacity;
  using IMemoryAllocator::allocate;
  using IMemoryAllocator::deallocate;
  using IMemoryAllocator::guarantedAlignment;
  using IMemoryAllocator::hasRealloc;
  using IMemoryAllocator::reallocate;

 private:

  static DefaultMemoryAllocator3 shared_null_instance;

 public:

  bool hasRealloc(MemoryAllocationArgs) const override;
  AllocatedMemoryInfo allocate(MemoryAllocationArgs, Int64 new_size) override;
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs, AllocatedMemoryInfo current_ptr, Int64 new_size) override;
  void deallocate(MemoryAllocationArgs, AllocatedMemoryInfo ptr) override;
  Int64 adjustedCapacity(MemoryAllocationArgs, Int64 wanted_capacity, Int64 element_size) const override;
  size_t guarantedAlignment(MemoryAllocationArgs) const override { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
  extern "C++" ARCCORE_COLLECTIONS_EXPORT size_t
  adjustMemoryCapacity(size_t wanted_capacity, size_t element_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur mémoire avec alignement mémoire spécifique.
 *
 * Cette classe s'utilise via les deux méthodes publiques Simd()
 * et CacheLine() qui retournent respectivement un allocateur avec
 * un alignement adéquat pour autoriser la vectorisation et un allocateur
 * aligné sur une ligne de cache.
 *
 * Cette classe est obsolète. Utiliser 'AlignedMemoryAllocator2' à la place
 */
class ARCCORE_COLLECTIONS_EXPORT AlignedMemoryAllocator
: public IMemoryAllocator
{
 public:

  using IMemoryAllocator::adjustCapacity;
  using IMemoryAllocator::allocate;
  using IMemoryAllocator::deallocate;
  using IMemoryAllocator::guarantedAlignment;
  using IMemoryAllocator::hasRealloc;
  using IMemoryAllocator::reallocate;

 private:

  static AlignedMemoryAllocator SimdAllocator;
  static AlignedMemoryAllocator CacheLineAllocator;

 public:

  // TODO: essayer de trouver les bonnes valeurs en fonction de la cible.
  // 64 est OK pour toutes les architectures x64 à la fois pour le SIMD
  // et la ligne de cache.

  // IMPORTANT : Si on change la valeur ici, il faut changer la taille de
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

  explicit AlignedMemoryAllocator(Integer alignment)
  : m_alignment((size_t)alignment)
  {}

 public:

  bool hasRealloc() const override;
  void* allocate(size_t new_size) override;
  void* reallocate(void* current_ptr, size_t new_size) override;
  void deallocate(void* ptr) override;
  size_t adjustCapacity(size_t wanted_capacity, size_t element_size) override;
  size_t guarantedAlignment() override { return m_alignment; }

 private:

  size_t m_alignment;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur mémoire avec alignement mémoire spécifique.
 *
 * Cette classe s'utilise via les deux méthodes publiques Simd()
 * et CacheLine() qui retournent respectivement un allocateur avec
 * un alignement adéquat pour autoriser la vectorisation et un allocateur
 * aligné sur une ligne de cache.
 */
class ARCCORE_COLLECTIONS_EXPORT AlignedMemoryAllocator3
: public IMemoryAllocator3
{
 public:

  using IMemoryAllocator::adjustCapacity;
  using IMemoryAllocator::allocate;
  using IMemoryAllocator::deallocate;
  using IMemoryAllocator::guarantedAlignment;
  using IMemoryAllocator::hasRealloc;
  using IMemoryAllocator::reallocate;

 private:

  static AlignedMemoryAllocator3 SimdAllocator;
  static AlignedMemoryAllocator3 CacheLineAllocator;

 public:

  // TODO: essayer de trouver les bonnes valeurs en fonction de la cible.
  // 64 est OK pour toutes les architectures x64 à la fois pour le SIMD
  // et la ligne de cache.

  // IMPORTANT : Si on change la valeur ici, il faut changer la taille de
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
  static AlignedMemoryAllocator3* Simd()
  {
    return &SimdAllocator;
  }

  /*!
   * \brief Allocateur garantissant l'alignement sur une ligne de cache.
   */
  static AlignedMemoryAllocator3* CacheLine()
  {
    return &CacheLineAllocator;
  }

 protected:

  explicit AlignedMemoryAllocator3(Integer alignment)
  : m_alignment((size_t)alignment)
  {}

 public:

  bool hasRealloc(MemoryAllocationArgs) const override { return false; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) override;
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) override;
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) override;
  Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const override;
  size_t guarantedAlignment(MemoryAllocationArgs) const override { return m_alignment; }

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

  using IMemoryAllocator::allocate;
  using IMemoryAllocator::deallocate;
  using IMemoryAllocator::reallocate;

 public:

  void* allocate(size_t new_size) override;
  void* reallocate(void* current_ptr, size_t new_size) override;
  void deallocate(void* ptr) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

