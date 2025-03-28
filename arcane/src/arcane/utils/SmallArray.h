// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SmallArray.h                                                (C) 2000-2025 */
/*                                                                           */
/* Tableau 1D de données avec buffer pré-alloué.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_STACKARRAY_H
#define ARCANE_UTILS_STACKARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/MemoryAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Allocateur avec buffer pré-alloué.
 *
 * Le buffer pré-alloué \a m_preallocated_buffer est utilisé lorsque la
 * taille demandée pour l'allocation est inférieure ou égale à
 * \a m_preallocated_size.
 *
 * Le buffer utilisé doit rester valide durant toute la durée de vie de l'allocateur.
 */
class ARCANE_UTILS_EXPORT StackMemoryAllocator final
: public IMemoryAllocator3
{
 public:

  StackMemoryAllocator(void* buf, size_t size)
  : m_preallocated_buffer(buf)
  , m_preallocated_size(size)
  {}

 public:

  bool hasRealloc(MemoryAllocationArgs) const final { return true; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) final;
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) final;
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) final;
  Int64 adjustedCapacity(MemoryAllocationArgs, Int64 wanted_capacity, Int64) const final { return wanted_capacity; }
  size_t guaranteedAlignment(MemoryAllocationArgs) const final { return 0; }

 private:

  void* m_preallocated_buffer = nullptr;
  Int64 m_preallocated_size = 0;
};

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Tableau 1D de données avec buffer pré-alloué sur la pile.
 *
 * Cette classe s'utilise comme UniqueArray mais contient un buffer de taille
 * fixe pour conserver \a NbElement éléments qui est utilisé si le tableau contient
 * au plus \a NbElement éléments. Cela permet d'éviter des allocations dynamiques
 * lorsque le nombre d'éléments est faible.
 *
 * Si le tableau doit contenir plus de \a NbElement éléments, alors on utilise
 * une allocation dynamique standard.
 */
template <typename T, Int32 NbElement = 32>
class SmallArray final
: public Array<T>
{
  using BaseClassType = AbstractArray<T>;
  static constexpr Int32 SizeOfType = static_cast<Int32>(sizeof(T));
  static constexpr Int32 nb_element_in_buf = NbElement;

 public:

  using typename BaseClassType::ConstReferenceType;
  static constexpr Int32 MemorySize = NbElement * SizeOfType;

 public:

  //! Créé un tableau vide
  SmallArray()
  : m_stack_allocator(m_stack_buffer, MemorySize)
  {
    this->_initFromAllocator(&m_stack_allocator, nb_element_in_buf);
  }

  //! Créé un tableau de \a size éléments contenant la valeur \a value.
  SmallArray(Int64 req_size, ConstReferenceType value)
  : SmallArray()
  {
    this->_resize(req_size, value);
  }

  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit SmallArray(Int64 asize)
  : SmallArray()
  {
    this->_resize(asize);
  }

  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit SmallArray(Int32 asize)
  : SmallArray((Int64)asize)
  {
  }

  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit SmallArray(size_t asize)
  : SmallArray((Int64)asize)
  {
  }

  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  SmallArray(const ConstArrayView<T>& aview)
  : SmallArray(Span<const T>(aview))
  {
  }

  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  SmallArray(const Span<const T>& aview)
  : SmallArray()
  {
    this->_initFromSpan(aview);
  }

  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  SmallArray(const ArrayView<T>& aview)
  : SmallArray(Span<const T>(aview))
  {
  }

  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  SmallArray(const Span<T>& aview)
  : SmallArray(Span<const T>(aview))
  {
  }

  SmallArray(std::initializer_list<T> alist)
  : SmallArray()
  {
    this->_initFromInitializerList(alist);
  }

  //! Créé un tableau en recopiant les valeurs \a rhs.
  SmallArray(const Array<T>& rhs)
  : SmallArray(rhs.constSpan())
  {
  }

  //! Copie les valeurs de \a rhs dans cette instance.
  void operator=(const Array<T>& rhs)
  {
    this->copy(rhs.constSpan());
  }

  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const ArrayView<T>& rhs)
  {
    this->copy(rhs);
  }

  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const Span<T>& rhs)
  {
    this->copy(rhs);
  }

  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const ConstArrayView<T>& rhs)
  {
    this->copy(rhs);
  }

  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const Span<const T>& rhs)
  {
    this->copy(rhs);
  }

  //! Détruit l'instance.
  ~SmallArray() override
  {
    // Il faut détruire explicitement car notre allocateur
    // sera détruit avant la classe de base et ne sera plus valide
    // lors de la désallocation.
    this->_destroy();
    this->_internalDeallocate();
    this->_reset();
  }

 public:

  template <Int32 N> SmallArray(SmallArray<T, N>&& rhs) = delete;
  template <Int32 N> SmallArray<T, NbElement> operator=(SmallArray<T, N>&& rhs) = delete;

 private:

  char m_stack_buffer[MemorySize];
  impl::StackMemoryAllocator m_stack_allocator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
