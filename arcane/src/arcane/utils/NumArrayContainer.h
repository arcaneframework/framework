﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayContainer.h                                         (C) 2000-2023 */
/*                                                                           */
/* Conteneur pour la classe 'NumArray'.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMARRAYCONTAINER_H
#define ARCANE_UTILS_NUMARRAYCONTAINER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/MemoryRessource.h"
#include "arcane/utils/ArcaneCxx20.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation commune à pour NumArray.
 */
class ARCANE_UTILS_EXPORT NumArrayBaseCommon
{
 protected:

  static IMemoryAllocator* _getDefaultAllocator();
  static IMemoryAllocator* _getDefaultAllocator(eMemoryRessource r);
  static void _checkHost(eMemoryRessource r);
  static void _memoryAwareCopy(Span<const std::byte> from, eMemoryRessource from_mem,
                               Span<std::byte> to, eMemoryRessource to_mem, RunQueue* queue);
  static void _memoryAwareFill(Span<std::byte> to, Int64 nb_element, const void* fill_address,
                               Int32 datatype_size, SmallSpan<const Int32> indexes, RunQueue* queue);
  static void _memoryAwareFill(Span<std::byte> to, Int64 nb_element, const void* fill_address,
                               Int32 datatype_size, RunQueue* queue);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Container pour la classe NumArray.
 *
 * Wrapper de Arccore::Array pour la classe NumArray.
 */
template <typename DataType>
class NumArrayContainer
: private Arccore::Array<DataType>
, private NumArrayBaseCommon
{
 private:

  using BaseClass = Arccore::Array<DataType>;
  using ThatClass = NumArrayContainer<DataType>;
  static constexpr Int32 _typeSize() { return static_cast<Int32>(sizeof(DataType)); }

 public:

  using BaseClass::capacity;
  using BaseClass::fill;

 private:

  explicit NumArrayContainer(IMemoryAllocator* a)
  : BaseClass()
  {
    this->_initFromAllocator(a, 0);
  }

 public:

  explicit NumArrayContainer()
  : NumArrayContainer(_getDefaultAllocator())
  {
  }

  explicit NumArrayContainer(eMemoryRessource r)
  : NumArrayContainer(_getDefaultAllocator(r))
  {
    m_memory_ressource = r;
  }

  NumArrayContainer(const ThatClass& rhs)
  : NumArrayContainer(rhs.allocator())
  {
    m_memory_ressource = rhs.m_memory_ressource;
    _resizeAndCopy(rhs);
  }

  NumArrayContainer(ThatClass&& rhs)
  : BaseClass(std::move(rhs))
  , m_memory_ressource(rhs.m_memory_ressource)
  {
  }

  // Cet opérateur est interdit car il faut gérer le potentiel
  // changement de l'allocateur et la recopie
  ThatClass& operator=(const ThatClass& rhs) = delete;

  ThatClass& operator=(ThatClass&& rhs)
  {
    this->_move(rhs);
    m_memory_ressource = rhs.m_memory_ressource;
    return (*this);
  }

 public:

  void resize(Int64 new_size) { BaseClass::_resizeNoInit(new_size); }
  Span<DataType> to1DSpan() { return BaseClass::span(); }
  Span<const DataType> to1DSpan() const { return BaseClass::constSpan(); }
  Span<std::byte> bytes() { return asWritableBytes(BaseClass::span()); }
  Span<const std::byte> bytes() const { return asBytes(BaseClass::constSpan()); }
  void swap(NumArrayContainer<DataType>& rhs)
  {
    BaseClass::_swap(rhs);
    std::swap(m_memory_ressource, rhs.m_memory_ressource);
  }
  void copy(Span<const DataType> rhs) { BaseClass::_copy(rhs.data()); }
  IMemoryAllocator* allocator() const { return BaseClass::allocator(); }
  eMemoryRessource memoryRessource() const { return m_memory_ressource; }
  void copyInitializerList(std::initializer_list<DataType> alist)
  {
    Span<DataType> s = to1DSpan();
    Int64 s1 = s.size();
    Int32 index = 0;
    for (auto x : alist) {
      s[index] = x;
      ++index;
      // S'assure qu'on ne déborde pas
      if (index >= s1)
        break;
    }
  }

  /*!
   * \brief Copie les valeurs de \a v dans l'instance.
   *
   * \a input_ressource indique l'origine de la zone mémoire (ou eMemoryRessource::Unknown si inconnu)
   */
  void copyOnly(const Span<const DataType>& v, eMemoryRessource input_ressource, RunQueue* queue = nullptr)
  {
    _memoryAwareCopy(v, input_ressource, queue);
  }
  /*!
   * \brief Remplit les indices données par \a indexes avec la valeur \a v.
   */
  void fill(const DataType& v, SmallSpan<const Int32> indexes, RunQueue* queue)
  {
    Span<DataType> destination = to1DSpan();
    NumArrayBaseCommon::_memoryAwareFill(asWritableBytes(destination), destination.size(), &v, _typeSize(), indexes, queue);
  }
  /*!
   * \brief Remplit les éléments de l'instance la valeur \a v.
   */
  void fill(const DataType& v, RunQueue* queue)
  {
    Span<DataType> destination = to1DSpan();
    NumArrayBaseCommon::_memoryAwareFill(asWritableBytes(destination), destination.size(), &v, _typeSize(), queue);
  }

 private:

  void _memoryAwareCopy(const Span<const DataType>& v, eMemoryRessource input_ressource, RunQueue* queue)
  {
    NumArrayBaseCommon::_memoryAwareCopy(asBytes(v), input_ressource,
                                         asWritableBytes(to1DSpan()), m_memory_ressource, queue);
  }
  void _resizeAndCopy(const ThatClass& v)
  {
    this->_resizeNoInit(v.to1DSpan().size());
    _memoryAwareCopy(v, v.memoryRessource(), nullptr);
  }

 private:

  eMemoryRessource m_memory_ressource = eMemoryRessource::UnifiedMemory;
};

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
