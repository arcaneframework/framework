// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArray.h                                                  (C) 2000-2022 */
/*                                                                           */
/* Tableaux multi-dimensionnel pour les types numériques.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMARRAY_H
#define ARCANE_UTILS_NUMARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/MemoryRessource.h"
#include "arcane/utils/MDSpan.h"

/*
 * ATTENTION:
 *
 * Toutes les classes de ce fichier sont expérimentales et l'API n'est pas
 * figée. A NE PAS UTILISER EN DEHORS DE ARCANE.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT NumArrayBaseCommon
{
 protected:
  static IMemoryAllocator* _getDefaultAllocator();
  static IMemoryAllocator* _getDefaultAllocator(eMemoryRessource r);
  static void _checkHost(eMemoryRessource r);
  static void _copy(Span<const std::byte> from, eMemoryRessource from_mem,
                    Span<std::byte> to, eMemoryRessource to_mem);
};

namespace impl
{
  // Wrapper de Arccore::Array pour la classe NumArray
  template<typename DataType>
  class NumArrayContainer
  : private Arccore::Array<DataType>
  {
   private:
    using BaseClass = Arccore::Array<DataType>;
   public:
    using BaseClass::capacity;
    using BaseClass::fill;
   public:
    explicit NumArrayContainer(IMemoryAllocator* a)
    {
      this->_initFromAllocator(a,0);
    }
    NumArrayContainer(const NumArrayContainer<DataType>& rhs) : BaseClass()
    {
      this->_initFromSpan(rhs.to1DSpan());
    }
   public:
    void resize(Int64 new_size) { BaseClass::_resize(new_size); }
    Span<DataType> to1DSpan() { return BaseClass::span(); }
    Span<const DataType> to1DSpan() const { return BaseClass::constSpan(); }
    Span<std::byte> bytes() { return asWritableBytes(BaseClass::span()); }
    Span<const std::byte> bytes() const { return asBytes(BaseClass::constSpan()); }
    void swap(NumArrayContainer<DataType>& rhs) { BaseClass::_swap(rhs); }
    void copy(Span<const DataType> rhs) { BaseClass::_copy(rhs.data()); }
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des tableaux multi-dimensionnels pour les types
 * numériques sur accélérateur.
 *
 * \warning API en cours de définition.
 *
 * En général cette classe n'est pas utilisée directement mais par l'intermédiaire
 * d'une de ses spécialisations suivant le rang comme NumArray<DataType,1>,
 * NumArray<DataType,2>, NumArray<DataType,3> ou NumArray<DataType,4>.
 *
 * Cette classe contient un nombre minimal de méthodes. Notamment, l'accès aux
 * valeurs du tableau se fait normalement via des vues (MDSpanBase).
 * Afin de faciliter l'utilisation sur CPU, l'opérateur 'operator()'
 * permet de retourner la valeur en lecture d'un élément. Pour modifier un élément,
 * il faut utiliser la méthode s().
 *
 * \warning Le redimensionnement via resize() ne conserve pas les valeurs existantes
 *
 * \warning Cette classe utilise par défaut un allocateur spécifique qui permet de
 * rendre accessible ces valeurs à la fois sur l'hôte (CPU) et l'accélérateur.
 * Néanmoins, il faut pour cela que le runtime associé à l'accélérateur ait été
 * initialisé (\ref arcanedoc_accelerator). C'est pourquoi il ne faut pas
 * utiliser de variables globales de cette classe ou d'une classe dérivée.
 */
template<typename DataType,int RankValue,typename LayoutType>
class NumArrayBase
: public NumArrayBaseCommon
{
 public:

  using ConstSpanType = MDSpan<const DataType,RankValue,LayoutType>;
  using SpanType = MDSpan<DataType,RankValue,LayoutType>;
  using ArrayWrapper = impl::NumArrayContainer<DataType>;

 public:

  //! Nombre total d'éléments du tableau
  Int64 totalNbElement() const { return m_total_nb_element; }
  //! Nombre d'éléments du rang \a i
  Int32 extent(int i) const { return m_span.extent(i); }
  /*!
   * \brief Modifie la taille du tableau.
   * \warning Les valeurs actuelles ne sont pas conservées lors de cette opération
   */
  void resize(ArrayExtents<RankValue> extents)
  {
    m_span.m_extents.setSize(extents);
    _resize();
  }
 protected:

  NumArrayBase() : m_data(_getDefaultAllocator()){}
  explicit NumArrayBase(eMemoryRessource r) : m_data(_getDefaultAllocator(r)), m_memory_ressource(r){}
  explicit NumArrayBase(ArrayExtents<RankValue> extents)
  : m_data(_getDefaultAllocator()), m_memory_ressource(eMemoryRessource::UnifiedMemory)
  {
    resize(extents);
  }
  NumArrayBase(ArrayExtents<RankValue> extents,eMemoryRessource r)
  : m_data(_getDefaultAllocator(r)), m_memory_ressource(r)
  {
    resize(extents);
  }

 private:
  void _resize()
  {
    Int32 dim1_size = extent(0);
    Int32 dim2_size = 1;
    // TODO: vérifier débordement.
    for (int i=1; i<RankValue; ++i )
      dim2_size *= extent(i);
    m_total_nb_element = dim1_size * dim2_size;
    m_data.resize(m_total_nb_element);
    m_span.m_ptr = m_data.to1DSpan().data();
  }
 public:
  void fill(const DataType& v) { m_data.fill(v); }
  Int32 nbDimension() const { return RankValue; }
  ArrayExtents<RankValue> extents() const { return m_span.extents(); }
  ArrayExtentsWithOffset<RankValue,LayoutType> extentsWithOffset() const
  {
    return m_span.extentsWithOffset();
  }
 public:
  SpanType span() { return m_span; }
  ConstSpanType span() const { return m_span.constSpan(); }
  ConstSpanType constSpan() const { return m_span.constSpan(); }
 public:
  Span<const DataType> to1DSpan() const { return m_span.to1DSpan(); }
  Span<DataType> to1DSpan() { return m_span.to1DSpan(); }
  void copy(ConstSpanType rhs)
  {
    _checkHost(m_memory_ressource);
    m_data.copy(rhs.to1DSpan());
  }
  void copy(const NumArrayBase<DataType,RankValue,LayoutType>& rhs)
  {
    this->resize(rhs.extents());
    _copy(asBytes(rhs.to1DSpan()),rhs.m_memory_ressource,
          asWritableBytes(to1DSpan()),m_memory_ressource);
  }
  const DataType& operator()(ArrayBoundsIndex<RankValue> idx) const
  {
    return m_span(idx);
  }
  DataType& operator()(ArrayBoundsIndex<RankValue> idx)
  {
    return m_span(idx);
  }
  DataType& s(ArrayBoundsIndex<RankValue> idx)
  {
    return m_span(idx);
  }
  void swap(NumArrayBase<DataType,RankValue>& rhs)
  {
    m_data.swap(rhs.m_data);
    std::swap(m_span,rhs.m_span);
    std::swap(m_total_nb_element,rhs.m_total_nb_element);
    std::swap(m_memory_ressource,rhs.m_memory_ressource);
  }
  Int64 capacity() const { return m_data.capacity(); }
  eMemoryRessource memoryRessource() const { return m_memory_ressource; }
  Span<std::byte> bytes() { return asWritableBytes(to1DSpan()); }
  Span<const std::byte> bytes() const { return asBytes(to1DSpan()); }
 public:
  //! \internal
  DataType* _internalData() { return m_span._internalData(); }
 protected:
  SpanType m_span;
  ArrayWrapper m_data;
  Int64 m_total_nb_element = 0;
  eMemoryRessource m_memory_ressource = eMemoryRessource::Unknown;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableau à 1 dimension pour les types numériques.
 *
 * Les tableaux à une dimension possèdent l'opérateur 'operator[]' pour
 * compatibilité avec les tableaux classiques du C++.
 *
 * \sa NumArrayBase
 */
template<class DataType,typename LayoutType>
class NumArray<DataType,1,LayoutType>
: public NumArrayBase<DataType,1,LayoutType>
{
 public:
  using BaseClass = NumArrayBase<DataType,1,LayoutType>;
  using BaseClass::extent;
  using BaseClass::resize;
  using BaseClass::operator();
  using BaseClass::s;
  using ConstSpanType = MDSpan<const DataType,1,LayoutType>;
  using SpanType = MDSpan<DataType,1,LayoutType>;
 private:
  using BaseClass::m_span;
 public:
  //! Construit un tableau vide
  NumArray() : NumArray(0){}
  explicit NumArray(eMemoryRessource r) : BaseClass(r){}
  //! Construit un tableau
  explicit NumArray(Int32 dim1_size)
  : BaseClass(ArrayExtents<1>(dim1_size)){}
  NumArray(Int32 dim1_size,eMemoryRessource r)
  : BaseClass(ArrayExtents<1>{dim1_size},r){}
 public:
  void resize(Int32 dim1_size)
  {
    this->resize(ArrayExtents<1>(dim1_size));
  }
 public:
  //! Valeur de la première dimension
  constexpr Int32 dim1Size() const { return this->extent(0); }
 public:
  //! Valeur pour l'élément \a i
  DataType operator()(Int32 i) const { return m_span(i); }
  //! Positionne la valeur pour l'élément \a i
  DataType& operator()(Int32 i) { return m_span(i); }
  //! Positionne la valeur pour l'élément \a i
  DataType& s(Int32 i) { return m_span(i); }
  //! Récupère une référence pour l'élément \a i
  DataType& operator[](Int32 i) { return m_span(i); }
  //! Valeur pour l'élément \a i
  DataType operator[](Int32 i) const { return m_span(i); }
 public:
  constexpr operator SpanType () { return this->span(); }
  constexpr operator ConstSpanType () const { return this->constSpan(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableau à 2 dimensions pour les types numériques.
 *
 * \sa NumArrayBase
 */
template<class DataType,typename LayoutType>
class NumArray<DataType,2,LayoutType>
: public NumArrayBase<DataType,2,LayoutType>
{
 public:
  using BaseClass = NumArrayBase<DataType,2,LayoutType>;
  using BaseClass::extent;
  using BaseClass::resize;
  using BaseClass::operator();
  using BaseClass::s;
 private:
  using BaseClass::m_span;
 public:
  //! Construit un tableau vide
  NumArray() = default;
  explicit NumArray(eMemoryRessource r) : BaseClass(r){}
  //! Construit une vue
  NumArray(Int32 dim1_size,Int32 dim2_size)
  : BaseClass(ArrayExtents<2>{dim1_size,dim2_size}){}
  NumArray(Int32 dim1_size,Int32 dim2_size,eMemoryRessource r)
  : BaseClass(ArrayExtents<2>{dim1_size,dim2_size},r){}
 public:
  void resize(Int32 dim1_size,Int32 dim2_size)
  {
    this->resize(ArrayExtents<2>(dim1_size,dim2_size));
  }

 public:
  //! Valeur de la première dimension
  constexpr Int32 dim1Size() const { return extent(0); }
  //! Valeur de la deuxième dimension
  constexpr Int32 dim2Size() const { return extent(1); }
 public:
  //! Valeur pour l'élément \a i,j
  DataType operator()(Int32 i,Int32 j) const
  {
    return m_span(i,j);
  }
  //! Positionne la valeur pour l'élément \a i,j
  DataType& operator()(Int32 i,Int32 j)
  {
    return m_span(i,j);
  }
  //! Positionne la valeur pour l'élément \a i,j
  DataType& s(Int32 i,Int32 j)
  {
    return m_span(i,j);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableau à 3 dimensions pour les types numériques.
 *
 * \sa NumArrayBase
 */
template<class DataType,typename LayoutType>
class NumArray<DataType,3,LayoutType>
: public NumArrayBase<DataType,3,LayoutType>
{
 public:
  using BaseClass = NumArrayBase<DataType,3,LayoutType>;
  using BaseClass::extent;
  using BaseClass::resize;
  using BaseClass::operator();
  using BaseClass::s;
 private:
  using BaseClass::m_span;
 public:
  //! Construit un tableau vide
  NumArray() = default;
  explicit NumArray(eMemoryRessource r) : BaseClass(r){}
  NumArray(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size)
  : BaseClass(ArrayExtents<3>(dim1_size,dim2_size,dim3_size)){}
  NumArray(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size,eMemoryRessource r)
  : BaseClass(ArrayExtents<3>{dim1_size,dim2_size,dim3_size},r){}
 public:
  void resize(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size)
  {
    this->resize(ArrayExtents<3>(dim1_size,dim2_size,dim3_size));
  }
 public:
  //! Valeur de la première dimension
  constexpr Int32 dim1Size() const { return extent(0); }
  //! Valeur de la deuxième dimension
  constexpr Int32 dim2Size() const { return extent(1); }
  //! Valeur de la troisième dimension
  constexpr Int32 dim3Size() const { return extent(2); }
 public:
  //! Valeur pour l'élément \a i,j,k
  DataType operator()(Int32 i,Int32 j,Int32 k) const
  {
    return m_span(i,j,k);
  }
  //! Positionne la valeur pour l'élément \a i,j,k
  DataType& operator()(Int32 i,Int32 j,Int32 k)
  {
    return m_span(i,j,k);
  }
  //! Positionne la valeur pour l'élément \a i,j,k
  DataType& s(Int32 i,Int32 j,Int32 k)
  {
    return m_span(i,j,k);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableau à 4 dimensions pour les types numériques.
 *
 * \sa NumArrayBase
 */
template<class DataType,typename LayoutType>
class NumArray<DataType,4,LayoutType>
: public NumArrayBase<DataType,4,LayoutType>
{
 public:
  using BaseClass = NumArrayBase<DataType,4,LayoutType>;
  using BaseClass::extent;
  using BaseClass::resize;
  using BaseClass::operator();
  using BaseClass::s;
 private:
  using BaseClass::m_span;
 public:
  //! Construit un tableau vide
  NumArray() = default;
  explicit NumArray(eMemoryRessource r) : BaseClass(r){}
  NumArray(Int32 dim1_size,Int32 dim2_size,
           Int32 dim3_size,Int32 dim4_size)
  : BaseClass(ArrayExtents<4>(dim1_size,dim2_size,dim3_size,dim4_size)){}
  NumArray(Int32 dim1_size,Int32 dim2_size,
           Int32 dim3_size,Int32 dim4_size,eMemoryRessource r)
  : BaseClass(ArrayExtents<4>{dim1_size,dim2_size,dim3_size,dim4_size},r){}
 public:
  void resize(Int32 dim1_size,Int32 dim2_size,Int32 dim3_size,Int32 dim4_size)
  {
    this->resize(ArrayExtents<4>(dim1_size,dim2_size,dim3_size,dim4_size));
  }

 public:
  //! Valeur de la première dimension
  constexpr Int32 dim1Size() const { return extent(0); }
  //! Valeur de la deuxième dimension
  constexpr Int32 dim2Size() const { return extent(1); }
  //! Valeur de la troisième dimension
  constexpr Int32 dim3Size() const { return extent(2); }
  //! Valeur de la quatrième dimension
  constexpr Int32 dim4Size() const { return extent(3); }
 public:
  //! Valeur pour l'élément \a i,j,k,l
  DataType operator()(Int32 i,Int32 j,Int32 k,Int32 l) const
  {
    return m_span(i,j,k,l);
  }
  //! Positionne la valeur pour l'élément \a i,j,k,l
  DataType& operator()(Int32 i,Int32 j,Int32 k,Int32 l)
  {
    return m_span(i,j,k,l);
  }
  //! Positionne la valeur pour l'élément \a i,j,k,l
  DataType& s(Int32 i,Int32 j,Int32 k,Int32 l)
  {
    return m_span(i,j,k,l);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
