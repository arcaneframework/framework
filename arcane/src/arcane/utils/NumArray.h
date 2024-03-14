﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArray.h                                                  (C) 2000-2024 */
/*                                                                           */
/* Tableaux multi-dimensionnel pour les types numériques.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMARRAY_H
#define ARCANE_UTILS_NUMARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MDSpan.h"
#include "arcane/utils/MDDim.h"
#include "arcane/utils/NumArrayContainer.h"
#include "arcane/utils/ArcaneCxx20.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableaux multi-dimensionnels pour les types numériques accessibles
 * sur accélérateurs.
 *
 * L'implémentation actuelle supporte des tableaux jusqu'à 4 dimensions. L'accès
 * aux éléments se fait via l'opérateur 'operator()'.
 *
 * \warning Le redimensionnement via resize() ne conserve pas les valeurs existantes
 * sauf pour les tableaux de rang 1.
 *
 * \warning Cette classe utilise par défaut un allocateur spécifique qui permet de
 * rendre accessible ces valeurs à la fois sur l'hôte (CPU) et l'accélérateur.
 * Néanmoins, il faut pour cela que le runtime associé à l'accélérateur ait été
 * initialisé (\ref arcanedoc_parallel_accelerator). C'est pourquoi il ne faut pas
 * utiliser de variables globales de cette classe ou d'une classe dérivée.
 *
 * Pour plus d'informations, se reporter à la page \ref arcanedoc_core_types_numarray.
 */
template <typename DataType, typename Extents, typename LayoutPolicy>
class NumArray
: private impl::NumArrayBaseCommon
{
 public:

  using ExtentsType = Extents;
  using ThatClass = NumArray<DataType, Extents, LayoutPolicy>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;
  using ConstMDSpanType = MDSpan<const DataType, ExtentsType, LayoutPolicy>;
  using MDSpanType = MDSpan<DataType, ExtentsType, LayoutPolicy>;
  using ArrayWrapper = impl::NumArrayContainer<DataType>;
  using ArrayBoundsIndexType = typename MDSpanType::ArrayBoundsIndexType;
  using value_type = DataType;
  using LayoutPolicyType = LayoutPolicy;

  using ConstSpanType ARCANE_DEPRECATED_REASON("Use 'ConstMDSpanType' instead") = ConstMDSpanType;
  using SpanType ARCANE_DEPRECATED_REASON("Use 'MDSpanType' instead") = MDSpanType;

 public:

  //! Nombre de dimensions du tableau
  static constexpr int rank() { return Extents::rank(); }

 public:

  //! Construit un tableau vide
  NumArray()
  {
    _resizeInit();
  }

  //! Construit un tableau en spécifiant directement la liste des dimensions
  explicit NumArray(DynamicDimsType extents)
  {
    resize(extents);
  }

  //! Construit un tableau en spécifiant directement la liste des dimensions
  NumArray(const DynamicDimsType& extents, eMemoryRessource r)
  : m_data(r)
  {
    resize(extents);
  }
  //! Créé un tableau vide utilisant la ressource mémoire \a r
  explicit NumArray(eMemoryRessource r)
  : m_data(r)
  {
    _resizeInit();
  }

  //! Construit un tableau avec 4 valeurs dynamiques
  NumArray(Int32 dim1_size, Int32 dim2_size,
           Int32 dim3_size, Int32 dim4_size) requires(Extents::nb_dynamic == 4)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size, dim3_size, dim4_size))
  {
  }

  //! Construit un tableau avec 4 valeurs dynamiques
  NumArray(Int32 dim1_size, Int32 dim2_size,
           Int32 dim3_size, Int32 dim4_size, eMemoryRessource r) requires(Extents::nb_dynamic == 4)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size, dim3_size, dim4_size), r)
  {
  }

  //! Construit un tableau avec 3 valeurs dynamiques
  NumArray(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size) requires(Extents::nb_dynamic == 3)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size, dim3_size))
  {
  }
  //! Construit un tableau avec 3 valeurs dynamiques
  NumArray(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size, eMemoryRessource r) requires(Extents::nb_dynamic == 3)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size, dim3_size), r)
  {
  }

  //! Construit un tableau avec 2 valeurs dynamiques
  NumArray(Int32 dim1_size, Int32 dim2_size) requires(Extents::nb_dynamic == 2)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size))
  {
  }
  //! Construit un tableau avec 2 valeurs dynamiques
  NumArray(Int32 dim1_size, Int32 dim2_size, eMemoryRessource r) requires(Extents::nb_dynamic == 2)
  : ThatClass(DynamicDimsType(dim1_size, dim2_size), r)
  {
  }

  //! Construit un tableau avec 1 valeur dynamique
  explicit NumArray(Int32 dim1_size) requires(Extents::nb_dynamic == 1)
  : ThatClass(DynamicDimsType(dim1_size))
  {
  }
  //! Construit un tableau avec 1 valeur dynamique
  NumArray(Int32 dim1_size, eMemoryRessource r) requires(Extents::nb_dynamic == 1)
  : ThatClass(DynamicDimsType(dim1_size), r)
  {
  }

  /*!
   * \brief Construit un tableau à partir de valeurs prédéfinies (tableaux 2D dynamiques).
   *
   * Les valeurs sont rangées de manière contigues en mémoire donc
   * la liste \a alist doit avoir un layout qui correspond à celui de cette classe.
   */
  NumArray(Int32 dim1_size, Int32 dim2_size, std::initializer_list<DataType> alist)
  requires(Extents::is_full_dynamic() && Extents::rank() == 2)
  : NumArray(dim1_size, dim2_size)
  {
    this->m_data.copyInitializerList(alist);
  }

  //! Construit un tableau à partir de valeurs prédéfinies (uniquement tableaux 1D dynamiques)
  NumArray(Int32 dim1_size, std::initializer_list<DataType> alist)
  requires(Extents::is_full_dynamic() && Extents::rank() == 1)
  : NumArray(dim1_size)
  {
    this->m_data.copyInitializerList(alist);
  }

  //! Construit une instance à partir d'une vue (uniquement tableaux 1D dynamiques)
  NumArray(SmallSpan<const DataType> v)
  requires(Extents::is_full_dynamic() && Extents::rank() == 1)
  : NumArray(v.size())
  {
    this->m_data.copy(v);
  }

  //! Construit une instance à partir d'une vue (uniquement tableaux 1D dynamiques)
  NumArray(Span<const DataType> v)
  requires(Extents::is_full_dynamic() && Extents::rank() == 1)
  : NumArray(arcaneCheckArraySize(v.size()))
  {
    this->m_data.copy(v);
  }

  NumArray(const ThatClass& rhs)
  : m_span(rhs.m_span)
  , m_data(rhs.m_data)
  , m_total_nb_element(rhs.m_total_nb_element)
  {
    _updateSpanPointerFromData();
  }

  NumArray(ThatClass&& rhs)
  : m_span(rhs.m_span)
  , m_data(std::move(rhs.m_data))
  , m_total_nb_element(rhs.m_total_nb_element)
  {
  }

  ThatClass& operator=(ThatClass&&) = default;

  /*!
   * \brief Opérateur de recopie.
   *
   * \warning Après appel à cette méthode, la ressource mémoire de l'instance
   * sera celle de \a rhs. Si on souhaite faire une recopie en conservant la
   * ressource mémoire associée il faut utiliser copy().
   */
  ThatClass& operator=(const ThatClass& rhs)
  {
    if (&rhs == this)
      return (*this);
    eMemoryRessource r = memoryRessource();
    eMemoryRessource rhs_r = rhs.memoryRessource();
    if (rhs_r != r)
      m_data = ArrayWrapper(rhs_r);
    this->copy(rhs);
    return (*this);
  }

  /*!
   * \brief Échange les données avec \a rhs.
   *
   * \warning L'allocateur mémoire est aussi échangé. Il est donc
   * préférable que les deux NumArray utilisent le même allocateur
   * et le même memoryRessource().
   */
  void swap(ThatClass& rhs)
  {
    m_data.swap(rhs.m_data);
    std::swap(m_span, rhs.m_span);
    std::swap(m_total_nb_element, rhs.m_total_nb_element);
  }

 public:

  //! Nombre total d'éléments du tableau
  constexpr Int64 totalNbElement() const { return m_total_nb_element; }
  //! Nombre de dimensions
  static constexpr Int32 nbDimension() { return Extents::rank(); }
  //! Valeurs des dimensions
  ArrayExtents<Extents> extents() const { return m_span.extents(); }
  ArrayExtentsWithOffset<Extents, LayoutPolicy> extentsWithOffset() const
  {
    return m_span.extentsWithOffset();
  }
  Int64 capacity() const { return m_data.capacity(); }
  eMemoryRessource memoryRessource() const { return m_data.memoryRessource(); }
  //! Vue sous forme d'octets
  Span<std::byte> bytes() { return asWritableBytes(to1DSpan()); }
  //! Vue constante forme d'octets
  Span<const std::byte> bytes() const { return asBytes(to1DSpan()); }

  //! Allocateur mémoire associé
  IMemoryAllocator* memoryAllocator() const { return m_data.allocator(); }

 public:

  //! Valeur de la première dimension
  constexpr Int32 dim1Size() const requires(Extents::rank() >= 1) { return m_span.extent0(); }
  //! Valeur de la deuxième dimension
  constexpr Int32 dim2Size() const requires(Extents::rank() >= 2) { return m_span.extent1(); }
  //! Valeur de la troisième dimension
  constexpr Int32 dim3Size() const requires(Extents::rank() >= 3) { return m_span.extent2(); }
  //! Valeur de la quatrième dimension
  constexpr Int32 dim4Size() const requires(Extents::rank() >= 4) { return m_span.extent3(); }

  //! Valeur de la première dimension
  constexpr Int32 extent0() const requires(Extents::rank() >= 1) { return m_span.extent0(); }
  //! Valeur de la deuxième dimension
  constexpr Int32 extent1() const requires(Extents::rank() >= 2) { return m_span.extent1(); }
  //! Valeur de la troisième dimension
  constexpr Int32 extent2() const requires(Extents::rank() >= 3) { return m_span.extent2(); }
  //! Valeur de la quatrième dimension
  constexpr Int32 extent3() const requires(Extents::rank() >= 4) { return m_span.extent3(); }

 public:

  /*!
   * \brief Modifie la taille du tableau.
   * \warning Les valeurs actuelles ne sont pas conservées lors de cette opération
   * et les nouvelles valeurs ne sont pas initialisées.
   */
  //@{
  void resize(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size, Int32 dim4_size) requires(Extents::nb_dynamic == 4)
  {
    this->resize(DynamicDimsType(dim1_size, dim2_size, dim3_size, dim4_size));
  }

  void resize(Int32 dim1_size, Int32 dim2_size, Int32 dim3_size) requires(Extents::nb_dynamic == 3)
  {
    this->resize(DynamicDimsType(dim1_size, dim2_size, dim3_size));
  }

  void resize(Int32 dim1_size, Int32 dim2_size) requires(Extents::nb_dynamic == 2)
  {
    this->resize(DynamicDimsType(dim1_size, dim2_size));
  }

  void resize(Int32 dim1_size) requires(Extents::nb_dynamic == 1)
  {
    this->resize(DynamicDimsType(dim1_size));
  }

  void resize(const DynamicDimsType& dims)
  {
    m_span.m_extents = dims;
    _resize();
  }
  //@}

 public:

  /*!
   * \brief Remplit les valeurs du tableau par \a v.
   *
   * \warning L'opération se fait sur l'hôte donc la mémoire associée
   * à l'instance doit être accessible sur l'hôte.
   */
  void fill(const DataType& v)
  {
    _checkHost(memoryRessource());
    m_data.fill(v);
  }
  /*!
   * \brief Remplit via la file \a queue, les valeurs du tableau d'indices
   * données par \a indexes par la valeur \a v .
   *
   * La mémoire associée à l'instance doit être accessible depuis la file \a queue.
   */
  void fill(const DataType& v, SmallSpan<const Int32> indexes, RunQueue* queue)
  {
    m_data.fill(v, indexes, queue);
  }

  /*!
   * \brief Remplit les éléments de l'instance la valeur \a v.
   */
  void fill(const DataType& v, RunQueue* queue)
  {
    m_data.fill(v, queue);
  }

 public:

  /*!
   * \brief Copie dans l'instance les valeurs de \a rhs.
   *
   * Cette opération est valide quelle que soit la mêmoire associée
   * associée à l'instance.
   */
  void copy(ConstMDSpanType rhs) { copy(rhs, nullptr); }

  /*!
   * \brief Copie dans l'instance les valeurs de \a rhs.
   *
   * Cette opération est valide quelle que soit la mêmoire associée
   * associée à l'instance.
   */
  void copy(const ThatClass& rhs) { copy(rhs, nullptr); }

  /*!
   * \brief Copie dans l'instance les valeurs de \a rhs via la file \a queue
   *
   * Cette opération est valide quelle que soit la mêmoire associée
   * associée à l'instance.
   * \a queue peut être nul. Si la file est asynchrone, il faudra la
   * synchroniser avant de pouvoir utiliser l'instance.
   */
  void copy(ConstMDSpanType rhs, RunQueue* queue)
  {
    _resizeAndCopy(rhs, eMemoryRessource::Unknown, queue);
  }

  /*!
   * \brief Copie dans l'instance les valeurs de \a rhs via la file \a queue
   *
   * Cette opération est valide quelle que soit la mêmoire associée
   * associée à l'instance.
   * \a queue peut être nul. Si la file est asynchrone, il faudra la
   * synchroniser avant de pouvoir utiliser l'instance.
   */
  void copy(const ThatClass& rhs, RunQueue* queue)
  {
    _resizeAndCopy(rhs.constSpan(), rhs.memoryRessource(), queue);
  }

 public:

  //! Récupère une référence pour l'élément \a i
  DataType& operator[](Int32 i) requires(Extents::rank() == 1) { return m_span(i); }
  //! Valeur pour l'élément \a i
  DataType operator[](Int32 i) const requires(Extents::rank() == 1) { return m_span(i); }

 public:

  //! Valeur pour l'élément \a i,j,k,l
  DataType operator()(Int32 i, Int32 j, Int32 k, Int32 l) const requires(Extents::rank() == 4)
  {
    return m_span(i, j, k, l);
  }
  //! Positionne la valeur pour l'élément \a i,j,k,l
  DataType& operator()(Int32 i, Int32 j, Int32 k, Int32 l) requires(Extents::rank() == 4)
  {
    return m_span(i, j, k, l);
  }

  //! Valeur pour l'élément \a i,j,k
  DataType operator()(Int32 i, Int32 j, Int32 k) const requires(Extents::rank() == 3)
  {
    return m_span(i, j, k);
  }
  //! Positionne la valeur pour l'élément \a i,j,k
  DataType& operator()(Int32 i, Int32 j, Int32 k) requires(Extents::rank() == 3)
  {
    return m_span(i, j, k);
  }

  //! Valeur pour l'élément \a i,j
  DataType operator()(Int32 i, Int32 j) const requires(Extents::rank() == 2)
  {
    return m_span(i, j);
  }
  //! Positionne la valeur pour l'élément \a i,j
  DataType& operator()(Int32 i, Int32 j) requires(Extents::rank() == 2)
  {
    return m_span(i, j);
  }
  //! Valeur pour l'élément \a i
  DataType operator()(Int32 i) const requires(Extents::rank() == 1) { return m_span(i); }
  //! Positionne la valeur pour l'élément \a i
  DataType& operator()(Int32 i) requires(Extents::rank() == 1) { return m_span(i); }

 public:

  //! Référence constante pour l'élément \a idx
  const DataType& operator()(ArrayBoundsIndexType idx) const
  {
    return m_span(idx);
  }
  //! Référence modifiable l'élément \a idx
  DataType& operator()(ArrayBoundsIndexType idx)
  {
    return m_span(idx);
  }

 public:

  // TODO: rendre obsolète
  //! Positionne la valeur pour l'élément \a i,j,k,l
  ARCANE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(Int32 i, Int32 j, Int32 k, Int32 l) requires(Extents::rank() == 4)
  {
    return m_span(i, j, k, l);
  }
  //! Positionne la valeur pour l'élément \a i,j,k
  ARCANE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(Int32 i, Int32 j, Int32 k) requires(Extents::rank() == 3)
  {
    return m_span(i, j, k);
  }
  //! Positionne la valeur pour l'élément \a i,j
  ARCANE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(Int32 i, Int32 j) requires(Extents::rank() == 2)
  {
    return m_span(i, j);
  }
  //! Positionne la valeur pour l'élément \a i
  ARCANE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(Int32 i) requires(Extents::rank() == 1) { return m_span(i); }

  //! Positionne la valeur pour l'élément \a idx
  ARCANE_DEPRECATED_REASON("Y2023: Use operator() instead")
  DataType& s(ArrayBoundsIndexType idx)
  {
    return m_span(idx);
  }

 public:

  MDSpanType span() { return m_span; }
  ConstMDSpanType span() const { return m_span.constMDSpan(); }
  ConstMDSpanType constSpan() const { return m_span.constMDSpan(); }

  MDSpanType mdspan() { return m_span; }
  ConstMDSpanType mdspan() const { return m_span.constMDSpan(); }
  ConstMDSpanType constMDSpan() const { return m_span.constMDSpan(); }

  Span<const DataType> to1DSpan() const { return m_span.to1DSpan(); }
  Span<DataType> to1DSpan() { return m_span.to1DSpan(); }

  constexpr operator MDSpanType() { return this->span(); }
  constexpr operator ConstMDSpanType() const { return this->constSpan(); }
  constexpr operator SmallSpan<DataType>() requires(Extents::rank() == 1) { return this->to1DSpan().smallView(); }
  constexpr operator SmallSpan<const DataType>() const requires(Extents::rank() == 1) { return this->to1DSpan().constSmallView(); }

  constexpr SmallSpan<DataType> to1DSmallSpan() requires(Extents::rank() == 1) { return m_span.to1DSmallSpan(); }
  constexpr SmallSpan<const DataType> to1DSmallSpan() const requires(Extents::rank() == 1) { return m_span.to1DSmallSpan(); }
  constexpr SmallSpan<const DataType> to1DConstSmallSpan() const requires(Extents::rank() == 1) { return m_span.to1DConstSmallSpan(); }

 public:

  //! \internal
  DataType* _internalData() { return m_span._internalData(); }

 private:

  MDSpanType m_span;
  ArrayWrapper m_data;
  Int64 m_total_nb_element = 0;

 private:

  void _updateSpanPointerFromData()
  {
    m_span.m_ptr = m_data.to1DSpan().data();
  }

  void _resizeAndCopy(ConstMDSpanType rhs, eMemoryRessource input_ressource, RunQueue* queue)
  {
    this->resize(rhs.extents().dynamicExtents());
    m_data.copyOnly(rhs.to1DSpan(), input_ressource, queue);
    _updateSpanPointerFromData();
  }

  //! Redimensionne le tableau à partir des valeurs de \a m_span.extents()
  void _resize()
  {
    m_total_nb_element = m_span.extents().totalNbElement();
    m_data.resize(m_total_nb_element);
    _updateSpanPointerFromData();
  }

  /*!
   * \brief Allocation éventuelle lors de l'initialisation.
   *
   * Il y a besoin de faire une allocation lors de l'initialisation
   * avec le constructeur par défaut dans le cas où toutes les
   * dimensions sont statiques.
   */
  void _resizeInit()
  {
    if constexpr (ExtentsType::nb_dynamic == 0) {
      resize(DynamicDimsType());
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
