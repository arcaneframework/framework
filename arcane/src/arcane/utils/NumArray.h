// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArray.h                                                  (C) 2000-2023 */
/*                                                                           */
/* Tableaux multi-dimensionnel pour les types numériques.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMARRAY_H
#define ARCANE_UTILS_NUMARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/MemoryRessource.h"
#include "arcane/utils/MDSpan.h"
#include "arcane/utils/MDDim.h"
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
 * \brief Wrapper de Arccore::Array pour la classe NumArray.
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
: public impl::NumArrayBaseCommon
{
 public:

  using ExtentsType = Extents;
  using ThatClass = NumArray<DataType, Extents, LayoutPolicy>;
  using DynamicDimsType = typename ExtentsType::DynamicDimsType;
  using ConstSpanType = MDSpan<const DataType, ExtentsType, LayoutPolicy>;
  using SpanType = MDSpan<DataType, ExtentsType, LayoutPolicy>;
  using ArrayWrapper = impl::NumArrayContainer<DataType>;
  using ArrayBoundsIndexType = typename SpanType::ArrayBoundsIndexType;
  using value_type = DataType;
  using LayoutPolicyType = LayoutPolicy;

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
  void copy(ConstSpanType rhs) { copy(rhs, nullptr); }

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
  void copy(ConstSpanType rhs, RunQueue* queue)
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

  //! Valeur pour l'élément \a i
  DataType operator()(Int32 i) const requires(Extents::rank() == 1) { return m_span(i); }
  //! Positionne la valeur pour l'élément \a i
  DataType& operator()(Int32 i) requires(Extents::rank() == 1) { return m_span(i); }
  //! Positionne la valeur pour l'élément \a i
  DataType& s(Int32 i) requires(Extents::rank() == 1) { return m_span(i); }
  //! Récupère une référence pour l'élément \a i
  DataType& operator[](Int32 i) requires(Extents::rank() == 1) { return m_span(i); }
  //! Valeur pour l'élément \a i
  DataType operator[](Int32 i) const requires(Extents::rank() == 1) { return m_span(i); }

 public:

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
  //! Positionne la valeur pour l'élément \a i,j
  DataType& s(Int32 i, Int32 j) requires(Extents::rank() == 2)
  {
    return m_span(i, j);
  }

 public:

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
  //! Positionne la valeur pour l'élément \a i,j,k
  DataType& s(Int32 i, Int32 j, Int32 k) requires(Extents::rank() == 3)
  {
    return m_span(i, j, k);
  }

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
  // TODO: rendre obsolète
  //! Positionne la valeur pour l'élément \a i,j,k,l
  DataType& s(Int32 i, Int32 j, Int32 k, Int32 l) requires(Extents::rank() == 4)
  {
    return m_span(i, j, k, l);
  }

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

  DataType& s(ArrayBoundsIndexType idx)
  {
    return m_span(idx);
  }

 public:

  SpanType span() { return m_span; }
  ConstSpanType span() const { return m_span.constSpan(); }
  ConstSpanType constSpan() const { return m_span.constSpan(); }
  Span<const DataType> to1DSpan() const { return m_span.to1DSpan(); }
  Span<DataType> to1DSpan() { return m_span.to1DSpan(); }

  constexpr operator SpanType() { return this->span(); }
  constexpr operator ConstSpanType() const { return this->constSpan(); }
  constexpr operator SmallSpan<DataType>() requires(Extents::rank() == 1) { return this->to1DSpan().smallView(); }
  constexpr operator SmallSpan<const DataType>() const requires(Extents::rank() == 1) { return this->to1DSpan().constSmallView(); }

 public:

  //! \internal
  DataType* _internalData() { return m_span._internalData(); }

 private:

  SpanType m_span;
  ArrayWrapper m_data;
  Int64 m_total_nb_element = 0;

 private:

  void _updateSpanPointerFromData()
  {
    m_span.m_ptr = m_data.to1DSpan().data();
  }

  void _resizeAndCopy(ConstSpanType rhs, eMemoryRessource input_ressource, RunQueue* queue)
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
