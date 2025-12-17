// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractArray.h                                             (C) 2000-2025 */
/*                                                                           */
/* Classe de base des tableaux.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ABSTRACTARRAY_H
#define ARCCORE_COMMON_ABSTRACTARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Span.h"

#include "arccore/common/ArrayTraits.h"
#include "arccore/common/ArrayMetaData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base interne pour les tableaux.
 *
 * Cette classe gère uniquement les meta-données pour les tableaux comme
 * le nombre d'éléments ou la capacité.
 *
 * \a m_md est un pointeur contenant les meta-donné du tableau. Si le
 * tableau est partagé (SharedArray, SharedArray2), alors ce pointeur
 * est alloué dynamiquement et dans ce cas _isUseOwnMetaData() doit
 * retourner \a false. Si le tableau n'est pas partagé (UniqueArray ou
 * UniqueArray2), alors les meta-données sont conservées directement
 * dans l'instance du tableau pour éviter des allocations inutiles
 * et \a m_md pointe alors vers \a m_meta_data. Dans tous les cas, il
 * ne faut pas utiliser \a m_meta_data directement, mais toujours passer
 * par \a m_md.
 */
class ARCCORE_COMMON_EXPORT AbstractArrayBase
{
 public:

  AbstractArrayBase()
  {
    m_md = &m_meta_data;
  }
  virtual ~AbstractArrayBase() = default;

 public:

  IMemoryAllocator* allocator() const
  {
    return m_md->allocation_options.allocator();
  }
  MemoryAllocationOptions allocationOptions() const
  {
    return m_md->allocation_options;
  }
  /*!
   * \brief Positionne le nom du tableau pour les informations de debug.
   *
   * Ce nom peut être utilisé par exemple pour les affichages listing.
   */
  void setDebugName(const String& name);
  //! Nom de debug (nul si aucun nom spécifié)
  String debugName() const;

 protected:

  ArrayMetaData* m_md = nullptr;
  ArrayMetaData m_meta_data;

 protected:

  //! Méthode explicite pour une RunQueue nulle.
  static constexpr RunQueue* _nullRunQueue() { return nullptr; }

  /*!
   * \brief Indique si \a m_md fait référence à \a m_meta_data.
   *
   * C'est le cas pour les UniqueArray et UniqueArray2 mais
   * pas pour les SharedArray et SharedArray2.
   */
  virtual bool _isUseOwnMetaData() const
  {
    return true;
  }

 protected:

  void _swapMetaData(AbstractArrayBase& rhs)
  {
    std::swap(m_md, rhs.m_md);
    std::swap(m_meta_data, rhs.m_meta_data);
    _checkSetUseOwnMetaData();
    rhs._checkSetUseOwnMetaData();
  }

  void _copyMetaData(const AbstractArrayBase& rhs)
  {
    // Déplace les meta-données
    // Attention si on utilise m_meta_data alors il
    // faut positionner m_md pour qu'il pointe vers notre propre m_meta_data.
    m_meta_data = rhs.m_meta_data;
    m_md = rhs.m_md;
    _checkSetUseOwnMetaData();
  }

  void _allocateMetaData()
  {
#ifdef ARCCORE_CHECK
    if (m_md->is_not_null)
      ArrayMetaData::throwNullExpected();
#endif
    if (_isUseOwnMetaData()) {
      m_meta_data = ArrayMetaData();
      m_md = &m_meta_data;
    }
    else {
      m_md = new ArrayMetaData();
      m_md->is_allocated_by_new = true;
    }
    m_md->is_not_null = true;
  }

  void _deallocateMetaData(ArrayMetaData* md)
  {
    if (md->is_allocated_by_new)
      delete md;
    else
      *md = ArrayMetaData();
  }

  void _checkValidSharedArray()
  {
#ifdef ARCCORE_CHECK
    if (m_md->is_not_null && !m_md->is_allocated_by_new)
      ArrayMetaData::throwInvalidMetaDataForSharedArray();
#endif
  }

 private:

  void _checkSetUseOwnMetaData()
  {
    if (!m_md->is_allocated_by_new)
      m_md = &m_meta_data;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Classe abstraite de base d'un vecteur.
 *
 * Cette classe ne peut pas être utilisée directement. Pour utiliser un
 * vecteur, choisissez la classe SharedArray ou UniqueArray.
 */
template <typename T>
class AbstractArray
: public AbstractArrayBase
{
 public:

  typedef typename ArrayTraits<T>::ConstReferenceType ConstReferenceType;
  typedef typename ArrayTraits<T>::IsPODType IsPODType;
  typedef AbstractArray<T> ThatClassType;
  using TrueImpl = T;

 public:

  //! Type des éléments du tableau
  typedef T value_type;
  //! Type pointeur d'un élément du tableau
  typedef value_type* pointer;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type de l'itérateur sur un élément du tableau
  typedef ArrayIterator<pointer> iterator;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef ArrayIterator<const_pointer> const_iterator;
  //! Type référence d'un élément du tableau
  typedef value_type& reference;
  //! Type référence constante d'un élément du tableau
  typedef ConstReferenceType const_reference;
  //! Type indexant le tableau
  typedef Int64 size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef ptrdiff_t difference_type;

  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

 protected:

  //! Construit un vecteur vide avec l'allocateur par défaut
  AbstractArray()
  {
  }
  //! Constructeur par déplacement. Ne doit être utilisé que par UniqueArray
  AbstractArray(ThatClassType&& rhs) ARCCORE_NOEXCEPT
  : m_ptr(rhs.m_ptr)
  {
    _copyMetaData(rhs);
    rhs._reset();
  }

  ~AbstractArray() override
  {
    --m_md->nb_ref;
    _checkFreeMemory();
  }

 public:

  AbstractArray(const AbstractArray<T>& rhs) = delete;
  AbstractArray<T>& operator=(const AbstractArray<T>& rhs) = delete;

 protected:

  static constexpr Int64 typeSize() { return static_cast<Int64>(sizeof(T)); }
  AllocatedMemoryInfo _currentMemoryInfo() const
  {
    return AllocatedMemoryInfo(m_ptr, m_md->size * typeSize(), m_md->capacity * typeSize());
  }

 protected:

  /*!
   * \brief Initialise le tableau avec la vue \a view.
   *
   * Cette méthode ne doit être appelée que dans un constructeur de la classe dérivée.
   */
  void _initFromSpan(const Span<const T>& view)
  {
    Int64 asize = view.size();
    if (asize != 0) {
      _internalAllocate(asize, _nullRunQueue());
      _createRange(0, asize, view.data());
      m_md->size = asize;
    }
  }

  /*!
   * \brief Construit un vecteur vide avec un allocateur spécifique \a a.
   *
   * Si \a acapacity n'est pas nul, la mémoire est allouée pour
   * contenir \a acapacity éléments (mais le tableau reste vide).
   *
   * Cette méthode ne doit être appelée que dans un constructeur de la classe dérivée
   * et uniquement par les classes utilisant une sémantique à la UniqueArray.
   */
  void _initFromAllocator(MemoryAllocationOptions o, Int64 acapacity,
                          void* pre_allocated_buffer = nullptr)
  {
    _directFirstAllocateWithAllocator(acapacity, o, pre_allocated_buffer);
  }

 public:

  //! Libère la mémoire utilisée par le tableau.
  void dispose()
  {
    _destroy();
    MemoryAllocationOptions options(m_md->allocation_options);
    _internalDeallocate();
    _setToSharedNull();
    // Si on a un allocateur spécifique, il faut allouer un
    // bloc pour conserver cette information.
    if (options.allocator() != m_md->_allocator())
      _directFirstAllocateWithAllocator(0, options);
    _updateReferences();
  }

 public:

  operator ConstArrayView<T>() const
  {
    return ConstArrayView<T>(ARCCORE_CAST_SMALL_SIZE(size()), m_ptr);
  }
  operator Span<const T>() const
  {
    return Span<const T>(m_ptr, m_md->size);
  }
  operator SmallSpan<const T>() const
  {
    return SmallSpan<const T>(m_ptr, ARCCORE_CAST_SMALL_SIZE(size()));
  }

 public:

  //! Nombre d'éléments du vecteur
  Integer size() const { return ARCCORE_CAST_SMALL_SIZE(m_md->size); }
  //! Nombre d'éléments du vecteur
  Integer length() const { return ARCCORE_CAST_SMALL_SIZE(m_md->size); }
  //! Capacité (nombre d'éléments alloués) du vecteur
  Integer capacity() const { return ARCCORE_CAST_SMALL_SIZE(m_md->capacity); }
  //! Nombre d'éléments du vecteur (en 64 bits)
  Int64 largeSize() const { return m_md->size; }
  //! Nombre d'éléments du vecteur (en 64 bits)
  Int64 largeLength() const { return m_md->size; }
  //! Capacité (nombre d'éléments alloués) du vecteur (en 64 bits)
  Int64 largeCapacity() const { return m_md->capacity; }
  //! Capacité (nombre d'éléments alloués) du vecteur
  bool empty() const { return m_md->size == 0; }
  //! Vrai si le tableau contient l'élément de valeur \a v
  bool contains(ConstReferenceType v) const
  {
    const T* ptr = m_ptr;
    for (Int64 i = 0, n = m_md->size; i < n; ++i) {
      if (ptr[i] == v)
        return true;
    }
    return false;
  }

 public:

  //! Elément d'indice \a i
  ConstReferenceType operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Elément d'indice \a i
  ConstReferenceType operator()(Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }

 public:

  //! Modifie les informations sur la localisation mémoire
  void setMemoryLocationHint(eMemoryLocationHint new_hint)
  {
    m_md->_setMemoryLocationHint(new_hint, m_ptr, sizeof(T));
  }

  /*!
   * \brief Positionne l'emplacement physique de la zone mémoire.
   *
   * \warning L'appelant doit garantir la cohérence entre l'allocateur
   * et la zone mémoire spécifiée.
   */
  void _internalSetHostDeviceMemoryLocation(eHostDeviceMemoryLocation location)
  {
    m_md->_setHostDeviceMemoryLocation(location);
  }

  //! Positionne l'emplacement physique de la zone mémoire.
  eHostDeviceMemoryLocation hostDeviceMemoryLocation() const
  {
    return m_md->allocation_options.hostDeviceMemoryLocation();
  }

 public:

  friend bool operator==(const AbstractArray<T>& rhs, const AbstractArray<T>& lhs)
  {
    return operator==(Span<const T>(rhs), Span<const T>(lhs));
  }

  friend bool operator!=(const AbstractArray<T>& rhs, const AbstractArray<T>& lhs)
  {
    return !(rhs == lhs);
  }

  friend bool operator==(const AbstractArray<T>& rhs, const Span<const T>& lhs)
  {
    return operator==(Span<const T>(rhs), lhs);
  }

  friend bool operator!=(const AbstractArray<T>& rhs, const Span<const T>& lhs)
  {
    return !(rhs == lhs);
  }

  friend bool operator==(const Span<const T>& rhs, const AbstractArray<T>& lhs)
  {
    return operator==(rhs, Span<const T>(lhs));
  }

  friend bool operator!=(const Span<const T>& rhs, const AbstractArray<T>& lhs)
  {
    return !(rhs == lhs);
  }

  friend std::ostream& operator<<(std::ostream& o, const AbstractArray<T>& val)
  {
    o << Span<const T>(val);
    return o;
  }

 private:

  using AbstractArrayBase::m_meta_data;

 protected:

  // NOTE: Ces deux champs sont utilisés pour l'affichage TTF de totalview.
  // Si on modifie leur ordre il faut mettre à jour la partie correspondante
  // dans l'afficheur totalview de Arcane.
  T* m_ptr = nullptr;

 protected:

  //! Réserve le mémoire pour \a new_capacity éléments
  void _reserve(Int64 new_capacity)
  {
    if (new_capacity <= m_md->capacity)
      return;
    _internalRealloc(new_capacity, false);
  }
  /*!
   * \brief Réalloue le tableau pour une nouvelle capacité égale à \a new_capacity.
   *
   * Si la nouvelle capacité est inférieure à l'ancienne, rien ne se passe.
   */
  template <typename PodType>
  void _internalRealloc(Int64 new_capacity, bool compute_capacity, PodType pod_type, RunQueue* queue = nullptr)
  {
    if (_isSharedNull()) {
      if (new_capacity != 0)
        _internalAllocate(new_capacity, queue);
      return;
    }

    Int64 acapacity = new_capacity;
    if (compute_capacity) {
      acapacity = m_md->capacity;
      //std::cout << " REALLOC: want=" << wanted_size << " current_capacity=" << capacity << '\n';
      while (new_capacity > acapacity)
        acapacity = (acapacity == 0) ? 4 : (acapacity + 1 + acapacity / 2);
      //std::cout << " REALLOC: want=" << wanted_size << " new_capacity=" << capacity << '\n';
    }
    // Si la nouvelle capacité est inférieure à la courante,ne fait rien.
    if (acapacity <= m_md->capacity)
      return;
    _internalReallocate(acapacity, pod_type, queue);
  }

  void _internalRealloc(Int64 new_capacity, bool compute_capacity)
  {
    _internalRealloc(new_capacity, compute_capacity, IsPODType());
  }

  //! Réallocation pour un type POD
  void _internalReallocate(Int64 new_capacity, TrueType, RunQueue* queue)
  {
    T* old_ptr = m_ptr;
    Int64 old_capacity = m_md->capacity;
    _directReAllocate(new_capacity, queue);
    bool update = (new_capacity < old_capacity) || (m_ptr != old_ptr);
    if (update) {
      _updateReferences();
    }
  }

  //! Réallocation pour un type complexe (non POD)
  void _internalReallocate(Int64 new_capacity, FalseType, RunQueue* queue)
  {
    T* old_ptr = m_ptr;
    ArrayMetaData* old_md = m_md;
    AllocatedMemoryInfo old_mem_info = _currentMemoryInfo();
    Int64 old_size = m_md->size;
    _directAllocate(new_capacity, queue);
    if (m_ptr != old_ptr) {
      for (Int64 i = 0; i < old_size; ++i) {
        new (m_ptr + i) T(old_ptr[i]);
        old_ptr[i].~T();
      }
      m_md->nb_ref = old_md->nb_ref;
      m_md->_deallocate(old_mem_info, queue);
      _updateReferences();
    }
  }
  // Libère la mémoire
  void _internalDeallocate(RunQueue* queue = nullptr)
  {
    if (!_isSharedNull())
      m_md->_deallocate(_currentMemoryInfo(), queue);
    if (m_md->is_not_null)
      _deallocateMetaData(m_md);
  }
  void _internalAllocate(Int64 new_capacity, RunQueue* queue)
  {
    _directAllocate(new_capacity, queue);
    m_md->nb_ref = _getNbRef();
    _updateReferences();
  }

  void _copyFromMemory(const T* source)
  {
    m_md->_copyFromMemory(m_ptr, source, sizeof(T), _nullRunQueue());
  }

 private:

  /*!
   * \brief Effectue la première allocation.
   *
   * Si \a pre_allocated_buffer est non nul, on l'utilise comme buffer
   * pour la première allocation. C'est à l'appelant de s'assurer que
   * ce buffer est valide pour la capacité demandée. Le \a pre_allocated_buffer
   * est utilisé notamment par l'allocateur de SmallArray.
   */
  void _directFirstAllocateWithAllocator(Int64 new_capacity, MemoryAllocationOptions options,
                                         void* pre_allocated_buffer = nullptr)
  {
    IMemoryAllocator* wanted_allocator = options.allocator();
    if (!wanted_allocator) {
      wanted_allocator = ArrayMetaData::_defaultAllocator();
      options.setAllocator(wanted_allocator);
    }
    _allocateMetaData();
    m_md->allocation_options = options;
    if (new_capacity > 0) {
      if (!pre_allocated_buffer)
        _allocateMP(new_capacity, options.runQueue());
      else
        _setMPCast(pre_allocated_buffer);
    }
    m_md->nb_ref = _getNbRef();
    m_md->size = 0;
    _updateReferences();
  }

  void _directAllocate(Int64 new_capacity, RunQueue* queue)
  {
    if (!m_md->is_not_null)
      _allocateMetaData();
    _allocateMP(new_capacity, queue);
  }

  void _allocateMP(Int64 new_capacity, RunQueue* queue)
  {
    _setMPCast(m_md->_allocate(new_capacity, typeSize(), queue));
  }

  void _directReAllocate(Int64 new_capacity, RunQueue* queue)
  {
    _setMPCast(m_md->_reallocate(_currentMemoryInfo(), new_capacity, typeSize(), queue));
  }

 public:

  void printInfos(std::ostream& o)
  {
    o << " Infos: size=" << m_md->size << " capacity=" << m_md->capacity << '\n';
  }

 protected:

  //! Mise à jour des références
  virtual void _updateReferences()
  {
  }
  //! Mise à jour des références
  virtual Integer _getNbRef()
  {
    return 1;
  }
  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void _addRange(ConstReferenceType val, Int64 n)
  {
    Int64 s = m_md->size;
    if ((s + n) > m_md->capacity)
      _internalRealloc(s + n, true);
    for (Int64 i = 0; i < n; ++i)
      new (m_ptr + s + i) T(val);
    m_md->size += n;
  }

  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void _addRange(Span<const T> val)
  {
    Int64 n = val.size();
    const T* ptr = val.data();
    Int64 s = m_md->size;
    if ((s + n) > m_md->capacity)
      _internalRealloc(s + n, true);
    _createRange(s, s + n, ptr);
    m_md->size += n;
  }

  //! Détruit l'instance si plus personne ne la référence
  void _checkFreeMemory()
  {
    if (m_md->nb_ref == 0) {
      _destroy();
      _internalDeallocate(_nullRunQueue());
    }
  }
  void _destroy()
  {
    _destroyRange(0, m_md->size, IsPODType());
  }
  void _destroyRange(Int64, Int64, TrueType)
  {
    // Rien à faire pour un type POD.
  }
  void _destroyRange(Int64 abegin, Int64 aend, FalseType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      m_ptr[i].~T();
  }
  void _createRangeDefault(Int64, Int64, TrueType)
  {
  }
  void _createRangeDefault(Int64 abegin, Int64 aend, FalseType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      new (m_ptr + i) T();
  }
  void _createRange(Int64 abegin, Int64 aend, ConstReferenceType value, TrueType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      m_ptr[i] = value;
  }
  void _createRange(Int64 abegin, Int64 aend, ConstReferenceType value, FalseType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      new (m_ptr + i) T(value);
  }
  void _createRange(Int64 abegin, Int64 aend, const T* values)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i) {
      new (m_ptr + i) T(*values);
      ++values;
    }
  }
  void _fill(ConstReferenceType value)
  {
    for (Int64 i = 0, n = size(); i < n; ++i)
      m_ptr[i] = value;
  }
  void _clone(const ThatClassType& orig_array)
  {
    Int64 that_size = orig_array.size();
    _internalAllocate(that_size, _nullRunQueue());
    m_md->size = that_size;
    m_md->dim1_size = orig_array.m_md->dim1_size;
    m_md->dim2_size = orig_array.m_md->dim2_size;
    _createRange(0, that_size, orig_array.m_ptr);
  }
  template <typename PodType>
  void _resizeHelper(Int64 s, PodType pod_type, RunQueue* queue)
  {
    if (s < 0)
      s = 0;
    if (s > m_md->size) {
      this->_internalRealloc(s, false, pod_type, queue);
      this->_createRangeDefault(m_md->size, s, pod_type);
    }
    else {
      this->_destroyRange(s, m_md->size, pod_type);
    }
    m_md->size = s;
  }
  void _resize(Int64 s)
  {
    _resizeHelper(s, IsPODType(), _nullRunQueue());
  }
  //! Redimensionne sans initialiser les nouvelles valeurs
  void _resizeNoInit(Int64 s, RunQueue* queue = nullptr)
  {
    _resizeHelper(s, TrueType{}, queue);
  }
  void _clear()
  {
    this->_destroyRange(0, m_md->size, IsPODType());
    m_md->size = 0;
  }
  //! Redimensionne et remplit les nouvelles valeurs avec \a value
  void _resize(Int64 s, ConstReferenceType value)
  {
    if (s < 0)
      s = 0;
    if (s > m_md->size) {
      this->_internalRealloc(s, false);
      this->_createRange(m_md->size, s, value, IsPODType());
    }
    else {
      this->_destroyRange(s, m_md->size, IsPODType());
    }
    m_md->size = s;
  }
  void _copy(const T* rhs_begin, TrueType)
  {
    _copyFromMemory(rhs_begin);
  }
  void _copy(const T* rhs_begin, FalseType)
  {
    for (Int64 i = 0, n = m_md->size; i < n; ++i)
      m_ptr[i] = rhs_begin[i];
  }
  void _copy(const T* rhs_begin)
  {
    _copy(rhs_begin, IsPODType());
  }

  /*!
   * \brief Redimensionne l'instance et recopie les valeurs de \a rhs.
   *
   * Si la taille diminue, les éléments compris entre size() et rhs.size()
   * sont détruits.
   *
   * \post size()==rhs.size()
   */
  void _resizeAndCopyView(Span<const T> rhs)
  {
    const T* rhs_begin = rhs.data();
    Int64 rhs_size = rhs.size();
    const Int64 current_size = m_md->size;
    T* abegin = m_ptr;
    // Vérifie que \a rhs n'est pas un élément à l'intérieur de ce tableau
    if (abegin >= rhs_begin && abegin < (rhs_begin + rhs_size))
      ArrayMetaData::overlapError(abegin, m_md->size, rhs_begin, rhs_size);

    if (rhs_size > current_size) {
      this->_internalRealloc(rhs_size, false);
      // Crée les nouveaux éléments
      this->_createRange(m_md->size, rhs_size, rhs_begin + current_size);
      // Copie les éléments déjà existant
      _copy(rhs_begin);
      m_md->size = rhs_size;
    }
    else {
      this->_destroyRange(rhs_size, current_size, IsPODType{});
      m_md->size = rhs_size;
      _copy(rhs_begin);
    }
  }

  /*!
   * \brief Implémente l'opérateur d'assignement par déplacement.
   *
   * Cet appel n'est valide que pour les tableaux de type UniqueArray
   * qui n'ont qu'une seule référence. Les infos de \a rhs sont directement
   * copiés cette l'instance. En retour, \a rhs contient le tableau vide.
   */
  void _move(ThatClassType& rhs) ARCCORE_NOEXCEPT
  {
    if (&rhs == this)
      return;

    // Comme il n'y a qu'une seule référence sur le tableau actuel, on peut
    // directement libérer la mémoire.
    _destroy();
    _internalDeallocate(_nullRunQueue());

    _setMP(rhs.m_ptr);

    _copyMetaData(rhs);

    // Indique que \a rhs est vide.
    rhs._reset();
  }
  /*!
   * \brief Échange les valeurs de l'instance avec celles de \a rhs.
   *
   * Cet appel n'est valide que pour les tableaux de type UniqueArray
   * et l'échange se fait uniquement par l'échange des pointeurs. L'opération
   * est donc de complexité constante.
   */
  void _swap(ThatClassType& rhs) ARCCORE_NOEXCEPT
  {
    std::swap(m_ptr, rhs.m_ptr);
    _swapMetaData(rhs);
  }

  void _shrink()
  {
    _shrink(size());
  }

  // Réalloue la mémoire pour avoir une capacité proche de \a new_capacity
  void _shrink(Int64 new_capacity)
  {
    if (_isSharedNull())
      return;
    // On n'augmente pas la capacité avec cette méthode
    if (new_capacity > this->capacity())
      return;
    if (new_capacity < 4)
      new_capacity = 4;
    _internalReallocate(new_capacity, IsPODType(), _nullRunQueue());
  }

  /*!
   * \brief Réinitialise le tableau à un tableau vide.
   * \warning Cette méthode n'est valide que pour les UniqueArray et pas
   * les SharedArray.
   */
  void _reset()
  {
    _setToSharedNull();
  }

  constexpr Integer _clampSizeOffet(Int64 offset, Int32 asize) const
  {
    Int64 max_size = m_md->size - offset;
    if (asize > max_size)
      // On est certain de ne pas dépasser 32 bits car on est inférieur à asize.
      asize = static_cast<Integer>(max_size);
    return asize;
  }

  // Uniquement pour UniqueArray et UniqueArray2
  void _assignFromArray(const AbstractArray<T>& rhs)
  {
    if (&rhs == this)
      return;
    Span<const T> rhs_span(rhs);
    if (rhs.allocator() == this->allocator()) {
      _resizeAndCopyView(rhs_span);
    }
    else {
      _destroy();
      _internalDeallocate(_nullRunQueue());
      _reset();
      _initFromAllocator(rhs.allocationOptions(), 0);
      _initFromSpan(rhs_span);
    }
  }

 protected:

  void _setMP(TrueImpl* new_mp)
  {
    m_ptr = new_mp;
  }

  void _setMP2(TrueImpl* new_mp, ArrayMetaData* new_md)
  {
    _setMP(new_mp);
    // Il ne faut garder le nouveau m_md que s'il est alloué
    // sinon on risque d'avoir des références sur des objets temporaires
    m_md = new_md;
    if (!m_md->is_allocated_by_new)
      m_md = &m_meta_data;
  }

  bool _isSharedNull()
  {
    return m_ptr == nullptr;
  }

 private:

  void _setToSharedNull()
  {
    m_ptr = nullptr;
    m_meta_data = ArrayMetaData();
    m_md = &m_meta_data;
  }
  void _setMPCast(void* p)
  {
    _setMP(reinterpret_cast<TrueImpl*>(p));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
