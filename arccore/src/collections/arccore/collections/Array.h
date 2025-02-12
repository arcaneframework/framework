// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Tableau 1D.                                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_ARRAY_H
#define ARCCORE_COLLECTIONS_ARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/Span.h"
#include "arccore/collections/MemoryAllocationOptions.h"
#include "arccore/collections/ArrayTraits.h"

#include <memory>
#include <initializer_list>
#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Meta-Données des tableaux.
 *
 * Cette classe sert pour contenir les meta-données communes à toutes les
 * implémentations qui dérivent de AbstractArray.
 *
 * Seules les classes qui implémentent une sémantique à la UniqueArray
 * ont le droit d'utiliser un allocateur autre que l'allocateur par défaut.
 */
class ARCCORE_COLLECTIONS_EXPORT ArrayMetaData
{
  // NOTE: Les champs de cette classe sont utilisés pour l'affichage TTF de totalview.
  // Si on modifie leur ordre il faut mettre à jour la copie de cette classe
  // dans l'afficheur totalview de Arcane.

  template <typename> friend class AbstractArray;
  template <typename> friend class Array2;
  template <typename> friend class Array;
  template <typename> friend class SharedArray;
  template <typename> friend class SharedArray2;
  friend class AbstractArrayBase;
  static IMemoryAllocator* _defaultAllocator();

 public:

  ArrayMetaData() : allocation_options(_defaultAllocator())
  {}

 protected:

  //! Nombre d'éléments du tableau (pour les tableaux 1D)
  Int64 size = 0;
  //! Taille de la première dimension (pour les tableaux 2D)
  Int64 dim1_size = 0;
  //! Taille de la deuxième dimension (pour les tableaux 2D)
  Int64 dim2_size = 0;
  //! Nombre d'éléments alloués
  Int64 capacity = 0;
  //! Allocateur mémoire et options associées
  MemoryAllocationOptions allocation_options;
  //! Nombre de références sur l'instance
  Int32 nb_ref = 0;
  //! Indique is cette instance a été allouée par l'opérateur new.
  bool is_allocated_by_new = false;
  //! Indique si cette instance n'est pas l'instance nulle (partagée par tous les SharedArray)
  bool is_not_null = false;

 protected:

  IMemoryAllocator* _allocator() const { return allocation_options.m_allocator; }

 public:

  static void throwInvalidMetaDataForSharedArray ARCCORE_NORETURN ();
  static void throwNullExpected ARCCORE_NORETURN ();
  static void throwNotNullExpected ARCCORE_NORETURN ();
  static void throwUnsupportedSpecificAllocator ARCCORE_NORETURN ();
  static void overlapError ARCCORE_NORETURN (const void* begin1,Int64 size1,
                                             const void* begin2,Int64 size2);
 protected:

  using MemoryPointer = void*;
  using ConstMemoryPointer = const void*;

 protected:

  MemoryPointer _allocate(Int64 nb, Int64 sizeof_true_type, RunQueue* queue);
  MemoryPointer _reallocate(const AllocatedMemoryInfo& mem_info, Int64 new_capacity, Int64 sizeof_true_type, RunQueue* queue);
  void _deallocate(const AllocatedMemoryInfo& mem_info, RunQueue* queue) ARCCORE_NOEXCEPT;
  void _setMemoryLocationHint(eMemoryLocationHint new_hint, void* ptr, Int64 sizeof_true_type);
  void _setHostDeviceMemoryLocation(eHostDeviceMemoryLocation location);
  void _copyFromMemory(MemoryPointer destination, ConstMemoryPointer source, Int64 sizeof_true_type, RunQueue* queue);

 private:

  void _checkAllocator() const;
  MemoryAllocationArgs _getAllocationArgs() const;
  MemoryAllocationArgs _getAllocationArgs(RunQueue* queue) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Ce type n'est plus utilisé.
 */
class ARCCORE_COLLECTIONS_EXPORT ArrayImplBase
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Cette classe n'est plus utilisée.
 */
template <typename T>
class ArrayImplT
: public ArrayImplBase
{
};

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
class ARCCORE_COLLECTIONS_EXPORT AbstractArrayBase
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
    std::swap(m_md,rhs.m_md);
    std::swap(m_meta_data,rhs.m_meta_data);
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
    if (_isUseOwnMetaData()){
      m_meta_data = ArrayMetaData();
      m_md = &m_meta_data;
    }
    else{
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
template<typename T>
class AbstractArray
: public AbstractArrayBase
{
 public:

  typedef typename ArrayTraits<T>::ConstReferenceType ConstReferenceType;
  typedef typename ArrayTraits<T>::IsPODType IsPODType;
  typedef AbstractArray<T> ThatClassType;
  using TrueImpl = T; //typedef ArrayImplT<T> TrueImpl;

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
    if (asize!=0){
      _internalAllocate(asize,_nullRunQueue());
      _createRange(0,asize,view.data());
      m_md->size = asize;
    }
  }

  /*!
   * \brief Construit un tableau avec un allocateur spécifique \a a.
   *
   * \sa _initFromAllocator(MemoryAllocationOptions o,Int64 acapacity);
   */
  // TODO A supprimer. Utiliser la surcharge avec MemoryAllocationOptions à la place.
  void _initFromAllocator(IMemoryAllocator* a, Int64 acapacity)
  {
    _initFromAllocator(MemoryAllocationOptions(a), acapacity);
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
  void _initFromAllocator(MemoryAllocationOptions o, Int64 acapacity)
  {
    _directFirstAllocateWithAllocator(acapacity,o);
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
      _directFirstAllocateWithAllocator(0,options);
    _updateReferences();
  }

 public:

  operator ConstArrayView<T>() const
  {
    return ConstArrayView<T>(ARCCORE_CAST_SMALL_SIZE(size()),m_ptr);
  }
  operator Span<const T>() const
  {
    return Span<const T>(m_ptr,m_md->size);
  }
  operator SmallSpan<const T>() const
  {
    return SmallSpan<const T>(m_ptr,ARCCORE_CAST_SMALL_SIZE(size()));
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
  bool empty() const { return m_md->size==0; }
  //! Vrai si le tableau contient l'élément de valeur \a v
  bool contains(ConstReferenceType v) const
  {
    const T* ptr = m_ptr;
    for (Int64 i = 0, n = m_md->size; i < n; ++i) {
      if (ptr[i]==v)
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
    m_md->_setMemoryLocationHint(new_hint,m_ptr,sizeof(T));
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
    return operator==(Span<const T>(rhs),Span<const T>(lhs));
  }

  friend bool operator!=(const AbstractArray<T>& rhs, const AbstractArray<T>& lhs)
  {
    return !(rhs==lhs);
  }

  friend bool operator==(const AbstractArray<T>& rhs, const Span<const T>& lhs)
  {
    return operator==(Span<const T>(rhs),lhs);
  }

  friend bool operator!=(const AbstractArray<T>& rhs, const Span<const T>& lhs)
  {
    return !(rhs==lhs);
  }

  friend bool operator==(const Span<const T>& rhs, const AbstractArray<T>& lhs)
  {
    return operator==(rhs,Span<const T>(lhs));
  }

  friend bool operator!=(const Span<const T>& rhs, const AbstractArray<T>& lhs)
  {
    return !(rhs==lhs);
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
    if (new_capacity<=m_md->capacity)
      return;
    _internalRealloc(new_capacity,false);
  }
  /*!
   * \brief Réalloue le tableau pour une nouvelle capacité égale à \a new_capacity.
   *
   * Si la nouvelle capacité est inférieure à l'ancienne, rien ne se passe.
   */
  template<typename PodType>
  void _internalRealloc(Int64 new_capacity, bool compute_capacity, PodType pod_type, RunQueue* queue = nullptr)
  {
    if (_isSharedNull()){
      if (new_capacity!=0)
        _internalAllocate(new_capacity,queue);
      return;
    }

    Int64 acapacity = new_capacity;
    if (compute_capacity){
      acapacity = m_md->capacity;
      //std::cout << " REALLOC: want=" << wanted_size << " current_capacity=" << capacity << '\n';
      while (new_capacity>acapacity)
        acapacity = (acapacity==0) ? 4 : (acapacity + 1 + acapacity / 2);
      //std::cout << " REALLOC: want=" << wanted_size << " new_capacity=" << capacity << '\n';
    }
    // Si la nouvelle capacité est inférieure à la courante,ne fait rien.
    if (acapacity <= m_md->capacity)
      return;
    _internalReallocate(acapacity, pod_type, queue);
  }

  void _internalRealloc(Int64 new_capacity, bool compute_capacity)
  {
    _internalRealloc(new_capacity, compute_capacity,IsPODType());
  }

  //! Réallocation pour un type POD
  void _internalReallocate(Int64 new_capacity, TrueType,RunQueue* queue)
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
    if (m_ptr!=old_ptr){
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
  void _internalAllocate(Int64 new_capacity,RunQueue* queue)
  {
    _directAllocate(new_capacity, queue);
    m_md->nb_ref = _getNbRef();
    _updateReferences();
  }

  void _copyFromMemory(const T* source)
  {
    m_md->_copyFromMemory(m_ptr, source,sizeof(T),_nullRunQueue());
  }

 private:

  void _directFirstAllocateWithAllocator(Int64 new_capacity,MemoryAllocationOptions options)
  {
    IMemoryAllocator* wanted_allocator = options.allocator();
    if (!wanted_allocator) {
      wanted_allocator = ArrayMetaData::_defaultAllocator();
      options.setAllocator(wanted_allocator);
    }
    _allocateMetaData();
    m_md->allocation_options = options;
    if (new_capacity > 0)
      _allocateMP(new_capacity, options.runQueue());
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
  void _addRange(ConstReferenceType val,Int64 n)
  {
    Int64 s = m_md->size;
    if ((s+n) > m_md->capacity)
      _internalRealloc(s+n,true);
    for( Int64 i=0; i<n; ++i )
      new (m_ptr + s + i) T(val);
    m_md->size += n;
  }

  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void _addRange(Span<const T> val)
  {
    Int64 n = val.size();
    const T* ptr = val.data();
    Int64 s = m_md->size;
    if ((s+n) > m_md->capacity)
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
  void _createRange(Int64 abegin, Int64 aend,ConstReferenceType value,FalseType)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i)
      new (m_ptr + i) T(value);
  }
  void _createRange(Int64 abegin,Int64 aend,const T* values)
  {
    if (abegin < 0)
      abegin = 0;
    for (Int64 i = abegin; i < aend; ++i ){
      new (m_ptr+i) T(*values);
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
      this->_internalRealloc(s,false,pod_type,queue);
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
  void _resizeNoInit(Int64 s,RunQueue* queue = nullptr)
  {
    _resizeHelper(s,TrueType{},queue);
  }
  void _clear()
  {
    this->_destroyRange(0,m_md->size,IsPODType());
    m_md->size = 0;
  }
  //! Redimensionne et remplit les nouvelles valeurs avec \a value
  void _resize(Int64 s,ConstReferenceType value)
  {
    if (s<0)
      s = 0;
    if (s>m_md->size){
      this->_internalRealloc(s,false);
      this->_createRange(m_md->size,s,value,IsPODType());
    }
    else{
      this->_destroyRange(s,m_md->size,IsPODType());
    }
    m_md->size = s;
  }
  void _copy(const T* rhs_begin,TrueType)
  {
    _copyFromMemory(rhs_begin);
  }
  void _copy(const T* rhs_begin,FalseType)
  {
    for( Int64 i=0, n=m_md->size; i<n; ++i )
      m_ptr[i] = rhs_begin[i];
  }
  void _copy(const T* rhs_begin)
  {
    _copy(rhs_begin,IsPODType());
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
    if (abegin>=rhs_begin && abegin<(rhs_begin+rhs_size))
      ArrayMetaData::overlapError(abegin,m_md->size,rhs_begin,rhs_size);

    if (rhs_size > current_size) {
      this->_internalRealloc(rhs_size, false);
      // Crée les nouveaux éléments
      this->_createRange(m_md->size, rhs_size, rhs_begin + current_size);
      // Copie les éléments déjà existant
      _copy(rhs_begin);
      m_md->size = rhs_size;
    }
    else{
      this->_destroyRange(rhs_size,current_size,IsPODType{});
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
    std::swap(m_ptr,rhs.m_ptr);
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
    if (new_capacity>this->capacity())
      return;
    if (new_capacity<4)
      new_capacity = 4;
    _internalReallocate(new_capacity,IsPODType(),_nullRunQueue());
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
    if (asize>max_size)
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
    if (rhs.allocator()==this->allocator()){
      _resizeAndCopyView(rhs_span);
    }
    else{
      _destroy();
      _internalDeallocate(_nullRunQueue());
      _reset();
      _initFromAllocator(rhs.allocationOptions(),0);
      _initFromSpan(rhs_span);
    }
  }

 protected:

  void _setMP(TrueImpl* new_mp)
  {
    m_ptr = new_mp;
  }

  void _setMP2(TrueImpl* new_mp,ArrayMetaData* new_md)
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
/*!
 * \ingroup Collection
 *
 * \brief Classe de base des vecteurs 1D de données.
 *
 * Cette classe manipule un vecteur (tableau) 1D de données.
 *
 * Les instances de cette classe ne sont pas copiables ni affectable. Pour créer un
 * tableau copiable, il faut utiliser SharedArray (pour une sémantique par
 * référence) ou UniqueArray (pour une sémantique par valeur comme la STL).
 */
template<typename T>
class Array
: public AbstractArray<T>
{
 protected:

  using AbstractArray<T>::m_ptr;
  using AbstractArray<T>::m_md;

 public:

  typedef AbstractArray<T> BaseClassType;
  using typename BaseClassType::ConstReferenceType;

 public:

  using typename BaseClassType::value_type;
  using typename BaseClassType::iterator;
  using typename BaseClassType::const_iterator;
  using typename BaseClassType::reverse_iterator;
  using typename BaseClassType::const_reverse_iterator;
  using typename BaseClassType::pointer;
  using typename BaseClassType::const_pointer;
  using typename BaseClassType::reference;
  using typename BaseClassType::const_reference;
  using typename BaseClassType::size_type;
  using typename BaseClassType::difference_type;

 protected:

  Array() {}

 protected:

  //! Constructeur par déplacement (uniquement pour UniqueArray)
  Array(Array<T>&& rhs) ARCCORE_NOEXCEPT : AbstractArray<T>(std::move(rhs)) {}

 protected:

  void _initFromInitializerList(std::initializer_list<T> alist)
  {
    Int64 nsize = arccoreCheckArraySize(alist.size());
    this->_reserve(nsize);
    for( const auto& x : alist )
      this->add(x);
  }

 private:

  Array(const Array<T>& rhs) = delete;
  void operator=(const Array<T>& rhs) = delete;

 public:

  ~Array()
  {
  }

 public:

  operator ConstArrayView<T>() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ConstArrayView<T>(s,m_ptr);
  }
  operator ArrayView<T>()
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ArrayView<T>(s,m_ptr);
  }
  operator Span<const T>() const
  {
    return Span<const T>(m_ptr,m_md->size);
  }
  operator Span<T>()
  {
    return Span<T>(m_ptr,m_md->size);
  }
  //! Vue constante sur ce tableau
  ConstArrayView<T> constView() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ConstArrayView<T>(s,m_ptr);
  }
  //! Vue constante sur ce tableau
  Span<const T> constSpan() const
  {
    return Span<const T>(m_ptr,m_md->size);
  }
  /*!
   * \brief Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments.
   *
   * Si \a (\a abegin + \a asize) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subConstView(Int64 abegin,Int32 asize) const
  {
    if (abegin>=m_md->size)
      return {};
    return { this->_clampSizeOffet(abegin,asize), m_ptr + abegin };
  }
  //! Vue mutable sur ce tableau
  ArrayView<T> view() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ArrayView<T>(s,m_ptr);
  }
  //! Vue immutable sur ce tableau
  Span<const T> span() const
  {
    return Span<const T>(m_ptr,m_md->size);
  }
  //! Vue mutable sur ce tableau
  Span<T> span()
  {
    return Span<T>(m_ptr,m_md->size);
  }
  //! Vue immutable sur ce tableau
  SmallSpan<const T> smallSpan() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return SmallSpan<const T>(m_ptr,s);
  }
  //! Vue immutable sur ce tableau
  SmallSpan<const T> constSmallSpan() const
  {
    return smallSpan();
  }
  //! Vue mutable sur ce tableau
  SmallSpan<T> smallSpan()
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return SmallSpan<T>(m_ptr,s);
  }
  /*!
   * \brief Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments.
   *
   * Si \a (\a abegin + \a asize) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ArrayView<T> subView(Int64 abegin,Integer asize)
  {
    if (abegin>=m_md->size)
      return {};
    return { this->_clampSizeOffet(abegin,asize), m_ptr + abegin };
  }
  /*!
   * \brief Extrait un sous-tableau à à partir d'une liste d'index.
   *
   * Le résultat est stocké dans \a result dont la taille doit être au moins
   * égale à celle de \a indexes.
   */
  void sample(ConstArrayView<Integer> indexes,ArrayView<T> result) const
  {
    const Integer result_size = indexes.size();
    [[maybe_unused]] const Int64 my_size = m_md->size;
    for( Integer i=0; i<result_size; ++i) {
      Int32 index = indexes[i];
      ARCCORE_CHECK_AT(index,my_size);
      result[i] = m_ptr[index];
    }
  }

 public:

  //! Ajoute l'élément \a val à la fin du tableau
  void add(ConstReferenceType val)
  {
    if (m_md->size >= m_md->capacity)
      this->_internalRealloc(m_md->size+1,true);
    new (m_ptr + m_md->size) T(val);
    ++m_md->size;
  }
  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void addRange(ConstReferenceType val,Int64 n)
  {
    this->_addRange(val,n);
  }
  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void addRange(ConstArrayView<T> val)
  {
    this->_addRange(val);
  }
  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void addRange(Span<const T> val)
  {
    this->_addRange(val);
  }
  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void addRange(ArrayView<T> val)
  {
    this->_addRange(val);
  }
  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void addRange(Span<T> val)
  {
    this->_addRange(val);
  }
  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void addRange(const Array<T>& val)
  {
    this->_addRange(val.constSpan());
  }
  /*!
   * \brief Change le nombre d'éléments du tableau à \a s.
   *
   * \note Si le nouveau tableau est plus grand que l'ancien, les nouveaux
   * éléments ne sont pas initialisés s'il s'agit d'un type POD.
   */
  void resize(Int64 s) { this->_resize(s); }
  /*!
   * \brief Change le nombre d'éléments du tableau à \a s.
   *
   * Si le nouveau tableau est plus grand que l'ancien, les nouveaux
   * éléments sont initialisé avec la valeur \a fill_value.
   */
  void resize(Int64 s, ConstReferenceType fill_value)
  {
    this->_resize(s, fill_value);
  }

  /*!
   * \brief Redimensionne sans initialiser les nouvelles valeurs.
   *
   * \warning Cela peut provoquer un comportement indéfini si le type
   * \a T n'est pas copiable trivialement car les
   * valeurs ne sont pas initialisées par la suite et le destructeur
   * de \a T sera appelé lors de la destruction de l'instance.
   */
  void resizeNoInit(Int64 s)
  {
    this->_resizeNoInit(s, nullptr);
  }

  //! Réserve le mémoire pour \a new_capacity éléments
  void reserve(Int64 new_capacity)
  {
    this->_reserve(new_capacity);
  }
  /*!
   * \brief Réalloue pour libérer la mémoire non utilisée.
   *
   * Après cet appel, capacity() sera équal à size(). Si size()
   * est nul ou est très petit, il est possible que capacity() soit
   * légèrement supérieur.
   */
  void shrink()
  {
    this->_shrink();
  }

  /*!
   * \brief Réalloue la mémoire avoir une capacité proche de \a new_capacity.
   */
  void shrink(Int64 new_capacity)
  {
    this->_shrink(new_capacity);
  }

  /*!
   * \brief Réalloue pour libérer la mémoire non utilisée.
   *
   * \sa shrink().
   */
  void shrink_to_fit()
  {
    this->_shrink();
  }

  /*!
   * \brief Supprime l'entité ayant l'indice \a index.
   *
   * Tous les éléments de ce tableau après celui supprimé sont
   * décalés.
   */
  void remove(Int64 index)
  {
    Int64 s = m_md->size;
    ARCCORE_CHECK_AT(index,s);
    for( Int64 i=index; i<(s-1); ++i )
      m_ptr[i] = m_ptr[i+1];
    --m_md->size;
    m_ptr[m_md->size].~T();
  }
  /*!
   * \brief Supprime la dernière entité du tableau.
   */
  void popBack()
  {
    ARCCORE_CHECK_AT(0,m_md->size);
    --m_md->size;
    m_ptr[m_md->size].~T();
  }
  //! Elément d'indice \a i. Vérifie toujours les débordements
  ConstReferenceType at(Int64 i) const
  {
    arccoreCheckAt(i,m_md->size);
    return m_ptr[i];
  }
  //! Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  void setAt(Int64 i,ConstReferenceType value)
  {
    arccoreCheckAt(i, m_md->size);
    m_ptr[i] = value;
  }
  //! Elément d'indice \a i
  ConstReferenceType item(Int64 i) const { return m_ptr[i]; }
  //! Elément d'indice \a i
  void setItem(Int64 i,ConstReferenceType v) { m_ptr[i] = v; }
  //! Elément d'indice \a i
  ConstReferenceType operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Elément d'indice \a i
  T& operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i,m_md->size);
    return m_ptr[i];
  }
  ConstReferenceType operator()(Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Elément d'indice \a i
  T& operator()(Int64 i)
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Dernier élément du tableau
  /*! Le tableau ne doit pas être vide */
  T& back()
  {
    ARCCORE_CHECK_AT(m_md->size-1,m_md->size);
    return m_ptr[m_md->size - 1];
  }
  //! Dernier élément du tableau (const)
  /*! Le tableau ne doit pas être vide */
  ConstReferenceType back() const
  {
    ARCCORE_CHECK_AT(m_md->size - 1, m_md->size);
    return m_ptr[m_md->size - 1];
  }

  //! Premier élément du tableau
  /*! Le tableau ne doit pas être vide */
  T& front()
  {
    ARCCORE_CHECK_AT(0, m_md->size);
    return m_ptr[0];
  }

  //! Premier élément du tableau (const)
  /*! Le tableau ne doit pas être vide */
  ConstReferenceType front() const
  {
    ARCCORE_CHECK_AT(0, m_md->size);
    return m_ptr[0];
  }

  //! Supprime les éléments du tableau
  void clear()
  {
    this->_clear();
  }

  //! Remplit le tableau avec la valeur \a value
  void fill(ConstReferenceType value)
  {
    this->_fill(value);
  }

  /*!
   * \brief Copie les valeurs de \a rhs dans l'instance.
   *
   * L'instance est redimensionnée pour que this->size()==rhs.size().
   */
  void copy(Span<const T> rhs)
  {
    this->_resizeAndCopyView(rhs);
  }

  //! Clone le tableau
  [[deprecated("Y2021: Use SharedArray::clone() or UniqueArray::clone()")]]
  Array<T> clone() const
  {
    Array<T> x;
    x.copy(this->constSpan());
    return x;
  }

  //! \internal Accès à la racine du tableau hors toute protection
  const T* unguardedBasePointer() const { return m_ptr; }
  //! \internal Accès à la racine du tableau hors toute protection
  T* unguardedBasePointer() { return m_ptr; }

  //! Accès à la racine du tableau hors toute protection
  const T* data() const { return m_ptr; }
  //! \internal Accès à la racine du tableau hors toute protection
  T* data() { return m_ptr; }

 public:

  //! Itérateur sur le premier élément du tableau.
  iterator begin() { return iterator(m_ptr); }

  //! Itérateur constant sur le premier élément du tableau.
  const_iterator begin() const { return const_iterator(m_ptr); }

  //! Itérateur sur le premier élément après la fin du tableau.
  iterator end() { return iterator(m_ptr+m_md->size); }

  //! Itérateur constant sur le premier élément après la fin du tableau.
  const_iterator end() const { return const_iterator(m_ptr+m_md->size); }

  //! Itérateur inverse sur le premier élément du tableau.
  reverse_iterator rbegin() { return std::make_reverse_iterator(end()); }

  //! Itérateur inverse sur le premier élément du tableau.
  const_reverse_iterator rbegin() const { return std::make_reverse_iterator(end()); }

  //! Itérateur inverse sur le premier élément après la fin du tableau.
  reverse_iterator rend() { return std::make_reverse_iterator(begin()); }

  //! Itérateur inverse sur le premier élément après la fin du tableau.
  const_reverse_iterator rend() const { return std::make_reverse_iterator(begin()); }

 public:

  //! Intervalle d'itération du premier au dernièr élément.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<pointer> range()
  {
    return ArrayRange<pointer>(m_ptr,m_ptr+m_md->size);
  }

  //! Intervalle d'itération du premier au dernièr élément.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr,m_ptr+m_md->size);
  }

 public:

  //@{ Méthodes pour compatibilité avec la STL.
  //! Ajoute l'élément \a val à la fin du tableau
  void push_back(ConstReferenceType val)
  {
    this->add(val);
  }
  //@}

 private:

  //! Method called from totalview debugger
  static int TV_ttf_display_type(const Arccore::Array<T> * obj);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Vecteur 1D de données avec sémantique par référence.
 *
 * Pour avoir un vecteur qui utilise une sémantique par valeur (à la std::vector),
 * il faut utiliser la classe UniqueArray.
 *
 * La sémantique par référence fonctionne comme suit:
 *
 * \code
 * SharedArray<int> a1(5);
 * SharedArray<int> a2;
 * a2 = a1; // a2 et a1 font référence à la même zone mémoire.
 * a1[3] = 1;
 * a2[3] = 2;
 * std::cout << a1[3]; // affiche '2'
 * \endcode
 *
 * Dans l'exemple précédent, \a a1 et \a a2 font référence à la même zone
 * mémoire et donc \a a2[3] aura la même valeur que \a a1[3] (soit la valeur \a 2),
 *
 * Un tableau partagée est désalloué lorsqu'il n'existe plus
 * de référence sur ce tableau.
 *
 * \warning les opérations de référencement/déréférencement (les opérateurs
 * d'affection, de recopie et les destructeurs) ne sont pas thread-safe. Par
 * conséquent ce type de tableau doit être utilisé avec précaution dans
 * le cas d'un environnement multi-thread.
 *
 * \sa UniqueArray.
 */
template<typename T>
class SharedArray
: public Array<T>
{
 protected:

  using AbstractArray<T>::m_md;
  using AbstractArray<T>::m_ptr;

 public:

  typedef SharedArray<T> ThatClassType;
  typedef AbstractArray<T> BaseClassType;
  using typename BaseClassType::ConstReferenceType;

 public:

  //! Créé un tableau vide
  SharedArray() = default;
  //! Créé un tableau de \a size éléments contenant la valeur \a value.
  SharedArray(Int64 asize, ConstReferenceType value)
  {
    this->_resize(asize, value);
    this->_checkValidSharedArray();
  }
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(long long asize)
  {
    this->_resize(asize);
    this->_checkValidSharedArray();
  }
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(long asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(int asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(unsigned long long asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(unsigned long asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(unsigned int asize)
  : SharedArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray(const ConstArrayView<T>& aview)
  : Array<T>()
  {
    this->_initFromSpan(Span<const T>(aview));
    this->_checkValidSharedArray();
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray(const Span<const T>& aview)
  : Array<T>()
  {
    this->_initFromSpan(Span<const T>(aview));
    this->_checkValidSharedArray();
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray(const ArrayView<T>& aview)
  : Array<T>()
  {
    this->_initFromSpan(Span<const T>(aview));
    this->_checkValidSharedArray();
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray(const Span<T>& aview)
  : Array<T>()
  {
    this->_initFromSpan(aview);
    this->_checkValidSharedArray();
  }
  SharedArray(std::initializer_list<T> alist)
  : Array<T>()
  {
    this->_initFromInitializerList(alist);
    this->_checkValidSharedArray();
  }
  //! Créé un tableau faisant référence à \a rhs.
  SharedArray(const SharedArray<T>& rhs)
  : Array<T>()
  {
    _initReference(rhs);
    this->_checkValidSharedArray();
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  inline SharedArray(const UniqueArray<T>& rhs);

  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   *
   * \warning Using specific allocator for SharedArray is experimental
   */
  explicit SharedArray(IMemoryAllocator* allocator)
  : Array<T>()
  {
    this->_initFromAllocator(allocator,0);
    this->_checkValidSharedArray();
  }

  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   *
   * \warning Using specific allocator for SharedArray is experimental
   */
  SharedArray(IMemoryAllocator* allocator,Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(allocator,asize);
    this->_resize(asize);
    this->_checkValidSharedArray();
  }

  /*!
   * \brief Créé un tableau avec l'allocateur \a allocator en recopiant les valeurs \a rhs.
   *
   * \warning Using specific allocator for SharedArray is experimental
   */
  SharedArray(IMemoryAllocator* allocator,Span<const T> rhs)
  {
    this->_initFromAllocator(allocator,0);
    this->_initFromSpan(rhs);
    this->_checkValidSharedArray();
  }

  //! Change la référence de cette instance pour qu'elle soit celle de \a rhs.
  void operator=(const SharedArray<T>& rhs)
  {
    this->_operatorEqual(rhs);
    this->_checkValidSharedArray();
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  inline void operator=(const UniqueArray<T>& rhs);
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const Span<const T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const Span<T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const ConstArrayView<T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const ArrayView<T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  void operator=(std::initializer_list<T> alist)
  {
    this->clear();
    for( const auto& x : alist )
      this->add(x);
    this->_checkValidSharedArray();
  }
  //! Détruit le tableau
  ~SharedArray() override
  {
    _removeReference();
  }

 public:

  //! Clone le tableau
  SharedArray<T> clone() const
  {
    return SharedArray<T>(this->allocator(),this->constSpan());
  }

 protected:

  void _initReference(const ThatClassType& rhs)
  {
    // TODO fusionner avec l'implémentation de SharedArray2
    this->_setMP(rhs.m_ptr);
    this->_copyMetaData(rhs);
    _addReference(&rhs);
    ++m_md->nb_ref;
  }
  //! Mise à jour des références
  void _updateReferences() final
  {
    // TODO fusionner avec l'implémentation de SharedArray2
    for( ThatClassType* i = m_prev; i; i = i->m_prev )
      i->_setMP2(m_ptr,m_md);
    for( ThatClassType* i = m_next; i; i = i->m_next )
      i->_setMP2(m_ptr,m_md);
  }
  //! Mise à jour des références
  Integer _getNbRef() final
  {
    // NOTE: à vérifier mais lorsque cette méthode est appelée
    // il n'y a toujours qu'une seule référence.
    // TODO fusionner avec l'implémentation de SharedArray2
    Integer nb_ref = 1;
    for( ThatClassType* i = m_prev; i; i = i->m_prev )
      ++nb_ref;
    for( ThatClassType* i = m_next; i; i = i->m_next )
      ++nb_ref;
    return nb_ref;
  }
  bool _isUseOwnMetaData() const final
  {
    return false;
  }
  /*!
   * \brief Insère cette instance dans la liste chaînée.
   * L'instance est insérée à la position de \a new_ref.
   * \pre m_prev==0
   * \pre m_next==0;
   */
  void _addReference(const ThatClassType* new_ref)
  {
    ThatClassType* nf = const_cast<ThatClassType*>(new_ref);
    ThatClassType* prev = nf->m_prev;
    nf->m_prev = this;
    m_prev = prev;
    m_next = nf;
    if (prev)
      prev->m_next = this;
  }
  //! Supprime cette instance de la liste chaînée des références
  void _removeReference()
  {
    if (m_prev)
      m_prev->m_next = m_next;
    if (m_next)
      m_next->m_prev = m_prev;
  }
  //! Détruit l'instance si plus personne ne la référence
  void _checkFreeMemory()
  {
    if (m_md->nb_ref==0){
      this->_destroy();
      this->_internalDeallocate();
    }
  }
  void _operatorEqual(const ThatClassType& rhs)
  {
    if (&rhs!=this){
      _removeReference();
      _addReference(&rhs);
      ++rhs.m_md->nb_ref;
      --m_md->nb_ref;
      _checkFreeMemory();
      this->_setMP2(rhs.m_ptr,rhs.m_md);
    }
  }
 private:

  ThatClassType* m_next = nullptr; //!< Référence suivante dans la liste chaînée
  ThatClassType* m_prev = nullptr; //!< Référence précédente dans la liste chaînée

 private:

  //! Interdit
  void operator=(const Array<T>& rhs) = delete;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Vecteur 1D de données avec sémantique par valeur (style STL).
 *
 * Cette classe gère un tableau de valeur de la même manière que la
 * classe stl::vector de la STL.

 * La sémantique par valeur fonctionne comme suit:
 *
 * \code
 * UniqueArray<int> a1(5);
 * UniqueArray<int> a2;
 * a2 = a1; // a2 devient une copie de a1.
 * a1[3] = 1;
 * a2[3] = 2;
 * std::cout << a1[3]; // affiche '1'
 * \endcode
 *
 * Il est possible de spécifier un allocateur mémoire spécifique via
 * le constructeur UniqueArray(IMemoryAllocator*). Dans ce cas, l'allocateur
 * spécifié en argument doit rester valide tant que cette instance
 * est utilisée.
 *
 * \warning L'allocateur est transféré à l'instance de destination lors d'un
 * appel aux constructeurs qui prennent en argument un Array, SharedArray ou
 * UniqueArray. Il en est de même avec l'opérateur d'assignement et lors
 * de l'appel à UniqueArray::swap(). Si ces appels sont envisagés, il
 * faut garantir que l'allocateur restera valide même après transfert. Il
 * est donc préférable dans tout les cas que l'allocateur spécifique utilisé
 * reste valide durant toute la durée de l'application.
 *
 * Si le type est un type Plain Object Data (POD) alors les données ne sont
 * pas initialisées en cas de réallocation. La classe template ArrayTraits
 * permet de spécifier si un type est POD suivant la valeur données par
 * le type ArrayTraits<T>::IsPODType qui peut être FalseType ou TrueType.
 * Sauf spécialisation, seuls les types de base du C++ sont POD.
 */
template<typename T>
class UniqueArray
: public Array<T>
{
 public:

  typedef AbstractArray<T> BaseClassType;
  using typename BaseClassType::ConstReferenceType;

 public:

  //! Créé un tableau vide
  UniqueArray() {}
  //! Créé un tableau de \a size éléments contenant la valeur \a value.
  UniqueArray(Int64 req_size,ConstReferenceType value)
  {
    this->_resize(req_size,value);
  }
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(long long asize)
  {
    this->_resize(asize);
  }
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(long asize)
  : UniqueArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(int asize)
  : UniqueArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(unsigned long long asize)
  : UniqueArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(unsigned long asize)
  : UniqueArray(static_cast<long long>(asize))
  {}
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(unsigned int asize)
  : UniqueArray(static_cast<long long>(asize))
  {}

  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  UniqueArray(const ConstArrayView<T>& aview)
  : UniqueArray(Span<const T>(aview))
  {
  }
  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  UniqueArray(const Span<const T>& aview)
  {
    this->_initFromSpan(aview);
  }
  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  UniqueArray(const ArrayView<T>& aview)
  : UniqueArray(Span<const T>(aview))
  {
  }
  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  UniqueArray(const Span<T>& aview)
  {
    this->_initFromSpan(aview);
  }
  UniqueArray(std::initializer_list<T> alist)
  {
    this->_initFromInitializerList(alist);
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray(const Array<T>& rhs)
  {
    this->_initFromAllocator(rhs.allocationOptions(),0);
    this->_initFromSpan(rhs);
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray(const UniqueArray<T>& rhs) : Array<T> {}
  {
    this->_initFromAllocator(rhs.allocationOptions(),0);
    this->_initFromSpan(rhs);
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray(const SharedArray<T>& rhs)
  {
    this->_initFromSpan(rhs);
  }
  //! Constructeur par déplacement. \a rhs est invalidé après cet appel
  UniqueArray(UniqueArray<T>&& rhs) ARCCORE_NOEXCEPT : Array<T>(std::move(rhs)) {}
  //! Créé un tableau vide avec un allocateur spécifique \a allocator
  explicit UniqueArray(IMemoryAllocator* allocator) : Array<T>()
  {
    this->_initFromAllocator(allocator,0);
  }
  //! Créé un tableau vide avec un allocateur spécifique \a allocator
  explicit UniqueArray(MemoryAllocationOptions allocate_options) : Array<T>()
  {
    this->_initFromAllocator(allocate_options,0);
  }
  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   */
  UniqueArray(IMemoryAllocator* allocator,Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(allocator,asize);
    this->_resize(asize);
  }
  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   */
  UniqueArray(MemoryAllocationOptions allocate_options,Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(allocate_options,asize);
    this->_resize(asize);
  }
  //! Créé un tableau avec l'allocateur \a allocator en recopiant les valeurs \a rhs.
  UniqueArray(IMemoryAllocator* allocator,Span<const T> rhs)
  {
    this->_initFromAllocator(allocator,0);
    this->_initFromSpan(rhs);
  }
  //! Créé un tableau avec l'allocateur \a allocator en recopiant les valeurs \a rhs.
  UniqueArray(MemoryAllocationOptions allocate_options,Span<const T> rhs)
  {
    this->_initFromAllocator(allocate_options,0);
    this->_initFromSpan(rhs);
  }

  //! Copie les valeurs de \a rhs dans cette instance.
  void operator=(const Array<T>& rhs)
  {
    this->_assignFromArray(rhs);
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  void operator=(const SharedArray<T>& rhs)
  {
    this->_assignFromArray(rhs);
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  void operator=(const UniqueArray<T>& rhs)
  {
    this->_assignFromArray(rhs);
  }
  //! Opérateur de recopie par déplacement. \a rhs est invalidé après cet appel.
  void operator=(UniqueArray<T>&& rhs) ARCCORE_NOEXCEPT
  {
    this->_move(rhs);
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
  void operator=(const SmallSpan<T>& rhs)
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
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const SmallSpan<const T>& rhs)
  {
    this->copy(rhs);
  }
  //! Copie les valeurs de la vue \a alist dans cette instance.
  void operator=(std::initializer_list<T> alist)
  {
    this->clear();
    for( const auto& x : alist )
      this->add(x);
  }
  //! Détruit l'instance.
  ~UniqueArray() override
  {
  }

 public:

  /*!
   * \brief Échange les valeurs de l'instance avec celles de \a rhs.
   *
   * L'échange se fait en temps constant et sans réallocation.
   */
  void swap(UniqueArray<T>& rhs)
  {
    this->_swap(rhs);
  }

  //! Clone le tableau
  UniqueArray<T> clone() const
  {
    return UniqueArray<T>(*this);
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Échange les valeurs de \a v1 et \a v2.
 *
 * L'échange se fait en temps constant et sans réallocation.
 */
template<typename T> inline void
swap(UniqueArray<T>& v1,UniqueArray<T>& v2)
{
  v1.swap(v2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline SharedArray<T>::
SharedArray(const UniqueArray<T>& rhs)
: Array<T>()
, m_next(nullptr)
, m_prev(nullptr)
{
  this->_initFromSpan(rhs.constSpan());
  this->_checkValidSharedArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline void SharedArray<T>::
operator=(const UniqueArray<T>& rhs)
{
  this->copy(rhs);
  this->_checkValidSharedArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue d'un tableau sous la forme d'octets non modifiables 
 *
 * T doit être un type POD.
 */
template<typename T> inline Span<const std::byte>
asBytes(const Array<T>& v)
{
  return asBytes(v.constSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vu d'un ableau sous la forme d'un tableau d'octets modifiables.
 *
 * T doit être un type POD.
 */
template<typename T> inline Span<std::byte>
asWritableBytes(Array<T>& v)
{
  return asWritableBytes(v.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
