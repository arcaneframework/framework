// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 IFPEN-CEA
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
/* Array.h                                                     (C) 2000-2021 */
/*                                                                           */
/* Tableau 1D.                                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_ARRAY_H
#define ARCCORE_COLLECTIONS_ARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/Span.h"
#include "arccore/collections/IMemoryAllocator.h"

#include <memory>
#include <initializer_list>
#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArrayImplBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Interface d'un allocateur pour un vecteur.
 * \deprecated Cette classe n'est plus utilisée.
 */
class ARCCORE_COLLECTIONS_EXPORT IArrayAllocator
{
 public:

  /*!
   * \brief Détruit l'allocateur.
   * 
   * Les objets alloués par l'allocateur doivent tous avoir été désalloués.
   */
  virtual ~IArrayAllocator() {}

 public:
  
  /*! \brief Alloue de la mémoire pour \a new_capacity objets.
   *
   * En cas de succès, retourne le pointeur sur le premier élément alloué.
   * En cas d'échec, une exception est levée (std::bad_alloc).
   * La valeur retournée n'est jamais nul.
   * \a new_capacity doit être strictement positif.
   */
  virtual ArrayImplBase* allocate(Int64 sizeof_true_impl,Int64 new_capacity,
                                  Int64 sizeof_true_type,ArrayImplBase* init) = 0;

  /*! \brief Libère la mémoire.
   *
   * Libère la mémoire dont le premier élément est donnée par \a ptr.
   */
  virtual void deallocate(ArrayImplBase* ptr) =0;

  /*!
   * \brief Réalloue de la mémoire pour \a new_capacity éléments.
   *
   */
  virtual ArrayImplBase* reallocate(Int64 sizeof_true_impl,Int64 new_capacity,
                                    Int64 sizeof_true_type,ArrayImplBase* ptr) =0;

  virtual Int64 computeCapacity(Int64 current,Int64 wanted) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Interface d'un allocateur pour un vecteur.
 * \deprecated Cette classe n'est plus utilisée.
 */
class ARCCORE_COLLECTIONS_EXPORT DefaultArrayAllocator
: public IArrayAllocator
{
 public:

  virtual ~DefaultArrayAllocator() {}

 public:
  
  ArrayImplBase* allocate(Int64 sizeof_true_impl,Int64 new_capacity,
                          Int64 sizeof_true_type,ArrayImplBase* init) override;

  void deallocate(ArrayImplBase* ptr) override;

  ArrayImplBase* reallocate(Int64 sizeof_true_impl,Int64 new_capacity,
                            Int64 sizeof_true_type,ArrayImplBase* current) override;

  Int64 computeCapacity(Int64 current,Int64 wanted) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Classe de base implémentant un vecteur.
 *
 * Cette classe sert d'implémentation pour tous les types tableaux
 * de Arccore, qu'ils soient 1D (Array),  2D simple (Array2)
 * ou 2D multiples (MultiArray2).
 *
 * \note Pour garantir l'alignement pour la vectorisation, cette structure
 * (et ArrayImplT) doit avoir une taille identique (sizeof) à celle spécifiée dans
 * AlignedMemoryAllocator pour le simd et le cache.
 */  
class ARCCORE_COLLECTIONS_EXPORT ArrayImplBase
{
 public:
  ArrayImplBase() : nb_ref(0), capacity(0), size(0),
    dim1_size(0), dim2_size(0), mutli_dims_size(0),
    allocator(&DefaultMemoryAllocator::shared_null_instance)
 {}

  // GG: note: normalement ce destructeur est inutile et on pourrait utiliser
  // le destructeur généré par le compilateur. Cependant, si on le supprime
  // les tests SIMD en AVX en optimisé avec gcc 4.9.3 plantent.
  // Il faudrait vérifier s'il s'agit d'un bug du compilateur ou d'un
  // problème dans Arccore.
  ~ArrayImplBase() {}
 public:
  static ArrayImplBase* shared_null;
 private:
  static ARCCORE_ALIGNAS(64) ArrayImplBase shared_null_instance;
 public:
  //! Nombre de références sur cet objet.
  Int64 nb_ref;
  //! Nombre d'éléments alloués
  Int64 capacity;
  //! Nombre d'éléments du tableau (pour les tableaux 1D)
  Int64 size;
  //! Taille de la première dimension (pour les tableaux 2D)
  Int64 dim1_size;
  //! Taille de la deuxième dimension (pour les tableaux 2D)
  Int64 dim2_size;
  //! Tableau des dimensions pour les tableaux multiples
  Int64* mutli_dims_size;
  //! Allocateur mémoire
  IMemoryAllocator* allocator;
  //! Padding pour que la structure ait une taille de 64 octets.
  Int64 padding1;

  static ArrayImplBase* allocate(Int64 sizeof_true_impl,Int64 nb,
                                 Int64 sizeof_true_type,ArrayImplBase* init);
  static ArrayImplBase* allocate(Int64 sizeof_true_impl,Int64 nb,
                                 Int64 sizeof_true_type,ArrayImplBase* init,
                                 IMemoryAllocator* allocator);
  static ArrayImplBase* reallocate(Int64 sizeof_true_impl,Int64 nb,
                                   Int64 sizeof_true_type,ArrayImplBase* current);
  static void deallocate(ArrayImplBase* current);
  static void overlapError(const void* begin1,Int64 size1,
                           const void* begin2,Int64 size2);

  static void checkSharedNull()
  {
    ArrayImplBase* s = shared_null;
    if (s->capacity!=0 || s->size!=0 || s->dim1_size!=0 || s->dim2_size!=0
        || s->mutli_dims_size || s->allocator)
      throwBadSharedNull();
  }
  static void throwBadSharedNull ARCCORE_NORETURN ();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Implémentation d'un vecteur d'un type T.
 */  
template <typename T>
class ArrayImplT
: public ArrayImplBase
{
 public:
  ArrayImplT() {}
  T ptr[1];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Caractéristiques pour un tableau
 */
template<typename T>
class ArrayTraits
{
 public:
  typedef const T& ConstReferenceType;
  typedef FalseType IsPODType;
};

#define ARCCORE_DEFINE_ARRAY_PODTYPE(datatype)\
template<>\
class ArrayTraits<datatype>\
{\
 public:\
  typedef datatype ConstReferenceType;\
  typedef TrueType IsPODType;\
}

template<typename T>
class ArrayTraits<T*>
{
 public:
  typedef T* Ptr;
  typedef const Ptr& ConstReferenceType;
  typedef FalseType IsPODType;
};

template<typename T>
class ArrayTraits<const T*>
{
 public:
  typedef T* Ptr;
  typedef const T* ConstReferenceType;
  typedef FalseType IsPODType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_DEFINE_ARRAY_PODTYPE(char);
ARCCORE_DEFINE_ARRAY_PODTYPE(signed char);
ARCCORE_DEFINE_ARRAY_PODTYPE(unsigned char);
ARCCORE_DEFINE_ARRAY_PODTYPE(short);
ARCCORE_DEFINE_ARRAY_PODTYPE(int);
ARCCORE_DEFINE_ARRAY_PODTYPE(long);
ARCCORE_DEFINE_ARRAY_PODTYPE(unsigned short);
ARCCORE_DEFINE_ARRAY_PODTYPE(unsigned int);
ARCCORE_DEFINE_ARRAY_PODTYPE(unsigned long);
ARCCORE_DEFINE_ARRAY_PODTYPE(float);
ARCCORE_DEFINE_ARRAY_PODTYPE(double);
ARCCORE_DEFINE_ARRAY_PODTYPE(long double);

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
{
 public:

  typedef typename ArrayTraits<T>::ConstReferenceType ConstReferenceType;
  typedef typename ArrayTraits<T>::IsPODType IsPODType;
  typedef AbstractArray<T> ThatClassType;
  typedef ArrayImplT<T> TrueImpl;

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

 private:

  static TrueImpl* _sharedNull()
  {
    return static_cast<TrueImpl*>(ArrayImplBase::shared_null);
  }
 protected:

  //! Construit un vecteur vide avec l'allocateur par défaut
  AbstractArray()
  : m_p(_sharedNull()), m_baseptr(m_p->ptr)
  {
  }
  /*!
   * \brief Construit un vecteur vide avec un allocateur spécifique \a a.
   * Si \a n'est pas nul, la mémoire est allouée pour
   * contenir \a acapacity éléments (mais le tableau reste vide).
   */
  AbstractArray(IMemoryAllocator* a,Int64 acapacity)
  : m_p(_sharedNull()), m_baseptr(m_p->ptr)
  {
    // Si un allocateur spécifique est utilisé et qu'il n'est pas
    // celui par défaut, il faut toujours allouer un objet pour
    // pouvoir conserver l'instance de l'allocateur. Par défaut
    // on utilise une taille de 1 élément.
    if (a && a!=m_p->allocator){
      Int64 c = (acapacity>1) ? acapacity : 1;
      _directFirstAllocateWithAllocator(c,a);
    }
  }
  //! Constructeur par déplacement. Ne doit être utilisé que par UniqueArray
  AbstractArray(ThatClassType&& rhs) ARCCORE_NOEXCEPT
  : m_p(rhs.m_p), m_baseptr(m_p->ptr)
  {
    rhs._reset();
  }
  AbstractArray(Int64 asize,const T* values)
  : m_p(_sharedNull()), m_baseptr(m_p->ptr)
  {
    if (asize!=0){
      _internalAllocate(asize);
      _createRange(0,asize,values);
      m_p->size = asize;
    }
  }
  AbstractArray(const Span<const T>& view)
  : m_p(_sharedNull()), m_baseptr(m_p->ptr)
  {
    Int64 asize = view.size();
    if (asize!=0){
      _internalAllocate(asize);
      _createRange(0,asize,view.data());
      m_p->size = asize;
    }
  }
  AbstractArray(const ConstArrayView<T>& view)
  : m_p(_sharedNull()), m_baseptr(m_p->ptr)
  {
    Int64 asize = view.size();
    if (asize!=0){
      _internalAllocate(asize);
      _createRange(0,asize,view.data());
      m_p->size = asize;
    }
  }

  virtual ~AbstractArray()
  {
    --m_p->nb_ref;
    _checkFreeMemory();
  }
 public:
  //! Libère la mémoire utilisée par le tableau.
  void dispose()
  {
    _destroy();
    IMemoryAllocator* a = m_p->allocator;
    _internalDeallocate();
    _setMP(_sharedNull());
    // Si on a un allocateur spécifique, il faut allouer un
    // bloc pour conserver cette information.
    if (a != m_p->allocator)
      _directFirstAllocateWithAllocator(1,a);
    _updateReferences();
  }
  IMemoryAllocator* allocator() const
  {
    return m_p->allocator;
  }
 public:
  operator ConstArrayView<T>() const
  {
    return ConstArrayView<T>(ARCCORE_CAST_SMALL_SIZE(size()),m_p->ptr);
  }
  operator Span<const T>() const
  {
    return Span<const T>(m_p->ptr,m_p->size);
  }
 public:
  //! Nombre d'éléments du vecteur
  Integer size() const { return ARCCORE_CAST_SMALL_SIZE(m_p->size); }
  //! Nombre d'éléments du vecteur
  Integer length() const { return ARCCORE_CAST_SMALL_SIZE(m_p->size); }
  //! Capacité (nombre d'éléments alloués) du vecteur
  Integer capacity() const { return ARCCORE_CAST_SMALL_SIZE(m_p->capacity); }
  //! Nombre d'éléments du vecteur (en 64 bits)
  Int64 largeSize() const { return m_p->size; }
  //! Nombre d'éléments du vecteur (en 64 bits)
  Int64 largeLength() const { return m_p->size; }
  //! Capacité (nombre d'éléments alloués) du vecteur (en 64 bits)
  Int64 largeCapacity() const { return m_p->capacity; }
  //! Capacité (nombre d'éléments alloués) du vecteur
  bool empty() const { return m_p->size==0; }
  //! Vrai si le tableau contient l'élément de valeur \a v
  bool contains(ConstReferenceType v) const
  {
    const T* ptr = m_p->ptr;
    for( Int64 i=0, n=m_p->size; i<n; ++i ){
      if (ptr[i]==v)
        return true;
    }
    return false;
  }
 public:
  //! Elément d'indice \a i
  ConstReferenceType operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_p->size);
    return m_p->ptr[i];
  }
 protected:
  TrueImpl* m_p;
 private:
  T* m_baseptr;
 protected:
  //! Réserve le mémoire pour \a new_capacity éléments
  void _reserve(Int64 new_capacity)
  {
    if (new_capacity<=m_p->capacity)
      return;
    _internalRealloc(new_capacity,false);
  }
  /*!
   * \brief Réalloue le tableau pour une nouvelle capacité égale à \a new_capacity.
   *
   * Si la nouvelle capacité est inférieure à l'ancienne, rien ne se passe.
   */
  virtual void _internalRealloc(Int64 new_capacity,bool compute_capacity)
  {
    if (m_p==TrueImpl::shared_null){
      if (new_capacity!=0)
        _internalAllocate(new_capacity);
      return;
    }

    Int64 acapacity = new_capacity;
    if (compute_capacity){
      acapacity = m_p->capacity;
      //std::cout << " REALLOC: want=" << wanted_size << " current_capacity=" << capacity << '\n';
      while (new_capacity>acapacity)
        acapacity = (acapacity==0) ? 4 : (acapacity + 1 + acapacity / 2);
      //std::cout << " REALLOC: want=" << wanted_size << " new_capacity=" << capacity << '\n';
    }
    // Si la nouvelle capacité est inférieure à la courante,ne fait rien.
    if (acapacity<m_p->capacity)
      return;
    _internalReallocate(acapacity,IsPODType());
  }

  //! Réallocation pour un type POD
  virtual void _internalReallocate(Int64 new_capacity,TrueType)
  {
    TrueImpl* old_p = m_p;
    Int64 old_capacity = m_p->capacity;
    _directReAllocate(new_capacity);
    bool update = (new_capacity < old_capacity) || (m_p != old_p);
    if (update){
      _updateReferences();
    }
  }

  //! Réallocation pour un type complexe (non POD)
  virtual void _internalReallocate(Int64 new_capacity,FalseType)
  {
    TrueImpl* old_p = m_p;
    Int64 old_size = m_p->size;
    _directAllocate(new_capacity);
    if (m_p!=old_p){
      for( Int64 i=0; i<old_size; ++i ){
        new (m_p->ptr+i) T(old_p->ptr[i]);
        old_p->ptr[i].~T();
      }
      m_p->nb_ref = old_p->nb_ref;
      ArrayImplBase::deallocate(old_p);
      _updateReferences();
    }
  }
  // Libère la mémoire
  virtual void _internalDeallocate()
  {
    if (m_p!=TrueImpl::shared_null)
      ArrayImplBase::deallocate(m_p);
  }
  virtual void _internalAllocate(Int64 new_capacity)
  {
    _directAllocate(new_capacity);
    m_p->nb_ref = _getNbRef();
    _updateReferences();
  }
 private:
  virtual void _directFirstAllocateWithAllocator(Int64 new_capacity,IMemoryAllocator* a)
  {
    //TODO: vérifier m_p vaut shared_null
    _setMP(static_cast<TrueImpl*>(ArrayImplBase::allocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p,a)));
    m_p->allocator = a;
    m_p->nb_ref = _getNbRef();
    m_p->size = 0;
    _updateReferences();
  }
  virtual void _directAllocate(Int64 new_capacity)
  {
    _setMP(static_cast<TrueImpl*>(ArrayImplBase::allocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p)));
  }
  virtual void _directReAllocate(Int64 new_capacity)
  {
    _setMP(static_cast<TrueImpl*>(ArrayImplBase::reallocate(sizeof(TrueImpl),new_capacity,sizeof(T),m_p)));
  }
 public:
  void printInfos(std::ostream& o)
  {
    o << " Infos: size=" << m_p->size << " capacity=" << m_p->capacity << '\n';
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
    Int64 s = m_p->size;
    if ((s+n) > m_p->capacity)
      _internalRealloc(s+n,true);
    for( Int64 i=0; i<n; ++i )
      new (m_p->ptr + s + i) T(val);
    m_p->size += n;
  }

  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void _addRange(Span<const T> val)
  {
    Int64 n = val.size();
    const T* ptr = val.data();
    Int64 s = m_p->size;
    if ((s+n) > m_p->capacity)
      _internalRealloc(s+n,true);
    _createRange(s,s+n,ptr);
    m_p->size += n;
  }

  //! Détruit l'instance si plus personne ne la référence
  void _checkFreeMemory()
  {
    if (m_p->nb_ref==0){
      _destroy();
      _internalDeallocate();
    }
  }
  void _destroy()
  {
    _destroyRange(0,m_p->size,IsPODType());
  }
  void _destroyRange(Int64,Int64,TrueType)
  {
    // Rien à faire pour un type POD.
  }
  void _destroyRange(Int64 abegin,Int64 aend,FalseType)
  {
    for( Int64 i=abegin; i<aend; ++i )
      m_p->ptr[i].~T();
  }
  void _createRangeDefault(Int64,Int64,TrueType)
  {
  }
  void _createRangeDefault(Int64 abegin,Int64 aend,FalseType)
  {
    for( Int64 i=abegin; i<aend; ++i )
      new (m_p->ptr+i) T();
  }
  void _createRange(Int64 abegin,Int64 aend,ConstReferenceType value,TrueType)
  {
    for( Int64 i=abegin; i<aend; ++i )
      m_p->ptr[i] = value;
  }
  void _createRange(Int64 abegin,Int64 aend,ConstReferenceType value,FalseType)
  {
    for( Int64 i=abegin; i<aend; ++i )
      new (m_p->ptr+i) T(value);
  }
  void _createRange(Int64 abegin,Int64 aend,const T* values)
  {
    for( Int64 i=abegin; i<aend; ++i ){
      new (m_p->ptr+i) T(*values);
      ++values;
    }
  }
  void _fill(ConstReferenceType value)
  {
    for( Int64 i=0, n=size(); i<n; ++i )
      m_p->ptr[i] = value;
  }
  void _clone(const ThatClassType& orig_array)
  {
    Int64 that_size = orig_array.size();
    _internalAllocate(that_size);
    m_p->size = that_size;
    m_p->dim1_size = orig_array.m_p->dim1_size;
    m_p->dim2_size = orig_array.m_p->dim2_size;
    _createRange(0,that_size,orig_array.m_p->ptr);
  }
  void _resize(Int64 s)
  {
    if (s<0)
      s = 0;
    if (s>m_p->size) {
      this->_internalRealloc(s,false);
      this->_createRangeDefault(m_p->size,s,IsPODType());
    }
    else{
      this->_destroyRange(s,m_p->size,IsPODType());
    }
    m_p->size = s;
  }
  void _clear()
  {
    this->_destroyRange(0,m_p->size,IsPODType());
    m_p->size = 0;
  }
  void _resize(Int64 s,ConstReferenceType value)
  {
    if (s<0)
      s = 0;
    if (s>m_p->size){
      this->_internalRealloc(s,false);
      this->_createRange(m_p->size,s,value,IsPODType());
    }
    else{
      this->_destroyRange(s,m_p->size,IsPODType());
    }
    m_p->size = s;
  }
  void _copy(const T* rhs_begin,TrueType)
  {
    std::memcpy(m_p->ptr,rhs_begin,((size_t)m_p->size)*sizeof(T));
  }
  void _copy(const T* rhs_begin,FalseType)
  {
    for( Int64 i=0, is=m_p->size; i<is; ++i )
      m_p->ptr[i] = rhs_begin[i];
  }
  void _copy(const T* rhs_begin)
  {
    _copy(rhs_begin,IsPODType());
  }
  void _copyView(Span<const T> rhs)
  {
    const T* rhs_begin = rhs.data();
    Int64 rhs_size = rhs.size();
    T* abegin = m_p->ptr;
    // Vérifie que \a rhs n'est pas un élément à l'intérieur de ce tableau
    if (abegin>=rhs_begin && abegin<(rhs_begin+rhs_size))
      ArrayImplBase::overlapError(abegin,m_p->size,rhs_begin,rhs_size);
    _resize(rhs_size);
    _copy(rhs_begin);
  }
  void _copyView(ConstArrayView<T> rhs)
  {
    this->_copyView(Span<const T>(rhs));
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
    if (&rhs==this)
      return;

    // Comme il n'y a qu'une seule référence sur le tableau actuel, on peut
    // directement libérer la mémoire.
    _destroy();
    _internalDeallocate();

    // Recopie bit à bit.
    _setMP(rhs.m_p);

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
    std::swap(m_p,rhs.m_p);
    std::swap(m_baseptr,rhs.m_baseptr);
  }

  void _shrink()
  {
    _shrink(size());
  }

  // Réalloue la mémoire pour avoir une capacité proche de \a new_capacity
  void _shrink(Int64 new_capacity)
  {
    if (m_p==TrueImpl::shared_null)
      return;
    // On n'augmente pas la capacité avec cette méthode
    if (new_capacity>this->capacity())
      return;
    if (new_capacity<4)
      new_capacity = 4;
    _internalReallocate(new_capacity,IsPODType());
  }

 private:
  /*!
   * \brief Réinitialise le tableau à un tableau vide.
   * \warning Cette méthode n'est valide que pour les UniqueArray et pas
   * les SharedArray.
   */
  void _reset()
  {
    _setMP(_sharedNull());
  }

 protected:

  void _setMP(TrueImpl* new_mp)
  {
    m_p = new_mp;
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

  using AbstractArray<T>::m_p;
  enum CloneBehaviour
    {
      CB_Clone,
      CB_Shared
    };
  enum BuildDeprecated
    {
      BD_NoWarning
    };
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
  Array(BuildDeprecated) : AbstractArray<T>() {}
  Array(Int64 asize,ConstReferenceType value,BuildDeprecated) : AbstractArray<T>()
  {
    this->_resize(asize,value);
  }
  //! Constructeur avec liste d'initialisation.
  Array(std::initializer_list<T> alist,BuildDeprecated) : AbstractArray<T>()
  {
    Int64 nsize = arccoreCheckArraySize(alist.size());
    this->_reserve(nsize);
    for( auto x : alist )
      this->add(x);
  }
  Array(Int64 asize,BuildDeprecated) : AbstractArray<T>()
  {
    this->_resize(asize);
  }
  Array(const ConstArrayView<T>& aview,BuildDeprecated) : AbstractArray<T>(aview)
  {
  }
  Array(const Span<const T>& aview,BuildDeprecated) : AbstractArray<T>(aview)
  {
  }
  /*!
   * \brief Créé un tableau de \a asize éléments avec un allocateur spécifique.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   */
  Array(IMemoryAllocator* allocator,Int64 asize,BuildDeprecated)
  : AbstractArray<T>(allocator,asize)
  {
    this->_resize(asize);
  }
 protected:
  //! Constructeur par déplacement (uniquement pour UniqueArray)
  Array(Array<T>&& rhs) ARCCORE_NOEXCEPT : AbstractArray<T>(std::move(rhs)) {}
 private:
  Array(const Array<T>& rhs);
  void operator=(const Array<T>& rhs);

 public:
  ~Array()
  {
  }
 public:
  operator ConstArrayView<T>() const
  {
    Integer s = arccoreCheckArraySize(m_p->size);
    return ConstArrayView<T>(s,m_p->ptr);
  }
  operator ArrayView<T>()
  {
    Integer s = arccoreCheckArraySize(m_p->size);
    return ArrayView<T>(s,m_p->ptr);
  }
  operator Span<const T>() const
  {
    return Span<const T>(m_p->ptr,m_p->size);
  }
  operator Span<T>()
  {
    return Span<T>(m_p->ptr,m_p->size);
  }
  //! Vue constante sur ce tableau
  ConstArrayView<T> constView() const
  {
    Integer s = arccoreCheckArraySize(m_p->size);
    return ConstArrayView<T>(s,m_p->ptr);
  }
  //! Vue constante sur ce tableau
  Span<const T> constSpan() const
  {
    return Span<const T>(m_p->ptr,m_p->size);
  }
  /*!
   * \brief Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments.
   *
   * Si \a (\a abegin + \a asize) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subConstView(Integer abegin,Integer asize) const
  {
    return constView().subView(abegin,asize);
  }
  //! Vue mutable sur ce tableau
  ArrayView<T> view() const
  {
    Integer s = arccoreCheckArraySize(m_p->size);
    return ArrayView<T>(s,m_p->ptr);
  }
  //! Vue immutable sur ce tableau
  Span<const T> span() const
  {
    return Span<const T>(m_p->ptr,m_p->size);
  }
  //! Vue mutable sur ce tableau
  Span<T> span()
  {
    return Span<T>(m_p->ptr,m_p->size);
  }
  /*!
   * \brief Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments.
   *
   * Si \a (\a abegin + \a asize) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ArrayView<T> subView(Integer abegin,Integer asize)
  {
    return view().subView(abegin,asize);
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
    [[maybe_unused]] const Int64 my_size = m_p->size;
    for( Integer i=0; i<result_size; ++i) {
      Int32 index = indexes[i];
      ARCCORE_CHECK_AT(index,my_size);
      result[i] = m_p->ptr[index];
    }
  }

 public:
  //! Ajoute l'élément \a val à la fin du tableau
  void add(ConstReferenceType val)
  {
    if (m_p->size >= m_p->capacity)
      this->_internalRealloc(m_p->size+1,true);
    new (m_p->ptr + m_p->size) T(val);
    ++m_p->size;
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
   * \brief Change le nombre d'élément du tableau à \a s.
   * Si le nouveau tableau est plus grand que l'ancien, les nouveaux
   * éléments ne sont pas initialisés s'il s'agit d'un type POD.
   */
  void resize(Int64 s) { this->_resize(s); }
  /*!
   * \brief Change le nombre d'élément du tableau à \a s.
   * Si le nouveau tableau est plus grand que l'ancien, les nouveaux
   * éléments sont initialisé avec la valeur \a fill_value.
   */
  void resize(Int64 s,ConstReferenceType fill_value)
  {
    this->_resize(s,fill_value);
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
    Int64 s = m_p->size;
    ARCCORE_CHECK_AT(index,s);
    for( Int64 i=index; i<(s-1); ++i )
      m_p->ptr[i] = m_p->ptr[i+1];
    --m_p->size;
    m_p->ptr[m_p->size].~T();
  }
  /*!
   * \brief Supprime la dernière entité du tableau.
   */
  void popBack()
  {
    ARCCORE_CHECK_AT(0,m_p->size);
    --m_p->size;
    m_p->ptr[m_p->size].~T();
  }
  //! Elément d'indice \a i. Vérifie toujours les débordements
  ConstReferenceType at(Int64 i) const
  {
    arccoreCheckAt(i,m_p->size);
    return m_p->ptr[i];
  }
  //! Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  void setAt(Int64 i,ConstReferenceType value)
  {
    arccoreCheckAt(i,m_p->size);
    m_p->ptr[i] = value;
  }
  //! Elément d'indice \a i
  ConstReferenceType item(Int64 i) const { return m_p->ptr[i]; }
  //! Elément d'indice \a i
  void setItem(Int64 i,ConstReferenceType v) { m_p->ptr[i] = v; }
  //! Elément d'indice \a i
  ConstReferenceType operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_p->size);
    return m_p->ptr[i];
  }
  //! Elément d'indice \a i
  T& operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i,m_p->size);
    return m_p->ptr[i];
  }
  //! Dernier élément du tableau
  /*! Le tableau ne doit pas être vide */
  T& back()
  {
    ARCCORE_CHECK_AT(m_p->size-1,m_p->size);
    return m_p->ptr[m_p->size-1];
  }
  //! Dernier élément du tableau (const)
  /*! Le tableau ne doit pas être vide */
  ConstReferenceType back() const
  {
    ARCCORE_CHECK_AT(m_p->size-1,m_p->size);
    return m_p->ptr[m_p->size-1];
  }

  //! Premier élément du tableau
  /*! Le tableau ne doit pas être vide */
  T& front()
  {
    ARCCORE_CHECK_AT(0,m_p->size);
    return m_p->ptr[0];
  }

  //! Premier élément du tableau (const)
  /*! Le tableau ne doit pas être vide */
  ConstReferenceType front() const
  {
    ARCCORE_CHECK_AT(0,m_p->size);
    return m_p->ptr[0];
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
    this->_copyView(rhs);
  }

  //! Clone le tableau
  Array<T> clone() const
  {
    return Array<T>(this->constSpan());
  }

  //! \internal Accès à la racine du tableau hors toute protection
  const T* unguardedBasePointer() const { return m_p->ptr; }
  //! \internal Accès à la racine du tableau hors toute protection
  T* unguardedBasePointer() { return m_p->ptr; }

  //! Accès à la racine du tableau hors toute protection
  const T* data() const { return m_p->ptr; }
  //! \internal Accès à la racine du tableau hors toute protection
  T* data() { return m_p->ptr; }

 public:

  //! Itérateur sur le premier élément du tableau.
  iterator begin() { return iterator(m_p->ptr); }

  //! Itérateur constant sur le premier élément du tableau.
  const_iterator begin() const { return const_iterator(m_p->ptr); }

  //! Itérateur sur le premier élément après la fin du tableau.
  iterator end() { return iterator(m_p->ptr+m_p->size); }

  //! Itérateur constant sur le premier élément après la fin du tableau.
  const_iterator end() const { return const_iterator(m_p->ptr+m_p->size); }

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
  ArrayRange<pointer> range()
  {
    return ArrayRange<pointer>(m_p->ptr,m_p->ptr+m_p->size);
  }
  //! Intervalle d'itération du premier au dernièr élément.
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_p->ptr,m_p->ptr+m_p->size);
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
 */
template<typename T>
class SharedArray
: public Array<T>
{
 protected:

  using AbstractArray<T>::m_p;
  using Array<T>::BD_NoWarning;

 public:

  typedef SharedArray<T> ThatClassType;
  typedef AbstractArray<T> BaseClassType;
  using typename BaseClassType::ConstReferenceType;

 public:

 public:
  //! Créé un tableau vide
  SharedArray() : Array<T>(BD_NoWarning), m_next(nullptr), m_prev(nullptr) {}
  //! Créé un tableau de \a size éléments contenant la valeur \a value.
  SharedArray(Int64 asize,ConstReferenceType value)
  : Array<T>(BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
    this->_resize(asize,value);
  }
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(Int64 asize)
  : Array<T>(BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
    this->_resize(asize);
  }
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(Int32 asize)
  : Array<T>(BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
    this->_resize(asize);
  }
  //! Créé un tableau de \a size éléments contenant la valeur par défaut du type T()
  explicit SharedArray(size_t asize)
  : Array<T>(BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
    this->_resize((Int64)asize);
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray(const ConstArrayView<T>& aview)
  : Array<T>(aview,BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray(const Span<const T>& aview)
  : Array<T>(aview,BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray(const ArrayView<T>& aview)
  : Array<T>(Span<const T>(aview),BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray(const Span<T>& aview)
  : Array<T>(aview,BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
  }
  SharedArray(std::initializer_list<T> alist)
  : Array<T>(alist,BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
  }
  //! Créé un tableau faisant référence à \a rhs.
  SharedArray(const SharedArray<T>& rhs)
  : Array<T>(BD_NoWarning), m_next(nullptr), m_prev(nullptr)
  {
    _initReference(rhs);
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  inline SharedArray(const UniqueArray<T>& rhs);
  //! Change la référence de cette instance pour qu'elle soit celle de \a rhs.
  void operator=(const SharedArray<T>& rhs)
  {
    this->_operatorEqual(rhs);
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  inline void operator=(const UniqueArray<T>& rhs);
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const Span<const T>& rhs)
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
  void operator=(const ArrayView<T>& rhs)
  {
    this->copy(rhs);
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
    return SharedArray<T>(this->constSpan());
  }
 protected:
  void _initReference(const ThatClassType& rhs)
  {
    this->_setMP(rhs.m_p);
    _addReference(&rhs);
    ++m_p->nb_ref;
  }
  //! Mise à jour des références
  void _updateReferences() final
  {
    for( ThatClassType* i = m_prev; i; i = i->m_prev )
      i->_setMP(m_p);
    for( ThatClassType* i = m_next; i; i = i->m_next )
      i->_setMP(m_p);
  }
  //! Mise à jour des références
  Integer _getNbRef() final
  {
    Integer nb_ref = 1;
    for( ThatClassType* i = m_prev; i; i = i->m_prev )
      ++nb_ref;
    for( ThatClassType* i = m_next; i; i = i->m_next )
      ++nb_ref;
    return nb_ref;
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
    if (m_p->nb_ref==0){
      this->_destroy();
      this->_internalDeallocate();
    }
  }
  void _operatorEqual(const ThatClassType& rhs)
  {
    if (&rhs!=this){
      _removeReference();
      _addReference(&rhs);
      ++rhs.m_p->nb_ref;
      --m_p->nb_ref;
      _checkFreeMemory();
      this->_setMP(rhs.m_p);
    }
  }
 private:
  ThatClassType* m_next; //!< Référence suivante dans la liste chaînée
  ThatClassType* m_prev; //!< Référence précédente dans la liste chaînée
 private:
  //! Interdit
  void operator=(const Array<T>& rhs);
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
 * appel au constructeur (UniqueArray(UniqueArray&&) ou assignement
 * (UniqueArray::operator=(UniqueArray&&) par déplacement ainsi que lors
 * de l'appel à UniqueArray::swap(). Si ces appels sont envisagés, il
 * faut garantir que l'allocateur restera valide même après transfert. Si
 * on ne peut pas garantir cela, il est préférable d'utiliser la
 * classe Array qui ne permet pas un tel transfert.
 */
template<typename T>
class UniqueArray
: public Array<T>
{
 protected:

  using AbstractArray<T>::m_p;
  using Array<T>::BD_NoWarning;

 public:

  typedef AbstractArray<T> BaseClassType;
  using typename BaseClassType::ConstReferenceType;

 public:

 public:
  //! Créé un tableau vide
  UniqueArray() : Array<T>(BD_NoWarning) {}
  //! Créé un tableau de \a size éléments contenant la valeur \a value.
  UniqueArray(Int64 req_size,ConstReferenceType value) : Array<T>(BD_NoWarning)
  {
    this->_resize(req_size,value);
  }
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(Int64 asize) : Array<T>(BD_NoWarning)
  {
    this->_resize(asize);
  }
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(Int32 asize) : Array<T>(BD_NoWarning)
  {
    this->_resize(asize);
  }
  //! Créé un tableau de \a asize éléments contenant la valeur par défaut du type T()
  explicit UniqueArray(size_t asize) : Array<T>(BD_NoWarning)
  {
    this->_resize((Int64)asize);
  }
  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  UniqueArray(const ConstArrayView<T>& aview) : Array<T>(Span<const T>(aview),BD_NoWarning)
  {
  }
  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  UniqueArray(const Span<const T>& aview) : Array<T>(aview,BD_NoWarning)
  {
  }
  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  UniqueArray(const ArrayView<T>& aview) : Array<T>(Span<const T>(aview),BD_NoWarning)
  {
  }
  //! Créé un tableau en recopiant les valeurs de la value \a aview.
  UniqueArray(const Span<T>& aview) : Array<T>(aview,BD_NoWarning)
  {
  }
  UniqueArray(std::initializer_list<T> alist) : Array<T>(alist,BD_NoWarning)
  {
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray(const Array<T>& rhs) : Array<T>(rhs.constView(),BD_NoWarning)
  {
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray(const UniqueArray<T>& rhs) : Array<T>(rhs.constView(),BD_NoWarning)
  {
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray(const SharedArray<T>& rhs) : Array<T>(rhs.constView(),BD_NoWarning)
  {
  }
  //! Constructeur par déplacement. \a rhs est invalidé après cet appel
  UniqueArray(UniqueArray<T>&& rhs) ARCCORE_NOEXCEPT : Array<T>(std::move(rhs)) {}
  //! Créé un tableau vide avec un allocateur spécifique \a allocator
  explicit UniqueArray(IMemoryAllocator* allocator) : Array<T>(allocator,0,BD_NoWarning) {}
  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   */
  UniqueArray(IMemoryAllocator* allocator,Int64 asize)
  : Array<T>(allocator,asize,BD_NoWarning) { }
  //! Copie les valeurs de \a rhs dans cette instance.
  void operator=(const Array<T>& rhs)
  {
    this->copy(rhs.constSpan());
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  void operator=(const SharedArray<T>& rhs)
  {
    this->copy(rhs.constSpan());
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  void operator=(const UniqueArray<T>& rhs)
  {
    this->copy(rhs.constSpan());
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
: Array<T>(rhs.constView(),BD_NoWarning)
, m_next(nullptr)
, m_prev(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline void SharedArray<T>::
operator=(const UniqueArray<T>& rhs)
{
  this->copy(rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline std::ostream&
operator<<(std::ostream& o, const AbstractArray<T>& val)
{
  o << ConstArrayView<T>(val);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline bool
operator==(const AbstractArray<T>& rhs, const AbstractArray<T>& lhs)
{
  return operator==(ConstArrayView<T>(rhs),ConstArrayView<T>(lhs));
}

template<typename T> inline bool
operator!=(const AbstractArray<T>& rhs, const AbstractArray<T>& lhs)
{
  return !(rhs==lhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
