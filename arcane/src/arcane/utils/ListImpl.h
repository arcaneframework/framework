// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ListImpl.h                                                  (C) 2000-2018 */
/*                                                                           */
/* Implémentation de la classe Tableau.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_LISTIMPL_H
#define ARCANE_UTILS_LISTIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CollectionImpl.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/DefaultAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCANE_UTILS_EXPORT void throwOutOfRangeException();
extern "C" ARCANE_UTILS_EXPORT void throwNullReference();

class IRessourceMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Tableau avec allocateur virtuel.
 *
 * C'est à la classe virtuelle de détruire les objets dans le
 * destructeur virtuel.
 */
template<class T>
class ListImplBase
: public CollectionImplT<T>
{
 public:

  typedef CollectionImplT<T> BaseClass;

 public:
  
  typedef const T& ObjectRef;

  //! Type des éléments du tableau
  typedef T value_type;
  //! Type de l'itérateur sur un élément du tableau
  typedef value_type* iterator;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef const value_type* const_iterator;
  //! Type pointeur d'un élément du tableau
  typedef value_type* pointer;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type référence d'un élément du tableau
  typedef value_type& reference;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;

  //! Type d'un itérateur sur tout le tableau
  typedef IterT< ListImplBase<T> > iter;
  //! Type d'un itérateur constant sur tout le tableau
  typedef ConstIterT< ListImplBase<T> > const_iter;

 public:

  //! Construit un tableau vide.
  ListImplBase() : BaseClass(), m_capacity(0), m_ptr(0), m_allocator(0) {}

  //! Libère les éléments du tableau.
  virtual ~ListImplBase() ARCANE_NOEXCEPT { _deallocate(_ptr()); m_allocator->destroy(); }

 public:

  //! Recopie le tableau \a s.
  void assign(const ListImplBase<T>& s)
    { _arrayCopy(s); }
  //! Recopie le tableau \a s.
  void assign(const ConstArrayView<T>& s)
    { _arrayCopy(s); }
  //! Recopie le tableau \a s.
  void assign(const ArrayView<T>& s)
    { _arrayCopy(s); }

 public:

  void setList(const ListImplBase<T>& v)
    { m_ptr = v.m_ptr; _setCount(v.count()); }

 public:

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  T& operator[](Integer i)
    {
#ifdef ARCANE_CHECK
      Arcane::arcaneCheckAt(i,this->count());
#endif
      return m_ptr[i];
    }

  /*!
   * \brief i-ème élément du tableau.
   *
   * En mode \a check, vérifie les débordements.
   */
  const T&  operator[](Integer i) const
    {
#ifdef ARCANE_CHECK
      Arcane::arcaneCheckAt(i,this->count());
#endif
      return m_ptr[i];
    }

  //! Retourne un iterateur sur le premier élément du tableau
  iterator begin() { return m_ptr; }
  //! Retourne un iterateur sur le premier élément après la fin du tableau
  iterator end() { return m_ptr+this->count(); }
  //! Retourne un iterateur constant sur le premier élément du tableau
  const_iterator begin() const { return m_ptr; }
  //! Retourne un iterateur constant sur le premier élément après la fin du tableau
  const_iterator end() const { return m_ptr+this->count(); }

  //! Retourne un iterateur sur le premier élément du tableau
  T* begin2() const { return m_ptr; }
  //! Retourne un iterateur sur le premier élément après la fin du tableau
  T* end2() const { return m_ptr+this->count(); }

 public:
  
  //! Applique le fonctor \a f à tous les éléments du tableau
  template<class Function> Function
  each(Function f)
  {
    std::for_each(begin(),end(),f);
    return f;
  }

 public:

  void setAllocator(IAllocatorT<T>* allocator)
  {
    m_allocator = allocator;
  }

 public:
  
  /*! \brief Signale qu'il faut réserver de la mémoire pour \a new_capacity éléments
   * Il s'agit juste d'une indication. La classe dérivée est libre de ne
   * pas en tenir compte.
   */
  void reserve(Integer new_capacity)
  {
    T* new_ptr = _allocate(new_capacity);
    _setPtr(new_ptr);
  }

  /*! \brief Retourne le nombre d'éléments alloués du tableau.
   * Il s'agit juste d'une indication. La classe dérivée est libre de ne
   * pas en tenir compte.
   */
  Integer capacity() const
  { return m_capacity; }

  //! Ajoute l'élément \a elem à la fin du tableau
  virtual void add(ObjectRef elem)
  {
    this->onInsert();
    Integer s = this->count();
    if (s>=m_capacity){
      _resize(_ptr(),s+1);
    }
    else
      this->_setCount(s+1);
    _ptr()[s] = elem;
    this->onInsertComplete(_ptr()+s,s);
  }

  virtual bool remove(ObjectRef element)
  {
    Integer i = 0;
    Integer s = this->count();
    for( ; i<s; ++i )
      if (_ptr()[i]==element){
        _removeAt(i);
        return true;
      }
    throwOutOfRangeException();
    return false;
  }

  virtual void removeAt(Integer index)
  {
    Integer s = this->count();
    if (index>=s)
      throwOutOfRangeException();
    _removeAt(index);
  }

  virtual iterator find(ObjectRef element)
  {
    Integer i = 0;
    Integer s = this->count();
    for( ; i<s; ++i )
      if (_ptr()[i]==element)
        return begin()+i;
    return end();
  }

  virtual const_iterator find(ObjectRef element) const
  {
    Integer i = 0;
    Integer s = this->count();
    for( ; i<s; ++i )
      if (_ptr()[i]==element)
        return begin()+i;
    return end();
  }

  virtual bool contains(ObjectRef element) const
  {
    const_iterator i = find(element);
    return (i!=end());
  }

  virtual EnumeratorImplBase* enumerator() const;

 protected:

  void _removeAt(Integer index)
  {
    Integer s = this->count();
    T* ptr = _ptr();
    T* remove_ob = ptr+index;
    this->onRemove();
    for( Integer i=index+1; i<s; ++i )
      ptr[i-1] = ptr[i];
    resize(s-1);
    this->onRemoveComplete(remove_ob,index);
  }

 public:

  /*!\brief Modifie la taille du tableau.
   * \a new_size est le nouveau nombre d'éléments du tableau.
   */
  void resize(Integer new_size)
  {
    _resize(_ptr(),new_size);
  }

 protected:

  void _setCapacity(Integer v) { m_capacity = v; }
  Integer _capacity() const { return m_capacity; }

 protected:

  void _arrayCopy(const ListImplBase<T>& array)
  {
    _arrayCopy(array.data(),array.count());
  }
  
  void _arrayCopy(const ConstArrayView<T>& array)
  {
    _arrayCopy(array.data(),array.size());
  }

  void _arrayCopy(const ArrayView<T>& array)
  {
    _arrayCopy(array.data(),array.size());
  }
  void _arrayCopy(const T* from_ptr,Integer from_size)
  {
    if (from_size==0){
      this->clear();
    }
    else{
      if (from_size<capacity()){
        this->_unguardedCopy(_ptr(),from_ptr,from_size);
        this->_setCount(from_size);
      }
      else{
        T* new_ptr = _resize(from_ptr,from_size);
        this->_setList(new_ptr,from_size);
      }
    }
  }

 protected:

  /*!
   * \brief Retourne un pointeur sur le tableau
   *
   * \warning Il est préférable de ne pas utiliser cette méthode pour
   * accéder à un élément du tableau car
   * ce pointeur peut être invalidé par un redimensionnement du tableau.
   * De plus, accéder aux éléments du tableau par ce pointeur ne permet
   * aucune vérification de dépassement, même en mode DEBUG.
   */
  inline T* _ptr() { return m_ptr; }

  inline T* _ptr() const { return m_ptr; }

  /*!
   * \brief Modifie le pointeur et la taille du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setList(T* v,Integer s){ m_ptr = v; this->_setCount(s); }

  /*!
   * \brief Modifie le pointeur du début du tableau.
   *
   * C'est à la classe dérivée de vérifier la cohérence entre le pointeur
   * alloué et la dimension donnée.
   */
  inline void _setPtr(T* v)
    { m_ptr = v; }

 protected:

  void _unguardedCopy(T* to_ptr,const T* from_ptr,Integer s)
  {
    T* p = to_ptr;
    _setPtr(to_ptr);
    Integer z = s;
    const T* optr = from_ptr;
    while (z--)
      *p++ = *optr++;
  }

 private:
  
  Integer m_capacity;
  T* m_ptr;  //!< Pointeur sur le tableau
  IAllocatorT<T>* m_allocator;

 private:

 private:
	
  /*! \brief Modifie la taille du tableau.
   * Cette méthode doit retourner un pointeur sur un tableau alloué par la classe
   * dérivée et ce tableau doit avoir une taille au moins égale à \a new_size.
   * La gestion mémoire est à la charge de la classe dérivée.
   * Il est permis de renvoyer le même pointeur que le pointeur actuel.
   * Il n'est pas permis de retourner un pointeur nul.
   * \param new_size nouvelle taille du tableau.
   * \return le pointeur alloué.
   */
  T* _resize(const T* from_ptr,Integer new_size)
  {
    Integer c = _capacity();
    while(new_size>c)
      c = (c==0) ? 4 : (c*2);
    _setCapacity(c);
    T* old_ptr = _ptr();
    T* new_ptr = _allocate(c);
    if (from_ptr){
      this->_unguardedCopy(new_ptr,from_ptr,this->count());
      this->_setCount(new_size);
      if (from_ptr==old_ptr)
        _deallocate(old_ptr);
    }
    else
      _setList(new_ptr,new_size);
    return new_ptr;
  }
  
  /*! \brief Signale qu'il faut libérer la mémoire associée au pointeur.
   * \param s le pointeur sur le tableau à libèrer. Il est identique à _ptr().
   */
  void _deallocate(T* ptr)
    { m_allocator->deallocate(ptr,m_capacity); }

  T* _allocate(Integer new_capacity)
    { return m_allocator->allocate(new_capacity); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 */
template<class T>
class ListImplT
: public ListImplBase<T>
{
 public:

  typedef ListImplBase<T> BaseClass;

 public:

  ListImplT() { _setAllocator(); }
  ListImplT(const ConstArrayView<T>& array)
    { _setAllocator(); this->_arrayCopy(array); }
  ListImplT(const ArrayView<T>& array)
    { _setAllocator(); this->_arrayCopy(array); }
  ListImplT(const ListImplT<T>& array)
  : BaseClass() { _setAllocator(); _arrayCopy(array); }
  ListImplT(const Collection<T>& array)
  : BaseClass()
  {
    _setAllocator();
    for( typename Collection<T>::Enumerator i(array); ++i; ){
      BaseClass::add(*i);
    }
  }
  ListImplT(const EnumeratorT<T>& enumerator)
  : BaseClass()
  {
    _setAllocator();
    for( EnumeratorT<T> i(enumerator); ++i; ){
      BaseClass::add(*i);
    }
  }
  virtual ~ListImplT() ARCANE_NOEXCEPT {}

  void assign(const Collection<T>& array)
  {
    this->clear();
    for( typename Collection<T>::Enumerator i(array); ++i; ){
      this->add(*i);
    }
  }
  void assign(const EnumeratorT<T>& enumerator)
  {
    this->clear();
    for( EnumeratorT<T> i(enumerator); ++i; ){
      this->add(*i);
    }
  }
 protected:
  
 private:

  void _setAllocator()
  {
    this->setAllocator(new DefaultAllocatorT<T>());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 */
template<class T>
class ListEnumeratorImplT
: public EnumeratorImplBase
{
 public:

  typedef T* Ptr;

 public:

  ListEnumeratorImplT(Ptr begin,Ptr end)
  : m_begin(begin), m_current(begin-1), m_end(end) {}
  virtual ~ListEnumeratorImplT() {}

 public:

  virtual void reset() { m_current = m_begin-1; }
  virtual bool moveNext() { ++m_current; return m_current<m_end; }
  virtual void* current() { return m_current; }
  virtual const void* current() const { return m_current; }

 private:

  Ptr m_begin;
  Ptr m_current;
  Ptr m_end;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T> EnumeratorImplBase* ListImplBase<T>::
enumerator() const
{
  return new ListEnumeratorImplT<T>(begin2(),end2());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
