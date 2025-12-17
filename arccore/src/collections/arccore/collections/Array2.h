// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2.h                                                    (C) 2000-2025 */
/*                                                                           */
/* Tableau 2D classique.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_ARRAY2_H
#define ARCCORE_COLLECTIONS_ARRAY2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NotImplementedException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/Span2.h"

#include "arccore/common/Array.h"
#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Classe représentant un tableau 2D classique.
 *
 * Les instances de cette classe ne sont ni copiables ni affectable. Pour créer un
 * tableau copiable, il faut utiliser SharedArray2 (pour une sémantique par
 * référence) ou UniqueArray2 (pour une sémantique par valeur comme la STL).
 */
template<typename DataType>
class Array2
: private AbstractArray<DataType>
{
 protected:

  enum CloneBehaviour
  {
    CB_Clone,
    CB_Shared
  };
  enum InitBehaviour
  {
    IB_InitWithDefault,
    IB_NoInit
  };

 private:

  using BaseClass = AbstractArray<DataType>;
  typedef AbstractArray<DataType> Base;
  typedef typename Base::ConstReferenceType ConstReferenceType;

 protected:

  using BaseClass::m_ptr;
  using BaseClass::m_md;
  using BaseClass::_setMP2;
  using BaseClass::_setMP;
  using BaseClass::_destroy;
  using BaseClass::_internalDeallocate;
  using BaseClass::_initFromAllocator;
  using BaseClass::_checkValidSharedArray;

 public:

  using AbstractArray<DataType>::allocator;
  using AbstractArray<DataType>::setMemoryLocationHint;
  using AbstractArrayBase::setDebugName;
  using AbstractArrayBase::debugName;
  using AbstractArrayBase::allocationOptions;

 protected:

  Array2() : AbstractArray<DataType>() {}
  //! Créé un tableau de \a size1 * \a size2 éléments.
  Array2(Int64 size1,Int64 size2)
  : AbstractArray<DataType>() 
  {
    resize(size1,size2);
  }
  Array2(ConstArray2View<DataType> rhs) : AbstractArray<DataType>()
  {
    this->copy(rhs);
  }
  Array2(const Span2<const DataType>& rhs) : AbstractArray<DataType>()
  {
    this->copy(rhs);
  }

 protected:

  //! Créé un tableau vide avec un allocateur spécifique \a allocator
  explicit Array2(IMemoryAllocator* allocator)
  : AbstractArray<DataType>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(allocator), 0);
  }
  /*!
   * \brief Créé un tableau de \a size1 * \a size2 éléments avec
   * un allocateur spécifique \a allocator.
   */
  Array2(IMemoryAllocator* allocator,Int64 size1,Int64 size2)
  : AbstractArray<DataType>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(allocator), size1 * size2);
    resize(size1,size2);
  }
  ~Array2() override = default;

 private:

  //! Interdit
  Array2<DataType>& operator=(const Array2<DataType>& rhs) = delete;
  //! Interdit
  Array2(const Array2<DataType>& rhs) = delete;

 protected:

  //! Constructeur par déplacement. Uniquement valide pour UniqueArray2.
  Array2(Array2<DataType>&& rhs) : AbstractArray<DataType>(std::move(rhs)) {}

 public:

  // TODO: retourner un Span.
  ArrayView<DataType> operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i,m_md->dim1_size);
    return ArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size),m_ptr + (m_md->dim2_size*i));
  }
  // TODO: retourner un Span
  ConstArrayView<DataType> operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_md->dim1_size);
    return ConstArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size),m_ptr + (m_md->dim2_size*i));
  }
  // TODO: retourner un Span.
  ArrayView<DataType> operator()(Int64 i)
  {
    ARCCORE_CHECK_AT(i,m_md->dim1_size);
    return ArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size),m_ptr + (m_md->dim2_size*i));
  }
  // TODO: retourner un Span
  ConstArrayView<DataType> operator()(Int64 i) const
  {
    ARCCORE_CHECK_AT(i,m_md->dim1_size);
    return ConstArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size),m_ptr + (m_md->dim2_size*i));
  }
  DataType& operator()(Int64 i,Int64 j)
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
  ConstReferenceType operator()(Int64 i,Int64 j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  DataType& operator[](Int64 i,Int64 j)
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
  ConstReferenceType operator[](Int64 i,Int64 j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
#endif
  DataType item(Int64 i,Int64 j)
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
  void setItem(Int64 i,Int64 j,ConstReferenceType v)
  {
    ARCCORE_CHECK_AT2(i,j,m_md->dim1_size,m_md->dim2_size);
    m_ptr[ (m_md->dim2_size*i) + j ] = v;
  }
  //! Elément d'indice \a i. Vérifie toujours les débordements
  ConstArrayView<DataType> at(Int64 i) const
  {
    arccoreCheckAt(i,m_md->dim1_size);
    return this->operator[](i);
  }
  //! Elément d'indice \a i. Vérifie toujours les débordements
  ArrayView<DataType> at(Int64 i)
  {
    arccoreCheckAt(i,m_md->dim1_size);
    return this->operator[](i);
  }
  DataType at(Int64 i,Int64 j)
  {
    arccoreCheckAt(i,m_md->dim1_size);
    arccoreCheckAt(j,m_md->dim1_size);
    return m_ptr[ (m_md->dim2_size*i) + j ];
  }
  void fill(ConstReferenceType v)
  {
    this->_fill(v);
  }
  void clear()
  {
    this->resize(0,0);
  }
  [[deprecated("Y2021: Use SharedArray2::clone() or UniqueArray2::clone()")]]
  Array2<DataType> clone()
  {
    return Array2<DataType>(this->constSpan());
  }
  /*!
   * \brief Redimensionne l'instance à partir des dimensions de \a rhs
   * et copie dedans les valeurs de \a rhs.
   */
  void copy(Span2<const DataType> rhs)
  {
    _resizeAndCopyView(rhs);
  }
  //! Capacité (nombre d'éléments alloués) du tableau
  Integer capacity() const { return Base::capacity(); }

  //! Capacité (nombre d'éléments alloués) du tableau
  Int64 largeCapacity() const { return Base::largeCapacity(); }

  //! Réserve de la mémoire pour \a new_capacity éléments
  void reserve(Int64 new_capacity) { Base::_reserve(new_capacity); }

  // Réalloue la mémoire au plus juste.
  void shrink() { Base::_shrink(); }

  //! Réalloue la mémoire avoir une capacité proche de \a new_capacity.
  void shrink(Int64 new_capacity) { Base::_shrink(new_capacity); }

  // Réalloue la mémoire au plus juste.
  void shrink_to_fit() { Base::_shrink(); }

  //! Vue du tableau sous forme de tableau 1D
  ArrayView<DataType> viewAsArray()
  {
    return ArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->size),m_ptr);
  }
  //! Vue du tableau sous forme de tableau 1D
  ConstArrayView<DataType> viewAsArray() const
  {
    return ConstArrayView<DataType>(ARCCORE_CAST_SMALL_SIZE(m_md->size),m_ptr);
  }
  //! Vue du tableau sous forme de tableau 1D
  Span<DataType> to1DSpan()
  {
    return Span<DataType>(m_ptr,m_md->size);
  }
  //! Vue du tableau sous forme de tableau 1D
  Span<const DataType> to1DSpan() const
  {
    return Span<const DataType>(m_ptr,m_md->size);
  }

 public:

  operator Array2View<DataType>()
  {
    return view();
  }
  operator ConstArray2View<DataType>() const
  {
    return constView();
  }
  operator Span2<const DataType>() const
  {
    return Span2<const DataType>(m_ptr,m_md->dim1_size,m_md->dim2_size);
  }
  operator Span2<DataType>()
  {
    return Span2<DataType>(m_ptr,m_md->dim1_size,m_md->dim2_size);
  }
  Array2View<DataType> view()
  {
    return Array2View<DataType>(m_ptr,ARCCORE_CAST_SMALL_SIZE(m_md->dim1_size),ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size));
  }
  ConstArray2View<DataType> constView() const
  {
    return ConstArray2View<DataType>(m_ptr,ARCCORE_CAST_SMALL_SIZE(m_md->dim1_size),ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size));
  }
  Span2<DataType> span()
  {
    return Span2<DataType>(m_ptr,m_md->dim1_size,m_md->dim2_size);
  }
  Span2<const DataType> constSpan() const
  {
    return Span2<const DataType>(m_ptr,m_md->dim1_size,m_md->dim2_size);
  }
 public:

  Integer dim2Size() const { return ARCCORE_CAST_SMALL_SIZE(m_md->dim2_size); }
  Integer dim1Size() const { return ARCCORE_CAST_SMALL_SIZE(m_md->dim1_size); }
  Int64 largeDim2Size() const { return m_md->dim2_size; }
  Int64 largeDim1Size() const { return m_md->dim1_size; }
  void add(const DataType& value)
  {
    Base::_addRange(value,m_md->dim2_size);
    ++m_md->dim1_size;
    _arccoreCheckSharedNull();
  }

  /*!
   * \brief Redimensionne uniquement la première dimension en laissant
   * la deuxième à l'identique.
   *
   * Les éventuelles nouvelles valeurs sont initialisées avec le constructeur par défaut.
   */
  void resize(Int64 new_size)
  {
    _resize(new_size,IB_InitWithDefault);
  }

  /*!
   * \brief Redimensionne uniquement la première dimension en laissant
   * la deuxième à l'identique.
   *
   * Les éventuelles nouvelles valeurs NE sont PAS initialisées.
   */
  void resizeNoInit(Int64 new_size)
  {
    _resize(new_size,IB_NoInit);
  }

  /*!
   * \brief Réalloue les deux dimensions.
   *
   * Les éventuelles nouvelles valeurs sont initialisées avec le constructeur par défaut.
   */
  void resize(Int64 new_size1,Int64 new_size2)
  {
    _resize(new_size1,new_size2,IB_InitWithDefault);
  }

  /*!
   * \brief Réalloue les deux dimensions.
   *
   * Les éventuelles nouvelles valeurs NE sont PAS initialisées.
   */
  void resizeNoInit(Int64 new_size1,Int64 new_size2)
  {
    _resize(new_size1,new_size2,IB_NoInit);
  }

  //! Nombre total d'éléments (dim1Size()*dim2Size())
  Integer totalNbElement() const { return ARCCORE_CAST_SMALL_SIZE(m_md->dim1_size*m_md->dim2_size); }

  //! Nombre total d'éléments (largeDim1Size()*largeDim2Size())
  Int64 largeTotalNbElement() const { return m_md->dim1_size*m_md->dim2_size; }

 protected:

  //! Redimensionne uniquement la première dimension en laissant la deuxième à l'identique
  void _resize(Int64 new_size,InitBehaviour rb)
  {
    Int64 old_size = m_md->dim1_size;
    if (new_size==old_size)
      return;
    _resize2(new_size,m_md->dim2_size,rb);
    m_md->dim1_size = new_size;
    _arccoreCheckSharedNull();
  }

  //! Réalloue les deux dimensions
  void _resize(Int64 new_size1,Int64 new_size2,InitBehaviour rb)
  {
    if (new_size2==m_md->dim2_size){
      _resize(new_size1,rb);
    }
    else if (totalNbElement()==0){
      _resizeFromEmpty(new_size1,new_size2,rb);
    }
    else if (new_size2<m_md->dim2_size){
      _resizeSameDim1ReduceDim2(new_size2,rb);
      _resize(new_size1,rb);
    }
    else if (new_size2>m_md->dim2_size){
      _resizeSameDim1IncreaseDim2(new_size2,rb);
      _resize(new_size1,rb);
    }
    else
      throw NotImplementedException("Array2::resize","already sized");
  }

  void _resizeFromEmpty(Int64 new_size1,Int64 new_size2,InitBehaviour rb)
  {
    _resize2(new_size1,new_size2,rb);
    m_md->dim1_size = new_size1;
    m_md->dim2_size = new_size2;
    _arccoreCheckSharedNull();
  }

  void _resizeSameDim1ReduceDim2(Int64 new_size2,InitBehaviour rb)
  {
    ARCCORE_ASSERT((new_size2<m_md->dim2_size),("Bad Size"));
    Int64 n = m_md->dim1_size;
    Int64 n2 = m_md->dim2_size;
    for( Int64 i=0; i<n; ++i ){
      for( Int64 j=0; j<new_size2; ++j )
        m_ptr[(i*new_size2)+j] = m_ptr[(i*n2)+j];
    }
    _resize2(n,new_size2,rb);
    m_md->dim2_size = new_size2;
    _arccoreCheckSharedNull();
  }

  void _resizeSameDim1IncreaseDim2(Int64 new_size2,InitBehaviour rb)
  {
    ARCCORE_ASSERT((new_size2>m_md->dim2_size),("Bad Size"));
    Int64 n = m_md->dim1_size;
    Int64 n2 = m_md->dim2_size;
    _resize2(n,new_size2,rb);
    // Recopie en partant de la fin pour éviter tout écrasement
    for( Int64 i=(n-1); i>=0; --i ){
      for( Int64 j=(n2-1); j>=0; --j )
        m_ptr[(i*new_size2)+j] = m_ptr[(i*n2)+j];
    }
    m_md->dim2_size = new_size2;
    _arccoreCheckSharedNull();
    if (rb==IB_InitWithDefault){
      // Remet les valeurs par défaut pour les nouveaux éléments
      for( Int64 i=0; i<n; ++i ){
        for( Int64 j=n2; j<new_size2; ++j ){
          m_ptr[(i*new_size2)+j] = DataType{};
        }
      }
      
    }
  }

  void _resize2(Int64 d1,Int64 d2,InitBehaviour rb)
  {
    Int64 new_size = d1*d2;
    // Si la nouvelle taille est nulle, il faut tout de meme faire une allocation
    // pour stocker les valeurs dim1_size et dim2_size (sinon, elle seraient
    // dans TrueImpl::shared_null
    if (new_size==0)
      this->_reserve(4);
    if (rb==IB_InitWithDefault)
      Base::_resize(new_size,DataType());
    else if (rb==IB_NoInit)
      Base::_resize(new_size);
    else
      throw NotSupportedException("Array2::_resize2","invalid value InitBehaviour");
  }

  void _move(Array2<DataType>& rhs)
  {
    Base::_move(rhs);
  }

  void _swap(Array2<DataType>& rhs)
  {
    Base::_swap(rhs);
  }

  // Uniquement valide pour UniqueArray2
  void _assignFromArray2(const Array2<DataType>& rhs)
  {
    if (&rhs==this)
      return;
    if (rhs.allocator()==this->allocator()){
      this->copy(rhs.constSpan());
    }
    else{
      this->_assignFromArray(rhs);
      m_md->dim1_size = rhs.dim1Size();
      m_md->dim2_size = rhs.dim2Size();
    }
  }

  void _resizeAndCopyView(Span2<const DataType> rhs)
  {
    Int64 total = rhs.totalNbElement();
    if (total==0){
      // Si la taille est nulle, il faut tout de meme faire une allocation
      // pour stocker les valeurs dim1_size et dim2_size (sinon, elle seraient
      // dans TrueImpl::shared_null)
      this->_reserve(4);
    }
    Span<const DataType> aview(rhs.data(),total);
    Base::_resizeAndCopyView(aview);
    m_md->dim1_size = rhs.dim1Size();
    m_md->dim2_size = rhs.dim2Size();
    _arccoreCheckSharedNull();
  }

 private:

  void _arccoreCheckSharedNull()
  {
    if (!m_ptr)
      ArrayMetaData::throwNullExpected();
    if (!m_md->is_not_null)
      ArrayMetaData::throwNotNullExpected();
  }

 protected:

  void _copyMetaData(const Array2<DataType>& rhs)
  {
    AbstractArray<DataType>::_copyMetaData(rhs);
  }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Vecteur de données 2D partagées avec sémantique par référence.
 *
 * \code
 * SharedArray2<int> a1(5,7);
 * SharedArray2<int> a2;
 * a2 = a1;
 * a1[3][6] = 1;
 * a2[1][2] = 2;
 * \endcode
 *
 * Dans l'exemple précédent, \a a1 et \a a2 font référence à la même zone
 * mémoire et donc a1[3][6] aura la même valeur que a2[1][2].
 *
 * Pour avoir un vecteur qui recopie les éléments lors de l'affectation,
 * il faut utiliser la classe UniqueArray2.
 *
 * Pour plus d'informations, se reporter à SharedArray.
 */
template<typename T>
class SharedArray2
: public Array2<T>
{
 protected:

  using Array2<T>::m_ptr;
  using Array2<T>::m_md;

 public:

  typedef SharedArray2<T> ThatClassType;
  typedef AbstractArray<T> BaseClassType;
  typedef typename BaseClassType::ConstReferenceType ConstReferenceType;

 public:

  //! Créé un tableau vide
  SharedArray2() = default;
  //! Créé un tableau de \a size1 * \a size2 éléments.
  SharedArray2(Int64 size1,Int64 size2)
  {
    this->resize(size1,size2);
    this->_checkValidSharedArray();
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray2(const ConstArray2View<T>& view)
  {
    this->copy(view);
    this->_checkValidSharedArray();
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  SharedArray2(const Span2<const T>& view)
  {
    this->copy(view);
    this->_checkValidSharedArray();
  }
  //! Créé un tableau faisant référence à \a rhs.
  SharedArray2(const SharedArray2<T>& rhs)
  : Array2<T>()
  {
    _initReference(rhs);
    this->_checkValidSharedArray();
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  inline SharedArray2(const UniqueArray2<T>& rhs);
  //! Change la référence de cette instance pour qu'elle soit celle de \a rhs.
  void operator=(const SharedArray2<T>& rhs)
  {
    this->_operatorEqual(rhs);
    this->_checkValidSharedArray();
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  inline void operator=(const UniqueArray2<T>& rhs);
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const ConstArray2View<T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  void operator=(const Span2<const T>& rhs)
  {
    this->copy(rhs);
    this->_checkValidSharedArray();
  }
  //! Détruit l'instance
  ~SharedArray2() override
  {
    _removeReference();
  }
 public:

  //! Clone le tableau
  SharedArray2<T> clone() const
  {
    return SharedArray2<T>(this->constSpan());
  }

 protected:

  void _initReference(const ThatClassType& rhs)
  {
    this->_setMP(rhs.m_ptr);
    this->_copyMetaData(rhs);
    _addReference(&rhs);
    ++m_md->nb_ref;
  }
  //! Mise à jour des références
  void _updateReferences() override final
  {
    for( ThatClassType* i = m_prev; i; i = i->m_prev )
      i->_setMP2(m_ptr,m_md);
    for( ThatClassType* i = m_next; i; i = i->m_next )
      i->_setMP2(m_ptr,m_md);
  }
  //! Mise à jour des références
  Integer _getNbRef() override final
  {
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
  void operator=(const Array2<T>& rhs);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 *
 * \brief Vecteur de données 2D avec sémantique par valeur (style STL).
 *
 * Cette classe est le pendant de UniqueArray pour les tableaux 2D.
 */
template<typename T>
class UniqueArray2
: public Array2<T>
{
 public:

  typedef AbstractArray<T> BaseClassType;
  typedef typename BaseClassType::ConstReferenceType ConstReferenceType;

 public:

 public:
  //! Créé un tableau vide
  UniqueArray2() : Array2<T>() {}
  //! Créé un tableau de \a size1 * \a size2 éléments.
  explicit UniqueArray2(Int64 size1,Int64 size2) : Array2<T>()
  {
    this->resize(size1,size2);
  }
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  UniqueArray2(const Span2<const T>& view) : Array2<T>(view) {}
  //! Créé un tableau en recopiant les valeurs de la value \a view.
  UniqueArray2(const ConstArray2View<T>& view) : Array2<T>(view) {}
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray2(const Array2<T>& rhs)
  : Array2<T>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(rhs.allocator()), 0);
    this->_resizeAndCopyView(rhs);
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray2(const UniqueArray2<T>& rhs)
  : Array2<T>()
  {
    this->_initFromAllocator(MemoryAllocationOptions(rhs.allocator()), 0);
    this->_resizeAndCopyView(rhs);
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray2(const SharedArray2<T>& rhs) : Array2<T>(rhs.constSpan()) {}
  //! Créé un tableau vide avec un allocateur spécifique \a allocator
  explicit UniqueArray2(IMemoryAllocator* allocator)
  : Array2<T>(allocator) {}
  /*!
   * \brief Créé un tableau de \a size1 * \a size2 éléments avec
   * un allocateur spécifique \a allocator.
   */
  UniqueArray2(IMemoryAllocator* allocator,Int64 size1,Int64 size2)
  : Array2<T>(allocator,size1,size2) { }
  //! Constructeur par déplacement. \a rhs est invalidé après cet appel
  UniqueArray2(UniqueArray2<T>&& rhs) ARCCORE_NOEXCEPT : Array2<T>(std::move(rhs)) {}
  //! Copie les valeurs de \a rhs dans cette instance.
  UniqueArray2& operator=(const Array2<T>& rhs)
  {
    this->_assignFromArray2(rhs);
    return (*this);
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  UniqueArray2& operator=(const SharedArray2<T>& rhs)
  {
    this->_assignFromArray2(rhs);
    return (*this);
  }
  //! Copie les valeurs de \a rhs dans cette instance.
  UniqueArray2& operator=(const UniqueArray2<T>& rhs)
  {
    this->_assignFromArray2(rhs);
    return (*this);
  }
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  UniqueArray2& operator=(ConstArray2View<T> rhs)
  {
    this->copy(rhs);
    return (*this);
  }
  //! Copie les valeurs de la vue \a rhs dans cette instance.
  UniqueArray2& operator=(const Span2<const T>& rhs)
  {
    this->copy(rhs);
    return (*this);
  }
  //! Opérateur de recopie par déplacement. \a rhs est invalidé après cet appel.
  UniqueArray2& operator=(UniqueArray2<T>&& rhs) ARCCORE_NOEXCEPT
  {
    this->_move(rhs);
    return (*this);
  }
  //! Détruit le tableau
  ~UniqueArray2() override = default;
 public:
  /*!
   * \brief Échange les valeurs de l'instance avec celles de \a rhs.
   *
   * L'échange se fait en temps constant et sans réallocation.
   */
  void swap(UniqueArray2<T>& rhs) ARCCORE_NOEXCEPT
  {
    this->_swap(rhs);
  }
  //! Clone le tableau
  UniqueArray2<T> clone()
  {
    return UniqueArray2<T>(this->constSpan());
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
swap(UniqueArray2<T>& v1,UniqueArray2<T>& v2)
{
  v1.swap(v2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline SharedArray2<T>::
SharedArray2(const UniqueArray2<T>& rhs)
{
  this->copy(rhs);
  this->_checkValidSharedArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> inline void SharedArray2<T>::
operator=(const UniqueArray2<T>& rhs)
{
  this->copy(rhs);
  this->_checkValidSharedArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
