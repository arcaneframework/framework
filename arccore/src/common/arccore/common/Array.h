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
#ifndef ARCCORE_COMMON_ARRAY_H
#define ARCCORE_COMMON_ARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/AbstractArray.h"

#include <initializer_list>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
template <typename T>
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

  using typename BaseClassType::const_iterator;
  using typename BaseClassType::const_pointer;
  using typename BaseClassType::const_reference;
  using typename BaseClassType::const_reverse_iterator;
  using typename BaseClassType::difference_type;
  using typename BaseClassType::iterator;
  using typename BaseClassType::pointer;
  using typename BaseClassType::reference;
  using typename BaseClassType::reverse_iterator;
  using typename BaseClassType::size_type;
  using typename BaseClassType::value_type;

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
    for (const auto& x : alist)
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
    return ConstArrayView<T>(s, m_ptr);
  }
  operator ArrayView<T>()
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ArrayView<T>(s, m_ptr);
  }
  operator Span<const T>() const
  {
    return Span<const T>(m_ptr, m_md->size);
  }
  operator Span<T>()
  {
    return Span<T>(m_ptr, m_md->size);
  }
  //! Vue constante sur ce tableau
  ConstArrayView<T> constView() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ConstArrayView<T>(s, m_ptr);
  }
  //! Vue constante sur ce tableau
  Span<const T> constSpan() const
  {
    return Span<const T>(m_ptr, m_md->size);
  }
  /*!
   * \brief Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments.
   *
   * Si \a (\a abegin + \a asize) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ConstArrayView<T> subConstView(Int64 abegin, Int32 asize) const
  {
    if (abegin >= m_md->size)
      return {};
    return { this->_clampSizeOffet(abegin, asize), m_ptr + abegin };
  }
  //! Vue mutable sur ce tableau
  ArrayView<T> view() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return ArrayView<T>(s, m_ptr);
  }
  //! Vue immutable sur ce tableau
  Span<const T> span() const
  {
    return Span<const T>(m_ptr, m_md->size);
  }
  //! Vue mutable sur ce tableau
  Span<T> span()
  {
    return Span<T>(m_ptr, m_md->size);
  }
  //! Vue immutable sur ce tableau
  SmallSpan<const T> smallSpan() const
  {
    Integer s = arccoreCheckArraySize(m_md->size);
    return SmallSpan<const T>(m_ptr, s);
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
    return SmallSpan<T>(m_ptr, s);
  }
  /*!
   * \brief Sous-vue à partir de l'élément \a abegin et contenant \a asize éléments.
   *
   * Si \a (\a abegin + \a asize) est supérieur à la taille du tableau,
   * la vue est tronqué à cette taille, retournant éventuellement une vue vide.
   */
  ArrayView<T> subView(Int64 abegin, Integer asize)
  {
    if (abegin >= m_md->size)
      return {};
    return { this->_clampSizeOffet(abegin, asize), m_ptr + abegin };
  }
  /*!
   * \brief Extrait un sous-tableau à à partir d'une liste d'index.
   *
   * Le résultat est stocké dans \a result dont la taille doit être au moins
   * égale à celle de \a indexes.
   */
  void sample(ConstArrayView<Integer> indexes, ArrayView<T> result) const
  {
    const Integer result_size = indexes.size();
    [[maybe_unused]] const Int64 my_size = m_md->size;
    for (Integer i = 0; i < result_size; ++i) {
      Int32 index = indexes[i];
      ARCCORE_CHECK_AT(index, my_size);
      result[i] = m_ptr[index];
    }
  }

 public:

  //! Ajoute l'élément \a val à la fin du tableau
  void add(ConstReferenceType val)
  {
    if (m_md->size >= m_md->capacity)
      this->_internalRealloc(m_md->size + 1, true);
    new (m_ptr + m_md->size) T(val);
    ++m_md->size;
  }
  //! Ajoute \a n élément de valeur \a val à la fin du tableau
  void addRange(ConstReferenceType val, Int64 n)
  {
    this->_addRange(val, n);
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
    ARCCORE_CHECK_AT(index, s);
    for (Int64 i = index; i < (s - 1); ++i)
      m_ptr[i] = m_ptr[i + 1];
    --m_md->size;
    m_ptr[m_md->size].~T();
  }
  /*!
   * \brief Supprime la dernière entité du tableau.
   */
  void popBack()
  {
    ARCCORE_CHECK_AT(0, m_md->size);
    --m_md->size;
    m_ptr[m_md->size].~T();
  }
  //! Elément d'indice \a i. Vérifie toujours les débordements
  T& at(Int64 i)
  {
    arccoreCheckAt(i, m_md->size);
    return m_ptr[i];
  }
  //! Elément d'indice \a i. Vérifie toujours les débordements
  ConstReferenceType at(Int64 i) const
  {
    arccoreCheckAt(i, m_md->size);
    return m_ptr[i];
  }
  //! Positionne l'élément d'indice \a i. Vérifie toujours les débordements
  void setAt(Int64 i, ConstReferenceType value)
  {
    arccoreCheckAt(i, m_md->size);
    m_ptr[i] = value;
  }
  //! Elément d'indice \a i
  ConstReferenceType item(Int64 i) const { return m_ptr[i]; }
  //! Elément d'indice \a i
  void setItem(Int64 i, ConstReferenceType v) { m_ptr[i] = v; }
  //! Elément d'indice \a i
  ConstReferenceType operator[](Int64 i) const
  {
    ARCCORE_CHECK_AT(i, m_md->size);
    return m_ptr[i];
  }
  //! Elément d'indice \a i
  T& operator[](Int64 i)
  {
    ARCCORE_CHECK_AT(i, m_md->size);
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
    ARCCORE_CHECK_AT(m_md->size - 1, m_md->size);
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
  iterator end() { return iterator(m_ptr + m_md->size); }

  //! Itérateur constant sur le premier élément après la fin du tableau.
  const_iterator end() const { return const_iterator(m_ptr + m_md->size); }

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
    return ArrayRange<pointer>(m_ptr, m_ptr + m_md->size);
  }

  //! Intervalle d'itération du premier au dernièr élément.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr, m_ptr + m_md->size);
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
  static int TV_ttf_display_type(const Array<T>* obj);
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
template <typename T>
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
   * \brief Créé un tableau vide avec un allocateur spécifique \a allocator.
   *
   * \warning Using specific allocator for SharedArray is experimental
   */
  explicit SharedArray(IMemoryAllocator* allocator)
  : SharedArray(MemoryAllocationOptions(allocator))
  {
  }

  /*!
   * \brief Créé un tableau vide avec un allocateur spécifique \a allocation_options.
   *
   * \warning Using specific allocator for SharedArray is experimental
   */
  explicit SharedArray(const MemoryAllocationOptions& allocation_options)
  : Array<T>()
  {
    this->_initFromAllocator(allocation_options, 0);
    this->_checkValidSharedArray();
  }

  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   */
  SharedArray(IMemoryAllocator* allocator, Int64 asize)
  : SharedArray(MemoryAllocationOptions(allocator), asize)
  {
  }

  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   */
  SharedArray(const MemoryAllocationOptions& allocation_options, Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(allocation_options, asize);
    this->_resize(asize);
    this->_checkValidSharedArray();
  }

  //!Créé un tableau avec l'allocateur \a allocator en recopiant les valeurs \a rhs.
  SharedArray(IMemoryAllocator* allocator, Span<const T> rhs)
  {
    this->_initFromAllocator(allocator, 0);
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
    for (const auto& x : alist)
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
    return SharedArray<T>(this->allocator(), this->constSpan());
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
    for (ThatClassType* i = m_prev; i; i = i->m_prev)
      i->_setMP2(m_ptr, m_md);
    for (ThatClassType* i = m_next; i; i = i->m_next)
      i->_setMP2(m_ptr, m_md);
  }
  //! Mise à jour des références
  Integer _getNbRef() final
  {
    // NOTE: à vérifier mais lorsque cette méthode est appelée
    // il n'y a toujours qu'une seule référence.
    // TODO fusionner avec l'implémentation de SharedArray2
    Integer nb_ref = 1;
    for (ThatClassType* i = m_prev; i; i = i->m_prev)
      ++nb_ref;
    for (ThatClassType* i = m_next; i; i = i->m_next)
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
    if (m_md->nb_ref == 0) {
      this->_destroy();
      this->_internalDeallocate();
    }
  }
  void _operatorEqual(const ThatClassType& rhs)
  {
    if (&rhs != this) {
      _removeReference();
      _addReference(&rhs);
      ++rhs.m_md->nb_ref;
      --m_md->nb_ref;
      _checkFreeMemory();
      this->_setMP2(rhs.m_ptr, rhs.m_md);
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
template <typename T>
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
  UniqueArray(Int64 req_size, ConstReferenceType value)
  {
    this->_resize(req_size, value);
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
    this->_initFromAllocator(rhs.allocationOptions(), 0);
    this->_initFromSpan(rhs);
  }
  //! Créé un tableau en recopiant les valeurs \a rhs.
  UniqueArray(const UniqueArray<T>& rhs)
  : Array<T>{}
  {
    this->_initFromAllocator(rhs.allocationOptions(), 0);
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
  explicit UniqueArray(IMemoryAllocator* allocator)
  : Array<T>()
  {
    this->_initFromAllocator(allocator, 0);
  }
  //! Créé un tableau vide avec un allocateur spécifique \a allocator
  explicit UniqueArray(MemoryAllocationOptions allocate_options)
  : Array<T>()
  {
    this->_initFromAllocator(allocate_options, 0);
  }
  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   */
  UniqueArray(IMemoryAllocator* allocator, Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(allocator, asize);
    this->_resize(asize);
  }
  /*!
   * \brief Créé un tableau de \a asize éléments avec un
   * allocateur spécifique \a allocator.
   *
   * Si ArrayTraits<T>::IsPODType vaut TrueType, les éléments ne sont pas
   * initialisés. Sinon, c'est le constructeur par défaut de T qui est utilisé.
   */
  UniqueArray(MemoryAllocationOptions allocate_options, Int64 asize)
  : Array<T>()
  {
    this->_initFromAllocator(allocate_options, asize);
    this->_resize(asize);
  }
  //! Créé un tableau avec l'allocateur \a allocator en recopiant les valeurs \a rhs.
  UniqueArray(IMemoryAllocator* allocator, Span<const T> rhs)
  {
    this->_initFromAllocator(allocator, 0);
    this->_initFromSpan(rhs);
  }
  //! Créé un tableau avec l'allocateur \a allocator en recopiant les valeurs \a rhs.
  UniqueArray(MemoryAllocationOptions allocate_options, Span<const T> rhs)
  {
    this->_initFromAllocator(allocate_options, 0);
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
    for (const auto& x : alist)
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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Échange les valeurs de \a v1 et \a v2.
 *
 * L'échange se fait en temps constant et sans réallocation.
 */
template <typename T> inline void
swap(UniqueArray<T>& v1, UniqueArray<T>& v2)
{
  v1.swap(v2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> inline SharedArray<T>::
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

template <typename T> inline void SharedArray<T>::
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
template <typename T> inline Span<const std::byte>
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
template <typename T> inline Span<std::byte>
asWritableBytes(Array<T>& v)
{
  return asWritableBytes(v.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
