// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiArray2.h                                               (C) 2000-2025 */
/*                                                                           */
/* Tableau 2D à taille multiple.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MULTIARRAY2_H
#define ARCANE_UTILS_MULTIARRAY2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/MultiArray2View.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Classe de base des tableau 2D à taille multiple.
 *
 * Cette classe gère les tableaux 2D dont le nombre d'éléments de la
 * deuxième dimension est variable.
 * Par exemple:
 * \code
 *  UniqueArray<Int32> sizes(3); // Tableau avec 3 éléments
 *  sizes[0] = 1; sizes[1] = 2; sizes[2] = 4;
 *  // Construit le tableau avec sizes comme tailles
 *  MultiArray2<Int32> v(sizes);
 *  info() << " size=" << v.dim1Size(); // affiche 3
 *  info() << " size[0]=" << v[0].size(); // affiche 1
 *  info() << " size[1]=" << v[1].size(); // affiche 2
 *  info() << " size[2]=" << v[2].size(); // affiche 4
 * \endcode
 *
 * \note Les indices sont conservés via le type Int32.
 * Le nombre total d'éléments du tableau est donc limité à 2^31
 *
 * Il est possible de redimensionner (via la méthode resize()) le
 * tableau tout en conservant ses valeurs mais pour des raisons de performance, ces
 * redimensionnements se font sur tout le tableau (il n'est pas possible
 * de redimensionner uniquement pour un seul élément, par exemple v[5].resize(3)).
 * 
 * Comme pour Array et Array2, les instances de cette classe ne sont
 * pas copiables ni assignables. Pour obtenir cette fonctionnalité, il faut
 * utiliser la classe SharedMultiArray2 pour une sémantique par référence
 * ou UniqueMultiArray2 pour une sémantique par valeur.
 */
template <typename DataType>
class MultiArray2
{
 public:

  using ConstReferenceType = typename UniqueArray<DataType>::ConstReferenceType;
  using ThatClass = MultiArray2<DataType>;

 public:

  MultiArray2() = default;
  // TODO: Rendre accessible uniquement à UniqueMultiArray2 ou SharedMultiArray2
  explicit MultiArray2(ConstArrayView<Int32> sizes)
  {
    _resize(sizes);
  }

 public:

  MultiArray2(const ThatClass& rhs) = delete;
  ThatClass& operator=(const ThatClass& rhs) = delete;

 protected:

  /*!
   * \brief Constructeur de recopie.
   * Méthode temporaire à supprimer une fois le constructeur et opérateur de recopie
   * supprimé.
   */
  MultiArray2(const MultiArray2<DataType>& rhs, bool do_clone)
  : m_buffer(do_clone ? rhs.m_buffer.clone() : rhs.m_buffer)
  , m_indexes(do_clone ? rhs.m_indexes.clone() : rhs.m_indexes)
  , m_sizes(do_clone ? rhs.m_sizes.clone() : rhs.m_sizes)
  {
  }
  explicit MultiArray2(ConstMultiArray2View<DataType> aview)
  : m_buffer(aview.m_buffer)
  , m_indexes(aview.m_indexes)
  , m_sizes(aview.m_sizes)
  {
  }
  explicit MultiArray2(const MemoryAllocationOptions& allocation_options)
  : m_buffer(allocation_options)
  , m_indexes(allocation_options)
  , m_sizes(allocation_options)
  {}
  // TODO: Rendre accessible uniquement à UniqueMultiArray2 ou SharedMultiArray2
  MultiArray2(const MemoryAllocationOptions& allocation_options, ConstArrayView<Int32> sizes)
  : MultiArray2(allocation_options)
  {
    _resize(sizes);
  }

 public:

  ArrayView<DataType> operator[](Integer i)
  {
    return ArrayView<DataType>(m_sizes[i], m_buffer.data() + (m_indexes[i]));
  }
  ConstArrayView<DataType> operator[](Integer i) const
  {
    return ConstArrayView<DataType>(m_sizes[i], m_buffer.data() + (m_indexes[i]));
  }

 public:

  //! Nombre total d'éléments
  Int32 totalNbElement() const { return m_buffer.size(); }

  //! Supprime les éléments du tableau.
  void clear()
  {
    m_buffer.clear();
    m_indexes.clear();
    m_sizes.clear();
  }
  //! Remplit les éléments du tableau avec la valeur \a v
  void fill(const DataType& v)
  {
    m_buffer.fill(v);
  }
  DataType& at(Integer i, Integer j)
  {
    return m_buffer[m_indexes[i] + j];
  }
  ConstReferenceType at(Integer i, Integer j) const
  {
    return m_buffer[m_indexes[i] + j];
  }
  void setAt(Integer i, Integer j, ConstReferenceType v)
  {
    return m_buffer.setAt(m_indexes[i] + j, v);
  }

 public:

  //! Nombre d'éléments suivant la première dimension
  Int32 dim1Size() const { return m_indexes.size(); }

  //! Tableau du nombre d'éléments suivant la deuxième dimension
  ConstArrayView<Int32> dim2Sizes() const { return m_sizes; }

  //! Opérateur de conversion vers une vue modifiable
  operator MultiArray2View<DataType>()
  {
    return view();
  }

  //! Opérateur de conversion vers une vue constante.
  operator ConstMultiArray2View<DataType>() const
  {
    return constView();
  }

  //! Vue modifiable du tableau
  MultiArray2View<DataType> view()
  {
    return MultiArray2View<DataType>(m_buffer, m_indexes, m_sizes);
  }

  //! Vue constante du tableau
  ConstMultiArray2View<DataType> constView() const
  {
    return ConstMultiArray2View<DataType>(m_buffer, m_indexes, m_sizes);
  }

  //! Vue modifiable du tableau
  MultiArray2SmallSpan<DataType> span()
  {
    return { m_buffer.smallSpan(), m_indexes, m_sizes };
  }

  //! Vue constante du tableau
  MultiArray2SmallSpan<const DataType> span() const
  {
    return { m_buffer, m_indexes, m_sizes };
  }

  //! Vue constante du tableau
  MultiArray2SmallSpan<const DataType> constSpan() const
  {
    return { m_buffer.constSmallSpan(), m_indexes, m_sizes };
  }

  //! Vue du tableau sous forme de tableau 1D
  ArrayView<DataType> viewAsArray()
  {
    return m_buffer.view();
  }

  //! Vue du tableau sous forme de tableau 1D
  ConstArrayView<DataType> viewAsArray() const
  {
    return m_buffer.constView();
  }

  //! Retaille le tableau avec comme nouvelles tailles \a new_sizes
  void resize(ConstArrayView<Int32> new_sizes)
  {
    if (new_sizes.empty()) {
      clear();
    }
    else
      _resize(new_sizes);
  }

 protected:

  ConstArrayView<DataType> _value(Integer i) const
  {
    return ConstArrayView<DataType>(m_sizes[i], m_buffer.data() + m_indexes[i]);
  }

 protected:

  void _resize(ConstArrayView<Int32> ar)
  {
    Integer size1 = ar.size();
    // Calcule le nombre d'éléments total
    // TODO: Vérifier qu'on ne dépasse pas la valeur max d'un Int32
    Integer total_size = 0;
    for (Integer i = 0; i < size1; ++i)
      total_size += ar[i];

    // Si on ne change pas le nombre total d'éléments, vérifie
    // si le resize est nécessaire
    if (total_size == totalNbElement() && size1 == m_indexes.size()) {
      bool is_same = true;
      for (Integer i = 0; i < size1; ++i)
        if (m_sizes[i] != ar[i]) {
          is_same = false;
          break;
        }
      if (is_same)
        return;
    }

    Integer old_size1 = m_indexes.size();

    SharedArray<DataType> new_buffer(m_buffer.allocationOptions(), total_size);

    // Recopie dans le nouveau tableau les valeurs de l'ancien.
    if (old_size1 > size1)
      old_size1 = size1;
    Integer index = 0;
    for (Integer i = 0; i < old_size1; ++i) {
      Integer size2 = ar[i];
      Integer old_size2 = m_sizes[i];
      if (old_size2 > size2)
        old_size2 = size2;
      ConstArrayView<DataType> cav(_value(i));
      for (Integer j = 0; j < old_size2; ++j)
        new_buffer[index + j] = cav[j];
      index += size2;
    }
    m_buffer = new_buffer;

    m_indexes.resize(size1);
    m_sizes.resize(size1);
    for (Integer i2 = 0, index2 = 0; i2 < size1; ++i2) {
      Integer size2 = ar[i2];
      m_indexes[i2] = index2;
      m_sizes[i2] = size2;
      index2 += size2;
    }
  }

 protected:

  void _copy(const MultiArray2<DataType>& rhs, bool do_clone)
  {
    m_buffer = do_clone ? rhs.m_buffer.clone() : rhs.m_buffer;
    m_indexes = do_clone ? rhs.m_indexes.clone() : rhs.m_indexes;
    m_sizes = do_clone ? rhs.m_sizes.clone() : rhs.m_sizes;
  }
  void _copy(ConstMultiArray2View<DataType> aview)
  {
    m_buffer = aview.m_buffer;
    m_indexes = aview.m_indexes;
    m_sizes = aview.m_sizes;
  }

 private:

  //! Tableau des Valeurs
  SharedArray<DataType> m_buffer;
  //! Tableau des indices dans \a m_buffer du premièr élément de la deuxième dimension
  SharedArray<Int32> m_indexes;
  //! Tableau des tailles de la deuxième dimension
  SharedArray<Int32> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Tableau 2D à taille multiple avec sémantique par référence.
 */
template <typename DataType>
class SharedMultiArray2
: public MultiArray2<DataType>
{
 public:

  using ThatClass = SharedMultiArray2<DataType>;

 public:

  SharedMultiArray2() = default;
  explicit SharedMultiArray2(ConstArrayView<Int32> sizes)
  : MultiArray2<DataType>(sizes)
  {}
  SharedMultiArray2(ConstMultiArray2View<DataType> view)
  : MultiArray2<DataType>(view)
  {}
  SharedMultiArray2(const SharedMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs, false)
  {}
  SharedMultiArray2(const UniqueMultiArray2<DataType>& rhs);

 public:

  ThatClass& operator=(const ThatClass& rhs)
  {
    if (&rhs != this)
      this->_copy(rhs, false);
    return (*this);
  }
  void operator=(ConstMultiArray2View<DataType> view)
  {
    this->_copy(view);
  }
  ThatClass& operator=(const UniqueMultiArray2<DataType>& rhs);
  void operator=(const MultiArray2<DataType>& rhs) = delete;

 public:

  //! Clone le tableau
  SharedMultiArray2<DataType> clone() const
  {
    return SharedMultiArray2<DataType>(this->constView());
  }

 private:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Tableau 2D à taille multiple avec sémantique par valeur.
 */
template <typename DataType>
class UniqueMultiArray2
: public MultiArray2<DataType>
{
 public:

  using ThatClass = UniqueMultiArray2<DataType>;

 public:

  UniqueMultiArray2() = default;
  explicit UniqueMultiArray2(ConstArrayView<Int32> sizes)
  : MultiArray2<DataType>(sizes)
  {}
  explicit UniqueMultiArray2(IMemoryAllocator* allocator)
  : UniqueMultiArray2(MemoryAllocationOptions(allocator))
  {}
  explicit UniqueMultiArray2(const MemoryAllocationOptions& allocation_options)
  : MultiArray2<DataType>(allocation_options)
  {}
  UniqueMultiArray2(const MemoryAllocationOptions& allocation_options,
                    ConstArrayView<Int32> sizes)
  : MultiArray2<DataType>(allocation_options, sizes)
  {}
  UniqueMultiArray2(ConstMultiArray2View<DataType> view)
  : MultiArray2<DataType>(view)
  {}
  UniqueMultiArray2(const SharedMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs, true)
  {}
  UniqueMultiArray2(const UniqueMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs, true)
  {}

 public:

  ThatClass& operator=(const SharedMultiArray2<DataType>& rhs)
  {
    this->_copy(rhs, true);
    return (*this);
  }
  ThatClass& operator=(ConstMultiArray2View<DataType> view)
  {
    // TODO: Vérifier que \a view n'est pas dans ce tableau
    this->_copy(view);
    return (*this);
  }
  ThatClass& operator=(const UniqueMultiArray2<DataType>& rhs)
  {
    if (&rhs != this)
      this->_copy(rhs, true);
    return (*this);
  }
  ThatClass& operator=(const MultiArray2<DataType>& rhs) = delete;

 public:

  //! Clone le tableau
  UniqueMultiArray2<DataType> clone() const
  {
    return UniqueMultiArray2<DataType>(this->constView());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> SharedMultiArray2<DataType>::
SharedMultiArray2(const UniqueMultiArray2<DataType>& rhs)
: MultiArray2<DataType>(rhs, true)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> SharedMultiArray2<DataType>& SharedMultiArray2<DataType>::
operator=(const UniqueMultiArray2<DataType>& rhs)
{
  this->_copy(rhs, true);
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
