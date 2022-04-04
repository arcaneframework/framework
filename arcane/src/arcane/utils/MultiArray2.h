// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiArray2.h                                               (C) 2000-2015 */
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

ARCANE_BEGIN_NAMESPACE

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
template<typename DataType>
class MultiArray2
{
 public:
  typedef typename UniqueArray<DataType>::ConstReferenceType ConstReferenceType;
 public:
  MultiArray2() {}
  MultiArray2(IntegerConstArrayView sizes)
  {
    _resize(sizes);
  }
 private:
  MultiArray2(const MultiArray2<DataType>& rhs);
  void operator=(const MultiArray2<DataType>& rhs);
 protected:
  /*!
   * \brief Constructeur de recopie.
   * Méthode temporaire à supprimer une fois le constructeur et opérateur de recopie
   * supprimé.
   */
  MultiArray2(const MultiArray2<DataType>& rhs,bool do_clone)
  : m_buffer(do_clone ? rhs.m_buffer.clone() : rhs.m_buffer),
    m_indexes(do_clone ? rhs.m_indexes.clone() : rhs.m_indexes),
    m_sizes(do_clone ? rhs.m_sizes.clone() : rhs.m_sizes)
  {
  }
  MultiArray2(ConstMultiArray2View<DataType> aview)
  : m_buffer(aview.m_buffer), m_indexes(aview.m_indexes), m_sizes(aview.m_sizes)
  {
  }
 public:
  ArrayView<DataType> operator[](Integer i)
  {
    return ArrayView<DataType>(m_sizes[i],m_buffer.data() + (m_indexes[i]));
  }
  ConstArrayView<DataType> operator[](Integer i) const
  {
    return ConstArrayView<DataType>(m_sizes[i],m_buffer.data()+ (m_indexes[i]));
  }
 public:
  //! Nombre total d'éléments
  Integer totalNbElement() const { return m_buffer.size(); }
  //! Supprime les éléments du tableau.
  void clear()
  {
    m_buffer.clear();
    m_indexes.clear();
    m_sizes.clear();
  }
  //! Clone le tableau
  MultiArray2<DataType> clone()
  {
    MultiArray2<DataType> new_array;
    new_array.m_buffer = m_buffer.clone();
    new_array.m_indexes = m_indexes.clone();
    new_array.m_sizes = m_sizes.clone();
    return new_array;
  }
  //! Remplit les éléments du tableau avec la valeur \a v
  void fill(const DataType& v)
  {
    m_buffer.fill(v);
  }
  DataType& at(Integer i,Integer j)
  {
    return m_buffer[m_indexes[i]+j];
  }
  void setAt(Integer i,Integer j,ConstReferenceType v)
  {
    return m_buffer.setAt(m_indexes[i]+j,v);
  }
 public:
  //! Nombre d'éléments suivant la première dimension
  Integer dim1Size() const { return m_indexes.size(); }
  
  //! Tableau du nombre d'éléments suivant la deuxième dimension
  IntegerConstArrayView dim2Sizes() const { return m_sizes; }

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
    return MultiArray2View<DataType>(m_buffer,m_indexes,m_sizes);
  }

  //! Vue constante du tableau
  ConstMultiArray2View<DataType> constView() const
  {
    return ConstMultiArray2View<DataType>(m_buffer,m_indexes,m_sizes);
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
  void resize(IntegerConstArrayView new_sizes)
  {
    if (new_sizes.size()==0){
      clear();
    }
    else
      _resize(new_sizes);
  }
 protected:
  ConstArrayView<DataType> _value(Integer i) const
  {
    return ConstArrayView<DataType>(m_sizes[i],m_buffer.data()+ m_indexes[i]);
  }
 protected:
  void _resize(IntegerConstArrayView ar)
  {
    Integer size1 = ar.size();
    // Calcule le nombre d'éléments total
    Integer total_size = 0;
    for( Integer i=0; i<size1; ++i )
      total_size += ar[i];

    // Si on ne change pas le nombre total d'élément, vérifie
    // si le resize est nécessaire
    if (total_size==totalNbElement() && size1==m_indexes.size()){
      bool is_same = true;
      for( Integer i=0; i<size1; ++i )
        if (m_sizes[i]!=ar[i]){
          is_same = false;
          break;
        }
      if (is_same)
        return;
    }
    //_setTotalNbElement(total_size);

    // Alloue le tampon correspondant.
    //T* anew_buffer = new T[total_size+1];
    Integer old_size1 = m_indexes.size();

    SharedArray<DataType> new_buffer(total_size);
			
    // Recopie dans le nouveau tableau les valeurs de l'ancien.
    if (old_size1>size1)
      old_size1 = size1;
    Integer index = 0;
    for( Integer i=0; i<old_size1; ++i ){
      Integer size2 = ar[i];
      Integer old_size2 = m_sizes[i];
      if (old_size2>size2)
        old_size2 = size2;
      ConstArrayView<DataType> cav(_value(i));
      for( Integer j=0; j<old_size2; ++j )
        new_buffer[index+j] = cav[j];
      index += size2;
    }
    m_buffer = new_buffer;

    m_indexes.resize(size1);
    m_sizes.resize(size1);
    for( Integer i2=0, index2=0; i2<size1; ++i2 ){
      Integer size2 = ar[i2];
      m_indexes[i2] = index2;
      m_sizes[i2] = size2;
      index2 += size2;
    }
  }

 protected:

  void _copy(const MultiArray2<DataType>& rhs,bool do_clone)
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
  //! Valeurs
  SharedArray<DataType> m_buffer;
  //! Tableau des indices dans \a m_buffer du premièr élément de la deuxième dimension
  SharedArray<Integer> m_indexes;
  //! Tableau des tailles de la deuxième dimension
  SharedArray<Integer> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Tableau 2D à taille multiple avec sémantique par référence.
 */
template<typename DataType>
class SharedMultiArray2
: public MultiArray2<DataType>
{
 public:

  SharedMultiArray2() {}
  SharedMultiArray2(IntegerConstArrayView sizes)
  : MultiArray2<DataType>(sizes){}
  SharedMultiArray2(ConstMultiArray2View<DataType> view)
  : MultiArray2<DataType>(view){}
  SharedMultiArray2(const SharedMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs,false){}
  SharedMultiArray2(const UniqueMultiArray2<DataType>& rhs);

 public:

  void operator=(const SharedMultiArray2<DataType>& rhs)
  {
    this->_copy(rhs,false);
  }
  void operator=(ConstMultiArray2View<DataType> view)
  {
    this->_copy(view);
  }
  void operator=(const UniqueMultiArray2<DataType>& rhs);

 public:

  //! Clone le tableau
  SharedMultiArray2<DataType> clone() const
  {
    return SharedMultiArray2<DataType>(this->constView());
  }

 private:

  void operator=(const MultiArray2<DataType>& rhs);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Tableau 2D à taille multiple avec sémantique par valeur.
 */
template<typename DataType>
class UniqueMultiArray2
: public MultiArray2<DataType>
{
 public:

  UniqueMultiArray2() {}
  UniqueMultiArray2(IntegerConstArrayView sizes)
  : MultiArray2<DataType>(sizes){}
  UniqueMultiArray2(ConstMultiArray2View<DataType> view)
  : MultiArray2<DataType>(view){}
  UniqueMultiArray2(const SharedMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs,true){}
  UniqueMultiArray2(const UniqueMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs,true){}

 public:

  void operator=(const SharedMultiArray2<DataType>& rhs)
  {
    this->_copy(rhs,true);
  }
  void operator=(ConstMultiArray2View<DataType> view)
  {
    this->_copy(view);
  }
  void operator=(const UniqueMultiArray2<DataType>& rhs)
  {
    this->_copy(rhs,true);
  }

 public:

  //! Clone le tableau
  UniqueMultiArray2<DataType> clone() const
  {
    return UniqueMultiArray2<DataType>(this->constView());
  }

 private:

  void operator=(const MultiArray2<DataType>& rhs);

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> SharedMultiArray2<DataType>::
SharedMultiArray2(const UniqueMultiArray2<DataType>& rhs)
: MultiArray2<DataType>(rhs,true){}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void SharedMultiArray2<DataType>::
operator=(const UniqueMultiArray2<DataType>& rhs)
{
  this->_copy(rhs,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
