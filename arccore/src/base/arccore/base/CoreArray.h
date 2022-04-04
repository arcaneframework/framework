﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CoreArray.h                                                 (C) 2000-2018 */
/*                                                                           */
/* Tableau simple pour Arccore.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_COREARRAY_H
#define ARCCORE_BASE_COREARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Span.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Tableau interne pour Arccore.
 *
 * Cette classe est privée et ne doit être utilisé que par les classes de Arccore.
 */
template<class DataType>
class CoreArray
{
 private:
  typedef std::vector<DataType> ContainerType;
 public:

  //! Type des éléments du tableau
  typedef DataType value_type;
  //! Type de l'itérateur sur un élément du tableau
  typedef typename ContainerType::iterator iterator;
  //! Type de l'itérateur constant sur un élément du tableau
  typedef typename ContainerType::const_iterator const_iterator;
  //! Type pointeur d'un élément du tableau
  typedef typename ContainerType::pointer pointer;
  //! Type pointeur constant d'un élément du tableau
  typedef const value_type* const_pointer;
  //! Type référence d'un élément du tableau
  typedef value_type& reference;
  //! Type référence constante d'un élément du tableau
  typedef const value_type& const_reference;
  //! Type indexant le tableau
  typedef Int64 size_type;
  //! Type d'une distance entre itérateur éléments du tableau
  typedef ptrdiff_t difference_type;

 public:

  //! Construit un tableau vide.
  CoreArray() {}
  //! Construit un tableau vide.
  CoreArray(ConstArrayView<DataType> v)
  : m_p(v.range().begin(),v.range().end()) {}
  CoreArray(Span<const DataType> v)
  : m_p(v.range().begin(),v.range().end()) {}
 public:

  //! Conversion vers un ConstArrayView
  operator Span<const DataType>() const
  {
    return CoreArray::_constView(m_p);
  }
  //! Conversion vers un ArrayView
  operator Span<DataType>()
  {
    return CoreArray::_view(m_p);
  }

 public:

  //! i-ème élément du tableau.
  inline DataType& operator[](Int64 i)
  {
    return m_p[i];
  }

  //! i-ème élément du tableau.
  inline const DataType& operator[](Int64 i) const
  {
    return m_p[i];
  }

  //! Retourne la taille du tableau
  inline Int64 size() const { return arccoreCheckArraySize(m_p.size()); }

  //! Retourne un iterateur sur le premier élément du tableau
  inline iterator begin() { return m_p.begin(); }
  //! Retourne un iterateur sur le premier élément après la fin du tableau
  inline iterator end() { return m_p.end(); }
  //! Retourne un iterateur constant sur le premier élément du tableau
  inline const_iterator begin() const { return m_p.begin(); }
  //! Retourne un iterateur constant sur le premier élément après la fin du tableau
  inline const_iterator end() const { return m_p.end(); }

  //! Vue constante
  Span<const DataType> constView() const
  {
    return CoreArray::_constView(m_p);
  }

  //! Vue modifiable
  Span<DataType> view()
  {
    return CoreArray::_view(m_p);
  }

  //! Retourne \a true si le tableau est vide
  bool empty() const
  {
    return m_p.empty();
  }

  void resize(Int64 new_size)
  {
    m_p.resize(new_size);
  }
  void reserve(Int64 new_size)
  {
    m_p.reserve(new_size);
  }
  void clear()
  {
    m_p.clear();
  }
  void add(const DataType& v)
  {
    CoreArray::_add(m_p,v);
  }
  DataType& back()
  {
    return m_p.back();
  }
  const DataType& back() const
  {
    return m_p.back();
  }
  const DataType* data() const
  {
    return _data(m_p);
  }
  DataType* data()
  {
    return _data(m_p);
  }
 private:
  static Span<const DataType> _constView(const std::vector<DataType>& c)
  {
    Int64 s = arccoreCheckArraySize(c.size());
    return Span<const DataType>(c.data(),s);
  }
  static ConstArrayView<DataType> _constSmallView(const std::vector<DataType>& c)
  {
    Integer s = arccoreCheckArraySize(c.size());
    return ConstArrayView<DataType>(s,c.data());
  }
  static Span<DataType> _view(std::vector<DataType>& c)
  {
    Int64 s = arccoreCheckArraySize(c.size());
    return Span<DataType>(c.data(),s);
  }
  static ArrayView<DataType> _smallView(std::vector<DataType>& c)
  {
    Integer s = arccoreCheckArraySize(c.size());
    return ArrayView<DataType>(s,c.data());
  }
  static void _add(std::vector<DataType>& c,const DataType& v)
  {
    c.push_back(v);
  }
  static DataType* _data(std::vector<DataType>& c)
  {
    return c.data();
  }
  static const DataType* _data(const std::vector<DataType>& c)
  {
    return c.data();
  }
 private:

  ContainerType m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
