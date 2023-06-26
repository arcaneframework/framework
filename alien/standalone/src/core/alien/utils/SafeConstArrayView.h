/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class SafeConstArrayView : private ConstArrayView<T>
{
 public:
  //! Classe de base
  typedef ConstArrayView<T> BaseArrayView;

 public:
  //! Constructeur par d�faut
  SafeConstArrayView() {}

  //! Constructeur � partir d'une vue
  SafeConstArrayView(ConstArrayView<T> view)
  : BaseArrayView(view)
  {}

  // Constructeur de copie
  SafeConstArrayView(const SafeConstArrayView<T>& src) = default;

  //! Constructeur � partir d'un array
  SafeConstArrayView(const SharedArray<T>& array)
  : m_array(array)
  {
    this->setArray(m_array.view());
  }

  //! Destructeur de la classe
  virtual ~SafeConstArrayView() {}

  //! Egalit� avec une vue (on lib�re le array)
  SafeConstArrayView& operator=(ConstArrayView<T> view)
  {
    m_array = SharedArray<T>();
    BaseArrayView::operator=(view);
    return *this;
  }

  //! Egalit� avec un array (on r�f�rence le array)
  SafeConstArrayView& operator=(const SharedArray<T>& array)
  {
    m_array = array;
    setArray(m_array.view());
    return *this;
  }

 public:
  //! Type des �l�ments du tableau
  typedef typename BaseArrayView::value_type value_type;
  //! Type de l'it�rateur constant sur un �l�ment du tableau
  typedef typename BaseArrayView::const_iterator const_iterator;
  //! Type pointeur constant d'un �l�ment du tableau
  typedef typename BaseArrayView::const_pointer const_pointer;
  //! Type r�f�rence constante d'un �l�ment du tableau
  typedef typename BaseArrayView::const_reference const_reference;
  //! Type indexant le tableau
  typedef typename BaseArrayView::size_type size_type;
  //! Type d'une distance entre it�rateur �l�ments du tableau
  typedef typename BaseArrayView::difference_type difference_type;

 public:
  //! \brief Sous-vue (constante) � partir de l'�l�ment \a begin et contenant \a size
  //! �l�ments.
  using BaseArrayView::subView;
  //! Sous-vue (constante) � partir de l'�l�ment \a begin et contenant \a size �l�ments.
  using BaseArrayView::subConstView;
  //! Sous-vue correspondant � l'interval \a index sur \a nb_interval
  using BaseArrayView::subViewInterval;
  //! i-�me �l�ment du tableau.
  using BaseArrayView::operator[];
  //! i-�me �l�ment du tableau.
  using BaseArrayView::item;
  //! Nombre d'�l�ments du tableau
  using BaseArrayView::size;
  //! Nombre d'�l�ments du tableau
  using BaseArrayView::length;
  //! Iterateur sur le premier �l�ment du tableau
  using BaseArrayView::begin;
  //! Iterateur sur le premier �l�ment apr�s la fin du tableau
  using BaseArrayView::end;
  //! \a true si le tableau est vide (size()==0)
  using BaseArrayView::empty;
  //! \a true si le tableau contient l'�l�ment de valeur \a v
  using BaseArrayView::contains;

 public:
  //! Retourne une vue du SafeConstArrayView
  BaseArrayView view() const { return *this; }

 private:
  SharedArray<T> m_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
