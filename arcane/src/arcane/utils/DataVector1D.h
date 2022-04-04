// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataVector1D.h                                              (C) 2000-2005 */
/*                                                                           */
/* Vecteur de données 1D.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_DATAVECTOR1D_H
#define ARCANE_UTILS_DATAVECTOR1D_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/DataVectorCommon1D.h"
#include "arcane/utils/DefaultAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Vecteur de données 1D d'un type \a T.
 *
 * Cette classe s'utilise par référence, chaque instance gardant
 * une réference sur les autres par l'intermédiaire d'une liste doublement
 * chaînée.
 *
 * Les opérations add() et resize() obligent à remettre à jour toutes
 * les instances qui font références au même tableau. Elle doivent donc
 * être utilisées le moins possible.
 *
 */
template<typename T>
class DataVector1D
: public DataVectorCommon1D<T>
{
 public:

  //! Type de la classe de base
  typedef DataVectorCommon1D<T> BaseClassType;

  //! Type de l'allocateur
  typedef typename BaseClassType::AllocatorType AllocatorType;

  //! Type de cette classe
  typedef DataVector1D<T> ThatClassType;

  static AllocatorType* _defaultAllocator()
  { return DefaultAllocatorT<T>::globalInstance(); }

 public:


 public:

  //! Créé un vecteur sans éléments
  DataVector1D(AllocatorType* allocator = _defaultAllocator())
  : BaseClassType(allocator), m_prev(0), m_next(0) {}

  //! Créé un vecteur de \a nb_elements
  DataVector1D(Integer nb_element,AllocatorType* allocator = _defaultAllocator())
  : BaseClassType(nb_element,allocator), m_prev(0), m_next(0) {}

  //! Constructeur de copie
  DataVector1D(ThatClassType& from)
  : BaseClassType(from), m_prev(0), m_next(0)
    {
      _addReference(from);
    }

  const ThatClassType& operator=(ThatClassType& from)
    {
      if (&from!=this){
        _removeReference();
        _addReference(from);
        BaseClassType* base = this;
        (*base) = from;
      }
      return (*this);
    }

  //! Supprime cette instance
  ~DataVector1D()
    {
      _removeReference();
      _checkFreeMemory();
    }

 public:

  //! Ajoute l'élément \a element en fin de liste
  void add(const T& element)
    {
      this->_add(element);
      _updateReferences();
    }

  //! Change le nombre d'éléments du tableau en \a new_size
  void resize(Integer new_size)
    {
      this->_resize(new_size);
      _updateReferences();
    }

  /*! \brief Réserve de la mémoire.
   *
   * Si la capacité actuelle est supérieure à \a new_capacity, la méthode
   * ne fait rien. Sinon, alloue de la mémoire pour \a max_capacity
   * éléments.
   */
  void reserve(Integer new_capacity)
    {
      this->_reserve(new_capacity);
      _updateReferences();
    }

 private:

  ThatClassType* m_prev; //!< Référence précédente dans la liste chaînée
  ThatClassType* m_next; //!< Référence suivante dans la liste chaînée

 private:

  //! Mise à jour des références
  void _updateReferences()
    {
      BaseClassType& that_ref = *this;
      for( ThatClassType* i = m_prev; i; i = m_prev->m_prev ){
        BaseClassType* bs = i;
        (*bs) = that_ref;
      }
      for( ThatClassType* i = m_next; i; i = m_next->m_next ){
        BaseClassType* bs = i;
        (*bs) = that_ref;
      }
    }

  /*! \brief Insère cette instance dans la liste chaînée.
   * L'instance est insérée à la position de \a new_ref.
   * \pre m_prev==0
   * \pre m_next==0;
   */
  void _addReference(ThatClassType& new_ref)
    {
      ThatClassType* prev = new_ref.m_prev;
      new_ref.m_prev = this;
      m_prev = prev;
      m_next = &new_ref;
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
      if (!m_prev && !m_next)
        this->_destroy();
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

