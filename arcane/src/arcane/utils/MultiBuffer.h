// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiBuffer.h                                               (C) 2000-2011 */
/*                                                                           */
/* Classe template d'un tableau avec tampon.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MULTIBUFFER_H
#define ARCANE_UTILS_MULTIBUFFER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Tampon pour allocation multiple

 Cette classe gère une liste pré-alloué d'élément afin de limiter
 les appels multiples à des allocations (new() ou malloc()).

 Les pré-allocation se font par bloc de \a m_buffer_size éléments

 Pour être utilisé par cette classe, un type doit posséder un constructeur
 par défaut et un opérateur de recopie. Cette classe garantit que les
 pointeurs retournés restent valides tant que cette instance existe.

 Les constructeurs et opérateurs de recopie ne dupliquent pas la mémoire
 mais se contentent juste de conserver la taille du tampon.
*/
template<class T>
class MultiBufferT
{
 public:

  typedef UniqueArray<T> BufferType;

 public:

  MultiBufferT()
  : m_buffer_size(1000), m_current_buffer_size(0), m_nb_in_buffer(0),m_current_buffer(0)
    {
    }
  MultiBufferT(Integer buf_size)
  : m_buffer_size(buf_size), m_current_buffer_size(0), m_nb_in_buffer(0),m_current_buffer(0)
    {
    }
  //! Constructeur de recopie
  MultiBufferT(const MultiBufferT<T>& ref)
  : m_buffer_size(ref.m_buffer_size), m_current_buffer_size(0), m_nb_in_buffer(0), m_current_buffer(0)
    {
    }
  ~MultiBufferT()
    {
      _freeAllocatedBuffers();
    }
 public:
  //! Opérateur de recopie (interdit)
  void operator=(const MultiBufferT<T>& ref)
    {
      if (&ref==this)
        return;
      clear();
      m_buffer_size = ref.m_buffer_size;
    }
 public:
  //! Alloue un nouvel élément
  T* allocOne()
    {
      if (!m_current_buffer)
        _allocateCurrentBuffer();
      T* v = &(*m_current_buffer)[m_nb_in_buffer];
      ++m_nb_in_buffer;
      if (m_nb_in_buffer==m_current_buffer_size)
        m_current_buffer = 0; // Indique que le tampon actuel est plein
      return v;
    }
  //! Alloue \a n éléments
  ArrayView<T> allocMany(Integer n)
    {
      // Si le nombre d'éléments souhaités est supérieure à la taille
      // du tampon, alloue spécifiquement un tampon de la bonne taille
      if (n>m_current_buffer_size){
        BufferType* bt = new BufferType(n);
        m_allocated_buffers.add(bt);
        return *bt;
      }
      if (!m_current_buffer)
        _allocateCurrentBuffer();
      // Si le tampon actuel n'est pas assez grand pour contenir les
      // \a n éléments souhaités, en alloue un autre
      if ((m_nb_in_buffer+n)>=m_current_buffer_size)
        _allocateCurrentBuffer();
      T* v = &(*m_current_buffer)[m_nb_in_buffer];
      m_nb_in_buffer += n;
      if (m_nb_in_buffer==m_current_buffer_size)
        m_current_buffer = 0; // Indique que le tampon actuel est plein
      return ArrayView<T>(n,v);
    }
  void clear()
    {
      m_nb_in_buffer = 0;
      m_current_buffer = 0;
      m_current_buffer_size = 0;
      _freeAllocatedBuffers();

    }
  Integer nbAllocatedBuffer() { return m_allocated_buffers.size(); }
  Integer bufferSize() const { return m_current_buffer_size; }

 protected:
  void _freeAllocatedBuffers()
    {
      for( Integer i=0, s=m_allocated_buffers.size(); i<s; ++i )
        delete m_allocated_buffers[i];
      m_allocated_buffers.clear();
    }
 private:
  Integer m_buffer_size; //!< Nombre d'élément d'un tampon
  Integer m_current_buffer_size; //!< Nombre d'éléments max du tampon courant
  Integer m_nb_in_buffer; //!< Nombre d'éléments dans le tampon actuel.
  BufferType* m_current_buffer; //!< Tampon actuel
  UniqueArray< BufferType* > m_allocated_buffers; //!< Liste de tous les tampons
 private:
  void _allocateCurrentBuffer()
    {
      m_current_buffer = new BufferType(m_buffer_size);
      m_current_buffer_size = m_buffer_size;
      m_allocated_buffers.add(m_current_buffer);
      m_nb_in_buffer = 0;
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

