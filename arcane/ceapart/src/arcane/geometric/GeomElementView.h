// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomElementView.h                                           (C) 2000-2014 */
/*                                                                           */
/* Vues sur les éléments géométriques.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMELEMENTVIEW_H
#define ARCANE_GEOMETRIC_GEOMELEMENTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"

#include "arcane/geometric/GeometricGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Classe de base des vues constantes sur des éléments géométriques.
 *
 * Les vues de ce type ne sont valides que tant que l'instance dont
 * elles sont issues existe. Par conséquent, les vues sont surtout
 * utilisées pour le passage d'argument et ne doivent pas être conservées.
 */
class GeomElementConstViewBase
{
 public:
  GeomElementConstViewBase(ARCANE_RESTRICT const Real3POD* ptr) : m_s(ptr){}
 public:
  //! Récupère la valeur du \a ième noeud
  inline const Real3 operator[](Integer i) const
  {
    return Real3(m_s[i].x,m_s[i].y,m_s[i].z);
  }
  /*!
   * \brief Récupère la valeur du \a ième noeud.
   * \deprecated Utiliser operator[] à la place.
   */
  //ARCANE_DEPRECATED inline const Real3 s(Integer i) const
  inline const Real3 s(Integer i) const
  {
    return Real3(m_s[i].x,m_s[i].y,m_s[i].z);
  }

 protected:
  ARCANE_RESTRICT const Real3POD* m_s;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Classe de base des vues modifiables sur des éléments géométriques.
 *
 * Les vues de ce type ne sont valides que tant que l'instance dont
 * elles sont issues existe. Par conséquent, les vues sont surtout
 * utilisées pour le passage d'argument et ne doivent pas être conservées.
 */
class GeomElementViewBase
{
 public:
  GeomElementViewBase(ARCANE_RESTRICT Real3POD* ptr) : m_s(ptr){}
 public:
  //! Récupère la valeur du \a ième noeud
  inline const Real3 operator[](Integer i) const
  {
    return Real3(m_s[i].x,m_s[i].y,m_s[i].z);
  }

 protected:
  ARCANE_RESTRICT Real3POD* m_s;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometric/GeneratedGeomElementView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
