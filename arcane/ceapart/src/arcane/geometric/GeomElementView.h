// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomElementView.h                                           (C) 2000-2023 */
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

namespace Arcane::geometric
{

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

  explicit GeomElementConstViewBase(ARCANE_RESTRICT const Real3POD* ptr) : m_s(ptr){}

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

  explicit GeomElementViewBase(ARCANE_RESTRICT Real3POD* ptr) : m_s(ptr){}

 public:

  //! Récupère la valeur du \a ième noeud
  const Real3 operator[](Integer i) const
  {
    return Real3(m_s[i].x,m_s[i].y,m_s[i].z);
  }

  //! Position la valeur du \a i-ème noeud à \a v.
  void setValue(Integer i,Real3 v)
  {
    m_s[i].x = v.x;
    m_s[i].y = v.y;
    m_s[i].z = v.z;
  }

 protected:

  ARCANE_RESTRICT Real3POD* m_s;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometric/GeneratedGeomElementView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
