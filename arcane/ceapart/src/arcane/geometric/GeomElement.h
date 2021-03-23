// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomElement.h                                               (C) 2000-2014 */
/*                                                                           */
/* Eléments géométriques.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMELEMENT_H
#define ARCANE_GEOMETRIC_GEOMELEMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Item.h"
#include "arcane/VariableTypes.h"

#include "arcane/geometric/GeomElementView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Classe de base des éléments géométriques.
 *
 * Un élément géométrique contient les coordoonnées des \a NbNode noeuds qui
 * composent cet élément.
 *
 * Pour des raisons de performance, le constructeur
 * par défaut n'initialise pas les coordonnées.
 *
 * Pour récupérer les coordonnées du i-ème noeud de l'élément géométrique,
 * il suffit d'utiliser l'opérateur []. La modification se fait via
 * setItem().
 */
template<int NbNode>
class GeomElementBase
{
 public:
  //! Constructeur vide.
  GeomElementBase(){}
  //! Constructeur à partir des coordonnées \a coords des noeuds de l'entité \a item
  GeomElementBase(const VariableNodeReal3& coords,ItemWithNodes item)
  {
    init(coords,item);
  }
  //! Constructeur à partir des coordonnées \a coords.
  GeomElementBase(Real3ConstArrayView coords)
  {
    for( Integer i=0; i<NbNode; ++i )
      setItem(i,coords[i]);
  }

  //! Récupère la valeur du \a ième noeud
  inline const Real3 s(Integer i) const { return Real3(m_s[i].x,m_s[i].y,m_s[i].z); }
  //! Récupère la valeur du \a ième noeud
  inline const Real3 operator[](Integer i) const { return Real3(m_s[i].x,m_s[i].y,m_s[i].z); }
  //! Positionne la valeur du \a ième noeud à \a v
  inline void setItem(Integer i,const Real3& v) { m_s[i] = v; }
  //! Positionne la valeur du \a ième noeud à Real3(\a x,\a y,\a z)
  inline void setItem(Integer i,Real x,Real y,Real z) { m_s[i] = Real3(x,y,z); }

  /*!
   * \brief Remplit la vue \a view avec les coordonnéees de l'instance.
   */
  void fillView(Real3ArrayView view) const
  {
    for( Integer i=0; i<NbNode; ++i )
      view[i] = s(i);
  }

  /*!
   * \brief Initialise les coordonnées avec celles des noeuds d'entité \a item
   */
  void init(const VariableNodeReal3& coords,ItemWithNodes item)
  {
    for( Integer i=0; i<NbNode; ++i )
      m_s[i] = coords[item.node(i)];
  }

 protected:

  Real3POD m_s[NbNode];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometric/GeneratedGeomElement.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
