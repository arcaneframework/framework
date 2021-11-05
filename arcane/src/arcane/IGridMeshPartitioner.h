// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGridMeshPartitioner.h                                      (C) 2000-2021 */
/*                                                                           */
/* Interface d'un partitionneur de maillage sur une grille.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IGRIDMESHPARTITIONER_H
#define ARCANE_IGRIDMESHPARTITIONER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMeshPartitionerBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un partitionneur de maillage sur une grille.
 *
 * Ce partitionneur redistribue le maillage dans une grille 2D ou 3D.
 *
 * La grille est composée d'un ensemble de partie, chaque partie étant
 * définie par sa boite englobante (les coordonnées des deux
 * points aux extrémités de la grille) et un indice (i,j,k). Les
 * dimensions de chaque partie peuvent être différentes mais elles
 * doivent être conformes: toutes les parties d'indice \a i identiques
 * doivent avoir la même coordonnée \a x de boite englobante.
 *
 * ------------------------
 * | 0,2 | 1,2   |2,2| 3,2 |
 * ------------------------
 * | 0,1 | 1,1   |2,1| 3,1 |
 * ------------------------
 * | 0,0 | 1,0   |2,0| 3,0 |
 * ------------------------
 *
 */
class ARCANE_CORE_EXPORT IGridMeshPartitioner
: public IMeshPartitionerBase
{
 public:
  /*!
   * \brief Positionne la bounding box de notre sous-domaine.
   *
   * Pour que l'algorithme fonctionne, il ne faut pas de recouvrement
   * entre les bounding box des sous-domaines.
   */
  virtual void setBoundingBox(Real3 min_val, Real3 max_val) = 0;
  /*!
   * \brief Indice (i,j,k) de la partie.
   *
   * Les indices commencent à zéro. En 1D ou 2D, la valeur de \a k vaut \a
   * (-1). En 1D, la valeur de \a j vaut \a (-1)
   */
  virtual void setPartIndex(Int32 i, Int32 j, Int32 k) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
