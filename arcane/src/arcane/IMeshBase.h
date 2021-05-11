// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshBase.h                                     (C) 2000-2021             */
/*                                                                           */
/* Interface for base mesh operations                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_IMESHBASE_H
#define ARCANE_IMESHBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*!
 * \brief Interface for base mesh operations.
 *
 * This interface is created to gradually implement IMesh operations in a
 * new implementation.
 *
 * This interface should be temporary.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshBase {
 public:

  virtual ~IMeshBase() = default;

 public:

  //! Handle sur ce maillage
  virtual const MeshHandle& handle() const =0;

 public:

  //! Nom du maillage
  virtual String name() const =0;

  //! Nombre de noeuds du maillage
  virtual Integer nbNode() =0;

  //! Nombre d'arêtes du maillage
  virtual Integer nbEdge() =0;

  //! Nombre de faces du maillage
  virtual Integer nbFace() =0;

  //! Nombre de mailles du maillage
  virtual Integer nbCell() =0;

  //! Nombre d'éléments du genre \a ik
  virtual Integer nbItem(eItemKind ik) =0;

  //! Gestionnaire de message associé
  virtual ITraceMng* traceMng() =0;

  /*!
   * \brief Dimension du maillage (1D, 2D ou 3D).
   *
   * La dimension correspond à la dimension des éléments mailles (Cell).
   * Si des mailles de plusieurs dimensions sont présentes, c'est la dimension
   * la plus importante qui est retournée.
   * Si la dimension n'a pas encore été positionnée, retourne -1;
   */
  virtual Integer dimension() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // end namespace ARcane

#endif //ARCANE_IMESHBASE_H
