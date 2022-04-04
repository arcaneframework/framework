// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshArea.h                                                 (C) 2000-2005 */
/*                                                                           */
/* Interface d'une zone du maillage.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHAREA_H
#define ARCANE_IMESHAREA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 *
 * \brief Interface d'une zone du maillage.
 *
 * Une zone du maillage est un sous-ensemble du maillage définit par
 * une liste de maille et de noeuds.
 */
class ARCANE_CORE_EXPORT IMeshArea
{
 public:

  virtual ~IMeshArea() {} //<! Libère les ressources

 public:

  //! Nombre de noeuds du maillage
  virtual Integer nbNode() =0;

  //! Nombre de mailles du maillage
  virtual Integer nbCell() =0;

 public:

  //! Sous-domaine associé
  virtual ISubDomain* subDomain() =0;

  //! Gestionnaire de trace associé
  virtual ITraceMng* traceMng() =0;

  //! Maillage auquel appartient la zone
  virtual IMesh* mesh() =0;

 public:

  //! Groupe de tous les noeuds
  virtual NodeGroup allNodes() =0;

  //! Groupe de toutes les mailles
  virtual CellGroup allCells() =0;

  //! Groupe de tous les noeuds propres au domaine
  virtual NodeGroup ownNodes() =0;

  //! Groupe de toutes les mailles propres au domaine
  virtual CellGroup ownCells() =0;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

