// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupsSynchronize.h                                     (C) 2000-2016 */
/*                                                                           */
/* Synchronisations des groupes.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMGROUPSSYNCHRONIZE_H
#define ARCANE_MESH_ITEMGROUPSSYNCHRONIZE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/MeshVariable.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour synchroniser les groupes entre sous-domaines.

 Synchroniser un groupe signifie que chaque sous-domaine qui possède un
 type d'entité envoie les infos des groupes aux autres.

 Pour pouvoir utiliser cette classe, il faut être certain que les infos
 de synchronisation sont à jour (IParallelMng::computeSynchronizeInfos()).

 Après avoir créer une instance, il suffit d'appeller la méthode
 synchronize() pour synchroniser le groupe. Par exemple, pour synchroniser
 les groupes de faces:
 \code
 ItemGroupsSynchronize igs(m_mesh->faceFamily());
 igs.synchronize();
 \endcode
 */
class ItemGroupsSynchronize
: public TraceAccessor
{
 public:

  /*!
   * \brief Créé une instance pour synchroniser tous les groupes
   * de la famille \a item_family.
   */
  ItemGroupsSynchronize(IItemFamily* item_family);
  /*!
   * \brief Créé une instance pour synchroniser les groupes \a groups
   * de la famille \a item_family.
   */
  ItemGroupsSynchronize(IItemFamily* item_family,ItemGroupCollection groups);
  ~ItemGroupsSynchronize();

 public:

  //! Synchronise les groupes
  void synchronize();
  /*!
   * \brief Vérifie si les groupes sont synchronisé.
   *
   * \retval le nombre d'entités qui sont désynchronisées.
   */
  Integer checkSynchronize();

 public:

  IItemFamily* m_item_family;
  typedef Int32 IntAggregator; //!< Type employé pour l'aggrégation des communications de groupes
  ItemVariableScalarRefT<IntAggregator> m_var;
  ItemGroupList m_groups;

 private:
  void _setGroups();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

