// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMng.h                                                  (C) 2000-2020 */
/*                                                                           */
/* Interface du gestionnaire des maillages.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHMNG_H
#define ARCANE_IMESHMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des maillages.
 *
 * Cette interface gère une liste de maillages et permet de créér des maillages
 * ou récupérer un maillage existant à partir de son nom.
 *
 * La création de maillage se fait via 'IMeshFactoryMng' dont une instance
 * est récupérable via meshFactoryMng(). La création effective de maillage ne peut
 * avoir lieu qu'après lecture du jeu de données. Il est par contre possible
 * de créér une référence (via createMeshHandle()) sur un maillage à n'importe quel
 * moment.
 */
class ARCANE_CORE_EXPORT IMeshMng
{
 public:

  //! Libère les ressources.
  virtual ~IMeshMng() = default;

 public:

  //! Gestionnaire de trace associé à ce gestionnaire
  virtual ITraceMng* traceMng() const = 0;

  //! Fabrique de maillages associée à ce gestionnaire
  virtual IMeshFactoryMng* meshFactoryMng() const = 0;

  //! Gestionnaire de variables associé à ce gestionnaire
  virtual IVariableMng* variableMng() const = 0;

 public:

  /*!
   * \brief Recherche le maillage de nom \a name.
   *
   * Si le maillage n'est pas trouvé, la méthode lance une exception
   * si \a throw_exception vaut \a true ou retourne *nullptr* si \a throw_exception
   * vaut \a false.
   */
  virtual MeshHandle* findMeshHandle(const String& name, bool throw_exception) = 0;

  /*!
   * \brief Recherche le maillage de nom \a name.
   *
   * Si le maillage n'est pas trouvé, la méthode lance une exception.
   */
  virtual MeshHandle findMeshHandle(const String& name) = 0;

  /*!
   * \brief Créé et retourne un handle pour un maillage de nom \a name.
   *
   * Lève une exception si un handle associé à ce nom existe déjà.
   */
  virtual MeshHandle createMeshHandle(const String& name) = 0;

  /*!
   * \brief Détruit le maillage associé à \a handle.
   *
   * Le maillage doit être un maillage implémentant IPrimaryMesh.
   *
   * \warning \a handle ne doit plus être utilisé après cet appel
   * et le maillage associé non plus. S'il reste des références à ces deux
   * objets, le comportement est indéfini.
   */
  virtual void destroyMesh(MeshHandle handle) = 0;

  //! Handle pour le maillage par défaut.
  virtual MeshHandle defaultMeshHandle() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
