// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeLoop.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Interface d'une boucle en temps.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMELOOP_H
#define ARCANE_CORE_ITIMELOOP_H
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
 * \ingroup Module
 * \brief Interface d'une boucle en temps.
 */
class ARCANE_CORE_EXPORT ITimeLoop
{
 public:

  /*! @name Point d'appel
    Endroit ou est utilisé le point d'entrée.
   */
  //@{
  //! appelé pendant la boucle de calcul
  static const char* WComputeLoop;
  //! appelé lors de la lecture du jeu de données
  static const char* WBuild;
  //! appelé pendant l'initialisation, l'initialisation d'une reprise ou d'un nouveau cas
  static const char* WInit;
  //! appelé pour restaurer les variables lors d'un retour arrière
  static const char* WRestore;
  //! appelé après un changement de maillage
  static const char* WOnMeshChanged;
  //! appelé après un raffinement de maillage
  static const char* WOnMeshRefinement;
  //! appelé lors de la terminaison du code.
  static const char* WExit;
  //@}

 public:

  virtual ~ITimeLoop() = default; //!< Libère les ressources.

 public:

  //! Construit la boucle en temps
  virtual void build() = 0;

 public:

  //! Application
  virtual IApplication* application() const = 0;

 public:

  //! Nom de la boucle en temps
  virtual String name() const = 0;

  //! Titre de la boucle en temps
  virtual String title() const = 0;

  //! Positionne le titre de la boucle en temps
  virtual void setTitle(const String&) = 0;

  //! Description de la boucle en temps
  virtual String description() const = 0;

  //! Positionne la description de la boucle en temps
  virtual void setDescription(const String&) = 0;

  //! Liste des noms des modules obligatoires.
  virtual StringCollection requiredModulesName() const = 0;

  //! Positionne la liste des des modules obligatoires.
  virtual void setRequiredModulesName(const StringCollection&) = 0;

  //! Liste des noms des modules facultatifs.
  virtual StringCollection optionalModulesName() const = 0;

  //! Positionne la liste des des modules facultatifs.
  virtual void setOptionalModulesName(const StringCollection&) = 0;

  //! Liste des noms des points d'entrée pour le point d'appel \a where.
  virtual TimeLoopEntryPointInfoCollection entryPoints(const String& where) const = 0;

  //! Positionne la liste des noms des points d'entrée pour le point d'appel \a where
  virtual void setEntryPoints(const String& where, const TimeLoopEntryPointInfoCollection&) = 0;

  //! Liste des classes utilisateurs associées à la boucle en temps.
  virtual StringCollection userClasses() const = 0;

  //! Retourne la liste des classes associées à la boucle en temps.
  virtual void setUserClasses(const StringCollection&) = 0;

  //! Liste services singletons
  virtual TimeLoopSingletonServiceInfoCollection singletonServices() const = 0;

  //! Positionne la liste des services singletons.
  virtual void setSingletonServices(const TimeLoopSingletonServiceInfoCollection& c) = 0;

  //! Options de configuration
  virtual IConfiguration* configuration() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

