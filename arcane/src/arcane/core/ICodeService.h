// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICodeService.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface du service du code.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICODESERVICE_H
#define ARCANE_CORE_ICODESERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un chargeur de cas.
 * \ingroup StandardService
 */
class ARCANE_CORE_EXPORT ICodeService
: public IService
{
 public:

  ~ICodeService() = default; //!< Libère les ressources

 public:

  /*! \brief Crée une session.
   *
   * L'instance doit appeler IApplication::addSession().
   */
  virtual ISession* createSession() = 0;

  /*!
   * \brief Analyse les arguments de la ligne de commandes.
   *
   * Le tableau \a args ne contient que les arguments qui n'ont
   * pas été interprétés par Arcane.
   *
   * Les arguments reconnus doivent être supprimés de la liste.
   *
   * \retval true si l'exéctution doit s'arrêter,
   * \retval false si elle continue normalement
   */
  virtual bool parseArgs(StringList& args) = 0;

  /*!
   * \brief Créé et charge le cas avec les infos \a sdbi
   * pour la session \a session.
   */
  virtual ISubDomain* createAndLoadCase(ISession* session, const SubDomainBuildInfo& sdbi) = 0;

  /*!
   * \brief Initialise la session \a session.
   *
   * \param is_continue indique si on est en reprise
   * Le cas doit déjà avoir été chargé par loadCase()
   */
  virtual void initCase(ISubDomain* sub_domain, bool is_continue) = 0;

  //! Retourne si le code accepte l'exécution.
  virtual bool allowExecution() const = 0;

  /*! \brief Retourne la liste des extensions de fichier traitées par l'instance.
   * L'extension ne comprend pas le '.'.
   */
  virtual StringCollection validExtensions() const = 0;

  /*!
   * \brief Unité de longueur utilisé par le code.
   *
   * Cela doit valoir 1.0 si le code utilise le système international et donc
   * le mêtre comme unité de longueur. Si l'unité est le centimètre par
   * exemple, la valeur est 0.01.
   *
   * Cette valeur peut être utilisée par exemple lors de la lecture du
   * maillage si le format de maillage supporte la notion d'unité de
   * longueur.
   */
  virtual Real ARCANE_DEPRECATED lengthUnit() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
