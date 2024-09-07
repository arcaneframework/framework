// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IUnitTest.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Interface d'un service de test unitaire.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IUNITTEST_H
#define ARCANE_CORE_IUNITTEST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class XmlNode;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface d'un service de test unitaire.
 */
class IUnitTest
{
 public:

  virtual ~IUnitTest() = default;

 public:

  //! Méthode appelée après lecture du jeu de données mais avant de lire le maillage
  virtual void buildInitializeTest() {}

  //! Méthode appelée après lecture du jeu de données pour initialiser le test
  virtual void initializeTest() =0;

  //! Méthode appelée pour exécuter le test
  virtual void executeTest() =0;

  //! Méthode appelée pour libérer les ressources en fin d'exécution
  virtual void finalizeTest() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface d'un service de test unitaire fournissant
 * un rapport de test sous forme d'un noeud XML.
 */
class IXmlUnitTest
{
 public:

  virtual ~IXmlUnitTest() = default;

 public:

  virtual void buildInitializeTest() {}
  virtual void initializeTest() =0;
  /*!
   * \brief Exécute le test et remplit le noeud XML fournit en paramètre.
   *
   * Retourne false pour que le code s'arrête en erreur, true sinon
   * (utile pour avoir une erreur dans CMake test...).
   */
  virtual bool executeTest(XmlNode& report) = 0;
  virtual void finalizeTest() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
