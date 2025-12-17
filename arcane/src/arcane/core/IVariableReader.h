// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableReader.h                                           (C) 2000-2025 */
/*                                                                           */
/* Lecture de variables pour l'initialisation et au cours du calcul.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEREADER_H
#define ARCANE_CORE_IVARIABLEREADER_H
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
 * \internal
 * \brief Lecture de variables au cours du calcul.
 */
class IVariableReader
{
 public:

  virtual ~IVariableReader() = default;

 public:

  //! Positionne le chemin du répertoire contenant les données
  virtual void setBaseDirectoryName(const String& path) = 0;
  //! Positionne le nom du fichier contenant les données.
  virtual void setBaseFileName(const String& filename) = 0;
  /*!
   * \brief Initialise le lecteur.
   *
   * \a is_start est vrai si on est au démarrage du calcul.
   */
  virtual void initialize(bool is_start) = 0;
  /*!
   * \brief.Positionne la liste des variables qu'on souhaite relire.
   * Cet appel doit avoir lieu avant initialize().
   */
  virtual void setVariables(VariableCollection vars) = 0;
  //! Mise à jour des variables pour le temps \a wanted_time
  virtual void updateVariables(Real wanted_time) = 0;
  /*!
   * \brief Interval de temps des valeurs pour la variable \a var.
   * Les données de la variable \a var existent pour les temps
   * comprit entre \a a.x et \a a.y avec \a a la valeur
   * de retour.
   *
   * Cet appel est valide uniquement après appel à initialize().
   */
  virtual Real2 timeInterval(IVariable* var) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
