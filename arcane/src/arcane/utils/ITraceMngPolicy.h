// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITraceMngPolicy.h                                           (C) 2000-2019 */
/*                                                                           */
/* Interface de la politique de gestion des traces.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ITRACEMNGPOLICY_H
#define ARCANE_UTILS_ITRACEMNGPOLICY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de configuration d'un gestionnnaire
 * de trace.
 *
 * Les propriétés définies par cette classe sont utilisées pour initialiser
 * les instances de ITraceMng. Modifier une propriété n'a pas d'influence
 * sur les ITraceMng déjà créés.
 */
class ARCANE_UTILS_EXPORT ITraceMngPolicy
{
 public:
  virtual ~ITraceMngPolicy(){}
 public:
  //! Construit l'instance
  virtual void build() =0;
  /*!
   * \brief Initialise \a trace.
   *
   * Si \a rank vaut 0, alors \a trace est considéré comme le ITraceMng
   * maître. En cas de sortie listing, le suffix aura comme valeur \a rank.
   */
  virtual void initializeTraceMng(ITraceMng* trace,Int32 rank) =0;

  /*!
   * \brief Initialise \a trace avec les infos du parent \a parent_trace.
   *
   * Si les sorties fichiers sont activées, \a trace sortira ses informations
   * dans un fichier suffixé par \a file_suffix.
   * \a parent_trace peut être nul.
   */
  virtual void initializeTraceMng(ITraceMng* trace,ITraceMng* parent_trace,
                                  const String& file_suffix) =0;
  /*!
   * \brief Positionne les valeurs des TraceClassConfig de \a trace via
   * les données contenues dans \a bytes.
   *
   * \a bytes est un buffer contenant une chaîne de caractères au format
   * XML tel que décrit dans la documentation \ref arcanedoc_general_traces.
   *
   * Les instance de TraceClassConfig de \a trace déjà enregistrées avant l'appel à cette
   * méthode sont supprimées.
   */
  virtual void setClassConfigFromXmlBuffer(ITraceMng* trace,ByteConstArrayView bytes) =0;

  /*!
   * \brief Indique si le parallélisme est actif.
   *
   * Cette propriété est positionnée par l'application lors de l'initialisation.
   */
  virtual void setIsParallel(bool v) =0;
  virtual bool isParallel() const =0;

  /*!
   * \brief Indique si les sorties de débug sont actives.
   *
   * Cette propriété est positionnée par l'application lors de l'initialisation.
   */
  virtual void setIsDebug(bool v) =0;
  virtual bool isDebug() const =0;

  /*!
   * \brief Indique si en parallèle tous les rangs sortent les traces dans
   * un fichier.
   */
  virtual void setIsParallelOutput(bool v) =0;
  virtual bool isParallelOutput() const =0;

  /*!
   * \brief Niveau de verbosité sur le flot de sortie standard (stdout).
   *
   * Cette propriété est utilisée lors des appels à initializeTraceMng()
   * pour positionner le niveau de verbosité des sorties standards
   */
  virtual void setStandardOutputVerbosityLevel(Int32 level) =0;
  virtual Int32 standardOutputVerbosityLevel() const =0;

  /*!
   * \brief Niveau de verbosité.
   *
   * Cette propriété est utilisée lors des appels à initializeTraceMng()
   * pour positionner le niveau de verbosité.
   */
  virtual void setVerbosityLevel(Int32 level) =0;
  virtual Int32 verbosityLevel() const =0;

  /*!
   * \brief Indique si un ITraceMng maître sort les traces dans un fichier
   * en plus de la sortie standard.
   *
   * Cette propriété a la valeur \a false par défaut.
   */
  virtual void setIsMasterHasOutputFile(bool active) =0;
  virtual bool isMasterHasOutputFile() const =0;

  /*!
   * Positionne le niveau de verbosité par défaut.
   *
   * Positionne pour \a trace les niveaux de verbosité au niveau \a minimal_level.
   * Si le niveau de verbosité est déjà supérieur \a minimal_level, rien
   * n'est fait.
   * Si \a minimal_level vaut Arccore::Trace::UNSPECIFIED_VERBOSITY_LEVEL,
   * remet le niveau de verbosité à celui spécifié par verbosityLevel()
   * et standardOutputVerbosityLevel().
   */
  virtual void setDefaultVerboseLevel(ITraceMng* trace,Int32 minimal_level) =0;

  virtual void setDefaultClassConfigXmlBuffer(ByteConstSpan bytes) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
