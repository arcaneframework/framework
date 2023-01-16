﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPostProcessorWriter.h                                      (C) 2000-2010 */
/*                                                                           */
/* Interface d'un écrivain pour les informations de post-traitement.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPOSTPROCESSORWRITER_H
#define ARCANE_IPOSTPROCESSORWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseOptionList;
class IDataWriter;
class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup IO
 * \brief Interface d'un écrivain pour les informations de post-traitement.
 *
 * L'instance doit retourner un IDataWriter (via dataWriter()) pour
 * gérer l'écriture.
 *
 * L'appelant doit positionner les champs de l'instant et lancer
 * L'écriture via l'appel IVariableMng::writePostProcessing(). Par exemple
 * \code
 * IPostProcessorWriter* pp = ...;
 * pp->setBaseDirectoryName(...);
 * pp->setTimes(...);
 * pp->setVariables(...);
 * pp->setGroups(...);
 * IVariableMng* vm = ...;
 * vm->writerPostProcessing(pp);
 * \endcode
 *
 * Avant d'écrire les variables, l'instance IVariableMng va appeler
 * notifyBeginWrite(). Après l'écriture, elle appelle notifyEndWrite().
 */
class ARCANE_CORE_EXPORT IPostProcessorWriter
{
 public:

  //! Libère les ressources
  virtual ~IPostProcessorWriter() {}

 public:

  //! Construit l'instance
  virtual void build() =0;

 public:

  /*!
   * \brief Retourne l'écrivain associé à ce post-processeur.
   */
  virtual IDataWriter* dataWriter() =0;

  /*!
   * \brief Positionne le nom du répertoire de sortie des fichiers.
   * Ce répertoire doit exister.
   */
  virtual void setBaseDirectoryName(const String& dirname) =0;

  //! Nom du répertoire de sortie des fichiers.
  virtual const String& baseDirectoryName() =0;

  /*!
   * \brief Positionne le nom du fichier contenant les sorties
   *
   * Tous les écrivains ne supportent pas de changer le nom
   * de fichier.
   */
  virtual void setBaseFileName(const String& filename) =0;

  //! Nom du fichier contenant les sorties.
  virtual const String& baseFileName() =0;

  //! Set mesh
  //GG:TODO: Mettre virtuel
  virtual void setMesh(IMesh * mesh) { ARCANE_UNUSED(mesh); }

  //! Positionne la liste des temps
  virtual void setTimes(RealConstArrayView times) =0;

  //! Liste des temps sauvés
  virtual RealConstArrayView times() =0;

  //! Positionne la liste des variables à sortir
  virtual void setVariables(VariableCollection variables) =0;

  //! Liste des variables à sauver
  virtual VariableCollection variables() =0;

  //! Positionne la liste des groupes à sortir
  virtual void setGroups(ItemGroupCollection groups) =0;

  //! Liste des groupes à sauver
  virtual ItemGroupCollection groups() =0;

 public:

  //! Notifie qu'une sortie va être effectuée avec les paramètres courants.
  virtual void notifyBeginWrite() =0;

  //! Notifie qu'une sortie vient d'être effectuée.
  virtual void notifyEndWrite() =0;

 public:

  //! Ferme l'écrivain. Après fermeture, il ne peut plus être utilisé
  virtual void close() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

