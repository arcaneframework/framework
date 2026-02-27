// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPostProcessorWriter.h                                      (C) 2000-2026 */
/*                                                                           */
/* Interface d'un écrivain pour les informations de post-traitement.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPOSTPROCESSORWRITER_H
#define ARCANE_CORE_IPOSTPROCESSORWRITER_H
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
  virtual ~IPostProcessorWriter() = default;

 public:

  //! Construit l'instance
  virtual void build() = 0;

 public:

  /*!
   * \brief Retourne l'écrivain associé à ce post-processeur.
   *
   * Le pointeur retourné n'est valide qu'entre appels
   * à notifyBeginWrite() et notifyEndWrite().
   */
  virtual IDataWriter* dataWriter() = 0;

  /*!
   * \brief Positionne le nom du répertoire de sortie des fichiers.
   *
   * Ce répertoire doit exister.
   */
  virtual void setBaseDirectoryName(const String& dirname) = 0;

  //! Nom du répertoire de sortie des fichiers.
  virtual String baseDirectoryName() = 0;

  /*!
   * \brief Positionne le nom du fichier contenant les sorties.
   *
   * Tous les écrivains ne supportent pas de changer le nom
   * de fichier.
   */
  virtual void setBaseFileName(const String& filename) = 0;

  //! Nom du fichier contenant les sorties.
  virtual String baseFileName() = 0;

  /*!
   * \brief Positionne le maillage.
   *
   * Si non surchargée, cette méthode ne fait rien.
   *
   * \deprecated Cette méthode est obsolète. Il n'est plus possible
   * de changer le maillage d'un service implémentant cette interface.
   * Le choix du maillage se fait lors de la création du service via
   * ServiceBuilder en passant le maillage souhaité en argument.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Choose the mesh during service creation via ServiceBuilder")
  virtual void setMesh(IMesh* mesh);

  //! Positionne la liste des temps
  virtual void setTimes(ConstArrayView<Real> times) = 0;

  //! Liste des temps sauvés
  virtual ConstArrayView<Real> times() = 0;

  //! Positionne la liste des variables à sortir
  virtual void setVariables(const VariableCollection& variables) = 0;

  //! Liste des variables à sauver
  virtual VariableCollection variables() = 0;

  /*!
   * \brief Positionne la liste des groupes à sortir.
   *
   * La collection passée en argument est clonée.
   */
  virtual void setGroups(const ItemGroupCollection& groups) = 0;

  //! Liste des groupes à sauver
  virtual ItemGroupCollection groups() = 0;

 public:

  //! Notifie qu'une sortie va être effectuée avec les paramètres courants.
  virtual void notifyBeginWrite() = 0;

  //! Notifie qu'une sortie vient d'être effectuée.
  virtual void notifyEndWrite() = 0;

 public:

  //! Ferme l'écrivain. Après fermeture, il ne peut plus être utilisé
  virtual void close() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
