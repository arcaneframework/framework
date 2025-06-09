// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICheckpointMng.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des informations des protections.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICHECKPOINTMNG_H
#define ARCANE_CORE_ICHECKPOINTMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des informations des protections.
 *
 * Ce gestionnaire gère les informations des protections, à savoir les
 * temps protégés, les services utilisés et d'autres informations nécessaires
 * pour les reprises. Il ne gère pas directement l'écriture ni la relecture
 * qui sont déléguées à un ICheckpointReader ou ICheckpointWriter.
 *
 * La lecture d'une protection provoque la modification de toutes les variables
 * et des maillages.
 */
class ARCANE_CORE_EXPORT ICheckpointMng
{
 public:

  virtual ~ICheckpointMng() = default; //!< Libère les ressources.

 public:

  /*!
   * \brief Lit une protection.
   *
   * Cette opération est collective.
   *
   * \deprecated Utiliser readDefaultCheckpoint() à la place
   */
  ARCANE_DEPRECATED_122 virtual void readCheckpoint() = 0;

  /*!
   * \brief Lit une protection.
   *
   * Lit une protection à partir du lecture \a reader.
   */
  virtual void readCheckpoint(ICheckpointReader* reader) = 0;

  /*!
   * \brief Lit une protection.
   *
   * Lit une protection dont les infos de lecture sont dans \a infos.
   *
   * \deprecated A la place, utiliser le code suivants:
   * \code
   * ICheckpointMng* cm = ...;
   * Span<const Byte> buffer;
   * CheckpointInfo checkpoint_info = cm->readChekpointInfo(buffer);
   * cm->readChekpoint(checkpoint_info);
   * \endcode
   */
  virtual ARCANE_DEPRECATED_2018 void readCheckpoint(ByteConstArrayView infos) = 0;

  /*!
   * \brief Lit les informations d'une protection.
   *
   * Lit les informations d'une protection contenant dans le buffer \a infos.
   * \a buf_name contient le nom du buffer utilisé dans les affichages en cas d'erreur.
   */
  virtual CheckpointInfo readCheckpointInfo(Span<const Byte> infos, const String& buf_name) = 0;

  /*!
   * \brief Lit une protection.
   *
   * Lit une protection dont les infos sont dans \a checkpoint_infos.
   */
  virtual void readCheckpoint(const CheckpointInfo& checkpoint_info) = 0;

  /*!
   * \brief Lit une protection par défaut
   *
   * Cette opération est collective.
   *
   * Dans l'implémentation par défaut, les informations pour la relecture
   * sont stockées dans un fichier de nom 'checkpoint_info.xml' et qui se trouve
   * dans le répertoire d'exportation du cas (ISubDomain::exportDirectory()).
   *
   * \deprecated A la place, utiliser le code suivants:
   * \code
   * ICheckpointMng* cm = ...;
   * CheckpointInfo checkpoint_info = cm->readDefaultChekpointInfo();
   * cm->readChekpoint(checkpoint_info);
   * \endcode
   */
  virtual ARCANE_DEPRECATED_2018 void readDefaultCheckpoint() = 0;

  /*!
   * \brief Lit les informations de protection par défaut.
   *
   * Cette opération est collective.
   *
   * Dans l'implémentation par défaut, les informations pour la relecture
   * sont stockées dans un fichier de nom 'checkpoint_info.xml' et qui se trouve
   * dans le répertoire d'exportation du cas (ISubDomain::exportDirectory()).
   *
   * Après lecture des informations, il est possible d'appeler
   * readCheckpoint(const CheckpointInfo& checkpoint_info) pour lire la protection.
   */
  virtual CheckpointInfo readDefaultCheckpointInfo() = 0;

  /*!
   * \brief Écrit une protection par défaut avec l'écrivain \a writer.
   *
   * Cette opération est collective.
   *
   * \deprecated Utiliser writeDefaultCheckpoint() à la place.
   */
  ARCANE_DEPRECATED_122 virtual void writeCheckpoint(ICheckpointWriter* writer) = 0;

  /*!
   * \brief Écrit une protection avec l'écrivain \a writer.
   *
   * Cette opération est collective.
   *
   * Les informations pour pouvoir le relire sont stockées dans le tableau
   * \a infos passé en argument. Il est ensuite possible de relire une protection
   * via readCheckpoint(ByteConstArrayView).
   *
   * L'implémentation par défaut stocke dans infos un fichier XML contenant en autre
   * le nom du lecteur correspondant, le nombre de sous-domaines, ...
   */
  virtual void writeCheckpoint(ICheckpointWriter* writer, ByteArray& infos) = 0;

  /*!
   * \brief Écrit une protection avec l'écrivain \a writer.
   *
   * Cette opération est collective.
   *
   * Il s'agit d'une protection classique qui pourra être relue via readDefaultCheckpoint().
   *
   * \sa readDefaultCheckpoint
   */
  virtual void writeDefaultCheckpoint(ICheckpointWriter* writer) = 0;

  /*!
   * \brief Observable en écriture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * avant d'écrire une protection.
   */
  virtual IObservable* writeObservable() = 0;

  /*!
   * \brief Observable en lecture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * après relecture complète d'une protection.
   */
  virtual IObservable* readObservable() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

