// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICheckpointReader.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface du service de lecture d'une protection/reprise.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICHECKPOINTREADER_H
#define ARCANE_CORE_ICHECKPOINTREADER_H
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
 * \ingroup StandardService
 * \brief Interface du service de lecture d'une protection/reprise.
 */
class ARCANE_CORE_EXPORT ICheckpointReader
{
 public:

  //! Libère les ressources
  virtual ~ICheckpointReader() = default;

 public:

  //! Retourne le lecteur associé
  virtual IDataReader* dataReader() = 0;

  //! Notifie qu'une protection va être lue avec les paramètres courants
  virtual void notifyBeginRead() = 0;

  //! Notifie qu'une protection vient d'être lue
  virtual void notifyEndRead() = 0;

  //! Positionne le nom du fichier de la protection
  virtual void setFileName(const String& file_name) = 0;

  //! Nom du fichier de la protection
  virtual String fileName() const = 0;

  //! Positionne le nom du répertoire de base de la protection
  virtual void setBaseDirectoryName(const String& dirname) = 0;

  //! Nom du répertoire de base de la protection
  virtual String baseDirectoryName() const = 0;

  //! Méta données associées à ce lecteur.
  virtual void setReaderMetaData(const String&) = 0;

  //! Positionne le temps et l'indice de la protection à lire
  virtual void setCurrentTimeAndIndex(Real current_time, Integer current_index) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface du service de lecture d'une protection/reprise (V2)
 */
class ARCANE_CORE_EXPORT ICheckpointReader2
{
 public:

  //! Libère les ressources
  virtual ~ICheckpointReader2() = default;

 public:

  //! Retourne le lecteur de données associé à ce lecteur de protection
  virtual IDataReader2* dataReader() = 0;

  /*!
   * \brief Notifie qu'une protection va être lue avec les informations
   * issues de \a checkpoint_info.
   */
  virtual void notifyBeginRead(const CheckpointReadInfo& cri) = 0;

  //! Notifie de la fin de la lecture d'une protection.
  virtual void notifyEndRead() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

