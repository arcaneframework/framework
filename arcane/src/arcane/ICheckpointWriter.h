// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICheckpointWriter.h                                         (C) 2000-2007 */
/*                                                                           */
/* Interface du service d'écriture d'une protection/reprise.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ICHECKPOINTWRITER_H
#define ARCANE_ICHECKPOINTWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IDataWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface du service d'écriture d'une protection/reprise.
 *
 * L'instance doit retourner un IDataWriter (via dataWriter()) pour
 * gérer l'écriture.
 *
 * L'enchainement des fonctions est le suivant:
 * \code
 * ICheckpointWriter* checkpoint_writer = ...;
 * checkpoint_writer->setCheckpointTimes();
 * checkpoint_writer->notifyBeginWrite();
 * checkpoint_writer->dataWriter();
 * // ...
 * // Ecriture avec le IDataWriter
 * // ...
 * checkpoint_writer->notifyBeginWrite();
 * checkpoint_writer->readerServiceName();
 * checkpoint_writer->readerMetaData();
 * \endcode
 */
class ICheckpointWriter
{
 public:

  //! Libère les ressources
  virtual ~ICheckpointWriter() {}

 public:

  /*!
   * \brief Retourne l'écrivain associé.
   */
  virtual IDataWriter* dataWriter() =0;
  
  //! Notifie qu'une protection va être écrite avec les paramètres courants
  virtual void notifyBeginWrite() =0;

  //! Notifie qu'une protection vient d'être écrite
  virtual void notifyEndWrite() =0;

  //! Positionne le nom du fichier de la protection
  virtual void setFileName(const String& file_name) =0;

  //! Nom du fichier de la protection
  virtual String fileName() const =0;

  //! Positionne le nom du répertoire de base de la protection
  virtual void setBaseDirectoryName(const String& dirname) =0;

  //! Nom du répertoire de base de la protection
  virtual String baseDirectoryName() const =0;

  /*! \brief Positionne les temps des protections.
   *
   * Le temps de la protection courante est le dernier élément du tableau
   */
  virtual void setCheckpointTimes(RealConstArrayView times) =0;

  //! Temps des protections
  virtual ConstArrayView<Real> checkpointTimes() const =0;

  //! Ferme les protections
  virtual void close() =0;

  //! Nom du service du lecteur associé à cet écrivain
  virtual String readerServiceName() const =0;
  
  //! Méta données pour le lecteur associé à cet écrivain
  virtual String readerMetaData() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

