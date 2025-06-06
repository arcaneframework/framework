// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshWriter.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service d'écriture du maillage.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHWRITER_H
#define ARCANE_CORE_IMESHWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface d'un service d'écriture d'un maillage.
 */
class IMeshWriter
{
 public:

  virtual ~IMeshWriter() {} //<! Libère les ressources

 public:

  //! Construit l'instance
  virtual void build() =0;

 public:

  /*!
   * \brief Ecrit un maillage sur un fichier.
   *
   * Le chemin du fichier de maillage doit être accessible en écriture et
   * le répertoire doit déja exiter.
   *
   * \param mesh maillage à sauver
   * \param file_name nom du fichier de maillage.
   *
   * \retval true en cas d'erreur
   * \retval false si tout est ok.
   */
  virtual bool writeMeshToFile(IMesh* mesh,const String& file_name) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

